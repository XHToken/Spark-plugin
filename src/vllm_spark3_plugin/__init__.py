# SPDX-License-Identifier: Apache-2.0
"""Out-of-tree Spark3 support for vLLM.

The entry point imports only the standard library.  vLLM loads general plugins in
the API process, engine core, and worker processes, so registration must stay
cheap, lazy, and safe to call more than once.
"""

from __future__ import annotations

import logging
import os

__all__ = [
    "ARCHITECTURE",
    "MODEL_TARGET",
    "OVERRIDE_ENV",
    "TESTED_VLLM",
    "register",
]

logger = logging.getLogger(__name__)

ARCHITECTURE = "Spark3ForCausalLM"
MODEL_TARGET = f"{__name__}.spark3:{ARCHITECTURE}"
TESTED_VLLM = "81efe7883f30582696b69f9b9ea93c4819a8c608"
OVERRIDE_ENV = "SPARK3_PLUGIN_OVERRIDE"

_TRUE = frozenset({"1", "true", "yes", "on"})


def _override_requested() -> bool:
    """Return whether the operator explicitly requested an override."""
    return os.environ.get(OVERRIDE_ENV, "").strip().lower() in _TRUE


def _register_config() -> None:
    """Expose Spark3Config through vLLM's model-type registry."""
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    from .spark3_config import Spark3Config

    _CONFIG_REGISTRY["spark3"] = Spark3Config


def _register_tool_parser() -> None:
    """Register the Spark3 parser without importing it until first use."""
    from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

    # If an upstream parser was already resolved before this entry point ran,
    # remove the eager cache so an explicit plugin override takes effect too.
    ToolParserManager.tool_parsers.pop("spark", None)
    ToolParserManager.register_lazy_module(
        name="spark",
        module_path=f"{__name__}.spark3_tool_parser",
        class_name="Spark3ToolParser",
    )


def register() -> None:
    """Register Spark3 with vLLM's ``vllm.general_plugins`` entry point."""
    import vllm
    from vllm import ModelRegistry

    if (
        ARCHITECTURE in ModelRegistry.get_supported_archs()
        and not _override_requested()
    ):
        logger.info(
            "vllm-spark3-plugin: %s is already supported in tree; standing down. "
            "Set %s=1 to override.",
            ARCHITECTURE,
            OVERRIDE_ENV,
        )
        return

    _register_config()
    _register_tool_parser()
    ModelRegistry.register_model(ARCHITECTURE, MODEL_TARGET)
    logger.info(
        "vllm-spark3-plugin: registered %s -> %s "
        "(tested against vLLM %s, running %s)",
        ARCHITECTURE,
        MODEL_TARGET,
        TESTED_VLLM,
        vllm.__version__,
    )
