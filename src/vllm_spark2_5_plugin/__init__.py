# SPDX-License-Identifier: Apache-2.0
"""Out-of-tree Spark2_5 support for vLLM.

The entry point imports only the standard library.  vLLM loads general plugins in
the API process, engine core, and worker processes, so registration must stay
cheap, lazy, and safe to call more than once.
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy

__all__ = [
    "ARCHITECTURE",
    "DRAFT_ARCHITECTURE",
    "DRAFT_MODEL_TARGET",
    "MODEL_TARGET",
    "OVERRIDE_ENV",
    "TESTED_VLLM",
    "TESTED_VLLM_ASCEND",
    "register",
]

logger = logging.getLogger(__name__)

ARCHITECTURE = "Spark2_5ForCausalLM"
MODEL_TARGET = f"{__name__}.spark2_5:{ARCHITECTURE}"
DRAFT_ARCHITECTURE = "Spark2_5DSparkForCausalLM"
DRAFT_MODEL_TARGET = f"{__name__}.spark2_5_dspark:{DRAFT_ARCHITECTURE}"
TESTED_VLLM = "v0.29.0"
TESTED_VLLM_ASCEND = "v0.29.0rc1"
OVERRIDE_ENV = "SPARK2_5_PLUGIN_OVERRIDE"

_TRUE = frozenset({"1", "true", "yes", "on"})


def _override_requested() -> bool:
    """Return whether the operator explicitly requested an override."""
    return os.environ.get(OVERRIDE_ENV, "").strip().lower() in _TRUE


def _register_config() -> None:
    """Expose Spark2_5Config through vLLM's model-type registry."""
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    from .spark2_5_config import Spark2_5Config

    _CONFIG_REGISTRY["spark2_5"] = Spark2_5Config


def _register_tool_parser() -> None:
    """Register the Spark2_5 parser without importing it until first use."""
    from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

    # If an upstream parser was already resolved before this entry point ran,
    # remove the eager cache so an explicit plugin override takes effect too.
    ToolParserManager.tool_parsers.pop("spark25", None)
    ToolParserManager.register_lazy_module(
        name="spark25",
        module_path=f"{__name__}.spark2_5_tool_parser",
        class_name="Spark2_5ToolParser",
    )


def _patch_dspark_dispatch() -> None:
    """Keep Spark DSpark configs out of vLLM's DeepSeek fallback branch.

    vLLM 0.29.0 rewrites every unrecognised DSpark architecture to
    ``DSparkDraftModel`` during ``SpeculativeConfig.__post_init__``. The
    plugin is loaded before engine configuration, so wrapping that hook lets
    us restore Spark after the upstream validation has completed.
    """
    from vllm.config.speculative import SpeculativeConfig

    if getattr(SpeculativeConfig, "_spark2_5_dspark_patched", False):
        return

    original_post_init = SpeculativeConfig.__post_init__

    def post_init(self):
        # For DSpark configured without an explicit ``model``, vLLM creates
        # ``draft_model_config`` inside ``original_post_init``. Capture the
        # target Spark config before that happens; inspecting
        # ``self.draft_model_config`` here returns None in that path.
        target_model_config = getattr(self, "target_model_config", None)
        target_hf_config = getattr(target_model_config, "hf_config", None)
        original_hf_config = None
        is_spark = target_hf_config is not None and (
            getattr(target_hf_config, "model_type", None) == "spark2_5"
            or "Spark2_5ForCausalLM" in getattr(target_hf_config, "architectures", ())
        )
        if is_spark:
            # DSpark without an external model aliases target_model_config, so
            # this snapshot also protects the target config from in-place edits.
            original_hf_config = deepcopy(target_hf_config)

        original_post_init(self)

        if original_hf_config is None or self.method != "dspark":
            return
        draft = self.draft_model_config
        current_hf = getattr(draft, "hf_config", None)
        if getattr(
            current_hf, "model_type", None
        ) != "deepseek_v4" or "DSparkDraftModel" not in getattr(
            current_hf, "architectures", ()
        ):
            return

        # Spark checkpoints use the native names ``markov_rank`` and
        # ``target_layer_ids``. The DSpark runner expects the normalized names.
        if not hasattr(original_hf_config, "dspark_markov_rank"):
            markov_rank = getattr(original_hf_config, "markov_rank", None)
            if markov_rank is not None:
                original_hf_config.dspark_markov_rank = markov_rank
        if not hasattr(original_hf_config, "dspark_target_layer_ids"):
            target_layer_ids = getattr(original_hf_config, "target_layer_ids", None)
            if target_layer_ids is not None:
                original_hf_config.dspark_target_layer_ids = target_layer_ids
        original_hf_config.is_causal = False
        original_hf_config.architectures = [DRAFT_ARCHITECTURE]
        draft.hf_config = original_hf_config
        # Restore the text/architecture snapshots that update_arch_ derives
        # from hf_config and make ModelRegistry resolve the plugin class.
        self.update_arch_()
        logger.info(
            "vllm-spark2_5-plugin: restored native Spark DSpark draft "
            "architecture %s after vLLM fallback detection",
            DRAFT_ARCHITECTURE,
        )

    SpeculativeConfig.__post_init__ = post_init
    SpeculativeConfig._spark2_5_dspark_patched = True


def register() -> None:
    """Register Spark2_5 with vLLM's ``vllm.general_plugins`` entry point."""
    import vllm
    from vllm import ModelRegistry

    _patch_dspark_dispatch()

    if (
        DRAFT_ARCHITECTURE not in ModelRegistry.get_supported_archs()
        or _override_requested()
    ):
        ModelRegistry.register_model(DRAFT_ARCHITECTURE, DRAFT_MODEL_TARGET)

    if (
        ARCHITECTURE in ModelRegistry.get_supported_archs()
        and not _override_requested()
    ):
        logger.info(
            "vllm-spark2_5-plugin: %s is already supported in tree; standing down. "
            "Set %s=1 to override.",
            ARCHITECTURE,
            OVERRIDE_ENV,
        )
        return

    _register_config()
    _register_tool_parser()
    ModelRegistry.register_model(ARCHITECTURE, MODEL_TARGET)
    logger.info(
        "vllm-spark2_5-plugin: registered %s -> %s "
        "(tested against vLLM %s / vllm-ascend %s, running %s)",
        ARCHITECTURE,
        MODEL_TARGET,
        TESTED_VLLM,
        TESTED_VLLM_ASCEND,
        vllm.__version__,
    )
