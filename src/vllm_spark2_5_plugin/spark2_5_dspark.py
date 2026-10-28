# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spark DSpark draft using the DeepSeek-V4 context/Markov/head contract.

The decoder is Spark GQA with attention-output gating and a GELU MLP. Each
draft layer attends to projected target context plus the non-causal query block.
Weights live under ``mtp.{i}.``; embedding and LM head are shared with the target.
"""

import re
from collections.abc import Iterable
from copy import copy
from typing import ClassVar

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.qwen3_dspark import (
    DSparkConfidenceHead,
    DSparkMarkovHead,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    maybe_prefix,
)

from .spark2_5 import Spark2_5DecoderLayer


class RNNSerialHead(nn.Module):
    """GRU serial head adapted to the vLLM DSpark markov-head contract."""

    def __init__(self, vocab_size: int, markov_rank: int, prefix: str,
                 quant_config=None) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        self.markov_head_type = "rnn"
        self.token_embedding = nn.Embedding(self.vocab_size, self.markov_rank)
        self.gru_cell = nn.GRUCell(self.markov_rank, self.markov_rank)
        self.output_proj = ReplicatedLinear(
            self.markov_rank, self.vocab_size, bias=False,
            quant_config=quant_config, prefix=maybe_prefix(prefix, "output_proj"),
            return_bias=False, disable_tp=True,
        )

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(token_ids.long())

    def _step(self, markov_embed: torch.Tensor) -> torch.Tensor:
        flat = markov_embed.reshape(-1, self.markov_rank)
        hidden = self.gru_cell(flat, torch.zeros_like(flat))
        return self.output_proj(hidden).view(*markov_embed.shape[:-1], self.vocab_size)

    def bias(self, markov_embed: torch.Tensor,
             logits_processor: LogitsProcessor | None = None) -> torch.Tensor:
        del logits_processor
        return self._step(markov_embed)

    def apply_bias_gathered(self, markov_embed: torch.Tensor, logits: torch.Tensor,
                            values: torch.Tensor, index: torch.Tensor,
                            scale: float = 1.0) -> torch.Tensor:
        flat = markov_embed.reshape(-1, self.markov_rank)
        hidden = self.gru_cell(flat, torch.zeros_like(flat))
        weight = self.output_proj.weight[index]
        corrected = values.unsqueeze(-1)
        corrected.baddbmm_(weight, hidden.unsqueeze(-1), beta=1.0, alpha=scale)
        return logits.scatter_(1, index, corrected.squeeze(-1))


def _validate_config(vllm_config: VllmConfig) -> tuple[int, tuple[int, ...]]:
    spec = vllm_config.speculative_config
    if spec is None or spec.method != "dspark":
        raise ValueError("Spark DSpark requires speculative method='dspark'.")
    config = spec.draft_model_config.hf_config
    target = vllm_config.model_config.hf_config
    if get_pp_group().world_size != 1:
        raise ValueError("Spark DSpark does not support pipeline parallelism.")
    if vllm_config.parallel_config.decode_context_parallel_size != 1:
        raise ValueError("Spark fixed-history Triton attention does not support DCP.")
    if target.model_type != "spark2_5":
        raise ValueError("Spark DSpark requires a Spark2_5 target.")
    for name in ("hidden_size", "vocab_size"):
        if getattr(config, name) != getattr(target, name):
            raise ValueError(f"Spark DSpark {name} must match the shared target.")
    draft_vocab_size = getattr(config, "draft_vocab_size", None)
    if draft_vocab_size is not None and draft_vocab_size != target.vocab_size:
        raise ValueError("Spark DSpark currently requires the full target vocabulary.")
    num_layers = getattr(config, "num_draft_layers", 3)
    if type(num_layers) is not int or num_layers < 1:
        raise ValueError("num_draft_layers must be a positive integer.")
    layer_ids = getattr(config, "dspark_target_layer_ids", None)
    if (
        not isinstance(layer_ids, (list, tuple))
        or not layer_ids
        or any(type(i) is not int for i in layer_ids)
        or list(layer_ids) != sorted(set(layer_ids))
        or layer_ids[0] < 0
        or layer_ids[-1] >= target.num_hidden_layers
    ):
        raise ValueError(
            "dspark_target_layer_ids must be nonempty, unique, ascending, "
            "zero-based target decoder layer IDs."
        )
    # The runner gives dflash/eagle aliases precedence over the DSpark field.
    # Reject conflicting selectors rather than silently projecting wrong features.
    from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
        get_eagle3_aux_layers_from_config,
    )

    if get_eagle3_aux_layers_from_config(spec) != tuple(i + 1 for i in layer_ids):
        raise ValueError("Conflicting Spark DSpark auxiliary layer selectors.")
    rank = getattr(config, "dspark_markov_rank", None)
    if type(rank) is not int or rank < 1:
        raise ValueError("dspark_markov_rank must be a positive integer.")
    if getattr(config, "is_causal", None) is not False:
        raise ValueError(
            "Spark DSpark requires is_causal=false on the draft config; "
            "apply the Spark DSpark speculative-config integration."
        )
    if not getattr(config, "sample_from_anchor", True) or getattr(
        config, "dspark_bonus_anchor", False
    ):
        raise ValueError("Spark DSpark requires anchor-as-first-prediction sampling.")
    block_size = getattr(config, "dspark_block_size", None)
    if block_size is None:
        block_size = getattr(config, "block_size", None)
    if block_size is not None and (
        type(block_size) is not int
        or block_size < 1
        or block_size != spec.num_speculative_tokens
    ):
        raise ValueError("num_speculative_tokens must match the DSpark block size.")
    from vllm.v1.worker.gpu.spec_decode.utils import get_parallel_drafting_token_id

    noise_id = get_parallel_drafting_token_id(config)
    if not 0 <= noise_id < config.vocab_size:
        raise ValueError("DSpark noise token must be in the shared target vocabulary.")
    return num_layers, tuple(layer_ids)


class Spark2_5DSparkModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        num_layers, self.target_layer_ids = _validate_config(vllm_config)
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.num_dspark_layers = num_layers
        self.hidden_size = config.hidden_size
        self.num_aux_hidden_states = len(self.target_layer_ids)
        use_fc_norm = getattr(config, "fc_norm", False)
        self.fc_norm = (
            nn.ModuleList(
                [
                    RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                    for _ in range(self.num_aux_hidden_states)
                ]
            )
            if use_fc_norm
            else None
        )
        # Replaced by load_dspark_model before profiling or inference. A parameter-
        # free placeholder avoids allocating/loading an unused vocabulary matrix.
        self.embed_tokens = PPMissingLayer()
        self.main_proj = ReplicatedLinear(
            config.hidden_size * len(self.target_layer_ids),
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "main_proj"),
        )
        self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        draft_types = getattr(config, "dspark_layer_types", None)
        if draft_types is None:
            draft_types = ["sliding_attention"] * num_layers
        if len(draft_types) != num_layers or any(
            kind not in ("sliding_attention", "full_attention") for kind in draft_types
        ):
            raise ValueError(
                "dspark_layer_types must specify one attention type per layer."
            )
        if "sliding_attention" in draft_types and (
            config.sliding_window is None or config.sliding_window < 1
        ):
            raise ValueError("Sliding DSpark layers require a positive sliding_window.")
        # Prefixes use global layer IDs to keep draft attention separate from target
        # layers. ModuleList/checkpoint indices remain draft-local (0..num_draft_layers).
        start = vllm_config.model_config.hf_config.num_hidden_layers
        layer_config = copy(config)
        layer_config.layer_types = ["full_attention"] * start + list(draft_types)
        layer_vllm_config = copy(vllm_config)
        layer_vllm_config.cache_config = copy(vllm_config.cache_config)
        # Attention(None) otherwise falls back to the target's global window,
        # silently making explicitly full-attention draft layers sliding too.
        layer_vllm_config.cache_config.sliding_window = None
        self.layers = nn.ModuleList(
            Spark2_5DecoderLayer(
                config=layer_config,
                vllm_config=layer_vllm_config,
                prefix=maybe_prefix(prefix, f"layers.{start + i}"),
            )
            for i in range(num_layers)
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        markov_head_type = str(getattr(config, "markov_head_type", "vanilla")).lower()
        markov_prefix = maybe_prefix(prefix, "markov_head")
        if markov_head_type == "rnn":
            self.markov_head = RNNSerialHead(
                config.vocab_size,
                config.dspark_markov_rank,
                prefix=markov_prefix,
                quant_config=vllm_config.quant_config,
            )
        elif markov_head_type in ("vanilla", ""):
            self.markov_head = DSparkMarkovHead(
                config.vocab_size,
                config.vocab_size,
                config.dspark_markov_rank,
                prefix=markov_prefix,
            )
        else:
            raise ValueError(
                "Unsupported markov_head_type for Spark DSpark: "
                f"{markov_head_type!r}; expected 'vanilla' or 'rnn'."
            )
        self.confidence_head = (
            DSparkConfidenceHead(
                config.hidden_size + config.dspark_markov_rank,
                prefix=maybe_prefix(prefix, "confidence_head"),
            )
            if getattr(config, "enable_confidence_head", True)
            else None
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        # Match Eagle3: normalize each target auxiliary hidden state before
        # concatenating them for the fusion projection.
        if self.fc_norm is not None:
            chunks = aux_hidden_states.chunk(self.num_aux_hidden_states, dim=-1)
            aux_hidden_states = torch.cat(
                [
                    norm(chunk)
                    for norm, chunk in zip(self.fc_norm, chunks, strict=True)
                ],
                dim=-1,
            )
        return self.main_norm(self.main_proj(aux_hidden_states))

    @torch.inference_mode()
    def precompute_and_store_context_kv(
        self,
        main_x: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: list[torch.Tensor | None] | None = None,
    ) -> None:
        if context_slot_mappings is not None and len(context_slot_mappings) != len(
            self.layers
        ):
            raise ValueError("Expected one context slot mapping per DSpark layer.")
        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            # Every layer projects the SAME main_x, without decoder input RMSNorm.
            # Spark has ordinary GQA K/V, so DSV4's MLA kv_norm/quant kernels do
            # not apply. Use the backend's cache writer (also handles FP8 caches).
            qkv, _ = attn.q_k_v_proj(main_x)
            q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
            _, k = attn.rotary_emb(context_positions, q, k)
            slots = None if context_slot_mappings is None else context_slot_mappings[i]
            if slots is None:
                continue
            cache = attn.core_attention
            cache.impl.do_kv_cache_update(
                cache,
                k.reshape(-1, attn.num_kv_heads, attn.head_dim).contiguous(),
                v.reshape(-1, attn.num_kv_heads, attn.head_dim).contiguous(),
                cache.kv_cache,
                slots,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = (
            self.embed_input_ids(input_ids) if inputs_embeds is None else inputs_embeds
        )
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        # Confidence consumes pre-norm states; only the vocabulary head uses norm.
        return hidden_states + residual


class Spark2_5DSparkForCausalLM(nn.Module):
    has_own_embed_tokens = False
    has_own_lm_head = False
    draft_id_to_target_id = None
    packed_modules_mapping: ClassVar[dict[str, list[str]]] = {
        "q_k_v_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".q_proj": (".q_k_v_proj", "q"),
            ".k_proj": (".q_k_v_proj", "k"),
            ".v_proj": (".q_k_v_proj", "v"),
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = Spark2_5DSparkModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(aux_hidden_states)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [
            layer.self_attn.core_attention.layer_name for layer in self.model.layers
        ]

    def get_draft_attn_causal(self) -> list[bool]:
        return [False] * len(self.model.layers)

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: list[torch.Tensor | None] | None = None,
    ) -> None:
        self.model.precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mappings
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, self.model.norm(hidden_states))

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.compute_logits(hidden_states)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embed, self.logits_processor)

    def compute_confidence(
        self, head_hidden: torch.Tensor, markov_embed: torch.Tensor
    ) -> torch.Tensor:
        if self.model.confidence_head is None:
            raise ValueError("This Spark DSpark checkpoint has no confidence head.")
        return torch.sigmoid(self.model.confidence_head(head_hidden, markov_embed))

    def _remap_dspark_name(self, name: str) -> str | None:
        # Checkpoints produced by the Spark-X25 exporter use
        # ``draft_model.*``; older merged checkpoints use ``mtp.*``.
        match = re.fullmatch(r"(?:model\.)?mtp\.(\d+)\.(.+)", name)
        if match is not None:
            stage, rest = int(match[1]), match[2]
        else:
            match = re.fullmatch(r"draft_model\.(.+)", name)
            if match is None:
                if name.startswith(("mtp.", "model.mtp.", "draft_model.")):
                    raise ValueError(f"Invalid DSpark weight name: {name}")
                return None
            stage = None
            rest = match[1]

        if stage is None:
            # The exported branch calls the projection ``fc`` and the input
            # normalization ``hidden_norm``. Map them to the equivalent names
            # used by the out-of-tree vLLM implementation.
            if rest == "fc.weight":
                return "model.main_proj.weight"
            if rest == "hidden_norm.weight":
                return "model.main_norm.weight"
            if rest == "norm.weight":
                return "model.norm.weight"
            if rest == "embed_tokens.weight":
                return None  # shared with the target embedding
            fc_norm_match = re.fullmatch(r"fc_norm\.(\d+)\.weight", rest)
            if fc_norm_match is not None:
                index = int(fc_norm_match[1])
                if not 0 <= index < self.model.num_aux_hidden_states:
                    raise ValueError(
                        f"DSpark fc_norm index exceeds auxiliary states: {name}"
                    )
                return f"model.fc_norm.{index}.weight"
            if rest.startswith("fc_norm."):
                raise ValueError(f"Invalid DSpark fc_norm weight name: {name}")
            layer_match = re.fullmatch(r"layers\.(\d+)\.(.+)", rest)
            if layer_match is not None:
                stage = int(layer_match[1])
                if not 0 <= stage < len(self.model.layers):
                    raise ValueError(f"DSpark weight stage exceeds num_draft_layers: {name}")
                return (
                    f"model.layers.{stage}.{layer_match[2]}"
                )
            return f"model.{rest}"
        if not 0 <= stage < len(self.model.layers):
            raise ValueError(f"DSpark weight stage exceeds num_draft_layers: {name}")
        if rest.startswith("confidence_head.") and self.model.confidence_head is None:
            return None
        if rest.startswith(
            ("main_proj.", "main_norm.", "norm.", "markov_head.", "confidence_head.")
        ):
            return f"model.{rest}"
        return f"model.layers.{stage}.{rest}"

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded_confidence = False
        found_draft = False

        def draft_weights():
            nonlocal loaded_confidence, found_draft
            for name, weight in weights:
                mapped = self._remap_dspark_name(name)
                if mapped is None:
                    continue
                found_draft = True
                loaded_confidence |= mapped.startswith("model.confidence_head.")
                yield mapped, weight

        loaded = AutoWeightsLoader(self).load_weights(
            draft_weights(), mapper=self.hf_to_vllm_mapper
        )
        if not found_draft:
            raise ValueError("No mtp.* Spark DSpark weights found in the checkpoint.")
        if not loaded_confidence:
            self.model.confidence_head = None
        return loaded
