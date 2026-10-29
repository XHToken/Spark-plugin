# SPDX-License-Identifier: Apache-2.0
"""Fixed-history paged attention for Spark DSpark draft layers."""

from __future__ import annotations

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
)

logger = init_logger(__name__)


@triton.jit
def _fixed_history_attention(
    out_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    table_ptr,
    seq_lens_ptr,
    q_starts_ptr,
    scale,
    table_stride,
    q_stride0,
    q_stride1,
    o_stride0,
    o_stride1,
    k_stride0,
    k_stride1,
    k_stride2,
    k_stride3,
    v_stride0,
    v_stride1,
    v_stride2,
    v_stride3,
    q_per_kv: tl.constexpr,
    block_size: tl.constexpr,
    head_size: tl.constexpr,
    window: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    req = tl.program_id(0)
    q_local = tl.program_id(1)
    q_head = tl.program_id(2)
    q_start = tl.load(q_starts_ptr + req)
    q_end = tl.load(q_starts_ptr + req + 1)
    if q_local >= q_end - q_start:
        return
    token = q_start + q_local
    seq_len = tl.load(seq_lens_ptr + req)
    context_len = seq_len - (q_end - q_start)
    # Keep exactly ``window`` context tokens, then include the full draft
    # block. The draft query positions are not charged against this window.
    history_start = tl.maximum(0, context_len - window)
    kv_head = q_head // q_per_kv
    d = tl.arange(0, head_size)
    q = tl.load(q_ptr + token * q_stride0 + q_head * q_stride1 + d).to(tl.float32)
    m = tl.full((), float("-inf"), tl.float32)
    l = tl.zeros((), tl.float32)
    acc = tl.zeros((head_size,), tl.float32)
    first_tile = (history_start // BLOCK_N) * BLOCK_N
    for start in range(first_tile, seq_len, BLOCK_N):
        pos = start + tl.arange(0, BLOCK_N)
        valid = pos < seq_len
        # ``tl.where`` wants a boolean predicate; comparing again yields an
        # explicit int1 instead of the int32 chain that Triton deprecates.
        keep = (valid & (pos >= history_start)) != 0
        block = tl.load(
            table_ptr + req * table_stride + pos // block_size,
            mask=valid,
            other=0,
        ).to(tl.int64)
        slot = pos % block_size
        k = tl.load(
            k_ptr
            + block[:, None] * k_stride0
            + slot[:, None] * k_stride1
            + kv_head * k_stride2
            + d[None, :] * k_stride3,
            mask=keep[:, None],
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            v_ptr
            + block[:, None] * v_stride0
            + slot[:, None] * v_stride1
            + kv_head * v_stride2
            + d[None, :] * v_stride3,
            mask=keep[:, None],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(k * q[None, :], axis=1) * scale
        scores = tl.where(keep, scores, float("-inf"))
        m_new = tl.maximum(m, tl.max(scores, axis=0))
        p = tl.exp(scores - m_new)
        alpha = tl.exp(m - m_new)
        l = l * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m = m_new
    tl.store(
        out_ptr + token * o_stride0 + q_head * o_stride1 + d,
        acc / l,
    )


class SparkTritonAttentionImpl(TritonAttentionImpl):
    def __init__(self, *args, dspark_window: int, **kwargs):
        super().__init__(*args, **kwargs)
        if dspark_window < 1:
            raise ValueError("dspark_window must be positive")
        if self.head_size & (self.head_size - 1):
            raise ValueError(
                "Spark fixed-history Triton attention requires power-of-two head size"
            )
        self.dspark_window = dspark_window
        logger.info_once(
            "Using Spark fixed-history Triton draft attention (window=%d)",
            dspark_window,
        )

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        """Write the draft K/V into an Ascend ``(k_cache, v_cache)`` pair.

        vLLM's Triton implementation only understands its own packed layout, so
        on NPU the update must go through the Ascend cache-update operator.
        """
        if isinstance(kv_cache, (tuple, list)) or (
            current_platform.device_type == "npu"
            and isinstance(kv_cache, torch.Tensor)
            and kv_cache.dim() == 5
            and kv_cache.shape[0] == 2
        ):
            from vllm_ascend.device.device_op import DeviceOperator

            DeviceOperator.reshape_and_cache(
                key=key,
                value=value,
                key_cache=kv_cache[0],
                value_cache=kv_cache[1],
                slot_mapping=slot_mapping,
                use_bnsd=False,
            )
            return
        return super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        if attn_metadata is None:
            return output.fill_(0)
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "Spark Triton path does not support output quantization"
            )
        # The DSpark draft queries must land in the KV cache *before* the
        # non-causal attention below reads "fixed history + full query block".
        # On Ascend the forward context carries no slot_mapping dict, so vLLM's
        # pre-backend ``unified_kv_cache_update`` is silently a no-op there;
        # the backend therefore claims the update itself
        # (``forward_includes_kv_cache_update``). Only real tokens are written,
        # graph padding is skipped.
        if key is not None and value is not None:
            num_actual_tokens = attn_metadata.num_actual_tokens
            self.do_kv_cache_update(
                layer,
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                kv_cache,
                attn_metadata.slot_mapping[:num_actual_tokens],
            )
        # GPU (vLLM): ``kv_cache`` is a single packed tensor with logical shape
        #   (num_blocks, num_kv_heads, block_size, 2 * head_size).
        # NPU (vllm-ascend): the runner reshapes the cache itself and hands over
        #   a ``(k_cache, v_cache)`` pair, each
        #   (num_blocks, block_size, num_kv_heads, head_size),
        # because it queries ``get_kv_cache_shape`` on the backend class.
        if isinstance(kv_cache, (tuple, list)):
            k_cache, v_cache = kv_cache[0], kv_cache[1]
        elif (
            current_platform.device_type == "npu"
            and isinstance(kv_cache, torch.Tensor)
            and kv_cache.dim() == 5
            and kv_cache.shape[0] == 2
        ):
            # Some vllm-ascend versions may hand over the whole
            # (2, blocks, block, kv_heads, head) tensor instead of a pair.
            k_cache, v_cache = kv_cache[0], kv_cache[1]
        else:
            cache = kv_cache.transpose(1, 2)
            k_cache, v_cache = cache.split(self.head_size, dim=-1)
        if (
            query.dtype not in (torch.float16, torch.bfloat16)
            or k_cache.dtype != query.dtype
            or v_cache.dtype != query.dtype
        ):
            raise NotImplementedError("Spark Triton path requires dense FP16/BF16 KV")
        if self._is_per_token_head_quant:
            raise NotImplementedError("Spark Triton path does not support quantized KV")
        _fixed_history_attention[
            (
                attn_metadata.seq_lens.shape[0],
                attn_metadata.max_query_len,
                self.num_heads,
            )
        ](
            output,
            query,
            k_cache,
            v_cache,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            attn_metadata.query_start_loc,
            self.scale,
            attn_metadata.block_table.stride(0),
            query.stride(0),
            query.stride(1),
            output.stride(0),
            output.stride(1),
            *k_cache.stride(),
            *v_cache.stride(),
            q_per_kv=self.num_queries_per_kv,
            block_size=k_cache.shape[1],
            head_size=self.head_size,
            window=self.dspark_window,
            BLOCK_N=32,
        )
        return output


class SparkTritonAttentionBackend(TritonAttentionBackend):
    # The plugin owns the draft K/V write. vLLM's Triton backend inherits
    # ``False`` here and relies on the pre-backend ``unified_kv_cache_update``,
    # which needs ``forward_context.slot_mapping`` -- a mapping vllm-ascend does
    # not populate, so the update would be skipped without any error.
    forward_includes_kv_cache_update = True

    @staticmethod
    def get_name() -> str:
        return "TRITON_ATTN"

    @staticmethod
    def get_impl_cls():
        return SparkTritonAttentionImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "",
    ) -> tuple[int, ...]:
        """KV cache shape used to reshape this layer's cache.

        vLLM's GPU model runner derives the layout from the ``AttentionSpec``
        and never calls this; vllm-ascend's NPU model runner *does* call it and
        expects the Ascend-style K/V separated 5-D shape.  The Spark kernel
        consumes ``(num_blocks, block_size, num_kv_heads, head_size)`` K and V
        tensors, which is exactly ``shape[1:]`` of the Ascend layout, so NBSD is
        requested explicitly (regardless of ``VLLM_KV_CACHE_LAYOUT``).
        """
        if current_platform.device_type == "npu":
            return (2, num_blocks, block_size, num_kv_heads, head_size)
        return (num_blocks, num_kv_heads, block_size, 2 * head_size)
