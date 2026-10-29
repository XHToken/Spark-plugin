"""Numerical checks for Spark's fixed-history draft attention."""

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vllm_spark2_5_plugin.spark_triton_attention import (
    SparkTritonAttentionImpl,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class SparkTritonAttentionTest(unittest.TestCase):
    def test_fixed_history_matches_reference_for_ragged_requests(self):
        self._check_reference("bfloat16", torch.bfloat16)

    def test_fp8_e4m3_matches_reference_for_ragged_requests(self):
        if torch.cuda.get_device_capability() < (8, 9):
            self.skipTest("FP8 KV requires SM89+")
        for query_dtype in (torch.float16, torch.bfloat16):
            with self.subTest(query_dtype=query_dtype):
                self._check_reference("fp8_e4m3", query_dtype)

    def _check_reference(self, cache_dtype, query_dtype):
        torch.manual_seed(42)
        impl = SparkTritonAttentionImpl(
            8, 256, 256**-0.5, 2, None, None, cache_dtype, dspark_window=512
        )
        self.assertEqual(impl.dspark_window, 512)
        # Attention must not pre-quantize Q just because KV storage is FP8.
        self.assertFalse(impl.supports_quant_query_input)
        fp8 = cache_dtype == "fp8_e4m3"
        device = "cuda"
        head_size, q_heads, kv_heads, block_size = 256, 8, 2, 16
        window, query_len = 512, 8
        context_lens = [1024, 73]
        seq_lens_cpu = [n + query_len for n in context_lens]
        q_starts_cpu = [0, query_len, 2 * query_len]
        seq_lens = torch.tensor(seq_lens_cpu, device=device, dtype=torch.int32)
        q_starts = torch.tensor(q_starts_cpu, device=device, dtype=torch.int32)
        q = torch.randn(
            2 * query_len, q_heads, head_size, device=device, dtype=query_dtype
        )
        out = torch.empty_like(q)

        blocks_per_req = [math.ceil(n / block_size) for n in seq_lens_cpu]
        table = torch.zeros(2, max(blocks_per_req), device=device, dtype=torch.int32)
        cache = torch.zeros(
            sum(blocks_per_req) + 1,
            kv_heads,
            block_size,
            2 * head_size,
            device=device,
            dtype=torch.uint8 if fp8 else query_dtype,
        )
        # Distinct non-unit scales catch missing or swapped K/V descales.
        layer = SimpleNamespace(
            _k_scale=torch.tensor(0.125, device=device),
            _v_scale=torch.tensor(0.25, device=device),
        )
        source = []
        next_block = 1
        for req, seq_len in enumerate(seq_lens_cpu):
            keys = torch.randn(
                seq_len, kv_heads, head_size, device=device, dtype=query_dtype
            )
            values = torch.randn_like(keys)
            ids = torch.arange(
                next_block, next_block + blocks_per_req[req], device=device
            )
            table[req, : blocks_per_req[req]] = ids
            positions = torch.arange(seq_len, device=device)
            slots = (next_block * block_size + positions).to(torch.int64)
            impl.do_kv_cache_update(layer, keys, values, cache, slots)
            if fp8:
                # Compare against the independently quantized/dequantized KV,
                # separating attention correctness from FP8 rounding error.
                keys = (keys.float() / layer._k_scale).to(impl.fp8_dtype).float()
                values = (values.float() / layer._v_scale).to(impl.fp8_dtype).float()
                keys = keys * layer._k_scale
                values = values * layer._v_scale
            source.append((keys, values))
            next_block += blocks_per_req[req]

        metadata = SimpleNamespace(
            seq_lens=seq_lens,
            max_query_len=query_len,
            block_table=table,
            query_start_loc=q_starts,
        )
        impl.forward(layer, q, None, None, cache, metadata, out)
        torch.cuda.synchronize()

        for req, context_len in enumerate(context_lens):
            keys, values = source[req]
            start = max(0, context_len - window)
            self.assertEqual(start, 512 if req == 0 else 0)
            for qi in range(query_len):
                for head in range(q_heads):
                    kv_head = head // (q_heads // kv_heads)
                    query = q[req * query_len + qi, head].float()
                    scores = keys[start:, kv_head].float() @ query * head_size**-0.5
                    probs = torch.softmax(scores, dim=0)
                    expected = probs @ values[start:, kv_head].float()
                    torch.testing.assert_close(
                        out[req * query_len + qi, head].float(),
                        expected,
                        atol=0.04,
                        rtol=0.04,
                    )


if __name__ == "__main__":
    unittest.main()
