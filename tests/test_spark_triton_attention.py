"""Numerical checks for Spark's fixed-history draft attention."""

import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vllm_spark2_5_plugin.spark_triton_attention import (
    SparkTritonAttentionImpl,
    _fixed_history_attention,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class SparkTritonAttentionTest(unittest.TestCase):
    def test_fixed_history_matches_reference_for_ragged_requests(self):
        impl = SparkTritonAttentionImpl(
            8, 256, 256**-0.5, 2, None, None, "bfloat16", dspark_window=512
        )
        self.assertEqual(impl.dspark_window, 512)
        device = "cuda"
        head_size, q_heads, kv_heads, block_size = 256, 8, 2, 16
        window, query_len = 512, 8
        context_lens = [1024, 73]
        seq_lens_cpu = [n + query_len for n in context_lens]
        q_starts_cpu = [0, query_len, 2 * query_len]
        seq_lens = torch.tensor(seq_lens_cpu, device=device, dtype=torch.int32)
        q_starts = torch.tensor(q_starts_cpu, device=device, dtype=torch.int32)
        q = torch.randn(2 * query_len, q_heads, head_size, device=device, dtype=torch.bfloat16)
        out = torch.empty_like(q)

        blocks_per_req = [math.ceil(n / block_size) for n in seq_lens_cpu]
        table = torch.zeros(2, max(blocks_per_req), device=device, dtype=torch.int32)
        cache = torch.zeros(
            sum(blocks_per_req) + 1, kv_heads, block_size, 2 * head_size,
            device=device, dtype=torch.bfloat16,
        )
        source = []
        next_block = 1
        for req, seq_len in enumerate(seq_lens_cpu):
            keys = torch.randn(seq_len, kv_heads, head_size, device=device, dtype=torch.bfloat16)
            values = torch.randn_like(keys)
            source.append((keys, values))
            ids = torch.arange(next_block, next_block + blocks_per_req[req], device=device)
            table[req, :blocks_per_req[req]] = ids
            for pos in range(seq_len):
                cache[next_block + pos // block_size, :, pos % block_size, :head_size] = keys[pos]
                cache[next_block + pos // block_size, :, pos % block_size, head_size:] = values[pos]
            next_block += blocks_per_req[req]

        k_cache, v_cache = cache.transpose(1, 2).split(head_size, dim=-1)
        _fixed_history_attention[(2, query_len, q_heads)](
            out, q, k_cache, v_cache, table, seq_lens, q_starts,
            head_size**-0.5, table.stride(0),
            q.stride(0), q.stride(1), out.stride(0), out.stride(1),
            *k_cache.stride(), *v_cache.stride(),
            q_per_kv=q_heads // kv_heads, block_size=block_size,
            head_size=head_size, window=window, BLOCK_N=32,
        )
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
                        out[req * query_len + qi, head].float(), expected,
                        atol=0.04, rtol=0.04,
                    )


if __name__ == "__main__":
    unittest.main()
