# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from torch import nn

from vllm_spark2_5_plugin.spark2_5 import Spark2_5ForCausalLM


def _parameter(values):
    return nn.Parameter(torch.tensor(values), requires_grad=False)


class _TinySparkModel(nn.Module):
    def __init__(self, calls):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=True)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        layer = self.model.layers[0]
        layer.mlp = nn.Module()
        layer.mlp.gate_up_proj = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.q_k_v_proj = nn.Module()
        self.model.embedding = nn.Module()

        gate_up = _parameter([0.0, 0.0, 0.0, 0.0])
        q_k_v = _parameter([0.0, 0.0, 0.0])
        embedding = _parameter([0.0, 0.0])

        def packed_loader(param, weight, shard_id=None):
            if shard_id is None:
                shard_id = getattr(weight, "shard_id", None)
            calls.append(("gate_up", shard_id))
            assert shard_id in (0, 1)
            start = shard_id * weight.numel()
            param.data[start : start + weight.numel()].copy_(weight)

        def direct_qkv_loader(param, weight):
            calls.append(("q_k_v", None))
            param.data.copy_(weight)

        gate_up.weight_loader = packed_loader
        q_k_v.weight_loader = direct_qkv_loader
        layer.mlp.gate_up_proj.register_parameter("weight", gate_up)
        layer.self_attn.q_k_v_proj.register_parameter("weight", q_k_v)
        self.model.embedding.register_parameter("weight", embedding)
        self.lm_head = self.model.embedding


def test_loader_preserves_spark_checkpoint_layout():
    calls = []
    model = _TinySparkModel(calls)
    weights = [
        ("model.layers.0.mlp.gate_proj.weight", torch.tensor([1.0, 2.0])),
        ("model.layers.0.mlp.up_proj.weight", torch.tensor([3.0, 4.0])),
        (
            "model.layers.0.self_attn.q_k_v_proj.weight",
            torch.tensor([5.0, 6.0, 7.0]),
        ),
        ("model.embedding.weight", torch.tensor([8.0, 9.0])),
        ("lm_head.weight", torch.tensor([98.0, 99.0])),
    ]

    loaded = Spark2_5ForCausalLM.load_weights(model, iter(weights))

    assert loaded == {
        "model.layers.0.mlp.gate_up_proj.weight",
        "model.layers.0.self_attn.q_k_v_proj.weight",
        "model.embedding.weight",
    }
    assert calls == [("gate_up", 0), ("gate_up", 1), ("q_k_v", None)]
    params = dict(model.named_parameters(remove_duplicate=False))
    assert params["model.layers.0.mlp.gate_up_proj.weight"].tolist() == [
        1.0,
        2.0,
        3.0,
        4.0,
    ]
    assert params["model.layers.0.self_attn.q_k_v_proj.weight"].tolist() == [
        5.0,
        6.0,
        7.0,
    ]
    assert params["lm_head.weight"].tolist() == [8.0, 9.0]
