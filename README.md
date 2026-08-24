# Spark-plugin
Spark-plugin for Spark‑3 model, enables seamless loading and serving Spark‑3 large language model within vLLM framework. Implements custom model plugin to support weights loading, tokenizer and inference runtime without modifying original vLLM source code.

# Spark3 Model
Spark3 adopts a causal language model architecture, supporting hybrid sliding‑window/full attention, head‑wise attention output gating, tensor parallelism, pipeline parallelism, RoPE configuration, and checkpoint weight loading.

## Key Technical Details

1. **Hybrid Attention**

    Spark3 supports per-layer hybrid attention consisting of sliding-window attention and full attention. The attention mode is selected according to `layer_types`, with configurable sliding-window size and RoPE parameters for different attention types.

2. **Head-wise Attention Output Gating**

    Spark3 optionally applies a learned head-wise gate to the attention output:

    `sigmoid(g_proj(hidden_states))`

    before merging the attention heads and applying the output projection.

3. **Grouped-Query Attention and Tensor Parallelism**

    Query and KV heads are partitioned or replicated according to the tensor-parallel world size using vLLM's `QKVParallelLinear`, `ColumnParallelLinear`, and `RowParallelLinear` implementations.

4. **Pipeline Parallelism**

    `Spark3ForCausalLM` implements `SupportsPP` and uses vLLM's pipeline-parallel layer construction and `IntermediateTensors` mechanism for transferring hidden states and residuals between pipeline stages.

5. **Optimized MLP**

    The MLP uses a merged gate/up projection with `MergedColumnParallelLinear`, followed by `GeluAndMul` and a row-parallel down projection.

6. **Checkpoint Loading**

    Spark3 uses `AutoWeightsLoader` and `WeightsMapper` to map Hugging Face checkpoint weights into vLLM's packed projection layout, including the merged gate/up projection.
