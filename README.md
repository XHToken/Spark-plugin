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

## Performance

A serving benchmark was performed with **500 concurrent requests**, and all requests completed successfully.

|Metric|Result|
|---|--:|
|Successful requests|500|
|Failed requests|0|
|Benchmark duration|4.70 s|
|Total input tokens|762,489|
|Total generated tokens|63,478|
|Request throughput|106.42 req/s|
|Output token throughput|13,510.08 tok/s|
|Peak output token throughput|16,452.00 tok/s|
|Total token throughput|175,791.36 tok/s|
|Peak concurrent requests|500|

### Latency

|Metric|Mean|Median|P99|
|---|--:|--:|--:|
|TTFT|2122.04 ms|2111.73 ms|4169.35 ms|
|TPOT|11.58 ms|13.39 ms|14.13 ms|
|ITL|12.34 ms|11.13 ms|32.92 ms|

The benchmark completed with a **100% request success rate**. Under a peak concurrency of 500 requests, the server achieved **106.42 requests/s**, approximately **13.5K output tokens/s**, and **175.8K total tokens/s**.

Decode latency remained relatively stable, with a mean TPOT of **11.58 ms** and a P99 TPOT of **14.13 ms**. TTFT was approximately **2.12 s on average** and **4.17 s at P99** under this high-concurrency workload.