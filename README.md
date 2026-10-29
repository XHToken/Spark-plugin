# vLLM Spark2_5 Plugin

> This document targets Linux systems and Bash-compatible shells. The commands
> use POSIX paths and assume a Linux vLLM environment at `.venv/`.

This package restores Spark2_5 support to vLLM as an out-of-tree general plugin.
It enables loading and serving Spark2_5 models without modifying the vLLM source
tree. The plugin registers:

- `Spark2_5ForCausalLM`, the Spark2_5 model implementation;
- `Spark2_5DSparkForCausalLM`, the Spark DSpark speculative draft model;
- the `spark2_5` Transformers configuration (`model_type: "spark2_5"`); and
- the Spark2_5 XML/KV tool-call parser, registered under the vLLM parser name
  `spark25`.

The implementation targets vLLM tag
[`v0.32.0`](https://github.com/vllm-project/vllm/tree/v0.32.0) and uses vLLM's
existing runtime layers. DSpark sliding draft layers use the plugin's Triton
JIT kernel; no separate extension build is required.

## Spark2_5 model capabilities

Spark2_5 is a causal language model that supports hybrid sliding-window/full
attention, head-wise attention output gating, tensor parallelism, pipeline
parallelism, configurable RoPE parameters, and checkpoint weight loading.

### Hybrid attention

Each layer can use either sliding-window attention or full attention. The mode
is selected from `layer_types`, with independent sliding-window and RoPE
configuration for the supported attention types.

### Head-wise attention output gating

Spark2_5 can apply a learned gate to each attention head before the heads are
merged and passed to the output projection:

```text
sigmoid(g_proj(hidden_states))
```

### Grouped-query attention and tensor parallelism

Query and KV heads are partitioned or replicated according to the tensor
parallel world size. The implementation uses vLLM's
`QKVParallelLinear`, `ColumnParallelLinear`, and `RowParallelLinear` layers.

### Pipeline parallelism

`Spark2_5ForCausalLM` implements `SupportsPP` and uses vLLM's pipeline-parallel
layer construction and `IntermediateTensors` to transfer hidden states and
residuals between pipeline stages.

### Optimized MLP and checkpoint loading

The MLP combines the gate and up projections with
`MergedColumnParallelLinear`, applies `GeluAndMul`, and uses a row-parallel
down projection. `AutoWeightsLoader` and `WeightsMapper` map Hugging Face
checkpoint weights into this packed projection layout.

### Spark DSpark integration

The DSpark implementation lives in
[`spark2_5_dspark.py`](src/vllm_spark2_5_plugin/spark2_5_dspark.py). It projects
selected target hidden states into a non-causal draft block, shares the target
embedding and LM head, and loads `draft_model.*`, `mtp.*`, or `model.mtp.*`
checkpoint weights. It supports `vanilla` and `rnn` Markov heads and an optional
confidence head.

The checkpoint must supply a positive `dspark_markov_rank` and nonempty,
unique, ascending, zero-based `dspark_target_layer_ids`. The registration hook
in `__init__.py` restores Spark after vLLM's DeepSeek fallback and normalizes
`markov_rank` and `target_layer_ids` to these DSpark field names. Draft and
target hidden sizes and full vocabularies must match.

[`spark_triton_attention.py`](src/vllm_spark2_5_plugin/spark_triton_attention.py)
implements fixed-history sliding draft attention. For target context length
`L` and draft block length `B`, every query attends to keys from
`max(0, L - sliding_window)` through `L + B - 1`, including the full non-causal
block. It uses vLLM's bundled Triton and requires dense FP16/BF16 KV and
power-of-two head dimensions. Target attention keeps vLLM's selected backend.
Draft layers default to sliding attention; `dspark_layer_types` can choose
sliding or full attention per layer. DSpark requires pipeline-parallel and
decode-context-parallel sizes of 1; the target-only pipeline-parallel support
above does not extend to DSpark.

## Linux prerequisites

Use a supported Linux Python 3.11 or newer environment with vLLM `0.32.0` installed. GPU
deployments must have a compatible NVIDIA driver and the CUDA/PyTorch runtime
required by that vLLM installation. The plugin itself adds no system packages
or compiled extensions.

From the vLLM repository root, activate the environment before running the
commands below:

```bash
source .venv/bin/activate
```

## Install

Install vLLM first, then install this package into the **same Python
environment**. The editable install is important because vLLM discovers the
plugin through the `vllm.general_plugins` entry-point metadata.

```bash
# From the vLLM checkout containing this directory.
uv pip install "vllm==0.32.0"
uv pip install -e ./Spark-plugin --no-deps
```

If the directory has been renamed to `vllm-spark2_5-plugin`, use that directory
name in the command instead. `--no-deps` prevents the plugin from replacing
the vLLM installation already selected for the environment. The vLLM parser
stack requires OpenAI Python SDK 2.25.0 or newer:

```bash
uv pip install -U "openai>=2.25.0"
```

Verify that the entry point is visible to the same interpreter used to launch
vLLM:

```bash
.venv/bin/python -c "from importlib.metadata import entry_points; \
print([(e.name, e.value) for e in entry_points(group='vllm.general_plugins')])"
```

The output should include an entry similar to:

```text
('spark2_5', 'vllm_spark2_5_plugin:register')
```

Adding `src/` to `PYTHONPATH` alone is not sufficient for normal plugin
discovery because entry points come from installed package metadata.

## First successful use

Start the OpenAI-compatible server with the parser name `spark25`:

```bash
vllm serve /path/to/spark2_5-model \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser spark25 \
  --chat-template /path/to/chat_template.jinja \
  --served-model-name Spark2_5
```

The parser name is `spark25`, not `spark2_5`: `spark2_5` identifies the model
configuration, while `spark25` is the name registered in `ToolParserManager`.

To enable Spark DSpark speculative decoding, use a Spark2_5 checkpoint that
includes DSpark draft weights (`draft_model.*` or `mtp.*`) and the corresponding
DSpark configuration:

```bash
vllm serve /path/to/spark2_5-dspark-model \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser spark25 \
  --chat-template /path/to/chat_template.jinja \
  --served-model-name Spark2_5 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7}'
```

When DSpark is enabled and the request's `temperature` is not `0.0`, we
recommend setting `draft_sample_method` to `probabilistic` so that the draft
sampler retains a probabilistic proposal distribution for rejection sampling.
The draft applies temperature; target top-k/top-p and penalties can still make
the two distributions differ. Set it in the server's
`--speculative-config`:

```text
--speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
```

The draft weights are loaded from the same checkpoint as the target model, so
no separate draft `model` path is needed. The plugin registers
`Spark2_5DSparkForCausalLM` and restores the Spark draft architecture if vLLM's
DSpark configuration selects its DeepSeek fallback. The override ensures that
the plugin's target model and tool parser are used even when vLLM already
provides Spark2_5 support.

The value `7` is an example; match `num_speculative_tokens` to the checkpoint's
`dspark_block_size` (or `block_size`) when configured. Spark DSpark currently
requires pipeline-parallel and decode-context-parallel sizes of 1.

For a small local test deployment, consider `--enforce-eager` and lower values
for `--max-model-len`, `--max-num-batched-tokens`, and
`--gpu-memory-utilization` as appropriate for the available device.

When the server starts, look for the plugin's registration log:

```text
vllm-spark2_5-plugin: registered Spark2_5ForCausalLM -> ...
```

If vLLM already provides `Spark2_5ForCausalLM`, the plugin logs a stand-down
message and leaves the in-tree implementation active. To explicitly use this
vendored implementation, set:

```bash
SPARK2_5_PLUGIN_OVERRIDE=1 vllm serve /path/to/spark2_5-model \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser spark25 \
  --chat-template /path/to/chat_template.jinja
```

## Tool-call format

The parser recognizes Spark2_5 XML/KV tool calls of this form:

```text
<tool_call>get_weather
<arg_key>city</arg_key><arg_value>Paris</arg_value>
<arg_key>days</arg_key><arg_value>3</arg_value>
</tool_call>
```

It converts declared integer, number, boolean, and JSON values to their
corresponding JSON types and returns standard vLLM `ToolCall` objects. Unknown
tool names are not emitted as tool calls.

## Performance

The following serving benchmark used 500 concurrent requests. All requests
completed successfully; hardware, model revision, and generation settings
should be recorded separately when reproducing these numbers.

| Metric | Result |
|---|---:|
| Successful requests | 500 |
| Failed requests | 0 |
| Benchmark duration | 4.70 s |
| Total input tokens | 762,489 |
| Total generated tokens | 63,478 |
| Request throughput | 106.42 req/s |
| Output token throughput | 13,510.08 tok/s |
| Peak output token throughput | 16,452.00 tok/s |
| Total token throughput | 175,791.36 tok/s |
| Peak concurrent requests | 500 |

### Latency

| Metric | Mean | Median | P99 |
|---|---:|---:|---:|
| TTFT | 2122.04 ms | 2111.73 ms | 4169.35 ms |
| TPOT | 11.58 ms | 13.39 ms | 14.13 ms |
| ITL | 12.34 ms | 11.13 ms | 32.92 ms |

These results represent a 100% request success rate, 106.42 requests/s,
approximately 13.5K output tokens/s, and 175.8K total tokens/s at peak
concurrency of 500. Decode latency had a mean TPOT of 11.58 ms and a P99 of
14.13 ms; TTFT was 2.12 s on average and 4.17 s at P99.

## Compatibility and maintenance

The model and DSpark implementations depend on vLLM's internal runtime APIs.
The current dependency baseline is vLLM tag `v0.32.0`. Install that version in
the serving environment before installing the plugin with `--no-deps`:

```bash
uv pip install "vllm==0.32.0"
```

The package metadata pins `vllm==0.32.0` and requires Python 3.11 or newer,
matching that runtime's minimum Python version. CI tests Python 3.11 and 3.12
against this pinned dependency. When upgrading vLLM, update the dependency pin
and retest model registration, tool parsing, and DSpark decoding against the
new runtime APIs.

The `TESTED_VLLM` constant in `src/vllm_spark2_5_plugin/__init__.py` records
`v0.32.0`, the current runtime baseline, and appears in registration logs
alongside the running vLLM version. Historical extraction provenance is retained
in `NOTICE` and source headers.

## Tests

The plugin smoke tests do not require a GPU or model weights. After installing
the plugin in the test environment, run:

```bash
.venv/bin/python -m pytest Spark-plugin/tests/test_smoke.py -q
```

The smoke tests verify both the public registration entry point and a complete
Spark2_5 XML tool-call parse.

[`test_spark_triton_attention.py`](tests/test_spark_triton_attention.py) checks
the fixed-history draft kernel against a PyTorch reference for requests with
different context lengths. This test requires CUDA and is skipped when CUDA
is unavailable; it is not an end-to-end DSpark decoding test.

```bash
# From the plugin repository root, in the vLLM 0.32.0 environment.
.venv/bin/python -m pytest tests/test_spark_triton_attention.py -q
```

## Plugin loading and allowlists

General plugins are loaded automatically by vLLM unless `VLLM_PLUGINS` is set
as an allowlist. If an allowlist is used, the entry-point name is `spark2_5`:

```bash
VLLM_PLUGINS=spark2_5 vllm serve /path/to/spark2_5-model \
  --tool-call-parser spark25 \
  --chat-template /path/to/chat_template.jinja
```

`VLLM_PLUGINS=spark25` is not equivalent: `spark25` is the tool-parser name, not
the general-plugin entry-point name.

## Troubleshooting

### `invalid tool call parser: spark2_5`

Use `--tool-call-parser spark25`. The plugin registers the parser under `spark25`.

### `NamespaceTool` cannot be imported from `openai.types.responses`

Upgrade the OpenAI Python SDK in the same environment that launches vLLM:

```bash
uv pip install -U "openai>=2.25.0"
```

### The model architecture is unsupported

The plugin was not discovered by the running interpreter. Reinstall it into
that environment and inspect the entry points:

```bash
uv pip install -e ./Spark-plugin --no-deps
.venv/bin/python -c "from importlib.metadata import entry_points; \
print([(e.name, e.value) for e in entry_points(group='vllm.general_plugins')])"
```

If `VLLM_PLUGINS` is set, include `spark2_5` in the allowlist.

### The plugin reports that Spark2_5 is already supported

This is expected when the installed vLLM contains the model. Set
`SPARK2_5_PLUGIN_OVERRIDE=1` only when the vendored implementation must be used.

### The plugin logs an import traceback

The entry-point loader catches plugin import failures and continues starting
vLLM. Check the complete server log for the original traceback, then compare
the running vLLM version with the current dependency baseline, `0.32.0`.

## Project layout

```text
Spark-plugin/
+-- src/vllm_spark2_5_plugin/
|   +-- __init__.py              # registration and DSpark config integration
|   +-- spark2_5.py              # Spark2_5 target model
|   +-- spark2_5_dspark.py       # DSpark draft, heads, and weight loading
|   +-- spark_triton_attention.py # fixed-history draft attention kernel
|   +-- spark2_5_config.py       # Transformers configuration
|   +-- spark2_5_tool_parser.py  # Spark2_5 XML/KV parser
+-- tests/
|   +-- test_smoke.py           # registration and parser smoke tests
|   +-- test_spark_triton_attention.py # CUDA kernel numerical test
+-- pyproject.toml
```

## License

Apache-2.0. The vendored vLLM implementation retains its upstream copyright
headers.
