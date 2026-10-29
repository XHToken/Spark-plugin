# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **DSpark draft model**: `spark2_5_dspark.py` provides `Spark2_5DSparkForCausalLM`, shared target embedding/LM head, auxiliary hidden-state projection, `vanilla`/`rnn` Markov heads, an optional confidence head, and draft-weight loading from `draft_model.*`, `mtp.*`, or `model.mtp.*` checkpoints. `__init__.py` registers the draft and restores Spark speculative configuration after the DeepSeek fallback, normalizing `markov_rank` and `target_layer_ids` to their DSpark field names.
- **DSpark attention**: `spark_triton_attention.py` implements Triton JIT fixed-history sliding draft attention over the context window and the full non-causal draft block. Draft layers default to sliding attention, with per-layer sliding/full attention selection through `dspark_layer_types`. It requires dense FP16/BF16 KV, power-of-two head dimensions, pipeline-parallel size 1, and decode-context-parallel size 1.
- **DSpark numerical test**: `tests/test_spark_triton_attention.py` compares ragged-request draft attention with a PyTorch reference on CUDA; it is skipped when CUDA is unavailable.

### Changed

- **vLLM dependency baseline**: Model and DSpark integration now target release tag [`v0.29.0`](https://github.com/vllm-project/vllm/tree/v0.29.0). `TESTED_VLLM` and registration logs identify the runtime tag; vendored file headers retain the source commit as provenance. The package metadata pins `vllm==0.29.0`; Ascend deployments use vllm-ascend `0.29.0rc1`.
- **Target model integration**: Expose auxiliary hidden states and shared embeddings for the vLLM V2 GPU DSpark runner, and exclude `draft_model.*` weights when loading the target model.
- **Tool parser compatibility**: Update chat utility and streaming response protocol imports for vLLM `0.29.0`.
- **Documentation**: Expand the README with DSpark checkpoint requirements, serving examples, sampling recommendations, parallelism restrictions, and CUDA test instructions; update third-party license documentation for the runtime baseline and DSpark/Triton additions.

### Fixed

- **Ascend DSpark integration**: Support separate and stacked NPU KV caches, write draft query K/V before attention, and accept both legacy and current vLLM parser import paths. Include the Ascend patch and adaptation guide in source distributions.

- **Transformers configuration validation**: Assign Spark's per-attention-type RoPE parameters after base configuration validation so the generic flat-parameter validator does not reject them.

## [0.1.0] - 2026-08-28

### Added

- **Spark2_5 vLLM Plugin**: Out-of-tree plugin that restores Spark2_5 model support to vLLM, registering `Spark2_5ForCausalLM`, the `spark2_5` Transformers config, and the Spark2_5 XML/KV tool-call parser
- **Vendored Model Implementation**: Spark2_5 model files vendored from vLLM commit [`81efe78`](https://github.com/vllm-project/vllm/tree/81efe7883f30582696b69f9b9ea93c4819a8c608), including hybrid sliding-window/full attention, head-wise attention gating, grouped-query attention, tensor and pipeline parallelism
- **Tool-Call Parser**: Spark2_5 XML/KV parser registered under the vLLM parser name `spark25`, with typed argument extraction (integer, number, boolean, JSON)
- **Project Governance**: LICENSE (Apache-2.0), NOTICE, THIRD_PARTY_LICENSES.md, CODE-OF-CONDUCT.md, SECURITY.md
- **GitHub Templates**: PR template (`.github/PULL_REQUEST_TEMPLATE.md`), Dependabot config for pip and github-actions
- **CI Pipeline**: `.github/workflows/check.yml` with lint (ruff), test (pytest, Python 3.10–3.12 matrix), and build jobs
- **Smoke Tests**: GPU-free tests covering plugin registration and tool-call parsing
- **Documentation**: Comprehensive README with install guide, usage, benchmarks, and troubleshooting

### Changed

- **Package Rename**: Renamed from Spark3 to Spark2_5 (`45a7d43`)
- **CI Actions**: Bumped `actions/checkout` to v7, `actions/upload-artifact` to v7, `actions/setup-python` to v7

[Unreleased]: https://github.com/XHToken/Spark-plugin/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/XHToken/Spark-plugin/releases/tag/v0.1.0
