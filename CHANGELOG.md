# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Restore weight loading on vLLM 0.23 while preserving the gate/up shard mapping
  and the checkpoint's already-fused QKV projection.
- Keep the `spark25` tool parser importable on vLLM releases that predate the
  shared `find_tool_name` helper.

### Changed

- Run and document the complete test suite, including packed-weight regression
  coverage for the legacy loader path.

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
