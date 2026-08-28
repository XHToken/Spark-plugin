# Third-Party Licenses

This file lists the third-party packages used by the **vLLM Spark2_5 Plugin**
project, together with the license under which each package is distributed.

The project itself is licensed under the Apache License 2.0 (see `LICENSE`).

---

## Direct Dependencies

| Package | Version / Reference | License |
|---|---|---|
| [vLLM](https://github.com/vllm-project/vllm) | commit `81efe788` (vendored) | Apache-2.0 |
| [PyTorch](https://github.com/pytorch/pytorch) | (transitive via vLLM) | BSD-3-Clause |
| [Transformers](https://github.com/huggingface/transformers) | (transitive via vLLM) | Apache-2.0 |
| [OpenAI Python SDK](https://github.com/openai/openai-python) | >= 2.25.0 | Apache-2.0 |
| [regex](https://github.com/mrabarnfield/mrab-regex) | (transitive via vLLM) | Apache-2.0 |
| [setuptools](https://github.com/pypa/setuptools) | >= 77.0.3 | MIT |
| [wheel](https://github.com/pypa/wheel) | (build dependency) | MIT |
| [pytest](https://github.com/pytest-dev/pytest) | >= 8.0 (optional, test) | MIT |

---

## Vendored Code

The following source files under `src/vllm_spark2_5_plugin/` were vendored
from the [vLLM](https://github.com/vllm-project/vllm) project at commit
[`81efe7883f30582696b69f9b9ea93c4819a8c608`](https://github.com/vllm-project/vllm/tree/81efe7883f30582696b69f9b9ea93c4819a8c608)
and are licensed under the Apache License 2.0:

- `spark2_5.py` -- Spark2_5 model implementation
- `spark2_5_config.py` -- Transformers configuration for Spark2_5
- `spark2_5_tool_parser.py` -- Spark2_5 XML/KV tool-call parser

Each vendored file carries the following SPDX headers:

```
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright contributors to the vLLM project
```

---

## Transitive Dependencies (via vLLM)

The following packages are pulled in transitively through vLLM and its
dependency tree. Only the most significant transitive dependencies are
listed here.

| Package | License |
|---|---|
| [torch](https://github.com/pytorch/pytorch) | BSD-3-Clause |
| [transformers](https://github.com/huggingface/transformers) | Apache-2.0 |
| [tokenizers](https://github.com/huggingface/tokenizers) | Apache-2.0 |
| [safetensors](https://github.com/huggingface/safetensors) | Apache-2.0 |
| [huggingface-hub](https://github.com/huggingface/huggingface_hub) | Apache-2.0 |
| [numpy](https://github.com/numpy/numpy) | BSD-3-Clause |
| [scipy](https://github.com/scipy/scipy) | BSD-3-Clause |
| [triton](https://github.com/triton-lang/triton) | MIT |
| [sentencepiece](https://github.com/google/sentencepiece) | Apache-2.0 |
| [protobuf](https://github.com/protocolbuffers/protobuf) | BSD-3-Clause |
| [pydantic](https://github.com/pydantic/pydantic) | MIT |
| [requests](https://github.com/psf/requests) | Apache-2.0 |
| [tqdm](https://github.com/tqdm/tqdm) | MIT |
| [Pillow](https://github.com/python-pillow/Pillow) | HPND (MIT-derived) |
| [packaging](https://github.com/pypa/packaging) | Apache-2.0 / BSD-2-Clause |
| [filelock](https://github.com/tox-dev/filelock) | Unlicense |
| [fsspec](https://github.com/fsspec/filesystem_spec) | BSD-3-Clause |
| [sympy](https://github.com/sympy/sympy) | BSD-3-Clause |
| [networkx](https://github.com/networkx/networkx) | BSD-3-Clause |
| [jinja2](https://github.com/pallets/jinja) | BSD-3-Clause |
| [prometheus-client](https://github.com/prometheus/client_python) | Apache-2.0 |
| [fastapi](https://github.com/fastapi/fastapi) | MIT |
| [uvicorn](https://github.com/encode/uvicorn) | BSD-3-Clause |
| [anyio](https://github.com/agronholm/anyio) | MIT |
| [httpx](https://github.com/encode/httpx) | BSD-3-Clause |
| [httptools](https://github.com/MagicStack/httptools) | MIT |
| [pyyaml](https://github.com/yaml/pyyaml) | MIT |
| [click](https://github.com/pallets/click) | BSD-3-Clause |
| [h11](https://github.com/python-hyper/h11) | MIT |
| [sniffio](https://github.com/python-trio/sniffio) | Apache-2.0 / MIT |
| [certifi](https://github.com/certifi/python-certifi) | MPL-2.0 |
| [charset-normalizer](https://github.com/ousret/charset_normalizer) | MIT |
| [idna](https://github.com/kjd/idna) | BSD-3-Clause |
| [urllib3](https://github.com/urllib3/urllib3) | MIT |
| [typing-extensions](https://github.com/python/typing_extensions) | PSF (Python) |
| [annotated-types](https://github.com/annotated-types/annotated-types) | MIT |
| [pydantic-core](https://github.com/pydantic/pydantic-core) | MIT |
| [distro](https://github.com/python-distro/distro) | Apache-2.0 |
| [tqdm](https://github.com/tqdm/tqdm) | MIT |
| [psutil](https://github.com/giampaolo/psutil) | BSD-3-Clause |

---

## Vendored Code License (Apache-2.0)

```
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

---

## BSD-3-Clause License (PyTorch, NumPy, SciPy, etc.)

```
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

---

## MIT License (setuptools, wheel, pytest, triton, pydantic, etc.)

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
