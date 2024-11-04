├── .github
    ├── CODEOWNERS
    └── workflows
    │   ├── e2e_tests.yaml
    │   ├── release.yaml
    │   ├── scripts
    │       └── create_release.js
    │   └── unit_tests.yaml
├── .gitignore
├── AUTHORS
├── CONTRIBUTING.md
├── LICENSE
├── MANIFEST.in
├── Makefile
├── README.md
├── benchmarks
    ├── README.md
    ├── __init__.py
    ├── benchmark_serving.py
    ├── eval_accuracy.py
    ├── open_orca_gpt4_tokenized_llama.calibration_1000.pkl
    └── requirements.in
├── docs
    ├── observability-prometheus-metrics-in-jetstream-server.md
    ├── online-inference-with-maxtext-engine.md
    └── profiling-with-jax-profiler-and-tensorboard.md
├── jetstream
    ├── __init__.py
    ├── core
    │   ├── README.md
    │   ├── __init__.py
    │   ├── config_lib.py
    │   ├── implementations
    │   │   ├── __init__.py
    │   │   └── mock
    │   │   │   ├── README.md
    │   │   │   ├── __init__.py
    │   │   │   ├── config.py
    │   │   │   └── server.py
    │   ├── metrics
    │   │   ├── __init__.py
    │   │   └── prometheus.py
    │   ├── orchestrator.py
    │   ├── proto
    │   │   ├── __init__.py
    │   │   ├── jetstream.proto
    │   │   ├── jetstream_pb2.py
    │   │   └── jetstream_pb2_grpc.py
    │   ├── server_lib.py
    │   └── utils
    │   │   ├── __init__.py
    │   │   ├── async_multifuture.py
    │   │   ├── proxy_util.py
    │   │   └── return_sample.py
    ├── engine
    │   ├── README.md
    │   ├── __init__.py
    │   ├── engine_api.py
    │   ├── mock_engine.py
    │   ├── mock_utils.py
    │   ├── sampling_utils.py
    │   ├── token_utils.py
    │   ├── tokenizer.proto
    │   ├── tokenizer_api.py
    │   ├── tokenizer_pb2.py
    │   ├── tokenizer_pb2_grpc.py
    │   └── warmup_utils.py
    ├── entrypoints
    │   ├── __init__.py
    │   ├── config.py
    │   └── http
    │   │   ├── __init__.py
    │   │   ├── api_server.py
    │   │   ├── protocol.py
    │   │   └── utils.py
    ├── tests
    │   ├── __init__.py
    │   ├── core
    │   │   ├── __init__.py
    │   │   ├── test_config_lib.py
    │   │   ├── test_orchestrator.py
    │   │   └── test_server.py
    │   ├── engine
    │   │   ├── __init__.py
    │   │   ├── test_mock_engine.py
    │   │   ├── test_sampling_utils.py
    │   │   ├── test_token_utils.py
    │   │   ├── test_utils.py
    │   │   └── third_party
    │   │   │   ├── llama2
    │   │   │       └── tokenizer.model
    │   │   │   └── llama3
    │   │   │       └── tokenizer.model
    │   └── entrypoints
    │   │   ├── __init__.py
    │   │   └── http
    │   │       ├── __init__.py
    │   │       └── test_api_server.py
    ├── third_party
    │   ├── __init__.py
    │   └── llama3
    │   │   ├── __init__.py
    │   │   └── llama3_tokenizer.py
    └── tools
    │   ├── load_tester.py
    │   ├── maxtext
    │       ├── model_ckpt_conversion.sh
    │       └── model_ckpt_finetune_with_aqt.sh
    │   ├── proxy_dev
    │       ├── base.Dockerfile
    │       └── dev.Dockerfile
    │   └── requester.py
├── license_preamble.txt
├── pylintrc
├── requirements.txt
└── setup.py


/.github/CODEOWNERS:
-----------------------

* @JoeZijunZhou
* @vipannalla


-----------------------

/.github/workflows/e2e_tests.yaml:
-----------------------

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Details for GCP Service Account Key setup: https://github.com/google-github-actions/auth?tab=readme-ov-file#sake

name: E2E Tests

on:
  push:
    branches: [main]

permissions:
  id-token: write
  contents: read

jobs:
  run_e2e_tests:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v3

      - name: GCP SA auth
        id: 'auth'
        uses: 'google-github-actions/auth@v2'
        with:
          credentials_json: '${{ secrets.GOOGLE_SA_KEY }}'

      - name: Setup gcloud
        uses: 'google-github-actions/setup-gcloud@v2'
        with:
          version: '>= 363.0.0'

      - name: Run E2E tests on ml-automation-solutions platform
        run: |
          gcloud composer environments run ml-automation-solutions \
            --project=cloud-ml-auto-solutions \
            --location=us-central1 dags trigger \
            -- \
            jetstream_e2e_inference


-----------------------

/.github/workflows/release.yaml:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# This workflow will create a release with git tag and publish a Python Package in PyPI.
# For more information see: https://help.github.com/en/actions/language-and-framework-guides/using-python-with-github-actions; https://docs.pypi.org/trusted-publishers/adding-a-publisher/

name: Create Release

on:
  push:
    tags:
      - v*

# Needed to create release and upload assets
permissions:
  contents: write

jobs:
  release:
    name: Create Release with tag
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-20.04]
        python-version: ['3.10']
    steps:
    - name: Checkout
      uses: actions/checkout@v4
    - name: Extract release tag from branch
      shell: bash
      run: |
        echo "release_tag=${GITHUB_REF#refs/*/}" >> $GITHUB_ENV
    - name: Create Github Release
      id: create_release
      uses: "actions/github-script@v6"
      env:
        RELEASE_TAG: ${{ env.release_tag }}
      with:
        github-token: "${{ secrets.GITHUB_TOKEN }}"
        script: |
          const script = require('.github/workflows/scripts/create_release.js')
          await script(github, context, core)

  pypi:
    name: Build and Publish JetStream Python Package
    runs-on: ${{ matrix.os }}
    needs: release
    strategy:
      matrix:
        os: [ubuntu-20.04]
        python-version: ['3.10']
    environment:
      name: pypi
      url: https://pypi.org/project/google-jetstream
    permissions:
      id-token: write  # IMPORTANT: this permission is mandatory for trusted publishing
    steps:
    - name: Checkout
      uses: actions/checkout@v4
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - name: Build Python distribution package
      run: |
        python -m pip install --upgrade build && python -m build
    - name: Publish Python package to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1


-----------------------

/.github/workflows/scripts/create_release.js:
-----------------------

/**
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Uses Github's API to create the release and wait for result.
// We use a JS script since github CLI doesn't provide a way to wait for the release's creation and returns immediately.

module.exports = async (github, context, core) => {
	try {
		const response = await github.rest.repos.createRelease({
			draft: false,
			generate_release_notes: true,
			name: process.env.RELEASE_TAG,
			owner: context.repo.owner,
			prerelease: false,
			repo: context.repo.repo,
			tag_name: process.env.RELEASE_TAG,
		});

		core.setOutput('upload_url', response.data.upload_url);
	} catch (error) {
		core.setFailed(error.message);
	}
}

-----------------------

/.github/workflows/unit_tests.yaml:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This workflow will install Python dependencies, run tests and lint with a variety of Python versions
# For more information see: https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python

name: Unit Tests

on:
  pull_request:
  push:
    branches: [ "main" ]
  workflow_dispatch:
  schedule:
    # Run the job every 12 hours
    - cron:  '0 */12 * * *'

jobs:
  py:
    name: "Python type/lint/format checks"
    strategy:
      matrix:
        os: [ubuntu-20.04]
        python-version: ['3.10']
    runs-on: ${{ matrix.os }}
    steps:
    - name: Checkout
      uses: actions/checkout@v4
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install Dependencies
      run: make install-deps
    - name: Typecheck the code with pytype
      run: make type-check
    - name: Analysing the code with pylint
      run: make linter-check
    - name: Format check with pyink
      run: make format-check

  cpu:
    name: "JetStream unit tests"
    strategy:
      matrix:
        os: [ubuntu-20.04]
        python-version: ['3.10']
    runs-on: ${{ matrix.os }}
    steps:
    - name: Checkout
      uses: actions/checkout@v4
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install Dependencies
      run: make install-deps
    - name: Run all unit tests in JetStream (jetstream/tests)
      run: make unit-tests
    - name: Create test coverage report
      run: make check-test-coverage

-----------------------

/.gitignore:
-----------------------

__pycache__
.env*
build/
dist/
google_jetstream.egg-info/
.coverage

# local folders
data/
logs/
tmp/
venv/
.vscode/


-----------------------

/AUTHORS:
-----------------------

Google LLC

-----------------------

/CONTRIBUTING.md:
-----------------------

# How to Contribute

We'd love to accept your patches and contributions to this project.

## Before you begin

### Sign our Contributor License Agreement

Contributions to this project must be accompanied by a
[Contributor License Agreement](https://cla.developers.google.com/about) (CLA).
You (or your employer) retain the copyright to your contribution; this simply
gives us permission to use and redistribute your contributions as part of the
project.

If you or your current employer have already signed the Google CLA (even if it
was for a different project), you probably don't need to do it again.

Visit <https://cla.developers.google.com/> to see your current agreements or to
sign a new one.

### Review our Community Guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).

## Contribution process

### Code Reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

-----------------------

/LICENSE:
-----------------------


                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.


-----------------------

/MANIFEST.in:
-----------------------

include requirements.txt

-----------------------

/Makefile:
-----------------------

PYTHON := python
PIP := $(PYTHON) -m pip
GRPC_TOOLS_VERSION := 1.62.1

all: install-deps generate-protos format check

# Dependency management targets
install-deps:
	$(PIP) install pytype pylint pyink -r requirements.txt -r benchmarks/requirements.in

# Code generation/formatting targets
generate-protos: generate-and-prepend-preambles format

generate-and-prepend-preambles:
	$(PIP) install grpcio-tools==$(GRPC_TOOLS_VERSION)
	for id in $$(find . -name "*.proto"); do \
		$(PYTHON) -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. $$id && \
		PROTO_FILE=$$(echo $$id | awk '{print substr($$0, 1, length($$0)-6)}') && \
		PB_GRPC_PY=$(addsuffix "_pb2_grpc.py",$$PROTO_FILE) && \
		PB_PY=$(addsuffix "_pb2.py",$$PROTO_FILE) && \
		cat license_preamble.txt $$PB_GRPC_PY >> $(addsuffix "_temp",$$PB_GRPC_PY) && \
		mv $(addsuffix "_temp",$$PB_GRPC_PY) $$PB_GRPC_PY; \
		cat license_preamble.txt $$PB_PY >> $(addsuffix "_temp",$$PB_PY) && \
		mv $(addsuffix "_temp",$$PB_PY) $$PB_PY; \
	done

format:
	$(PIP) install pyink
	pyink --pyink-indentation 2 --line-length 80 --verbose .

# Code checking related targets
check: type-check format-check linter-check

type-check:
	$(PIP) install pytype
	pytype --jobs auto --disable=import-error,module-attr jetstream/ benchmarks/

format-check:
	$(PIP) install pyink
	pyink --pyink-indentation 2 --line-length 80 --check --verbose .

linter-check:
	$(PIP) install pylint
	pylint --ignore-patterns=".*_pb2.py,.*_pb2_grpc.py" jetstream/ benchmarks/


# Testing related targets
tests: unit-tests check-test-coverage

unit-tests:
	coverage run -m unittest -v

check-test-coverage:
	coverage report -m --omit="jetstream/core/proto/*,jetstream/engine/tokenizer_pb2.py,jetstream/third_party/*" --fail-under=96


-----------------------

/README.md:
-----------------------

[![Unit Tests](https://github.com/google/JetStream/actions/workflows/unit_tests.yaml/badge.svg?branch=main)](https://github.com/google/JetStream/actions/workflows/unit_tests.yaml?query=branch:main)
[![PyPI version](https://badge.fury.io/py/google-jetstream.svg)](https://badge.fury.io/py/google-jetstream)
[![PyPi downloads](https://img.shields.io/pypi/dm/google-jetstream?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/google-jetstream/)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)

# JetStream is a throughput and memory optimized engine for LLM inference on XLA devices.

## About

JetStream is a throughput and memory optimized engine for LLM inference on XLA devices, starting with TPUs (and GPUs in future -- PRs welcome).

## JetStream Engine Implementation 

Currently, there are two reference engine implementations available -- one for Jax models and another for Pytorch models.

### Jax

- Git: https://github.com/google/maxtext
- README: https://github.com/google/JetStream/blob/main/docs/online-inference-with-maxtext-engine.md

### Pytorch

- Git: https://github.com/google/jetstream-pytorch 
- README: https://github.com/google/jetstream-pytorch/blob/main/README.md 

## Documentation
- [Online Inference with MaxText on v5e Cloud TPU VM](https://cloud.google.com/tpu/docs/tutorials/LLM/jetstream) [[README](https://github.com/google/JetStream/blob/main/docs/online-inference-with-maxtext-engine.md)]
- [Online Inference with Pytorch on v5e Cloud TPU VM](https://cloud.google.com/tpu/docs/tutorials/LLM/jetstream-pytorch) [[README](https://github.com/google/jetstream-pytorch/tree/main?tab=readme-ov-file#jetstream-pytorch)]
- [Serve Gemma using TPUs on GKE with JetStream](https://cloud.google.com/kubernetes-engine/docs/tutorials/serve-gemma-tpu-jetstream)
- [Benchmark JetStream Server](https://github.com/google/JetStream/blob/main/benchmarks/README.md)
- [Observability in JetStream Server](https://github.com/google/JetStream/blob/main/docs/observability-prometheus-metrics-in-jetstream-server.md)
- [Profiling in JetStream Server](https://github.com/google/JetStream/blob/main/docs/profiling-with-jax-profiler-and-tensorboard.md)
- [JetStream Standalone Local Setup](#jetstream-standalone-local-setup)


# JetStream Standalone Local Setup

## Getting Started

### Setup
```
make install-deps
```

### Run local server & Testing

Use the following commands to run a server locally:
```
# Start a server
python -m jetstream.core.implementations.mock.server

# Test local mock server
python -m jetstream.tools.requester

# Load test local mock server
python -m jetstream.tools.load_tester

```

### Test core modules
```
# Test JetStream core orchestrator
python -m unittest -v jetstream.tests.core.test_orchestrator

# Test JetStream core server library
python -m unittest -v jetstream.tests.core.test_server

# Test mock JetStream engine implementation
python -m unittest -v jetstream.tests.engine.test_mock_engine

# Test mock JetStream token utils
python -m unittest -v jetstream.tests.engine.test_token_utils
python -m unittest -v jetstream.tests.engine.test_utils

```


-----------------------

/benchmarks/README.md:
-----------------------

# JetStream Benchmark And Eval

## Install Dependencies

```
cd ~/JetStream/benchmarks
pip install -r requirements.in
```

## Benchmark with shareGPT

### Prepare DataSet

```
cd ~/data
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

```

### Run Benchmark with maxtext tokenizer

```
python benchmark_serving.py \
--tokenizer /home/{username}/maxtext/assets/tokenizer \
--num-prompts 10  \
--dataset sharegpt \
--dataset-path ~/data/ShareGPT_V3_unfiltered_cleaned_split.json \
--max-output-length 1024

```

### Run Benchmark for Llama 3

```
python benchmark_serving.py \
--tokenizer <llama3 tokenizer path> \
--num-prompts 10  \
--dataset sharegpt \
--dataset-path ~/data/ShareGPT_V3_unfiltered_cleaned_split.json \
--max-output-length 1024 \
--model llama-3

```

### Save request outputs in Benchmark

Please use `--save-request-outputs` flag to save predictions to a file.

```
python benchmark_serving.py \
--tokenizer /home/{username}/maxtext/assets/tokenizer \
--num-prompts 10  \
--dataset sharegpt \
--dataset-path ~/data/ShareGPT_V3_unfiltered_cleaned_split.json \
--max-output-length 1024  \
--save-request-outputs

```

### Automatically run evaluation after Benchmark

To automatically evaluate the outputs against the ROUGE evaluation metric, add the `--run-eval true` flag.
Note: If `--save-result` is used, the evaluation scores will be saved as well.

```
python benchmark_serving.py \
--tokenizer /home/{username}/maxtext/assets/tokenizer \
--num-prompts 10  \
--dataset sharegpt \
--dataset-path ~/data/ShareGPT_V3_unfiltered_cleaned_split.json \
--max-output-length 1024  \
--save-request-outputs \
--run-eval true

```

## Benchmark with openorca dataset (openorca is used by MLPerf inference for LLaMA2 models)
```
python JetStream/benchmarks/benchmark_serving.py   \
--tokenizer ~/maxtext/assets/tokenizer.llama2  \
--warmup-mode sampled   \
--save-result   \
--save-request-outputs   \
--request-outputs-file-path outputs.json   \
--num-prompts 1000   \
--max-output-length 1024   \
--dataset openorca

```

## Benchmark warmup mode

The benchmark has better performance if it first conducts a warmup of the JetStream server. We currently support `sampled` and `full` warmup modes. `full` mode would warmup up the JetStream server with all the input requests. `sampled` mode would warmup up the JetStream server with a sampling of the input requests across different bucket sizes of input lengths.

Example to run benchmark with `full` warmup mode:
```
python JetStream/benchmarks/benchmark_serving.py   \
--tokenizer ~/maxtext/assets/tokenizer.llama2  \
--warmup-mode full   \
--save-result   \
--save-request-outputs   \
--request-outputs-file-path outputs.json   \
--num-prompts 1000   \
--max-output-length 1024   \
--dataset openorca
```

## Standalone Evaluation Run

If you used `--save-request-outputs`, you can separately evaluate against the generated outputs.

```
python eval_accuracy.py outputs.json

```

With openorca dataset and llama2-chat models (used by MLPerf), here are the reference accuracy numbers:
```
llama2-7b-chat {'rouge1': 42.0706, 'rouge2': 19.8021, 'rougeL': 26.8474, 'rougeLsum': 39.5952, 'gen_len': 1146679, 'gen_num': 998}
llama2-70b-chat {'rouge1': 44.4312, 'rouge2': 22.0352, 'rougeL': 28.6162}
``` 

-----------------------

/benchmarks/__init__.py:
-----------------------



-----------------------

/benchmarks/benchmark_serving.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark JetStream online serving.

On the server side, run one of the following commands:
    * For real server, you need to pass correct server config (include the
      model config that being passed into your engine impl) to the command
      below. Refer to config_lib.py and implementations/mock/config.py for
      config impl detail.

    (run with real server)
    python -m jetstream.core.implementations.<your_impl>.server \
        --config <your_server_config>

    (run with mock server)
    python -m jetstream.core.implementations.mock.server

On the client side, run:
    * For real server and shareGPT dataset, you need to pass the tokenizer,
      server config, and dataset flags to the command below, and make some
      changes to the tokenizer logic in the benchmark script (get_tokenizer
      and sample_requests func) to use your tokenizer correctly.
    * Add `--save-result` flag to save the benchmark result to a json file in
      current folder.
    * You can also add `--run_eval true` if you want to calculate ROUGE score
      on the predicted outputs.

    (run with real model and engines)
    python -m benchmarks.benchmark_serving \
        --tokenizer <your_tokenizer> \
        --dataset <target_dataset_name> \
        --dataset-path <target_dataset_path> \
        --request-rate <request_rate>

    (run with mock)
    python -m benchmarks.benchmark_serving \
        --request-rate 1

e2e example:
python3 benchmark_serving.py \
    --tokenizer /home/{username}/maxtext/assets/tokenizer \
    --num-prompts 100 \
    --dataset sharegpt \
    --dataset-path ~/ShareGPT_V3_unfiltered_cleaned_split.json

"""


import argparse
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
import json
import random
import time
from typing import Any, AsyncGenerator, Optional
import os


import grpc
from jetstream.core.proto import jetstream_pb2
from jetstream.core.proto import jetstream_pb2_grpc
from jetstream.engine.token_utils import load_vocab
from jetstream.third_party.llama3 import llama3_tokenizer
import numpy as np
from tqdm.asyncio import tqdm  # pytype: disable=pyi-error
import pandas

from eval_accuracy import eval_accuracy


def str2bool(v: str) -> bool:
  """Convert a string of truth to True or False.

  Args:
    - v (str):
      - True values are 'y', 'yes', 't', 'true', and '1';
      - False values are 'n', 'no', 'f', 'false', and '0'.

  Returns:
    bool: True or False

  Raises:
    ValueError if v is anything else.
  """
  v = v.lower()
  true_values = ["y", "yes", "t", "true", "1"]
  false_values = ["n", "no", "f", "false", "0"]
  if v in true_values:
    return True
  elif v in false_values:
    return False
  else:
    raise ValueError(f"Invalid value '{v}'!")


@dataclass
class BenchmarkMetrics:
  """Data class to store benchmark metrics."""

  completed: int
  total_input: int
  total_output: int
  request_throughput: float
  input_throughput: float
  output_throughput: float
  mean_ttft_ms: float
  median_ttft_ms: float
  p99_ttft_ms: float
  mean_tpot_ms: float
  median_tpot_ms: float
  p99_tpot_ms: float


@dataclass
class InputRequest:
  prompt: str = ""
  prompt_len: int = 0
  output: str = ""
  output_len: int = 0
  sample_idx: int = -1


@dataclass
class RequestFuncOutput:
  input_request: Optional[InputRequest] = None
  generated_token_list: list[str] = field(default_factory=list)
  generated_text: str = ""
  success: bool = False
  latency: float = 0
  ttft: float = 0
  prompt_len: int = 0

  # Flatten the structure and return only the necessary results
  def to_dict(self):
    return {
        "prompt": self.input_request.prompt,
        "original_output": self.input_request.output,
        "generated_text": self.generated_text,
        "success": self.success,
        "latency": self.latency,
        "prompt_len": self.prompt_len,
        "sample_idx": self.input_request.sample_idx,
    }


def get_tokenizer(model_id: str, tokenizer_name: str) -> Any:
  """Return a tokenizer or a tokenizer placholder."""
  if tokenizer_name == "test":
    return "test"
  elif model_id == "llama-3":
    # Llama 3 uses a tiktoken tokenizer.
    return llama3_tokenizer.Tokenizer(tokenizer_name)
  else:
    # Use JetStream tokenizer util. It's using the sentencepiece wrapper in
    # seqio library.
    vocab = load_vocab(tokenizer_name)
    return vocab.tokenizer


def load_sharegpt_dataset(
    dataset_path: str,
    conversation_starter: str,
) -> list[tuple[Any, Any]]:
  # Load the dataset.
  with open(dataset_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)
  # Filter out the conversations with less than 2 turns.
  dataset = [data for data in dataset if len(data["conversations"]) >= 2]

  # Filter based on conversation starter
  if conversation_starter != "both":
    dataset = [
        data
        for data in dataset
        if data["conversations"][0]["from"] == conversation_starter
    ]
  # Only keep the first two turns of each conversation.
  dataset = [
      (data["conversations"][0]["value"], data["conversations"][1]["value"])
      for data in dataset
  ]

  return dataset


def load_openorca_dataset_pkl():
  # read pickle file
  samples = pandas.read_pickle(
      os.path.join(
          os.path.dirname(os.path.relpath(__file__)),
          "open_orca_gpt4_tokenized_llama.calibration_1000.pkl",
      )
  )

  prompts = []
  outputs = []
  for _, row in samples.iterrows():
    prompts.append(row["input"])
    outputs.append(row["output"])

  return [(prompt, output) for prompt, output in zip(prompts, outputs)]


def tokenize_dataset(
    dataset: list[tuple[Any, Any, Any]],
    tokenizer: Any,
) -> list[tuple[str, Any, str, int, int, int]]:

  n = len(dataset)

  prompts = []
  outputs = []
  indices = []
  prompt_token_ids = []
  outputs_token_ids = []
  for prompt, output, idx in dataset:
    prompts.append(prompt)
    outputs.append(output)
    indices.append(idx)
    prompt_token_ids.append(tokenizer.encode(prompt))
    outputs_token_ids.append(tokenizer.encode(output))

  tokenized_dataset = []
  for i in range(n):
    prompt_len = len(prompt_token_ids[i])
    output_len = len(outputs_token_ids[i])
    tokenized_data = (
        prompts[i],
        prompt_token_ids[i],
        outputs[i],
        prompt_len,
        output_len,
        indices[i],
    )
    tokenized_dataset.append(tokenized_data)
  return tokenized_dataset


def filter_dataset(
    tokenized_dataset: list[tuple[str, Any, str, int, int, int]],
    max_output_length: int = 0,
) -> list[InputRequest]:
  if max_output_length != 0:
    print("In InputRequest, pass in actual output_length for each sample")
  else:
    print(
        f"In InputRequest, pass in max_output_length: {max_output_length} for"
        " each sample"
    )

  # Filter out too long sequences.
  filtered_dataset: list[InputRequest] = []
  for (
      prompt,
      _,
      output,
      prompt_len,
      output_len,
      sample_idx,
  ) in tokenized_dataset:
    if prompt_len < 4 or output_len < 4:
      # Prune too short sequences.
      # This is because TGI causes errors when the input or output length
      # is too short.
      continue
    if prompt_len > 1024 or prompt_len + output_len > 2048:
      # Prune too long sequences.
      continue
    request = InputRequest(
        prompt, prompt_len, output, max_output_length or output_len, sample_idx
    )
    filtered_dataset.append(request)

  print(f"The dataset contains {len(tokenized_dataset)} samples.")
  print(f"The filtered dataset contains {len(filtered_dataset)} samples.")

  return filtered_dataset


def sample_requests(
    dataset: list[tuple[Any, Any]],
    tokenizer: Any,
    num_requests: int,
    max_output_length: int = 0,
    oversample_multiplier: float = 1.2,
) -> list[InputRequest]:

  # Original dataset size
  n = len(dataset)
  dataset_indices = range(n)

  # Create necessary number of requests even if bigger than dataset size
  sampled_indices = random.sample(
      dataset_indices, min(int(num_requests * oversample_multiplier), n)
  )

  if num_requests > len(sampled_indices):
    print(
        f"Number of requests {num_requests} is larger than size of dataset"
        f" {n}.\n",
        "Repeating data to meet number of requests.\n",
    )
    sampled_indices = sampled_indices * int(
        np.ceil(num_requests / len(sampled_indices))
    )

  print(f"{len(sampled_indices)=}")
  # some of these will be filtered out, so sample more than we need

  sampled_dataset = []
  for i in sampled_indices:
    sampled_data = dataset[i] + (dataset_indices[i],)
    sampled_dataset.append(sampled_data)

  tokenized_dataset = tokenize_dataset(sampled_dataset, tokenizer)

  input_requests = filter_dataset(tokenized_dataset, max_output_length)

  # Sample the requests.
  if len(input_requests) > num_requests:
    input_requests = random.sample(input_requests, num_requests)

  return input_requests


async def get_request(
    input_requests: list[InputRequest],
    request_rate: float,
) -> AsyncGenerator[InputRequest, None]:
  input_requests = iter(input_requests)
  for request in input_requests:
    yield request

    if request_rate == 0.0:
      # If the request rate is infinity, then we don't need to wait.
      continue
    # Sample the request interval from the exponential distribution.
    interval = np.random.exponential(1.0 / request_rate)
    # The next request will be sent after the interval.
    await asyncio.sleep(interval)


def calculate_metrics(
    input_requests: list[InputRequest],
    outputs: list[RequestFuncOutput],
    dur_s: float,
    tokenizer: Any,
) -> BenchmarkMetrics:
  total_output = 0
  total_input = 0
  completed = 0
  per_token_latencies = []
  ttfts = []
  for i in range(len(outputs)):
    if outputs[i].success:
      output_len = len(
          outputs[i].generated_token_list
          if tokenizer != "test"
          else ["Ċ", "Ō", "Ɵ"]
      )
      total_output += output_len
      total_input += input_requests[i].prompt_len
      if output_len == 0:
        print(
            f"""-------- output_len is zero for {i}th request:,
             output: {outputs[i]}"""
        )
        continue
      per_token_latencies.append(outputs[i].latency / output_len)
      ttfts.append(outputs[i].ttft)
      completed += 1

  metrics = BenchmarkMetrics(
      completed=completed,
      total_input=total_input,
      total_output=total_output,
      request_throughput=completed / dur_s,
      input_throughput=total_input / dur_s,
      output_throughput=total_output / dur_s,
      mean_ttft_ms=float(np.mean(ttfts) * 1000),
      median_ttft_ms=float(np.median(ttfts) * 1000),
      p99_ttft_ms=float(np.percentile(ttfts, 99) * 1000),
      mean_tpot_ms=float(np.mean(per_token_latencies) * 1000),
      median_tpot_ms=float(np.median(per_token_latencies) * 1000),
      p99_tpot_ms=float(np.percentile(per_token_latencies, 99) * 1000),
  )

  return metrics


async def grpc_async_request(
    api_url: str, request: Any
) -> tuple[list[str], float, float]:
  """Send grpc synchronous request since the current grpc server is sync."""
  options = [("grpc.keepalive_timeout_ms", 10000)]
  async with grpc.aio.insecure_channel(api_url, options=options) as channel:
    stub = jetstream_pb2_grpc.OrchestratorStub(channel)
    print("Making request")
    ttft = 0
    token_list = []
    request_start_time = time.perf_counter()
    response = stub.Decode(request)
    async for resp in response:
      if ttft == 0:
        ttft = time.perf_counter() - request_start_time
      token_list.extend(resp.stream_content.samples[0].token_ids)
    latency = time.perf_counter() - request_start_time
    return token_list, ttft, latency


async def send_request(
    api_url: str,
    tokenizer: Any,
    input_request: InputRequest,
    pbar: tqdm,
) -> RequestFuncOutput:
  """Send the request to JetStream server."""
  # Tokenization on client side following MLPerf standard.
  token_ids = tokenizer.encode(input_request.prompt)
  request = jetstream_pb2.DecodeRequest(
      token_content=jetstream_pb2.DecodeRequest.TokenContent(
          token_ids=token_ids
      ),
      max_tokens=input_request.output_len,
  )
  output = RequestFuncOutput()
  output.input_request = input_request
  output.prompt_len = input_request.prompt_len
  generated_token_list, ttft, latency = await grpc_async_request(
      api_url, request
  )
  output.ttft = ttft
  output.latency = latency
  output.generated_token_list = generated_token_list
  # generated_token_list is a list of token ids, decode it to generated_text.
  output.generated_text = tokenizer.decode(generated_token_list)
  output.success = True
  if pbar:
    pbar.update(1)
  return output


async def benchmark(
    api_url: str,
    tokenizer: Any,
    input_requests: list[InputRequest],
    request_rate: float,
    disable_tqdm: bool,
):
  """Benchmark the online serving performance."""
  pbar = None if disable_tqdm else tqdm(total=len(input_requests))

  print(f"Traffic request rate: {request_rate}")

  benchmark_start_time = time.perf_counter()
  tasks = []
  async for request in get_request(input_requests, request_rate):
    tasks.append(
        asyncio.create_task(
            send_request(
                api_url=api_url,
                tokenizer=tokenizer,
                input_request=request,
                pbar=pbar,
            )
        )
    )
  outputs = await asyncio.gather(*tasks)

  if not disable_tqdm and pbar:
    pbar.close()

  benchmark_duration = time.perf_counter() - benchmark_start_time

  metrics = calculate_metrics(
      input_requests=input_requests,
      outputs=outputs,
      dur_s=benchmark_duration,
      tokenizer=tokenizer,
  )

  print(f"Successful requests: {metrics.completed}")
  print(f"Benchmark duration: {benchmark_duration:2f} s")
  print(f"Total input tokens: {metrics.total_input}")
  print(f"Total generated tokens: {metrics.total_output}")
  print(f"Request throughput: {metrics.request_throughput:.2f} requests/s")
  print(f"Input token throughput: {metrics.input_throughput:.2f} tokens/s")
  print(f"Output token throughput: {metrics.output_throughput:.2f} tokens/s")
  print(f"Mean TTFT: {metrics.mean_ttft_ms:.2f} ms")
  print(f"Median TTFT: {metrics.median_ttft_ms:.2f} ms")
  print(f"P99 TTFT: {metrics.p99_ttft_ms:.2f} ms")
  print(f"Mean TPOT: {metrics.mean_tpot_ms:.2f} ms")
  print(f"Median TPOT: {metrics.median_tpot_ms:.2f} ms")
  print(f"P99 TPOT: {metrics.p99_tpot_ms:.2f} ms")

  result = {
      "duration": benchmark_duration,
      "completed": metrics.completed,
      "total_input_tokens": metrics.total_input,
      "total_output_tokens": metrics.total_output,
      "request_throughput": metrics.request_throughput,
      "input_throughput": metrics.input_throughput,
      "output_throughput": metrics.output_throughput,
      "mean_ttft_ms": metrics.mean_ttft_ms,
      "median_ttft_ms": metrics.median_ttft_ms,
      "p99_ttft_ms": metrics.p99_ttft_ms,
      "mean_tpot_ms": metrics.mean_tpot_ms,
      "median_tpot_ms": metrics.median_tpot_ms,
      "p99_tpot_ms": metrics.p99_tpot_ms,
  }
  return result, outputs


def mock_requests(total_mock_requests: int):
  """Generates a list of mock requests containing mock data."""
  data = []
  for _ in range(total_mock_requests):
    reqeust = InputRequest()
    reqeust.prompt = f"Prompt {random.randint(1, 1000)}"
    reqeust.prompt_len = random.randint(10, 100)
    reqeust.out = f"Output {random.randint(1, 1000)}"
    reqeust.output_len = random.randint(1, 10)
    data.append(reqeust)
  return data


def sample_warmup_requests(requests):
  interesting_buckets = [
      0,
      16,
      32,
      64,
      128,
      256,
      512,
      1024,
  ]

  for start, end in zip(interesting_buckets[:-1], interesting_buckets[1:]):
    for request in requests:
      if start < request.prompt_len <= end:
        yield request
        break


def main(args: argparse.Namespace):
  print(args)
  random.seed(args.seed)
  np.random.seed(args.seed)

  model_id = args.model
  tokenizer_id = args.tokenizer

  api_url = f"{args.server}:{args.port}"

  tokenizer = get_tokenizer(model_id, tokenizer_id)
  if tokenizer == "test" or args.dataset == "test":
    input_requests = mock_requests(
        args.total_mock_requests
    )  # e.g. [("AB", 2, "AB", 3)]
  else:
    dataset = []
    if args.dataset == "openorca":
      dataset = load_openorca_dataset_pkl()
    elif args.dataset == "sharegpt":
      dataset = load_sharegpt_dataset(
          args.dataset_path,
          args.conversation_starter,
      )

    # A given args.max_output_length value is the max generation step,
    # when the args.max_output_length is default to None, the sample's golden
    # output length will be used to decide the generation step.
    input_requests = sample_requests(
        dataset=dataset,
        tokenizer=tokenizer,
        num_requests=args.num_prompts,
        max_output_length=args.max_output_length,
    )

  warmup_requests = None
  if args.warmup_mode == "full":
    warmup_requests = input_requests
  elif args.warmup_mode == "sampled":
    warmup_requests = list(sample_warmup_requests(input_requests)) * 2

  if warmup_requests:
    print(f"Starting {args.warmup_mode} warmup:")
    benchmark_result, request_outputs = asyncio.run(
        benchmark(
            api_url=api_url,
            tokenizer=tokenizer,
            input_requests=warmup_requests,
            request_rate=args.request_rate,
            disable_tqdm=args.disable_tqdm,
        )
    )
    print(f"{args.warmup_mode} warmup completed.")

  # TODO: Replace this with warmup complete signal once supported.
  # Wait for server completely warmup before running the benchmark.
  time.sleep(5)

  benchmark_result, request_outputs = asyncio.run(
      benchmark(
          api_url=api_url,
          tokenizer=tokenizer,
          input_requests=input_requests,
          request_rate=args.request_rate,
          disable_tqdm=args.disable_tqdm,
      )
  )

  # Process output
  output = [output.to_dict() for output in request_outputs]
  if args.run_eval:
    eval_json = eval_accuracy(output)

  # Save config and results to json
  if args.save_result:
    # dimensions values are strings
    dimensions_json = {}
    # metrics values are numerical
    metrics_json = {}

    # Setup
    current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    dimensions_json["date"] = current_dt
    dimensions_json["model_id"] = model_id
    dimensions_json["tokenizer_id"] = tokenizer_id
    if args.additional_metadata_metrics_to_save is not None:
      dimensions_json = {
          **dimensions_json,
          **json.loads(args.additional_metadata_metrics_to_save),
      }
    metrics_json["num_prompts"] = args.num_prompts

    # Traffic
    metrics_json["request_rate"] = args.request_rate
    metrics_json = {**metrics_json, **benchmark_result}
    if args.run_eval:
      metrics_json = {**metrics_json, **eval_json}

    final_json = {}
    final_json["metrics"] = metrics_json
    final_json["dimensions"] = dimensions_json

    # Save to file
    base_model_id = model_id.split("/")[-1]
    file_name = (
        f"JetStream-{args.request_rate}qps-{base_model_id}-{current_dt}.json"
    )
    with open(file_name, "w", encoding="utf-8") as outfile:
      json.dump(final_json, outfile)

  if args.save_request_outputs:
    file_path = args.request_outputs_file_path
    with open(file_path, "w", encoding="utf-8") as output_file:
      json.dump(
          output,
          output_file,
          indent=4,
      )


if __name__ == "__main__":

  parser = argparse.ArgumentParser(
      description="Benchmark the online serving throughput."
  )
  parser.add_argument(
      "--server",
      type=str,
      default="0.0.0.0",
      help="Server address.",
  )
  parser.add_argument("--port", type=str, default=9000)
  parser.add_argument(
      "--dataset",
      type=str,
      default="test",
      choices=["test", "sharegpt", "openorca"],
      help="The dataset name.",
  )
  parser.add_argument("--dataset-path", type=str, help="Path to the dataset.")
  parser.add_argument(
      "--model",
      type=str,
      default="no_model",
      help=(
          "Name of the model like llama-2, llama-3, gemma. (it's just used to"
          " label the benchmark, pick the tokenizer, the model config is"
          " defined in config_lib, and passed as the server config flag when"
          " we run the JetStream server)"
      ),
  )
  parser.add_argument(
      "--tokenizer",
      type=str,
      default="test",
      help=(
          "Name or path of the tokenizer. (For mock model testing, use the"
          " default value)"
      ),
  )
  parser.add_argument(
      "--num-prompts",
      type=int,
      default=1000,
      help=(
          "Number of prompts to process. (number of sample requests we randomly"
          " collect from dataset)"
      ),
  )
  parser.add_argument(
      "--request-rate",
      type=float,
      default=0.0,
      help=(
          "Number of requests per second. If this is 0., "
          "then all the requests are sent at time 0. "
          "Otherwise, we use Poisson process to synthesize "
          "the request arrival times."
      ),
  )
  parser.add_argument(
      "--total-mock-requests",
      type=int,
      default=150,
      help="The maximum number of mock requests to send for benchmark testing.",
  )

  parser.add_argument(
      "--max-output-length",
      type=int,
      default=0,
      help=(
          "The maximum output length for reference request. It would be passed"
          " to `max_tokens` parameter of the JetStream's DecodeRequest proto,"
          " and used in JetStream to control the output/decode length of a"
          " sequence. It would not be used in the engine. We should always set"
          " max_tokens <= (max_target_length - max_prefill_predict_length)."
          " max_target_length is the maximum length of a sequence;"
          " max_prefill_predict_length is the maximum length of the"
          " input/prefill of a sequence. Default to 0, in this case, "
          "the output length of the golden dataset would be passed."
      ),
  )

  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
      "--disable-tqdm",
      action="store_true",
      help="Specify to disable tqdm progress bar.",
  )
  parser.add_argument(
      "--save-result",
      action="store_true",
      help="Specify to save benchmark results to a json file",
  )
  parser.add_argument(
      "--additional-metadata-metrics-to-save",
      type=str,
      help=(
          "Additional metadata about the workload. Should be a dictionary in"
          " the form of a string."
      ),
  )
  parser.add_argument(
      "--save-request-outputs",
      action="store_true",
      help="Specify to store request outputs into a json file",
  )
  parser.add_argument(
      "--request-outputs-file-path",
      type=str,
      default="/tmp/request-outputs.json",
      help="File path to store request outputs",
  )
  parser.add_argument(
      "--run-eval",
      type=str2bool,
      default=False,
      help="Whether to run evaluation script on the saved outputs",
  )
  parser.add_argument(
      "--warmup-mode",
      type=str,
      default="none",
      choices=["none", "sampled", "full"],
      help="Whether to warmup first, and set the warmup mode",
  )
  parser.add_argument(
      "--conversation-starter",
      type=str,
      default="human",
      choices=["human", "gpt", "both"],
      help="What entity should be the one starting the conversations.",
  )

  parsed_args = parser.parse_args()
  main(parsed_args)


-----------------------

/benchmarks/eval_accuracy.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluate accuracy of JetStream online serving."""

import argparse
import nltk
import evaluate
import json

import numpy as np


def postprocess_text(preds, targets):
  preds = [pred.strip() for pred in preds]
  targets = [target.strip() for target in targets]

  # rougeLSum expects newline after each sentence
  preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
  targets = ["\n".join(nltk.sent_tokenize(target)) for target in targets]

  return preds, targets


def eval_accuracy(request_outputs_dict):
  metric = evaluate.load("rouge")
  nltk.download("punkt")
  preds = []
  targets = []

  for output in request_outputs_dict:
    preds.append(output["generated_text"])
    targets.append(output["original_output"])
  preds, targets = postprocess_text(preds, targets)
  result = metric.compute(
      predictions=preds,
      references=targets,
      use_stemmer=True,
      use_aggregator=False,
  )
  result = {k: float(round(np.mean(v) * 100, 4)) for k, v in result.items()}
  prediction_lens = [len(pred) for pred in preds]
  result["gen_len"] = int(np.sum(prediction_lens))
  result["gen_num"] = len(preds)
  print("\nResults\n")
  print(result)
  return result


def main(args):
  with open(args.output_path, "r", encoding="utf-8") as f:
    request_outputs_dict = json.load(f)

  eval_accuracy(request_outputs_dict)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--output_path",
      type=str,
      default="/tmp/request-outputs.json",
      help="File path which has original_output and inference generated_text.",
  )

  parsed_args = parser.parse_args()

  main(parsed_args)


-----------------------

/benchmarks/open_orca_gpt4_tokenized_llama.calibration_1000.pkl:
-----------------------



-----------------------

/benchmarks/requirements.in:
-----------------------

nltk
evaluate
rouge-score
tqdm

-----------------------

/docs/observability-prometheus-metrics-in-jetstream-server.md:
-----------------------

# Observability in JetStream Server

In JetStream Server, we use [Prometheus](https://prometheus.io/docs/introduction/overview/) to collect key metrics within JetStream orchestrator and engines. We implemented a [Prometheus client server](https://prometheus.github.io/client_python/exporting/http/) in JetStream `server_lib.py` and use `MetricsServerConfig` (by passing `prometheus_port` in server entrypoint) to gaurd the metrics observability feature.

## Enable Prometheus server to observe Jetstream metrics

Metrics are not exported by default, here is an example to run JetStream MaxText server with metrics observability:

```bash
# Refer to JetStream MaxText User Guide for the following server config.
export TOKENIZER_PATH=assets/tokenizer.gemma
export LOAD_PARAMETERS_PATH=${UNSCANNED_CKPT_PATH}
export MAX_PREFILL_PREDICT_LENGTH=1024
export MAX_TARGET_LENGTH=2048
export MODEL_NAME=gemma-7b
export ICI_FSDP_PARALLELISM=1
export ICI_AUTOREGRESSIVE_PARALLELISM=-1
export ICI_TENSOR_PARALLELISM=1
export SCAN_LAYERS=false
export WEIGHT_DTYPE=bfloat16
export PER_DEVICE_BATCH_SIZE=11
# Set PROMETHEUS_PORT to enable Prometheus metrics.
export PROMETHEUS_PORT=9090

cd ~/maxtext
python MaxText/maxengine_server.py \
  MaxText/configs/base.yml \
  tokenizer_path=${TOKENIZER_PATH} \
  load_parameters_path=${LOAD_PARAMETERS_PATH} \
  max_prefill_predict_length=${MAX_PREFILL_PREDICT_LENGTH} \
  max_target_length=${MAX_TARGET_LENGTH} \
  model_name=${MODEL_NAME} \
  ici_fsdp_parallelism=${ICI_FSDP_PARALLELISM} \
  ici_autoregressive_parallelism=${ICI_AUTOREGRESSIVE_PARALLELISM} \
  ici_tensor_parallelism=${ICI_TENSOR_PARALLELISM} \
  scan_layers=${SCAN_LAYERS} \
  weight_dtype=${WEIGHT_DTYPE} \
  per_device_batch_size=${PER_DEVICE_BATCH_SIZE} \
  prometheus_port=${PROMETHEUS_PORT}
```

Now that we configured `prometheus_port=9090` above, we can observe various Jetstream metrics via HTTP requests to `0.0.0.0:9000`. Towards the end, the response should have content similar to the following:

```
# HELP jetstream_prefill_backlog_size Size of prefill queue
# TYPE jetstream_prefill_backlog_size gauge
jetstream_prefill_backlog_size{id="SOME-HOSTNAME-HERE>"} 0.0
# HELP jetstream_slots_used_percentage The percentage of decode slots currently being used
# TYPE jetstream_slots_used_percentage gauge
jetstream_slots_used_percentage{id="<SOME-HOSTNAME-HERE>",idx="0"} 0.04166666666666663
```

## Observe metrics on GKE clusters

The following applies only for Jetstream deployed on a GKE cluster. Currently [Google Cloud Managed Service for Prometheus](https://cloud.google.com/stackdriver/docs/managed-prometheus) is enabled by default on all GKE clusters, it determines scrape targets via the [PodMonitoring](https://github.com/GoogleCloudPlatform/prometheus-engine/blob/v0.10.0/doc/api.md#podmonitoring) custom resource. After you deployed the JetStream GKE workload, you need to apply the PodMonitoring resource to your cluster as follows:

```
echo '{
    "apiVersion": "monitoring.googleapis.com/v1",
    "kind": "PodMonitoring",
    "metadata": {
      "name": "jetstream-podmonitoring"
    },
    "spec": {
      "endpoints": [
        {
          "interval": "1s",
          "path": "/",
          "port": <your-prometheus-port>
        }
      ],
      "targetLabels": {
        "metadata": [
          "pod",
          "container",
          "node"
        ]
      }
    }
  }' | kubectl apply -f -
  ```

The metrics can now be queried in the [Google Cloud Metrics Explorer](https://pantheon.corp.google.com/monitoring/metrics-explorer). When adding a metrics query with the `+Add Query` button the new metrics should be found under the `Prometheus Target > Jetstream` submenu.

Additional guides on the metrics explorer can be found [here](https://cloud.google.com/monitoring/charts/metrics-selector).

-----------------------

/docs/online-inference-with-maxtext-engine.md:
-----------------------

# JetStream MaxText Inference on v5e Cloud TPU VM User Guide

## Outline


1. Prerequisites: Prepare your GCP project and connect to Cloud TPU VM
2. Download the JetStream and MaxText github repository
3. Setup your MaxText JetStream environment
4. Convert Model Checkpoints
5. Run the JetStream MaxText server
6. Send a test request to the JetStream MaxText server
7. Run benchmarks with the JetStream MaxText server
8. Clean up


## Prerequisites: Prepare your GCP project and connect to Cloud TPU VM

Follow the steps in [Manage TPU resources | Google Cloud](https://cloud.google.com/tpu/docs/managing-tpus-tpu-vm) to create a Cloud TPU VM (Recommend TPU type: `v5litepod-8`) and connect to the Cloud TPU VM.


## Step 1: Download JetStream and the MaxText github repository

```bash
git clone https://github.com/google/maxtext.git
git clone https://github.com/google/JetStream.git
```

## Step 2: Setup MaxText and JetStream

```bash
# Create a python virtual environment for the demo.
sudo apt install python3.10-venv
python -m venv .env
source .env/bin/activate

# Setup MaxText.
cd maxtext/
bash setup.sh

# Setup JetStream
cd JetStream
pip install -e .
cd benchmarks
pip install -r requirements.in
```

## Step 3: Convert Model Checkpoints 

You can run the JetStream MaxText Server with Gemma and Llama2 models. This section describes how to run the JetStream MaxText server with various sizes of these models.

### Use a Gemma model checkpoint

*   You can download a [Gemma checkpoint from Kaggle](https://www.kaggle.com/models/google/gemma/frameworks/maxText/variations/7b). 
*   After downloading orbax Gemma checkpoints, copy them to your GCS bucket at `$CHKPT_BUCKET`. You should also set two more paths `$MAXTEXT_BUCKET_SCANNED` and `$MAXTEXT_BUCKET_UNSCANNED` that point to the locations of the maxtext checkpoints for the scanned and unscanned (inference-optimized) versions, respectively.
    *   `gsutil -m cp -r ${YOUR_CKPT_PATH} ${CHKPT_BUCKET}`
    *   Please refer to the [conversion script](https://github.com/google/JetStream/blob/main/jetstream/tools/maxtext/model_ckpt_conversion.sh) for an example of `$CHKPT_BUCKET`.
*   Then, using the following command to convert the Gemma checkpoint into a MaxText compatible unscanned checkpoint.

```bash
# bash ../JetStream/jetstream/tools/maxtext/model_ckpt_conversion.sh ${MODEL} ${MODEL_VARIATION} ${CHKPT_BUCKET} ${MAXTEXT_BUCKET_SCANNED} ${MAXTEXT_BUCKET_UNSCANNED}

# For gemma-7b
bash ../JetStream/jetstream/tools/maxtext/model_ckpt_conversion.sh gemma 7b ${CHKPT_BUCKET} ${MAXTEXT_BUCKET_SCANNED} ${MAXTEXT_BUCKET_UNSCANNED}
```

Note: For more information about the Gemma model and checkpoints, see [About Gemma](https://github.com/google/maxtext/blob/main/end_to_end/gemma/Run_Gemma.md).


### Use a Llama2 model checkpoint

*   You can use a Llama2 checkpoint you have generated or one from [the open source community](https://llama.meta.com/llama-downloads/). 
*   After downloading PyTorch checkpoints, copy them to your GCS bucket at `$CHKPT_BUCKET`. You should also set two more paths `$MAXTEXT_BUCKET_SCANNED` and `$MAXTEXT_BUCKET_UNSCANNED` that point to the locations of the maxtext checkpoints for the scanned and unscanned (inference-optimized) versions, respectively.
    *   `gsutil -m cp -r ${YOUR_CKPT_PATH} ${CHKPT_BUCKET}`
    *   Please refer to the [conversion script](https://github.com/google/JetStream/blob/main/jetstream/tools/maxtext/model_ckpt_conversion.sh) for an example of `$CHKPT_BUCKET`.
*   Then, using the following command to convert the Llama2 checkpoint into a MaxText compatible unscanned checkpoint.

```bash
# bash ../JetStream/jetstream/tools/maxtext/model_ckpt_conversion.sh ${MODEL} ${MODEL_VARIATION} ${CHKPT_BUCKET} ${MAXTEXT_BUCKET_SCANNED} ${MAXTEXT_BUCKET_UNSCANNED}

# For llama2-7b
bash ../JetStream/jetstream/tools/maxtext/model_ckpt_conversion.sh llama2 7b ${CHKPT_BUCKET} ${MAXTEXT_BUCKET_SCANNED} ${MAXTEXT_BUCKET_UNSCANNED}

# For llama2-13b
bash ../JetStream/jetstream/tools/maxtext/model_ckpt_conversion.sh llama2 13b ${CHKPT_BUCKET} ${MAXTEXT_BUCKET_SCANNED} ${MAXTEXT_BUCKET_UNSCANNED}
```

Note: For more information about the Llama2 model and checkpoints, see [About Llama2](https://github.com/google/maxtext/blob/main/getting_started/Run_Llama2.md).


## Step 4: Run the JetStream MaxText server


### Create model config environment variables for server flags

You can export the following environment variables based on the model you used.

*   You can copy and export the `UNSCANNED_CKPT_PATH` from the model\_ckpt\_conversion.sh output.


#### Create Gemma-7b environment variables for server flags



*   Configure the [flags](#jetstream-maxtext-server-flag-descriptions) passing into the JetStream MaxText server

```bash
export TOKENIZER_PATH=assets/tokenizer.gemma
export LOAD_PARAMETERS_PATH=${UNSCANNED_CKPT_PATH}
export MAX_PREFILL_PREDICT_LENGTH=1024
export MAX_TARGET_LENGTH=2048
export MODEL_NAME=gemma-7b
export ICI_FSDP_PARALLELISM=1
export ICI_AUTOREGRESSIVE_PARALLELISM=1
export ICI_TENSOR_PARALLELISM=-1
export SCAN_LAYERS=false
export WEIGHT_DTYPE=bfloat16
export PER_DEVICE_BATCH_SIZE=11
```

#### Create Llama2-7b environment variables for server flags

*   Configure the [flags](#jetstream-maxtext-server-flag-descriptions) passing into the JetStream MaxText server

```bash
export TOKENIZER_PATH=assets/tokenizer.llama2
export LOAD_PARAMETERS_PATH=${UNSCANNED_CKPT_PATH}
export MAX_PREFILL_PREDICT_LENGTH=1024
export MAX_TARGET_LENGTH=2048
export MODEL_NAME=llama2-7b
export ICI_FSDP_PARALLELISM=1
export ICI_AUTOREGRESSIVE_PARALLELISM=1
export ICI_TENSOR_PARALLELISM=-1
export SCAN_LAYERS=false
export WEIGHT_DTYPE=bfloat16
export PER_DEVICE_BATCH_SIZE=11
```

#### Create Llama2-13b environment variables for server flags

*   Configure the [flags](#jetstream-maxtext-server-flag-descriptions) passing into the JetStream MaxText server

```bash
export TOKENIZER_PATH=assets/tokenizer.llama2
export LOAD_PARAMETERS_PATH=${UNSCANNED_CKPT_PATH}
export MAX_PREFILL_PREDICT_LENGTH=1024
export MAX_TARGET_LENGTH=2048
export MODEL_NAME=llama2-13b
export ICI_FSDP_PARALLELISM=1
export ICI_AUTOREGRESSIVE_PARALLELISM=1
export ICI_TENSOR_PARALLELISM=-1
export SCAN_LAYERS=false
export WEIGHT_DTYPE=bfloat16
export PER_DEVICE_BATCH_SIZE=4
```

### Run the following command to start the JetStream MaxText server

```bash
cd ~/maxtext
python MaxText/maxengine_server.py \
  MaxText/configs/base.yml \
  tokenizer_path=${TOKENIZER_PATH} \
  load_parameters_path=${LOAD_PARAMETERS_PATH} \
  max_prefill_predict_length=${MAX_PREFILL_PREDICT_LENGTH} \
  max_target_length=${MAX_TARGET_LENGTH} \
  model_name=${MODEL_NAME} \
  ici_fsdp_parallelism=${ICI_FSDP_PARALLELISM} \
  ici_autoregressive_parallelism=${ICI_AUTOREGRESSIVE_PARALLELISM} \
  ici_tensor_parallelism=${ICI_TENSOR_PARALLELISM} \
  scan_layers=${SCAN_LAYERS} \
  weight_dtype=${WEIGHT_DTYPE} \
  per_device_batch_size=${PER_DEVICE_BATCH_SIZE}
```

### JetStream MaxText Server flag descriptions:



*   tokenizer\_path: file path to a tokenizer (should match your model)
*   load\_parameters\_path: Loads the parameters (no optimizer states) from a specific directory
*   per\_device\_batch\_size: decoding batch size per device (1 TPU chip = 1 device)
*   max\_prefill\_predict\_length: Maximum length for the prefill when doing autoregression
*   max\_target\_length: Maximum sequence length
*   model\_name: Model name
*   ici\_fsdp\_parallelism: The number of shards for FSDP parallelism
*   ici\_autoregressive\_parallelism: The number of shards for autoregressive parallelism
*   ici\_tensor\_parallelism: The number of shards for tensor parallelism
*   weight\_dtype: Weight data type (e.g. bfloat16)
*   scan\_layers: Scan layers boolean flag (set to `false` for inference)

Note: these flags are from [MaxText config](https://github.com/google/maxtext/blob/f9e04cdc1eec74a0e648411857c09403c3358461/MaxText/configs/base.yml)


## Step 5: Send a test request to JetStream MaxText server
In a new tab in your terminal, run the following command

```bash
cd ~
# For Gemma model
python JetStream/jetstream/tools/requester.py --tokenizer maxtext/assets/tokenizer.gemma
# For Llama2 model
python JetStream/jetstream/tools/requester.py --tokenizer maxtext/assets/tokenizer.llama2
```

The output will be similar to the following:

```bash
Sending request to: 0.0.0.0:9000
Prompt: Today is a good day
Response:  to be a fan
```

## Step 6: Run benchmarks with JetStream MaxText server

Note: The JetStream MaxText Server commands from Step 4 are not running with any quantization optimizations. To get the best benchmark results, we need to enable quantization for weights and KV cache. To do this, first generate AQT trained or fine-tuned checkpoints. Then, add the quantization flags and restart the server.

### Generating a quantized checkpoint

First, define the path to which the quantized checkpoint
```bash
export SAVE_QUANT_PARAMS_PATH=gs://${USER}-bkt/quantized/llama2-7b-chat
```

There are several different quantization configurations to choose from:

#### int8 DRQ quantized checkpoint
```bash
python MaxText/decode.py MaxText/configs/base.yml tokenizer_path=assets/tokenizer.llama2 load_parameters_path=${LOAD_PARAMETERS_PATH} max_prefill_predict_length=1024 max_target_length=2048 model_name=llama2-7b ici_fsdp_parallelism=1 ici_autoregressive_parallelism=1 ici_tensor_parallelism=-1 scan_layers=false weight_dtype=bfloat16 per_device_batch_size=11 attention=dot_product quantization=int8 save_quantized_params_path=${SAVE_QUANT_PARAMS_PATH}
```

#### Weights-only int8 quantized checkpoint
```bash
python MaxText/decode.py MaxText/configs/base.yml tokenizer_path=assets/tokenizer.llama2 load_parameters_path=${LOAD_PARAMETERS_PATH} max_prefill_predict_length=1024 max_target_length=2048 model_name=llama2-7b ici_fsdp_parallelism=1 ici_autoregressive_parallelism=1 ici_tensor_parallelism=-1 scan_layers=false weight_dtype=bfloat16 per_device_batch_size=11 attention=dot_product quantization=int8w save_quantized_params_path=${SAVE_QUANT_PARAMS_PATH}
```

#### Mixed precision weight-only quantized checkpoint
First, update the mixed precision config file (`MaxText/configs/quantization/mp_scale.json`)  in MaxText repo to the mixed-precision-config defined below.
```
{
  ".*/query": {"bits": 4, "scale": 0.8},
  ".*/key": {"bits": 4, "scale": 0.9},
  ".*/value": {"bits": 8},
  ".*/out": {"bits": 4},
  ".*/wi_0": {"bits": 4},
  ".*/wo": {"bits": 8}
}
```
Then run the following command:
```bash
python MaxText/decode.py MaxText/configs/base.yml tokenizer_path=assets/tokenizer.llama2 load_parameters_path=${LOAD_PARAMETERS_PATH} max_prefill_predict_length=1024 max_target_length=2048 model_name=llama2-7b ici_fsdp_parallelism=1 ici_autoregressive_parallelism=1 ici_tensor_parallelism=-1 scan_layers=false weight_dtype=bfloat16 per_device_batch_size=11 attention=dot_product quantization=intmp
quant_cfg_path=configs/quantization/mp_scale.json save_quantized_params_path=${SAVE_QUANT_PARAMS_PATH}
```

### Restart the server with quantization flags

#### Set flags

Setting base quantization flags
```bash
# To load an int8 DRQcheckpoint
export QUANTIZATION=int8
export LOAD_PARAMETERS_PATH${SAVE_QUANT_PARAMS_PATH}
export CHECKPOINT_IS_QUANTIZED=True

# To load an int8 weight-only checkpoint
export QUANTIZATION=int8w
export LOAD_PARAMETERS_PATH${SAVE_QUANT_PARAMS_PATH}
export CHECKPOINT_IS_QUANTIZED=True

# To load a Mixed-Precision quantized checkpoint
# If using Mixed-Precision mode, make sure to update the mixed precision config file to the same file as used for quantizing the checkpoint (MaxText/configs/quantization/mp_scale.json) 
export QUANTIZATION=intmp
export LOAD_PARAMETERS_PATH${SAVE_QUANT_PARAMS_PATH}
export CHECKPOINT_IS_QUANTIZED=True
export QUANT_CFG_PATH=configs/quantization/mp_scale.json
```

The KV-cache is quantized to int8 by using the following config params
```bash
export QUANTIZE_KVCACHE=True
```
If you don't want to quantize the KV-cache, set
```bash
export QUANTIZE_KVCACHE=False
```


#### Restart server
```bash
# For Gemma 7b model, change per_device_batch_size to 12 to optimize performance. 
export PER_DEVICE_BATCH_SIZE=12

cd ~/maxtext
python MaxText/maxengine_server.py \
  MaxText/configs/base.yml \
  tokenizer_path=${TOKENIZER_PATH} \
  load_parameters_path=${LOAD_PARAMETERS_PATH} \
  max_prefill_predict_length=${MAX_PREFILL_PREDICT_LENGTH} \
  max_target_length=${MAX_TARGET_LENGTH} \
  model_name=${MODEL_NAME} \
  ici_fsdp_parallelism=${ICI_FSDP_PARALLELISM} \
  ici_autoregressive_parallelism=${ICI_AUTOREGRESSIVE_PARALLELISM} \
  ici_tensor_parallelism=${ICI_TENSOR_PARALLELISM} \
  scan_layers=${SCAN_LAYERS} \
  weight_dtype=${WEIGHT_DTYPE} \
  per_device_batch_size=${PER_DEVICE_BATCH_SIZE} \
  quantization=${QUANTIZATION} \
  quantize_kvcache=${QUANTIZE_KVCACHE} \
  checkpoint_is_quantized=${CHECKPOINT_IS_QUANTIZED}
```

For the mixed precision quantized model
```bash
python MaxText/maxengine_server.py \
  MaxText/configs/base.yml \
  tokenizer_path=${TOKENIZER_PATH} \
  load_parameters_path=${LOAD_PARAMETERS_PATH} \
  max_prefill_predict_length=${MAX_PREFILL_PREDICT_LENGTH} \
  max_target_length=${MAX_TARGET_LENGTH} \
  model_name=${MODEL_NAME} \
  ici_fsdp_parallelism=${ICI_FSDP_PARALLELISM} \
  ici_autoregressive_parallelism=${ICI_AUTOREGRESSIVE_PARALLELISM} \
  ici_tensor_parallelism=${ICI_TENSOR_PARALLELISM} \
  scan_layers=${SCAN_LAYERS} \
  weight_dtype=${WEIGHT_DTYPE} \
  per_device_batch_size=${PER_DEVICE_BATCH_SIZE} \
  quantization=${QUANTIZATION} \
  quantize_kvcache=${QUANTIZE_KVCACHE} \
  checkpoint_is_quantized=${CHECKPOINT_IS_QUANTIZED} \
  quant_cfg_path=${QUANT_CFG_PATH}
```


### Benchmarking Gemma-7b

Instructions
- Download the ShareGPT dataset
- Make sure to use the Gemma tokenizer (tokenizer.gemma) when running Gemma 7b.
- Add `--warmup-first` flag for your 1st run to warmup the server

```bash
# Activate the python virtual environment we created in Step 2.
cd ~
source .env/bin/activate

# download dataset
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# run benchmark with the downloaded dataset and the tokenizer in maxtext
# You can control the qps by setting `--request-rate`, the default value is inf.
python JetStream/benchmarks/benchmark_serving.py \
--tokenizer maxtext/assets/tokenizer.gemma \
--num-prompts 1000 \
--dataset sharegpt \
--dataset-path ~/ShareGPT_V3_unfiltered_cleaned_split.json \
--max-output-length 1024 \
--request-rate 5 \
--warmup-mode sampled
```
For details, please see https://github.com/google/JetStream/blob/main/benchmarks/README.md

### Benchmarking Llama2

```bash
# The command is the same as that for the Gemma-7b, except for the tokenizer. Since we need to use a tokenizer that matches the model, it should now be tokenizer.llama2. 

python JetStream/benchmarks/benchmark_serving.py \
--tokenizer maxtext/assets/tokenizer.llama2 \
--num-prompts 1000  \
--dataset sharegpt \
--dataset-path ~/ShareGPT_V3_unfiltered_cleaned_split.json \
--max-output-length 1024 \
--request-rate 5 \
--warmup-mode sampled
```
For details, please see https://github.com/google/JetStream/blob/main/benchmarks/README.md

## Clean Up

```bash
# Clean up gcs buckets.
gcloud storage buckets delete ${MODEL_BUCKET}
gcloud storage buckets delete ${BASE_OUTPUT_DIRECTORY}

# Clean up repositories.
rm -rf maxtext
rm -rf JetStream

# Clean up python virtual environment
rm -rf .env
```


-----------------------

/docs/profiling-with-jax-profiler-and-tensorboard.md:
-----------------------

# Profiling in JetStream Server

In JetStream server, we have implemented JAX profiler server to support profiling JAX program with tensorboard.

## Profiling with JAX profiler server and tenorboard server

Following the [JAX official manual profiling approach](https://jax.readthedocs.io/en/latest/profiling.html#manual-capture-via-tensorboard), here is an example of JetStream MaxText server profiling with tensorboard:

1. Start a TensorBoard server:
```bash
tensorboard --logdir /tmp/tensorboard/
```
You should be able to load TensorBoard at http://localhost:6006/. You can specify a different port with the `--port` flag. If you are running on a remote Cloud TPU VM, the `tensorboard-plugin-profile` python package enables remote access to tensorboard endpoints (JetStream deps include this package).

When you can not access the tensorboard and the profiling code is run remotely, please run below command setup an SSH tunnel on port 6006 to work. If you run with vs code remote debug commandline, the vs code did ssh forward port for you.

```bash
 gcloud compute ssh <machine-name> -- -L 6006:127.0.0.1:6006
 ```


2. Start JetStream MaxText server:
```bash
# Refer to JetStream MaxText User Guide for the following server config.
export TOKENIZER_PATH=assets/tokenizer.gemma
export LOAD_PARAMETERS_PATH=${UNSCANNED_CKPT_PATH}
export MAX_PREFILL_PREDICT_LENGTH=1024
export MAX_TARGET_LENGTH=2048
export MODEL_NAME=gemma-7b
export ICI_FSDP_PARALLELISM=1
export ICI_AUTOREGRESSIVE_PARALLELISM=-1
export ICI_TENSOR_PARALLELISM=1
export SCAN_LAYERS=false
export WEIGHT_DTYPE=bfloat16
export PER_DEVICE_BATCH_SIZE=11
# Set ENABLE_JAX_PROFILER to enable JAX profiler server at port 9999.
export ENABLE_JAX_PROFILER=true
# Set JAX_PROFILER_PORT to customize JAX profiler server port.
export JAX_PROFILER_PORT=9999

cd ~/maxtext
python MaxText/maxengine_server.py \
  MaxText/configs/base.yml \
  tokenizer_path=${TOKENIZER_PATH} \
  load_parameters_path=${LOAD_PARAMETERS_PATH} \
  max_prefill_predict_length=${MAX_PREFILL_PREDICT_LENGTH} \
  max_target_length=${MAX_TARGET_LENGTH} \
  model_name=${MODEL_NAME} \
  ici_fsdp_parallelism=${ICI_FSDP_PARALLELISM} \
  ici_autoregressive_parallelism=${ICI_AUTOREGRESSIVE_PARALLELISM} \
  ici_tensor_parallelism=${ICI_TENSOR_PARALLELISM} \
  scan_layers=${SCAN_LAYERS} \
  weight_dtype=${WEIGHT_DTYPE} \
  per_device_batch_size=${PER_DEVICE_BATCH_SIZE} \
  enable_jax_profiler=${ENABLE_JAX_PROFILER} \
  jax_profiler_port=${JAX_PROFILER_PORT}
```

3. Open http://localhost:6006/#profile, and click the “CAPTURE PROFILE” button in the upper left. Enter “localhost:9999” as the profile service URL (this is the address of the profiler server you started in the previous step). Enter the number of milliseconds you’d like to profile for, and click “CAPTURE”.

4. After the capture finishes, TensorBoard should automatically refresh. (Not all of the TensorBoard profiling features are hooked up with JAX, so it may initially look like nothing was captured.) On the left under “Tools”, select `trace_viewer`.

-----------------------

/jetstream/__init__.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


-----------------------

/jetstream/core/README.md:
-----------------------

# JetStream core Subpackage - Server and Library that support continuous batching serving.

Interleaved mode: Provide continuous batching to optimize inference. Uses JAX directy on single-host TPU.

-----------------------

/jetstream/core/__init__.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


-----------------------

/jetstream/core/config_lib.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configs of engines for the orchestrator to load."""

import dataclasses
import functools
from typing import Any, Callable, List, Tuple, Type
from numpy import uint16

from jetstream.engine import engine_api
from jetstream.engine import mock_engine


Devices = Any

CreateEngineFn = Callable[[Devices], engine_api.Engine]


@dataclasses.dataclass
class ServerConfig:
  """Configs for slices to put engines on."""

  prefill_slices: Tuple[str, ...] = ()
  generate_slices: Tuple[str, ...] = ()
  interleaved_slices: Tuple[str, ...] = ()
  prefill_engine_create_fns: Tuple[CreateEngineFn, ...] = ()
  generate_engine_create_fns: Tuple[CreateEngineFn, ...] = ()
  interleaved_engine_create_fns: Tuple[CreateEngineFn, ...] = ()
  is_ray_backend: bool = False


@dataclasses.dataclass
class InstantiatedEngines:
  prefill_engines: List[engine_api.Engine]
  generate_engines: List[engine_api.Engine]
  interleaved_engines: List[engine_api.Engine]


@dataclasses.dataclass
class MetricsServerConfig:
  port: uint16


# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼#


def get_test_engine(devices: Devices, weight: int) -> engine_api.Engine:
  del devices
  return mock_engine.TestEngine(
      batch_size=8,
      cache_length=32,
      weight=weight,
  )


@dataclasses.dataclass
class CPUTestServer(ServerConfig):
  prefill_slices = ("cpu=1",)
  generate_slices = ("cpu=1",)
  prefill_engine_create_fns = (functools.partial(get_test_engine, weight=2),)
  generate_engine_create_fns = (functools.partial(get_test_engine, weight=4),)


@dataclasses.dataclass
class InterleavedCPUTestServer(ServerConfig):
  interleaved_slices = ("cpu=1",)
  interleaved_engine_create_fns = (
      functools.partial(get_test_engine, weight=2),
  )


# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼#


def slice_to_num_chips(s: str) -> int:
  """Converts a TPU spec like v5e-8 or v5e=8 to the number of chips, 8."""
  # Account for the case where it is written 'tpu=8' for compatibility.
  delim = "-" if "-" in s else "="
  # TODO: Support more accelerator type check.
  accelerator_type, num_devices = s.split(delim)
  return int(num_devices) if accelerator_type != "v4" else int(num_devices) // 2


def _split_devices_by_slices(
    devices: list[Devices], slices: list[int]
) -> List[List[Devices]]:
  """Converts an ordered list of devices into slices."""
  assert sum(slices) == len(devices), f"{sum(slices)} != {len(devices)}"
  cumsum = 0
  slice_split_devices = []
  for sl in slices:
    slice_split_devices.append(devices[cumsum : cumsum + sl])
    cumsum += sl
  return slice_split_devices


def get_engines(
    server_config: Type[ServerConfig], devices: List[Devices]
) -> InstantiatedEngines:
  """Processes config to get the appropriate engines.

  Args:
    server_config: ServerConfig.
    devices: Device objects.

  Returns:
    Instantiated engines!

  Devices are popped in order!
  """
  # Now, we need to split devices by slice due to TPU backend config.
  slices: list[int] = [
      slice_to_num_chips(s)
      for s in list(server_config.prefill_slices)
      + list(server_config.generate_slices)
      + list(server_config.interleaved_slices)
  ]
  if sum(slices) != len(devices):
    raise ValueError(
        f"The number of available devices ({len(devices)}) do not match the "
        f"expected number of devices on all the slices ({sum(slices)}) "
        "specified in the server_config:\n"
        f"{server_config.prefill_slices=}\n"
        f"{server_config.generate_slices=}\n"
        f"{server_config.interleaved_slices=}\n"
    )
  # e.g. [[tpu_0], [tpu_1]] corresponding to prefill: v5e=1x1
  # generate = v5e=1x1; or [[tpu_0, tpu_1, tpu_2, tpu_3]] corresponding to
  # interleaved: v5e=2x2
  split_devices = _split_devices_by_slices(devices, slices)
  prefill_engines = [
      e(split_devices.pop(0)) for e in server_config.prefill_engine_create_fns
  ]
  generate_engines = [
      e(split_devices.pop(0)) for e in server_config.generate_engine_create_fns
  ]
  # These share chips and weights for prefill and generation.
  interleaved_engines = [
      e(split_devices.pop(0))
      for e in server_config.interleaved_engine_create_fns
  ]
  return InstantiatedEngines(
      prefill_engines, generate_engines, interleaved_engines
  )


-----------------------

/jetstream/core/implementations/__init__.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


-----------------------

/jetstream/core/implementations/mock/README.md:
-----------------------

# Mock server

Implements Mock JetStream core with mock model config.

-----------------------

/jetstream/core/implementations/mock/__init__.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


-----------------------

/jetstream/core/implementations/mock/config.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Config for mock."""

from typing import Type


from jetstream.core import config_lib


def get_server_config(config_str: str) -> Type[config_lib.ServerConfig]:
  match config_str:
    case "InterleavedCPUTestServer":
      server_config = config_lib.InterleavedCPUTestServer
    case "CPUTestServer":
      server_config = config_lib.CPUTestServer
    case _:
      raise NotImplementedError
  return server_config


-----------------------

/jetstream/core/implementations/mock/server.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runs a mock server."""

from typing import Sequence

from absl import app
from absl import flags

from jetstream.core.implementations.mock import config as mock_config
from jetstream.core import server_lib


_PORT = flags.DEFINE_integer("port", 9000, "port to listen on")
_CONFIG = flags.DEFINE_string(
    "config",
    "InterleavedCPUTestServer",
    "available servers",
)


def main(argv: Sequence[str]):
  del argv
  # No devices for local cpu test. A None for prefill and a None for generate.
  devices = server_lib.get_devices()
  server_config = mock_config.get_server_config(_CONFIG.value)
  # We separate credential from run so that we can unit test it with local
  # credentials.
  # TODO: Add grpc credentials for OSS.
  jetstream_server = server_lib.run(
      port=_PORT.value,
      config=server_config,
      devices=devices,
  )
  jetstream_server.wait_for_termination()


if __name__ == "__main__":
  app.run(main)


-----------------------

/jetstream/core/metrics/__init__.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


-----------------------

/jetstream/core/metrics/prometheus.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains common functions for configuring Jetstream server metrics"""

import os
import shortuuid
from prometheus_client import Counter, Gauge, Histogram
from jetstream.engine.token_utils import DEFAULT_PREFILL_BUCKETS


class JetstreamMetricsCollector:
  """Wrapper class should be used to assure all metrics have proper tags"""

  _id: str = os.getenv("HOSTNAME", shortuuid.uuid())

  def __new__(cls):
    if not hasattr(cls, "instance"):
      cls.instance = super(JetstreamMetricsCollector, cls).__new__(cls)
    return cls.instance

  # Metric definitions
  _prefill_backlog = Gauge(
      name="jetstream_prefill_backlog_size",
      documentation="Size of prefill queue",
      labelnames=["id"],
  )

  _transfer_backlog = Gauge(
      name="jetstream_transfer_backlog_size",
      documentation="Size of transfer queue",
      labelnames=["id", "idx"],
  )

  _generate_backlog = Gauge(
      name="jetstream_generate_backlog_size",
      documentation="Size of generate queue",
      labelnames=["id", "idx"],
  )

  _queue_duration = Histogram(
      name="jetstream_queue_duration",
      documentation="The total time each request spends enqueued in seconds",
      labelnames=["id"],
      buckets=[
          0.01,
          0.02,
          0.05,
          0.1,
          0.2,
          0.5,
          1.0,
          2.0,
          5.0,
          10.0,
          20.0,
          50.0,
          100.0,
      ],
  )

  _slots_used_percentage = Gauge(
      name="jetstream_slots_used_percentage",
      documentation="The percentage of decode slots currently being used",
      labelnames=["id", "idx"],
  )

  _server_startup_latency = Gauge(
      name="jetstream_server_startup_latency",
      documentation="Total time taken to start the Jetstream server",
      labelnames=["id"],
  )
  _request_input_length = Histogram(
      name="jetstream_request_input_length",
      documentation="Number of input tokens per request",
      labelnames=["id"],
      buckets=DEFAULT_PREFILL_BUCKETS,
  )
  _request_output_length = Histogram(
      name="jetstream_request_output_length",
      documentation="Number of output tokens per request",
      labelnames=["id"],
      buckets=[
          1,
          2,
          5,
          10,
          20,
          50,
          100,
          200,
          500,
          1000,
          2000,
          5000,
          10000,
          20000,
          50000,
          100000,
          200000,
          500000,
          1000000,
          2000000,
      ],
  )
  _request_success_count = Counter(
      name="jetstream_request_success_count",
      documentation="Number of requests successfully completed",
      labelnames=["id"],
  )

  _time_to_first_token = Histogram(
      name="jetstream_time_to_first_token",
      documentation="Time to first token per request in seconds",
      labelnames=["id"],
      buckets=[
          0.001,
          0.005,
          0.01,
          0.02,
          0.04,
          0.06,
          0.08,
          0.1,
          0.25,
          0.5,
          0.75,
          1.0,
          2.5,
          5.0,
          7.5,
          10.0,
      ],
  )

  _time_per_output_token = Histogram(
      name="jetstream_time_per_output_token",
      documentation="Average time per output token per request in seconds",
      labelnames=["id"],
      buckets=[
          0.01,
          0.025,
          0.05,
          0.075,
          0.1,
          0.15,
          0.2,
          0.3,
          0.4,
          0.5,
          0.75,
          1.0,
          2.5,
      ],
  )

  _time_per_prefill_token = Histogram(
      name="jetstream_time_per_prefill_token",
      documentation="Prefill time per token per request in seconds",
      labelnames=["id"],
      buckets=[
          0.00001,
          0.00002,
          0.00005,
          0.0001,
          0.0002,
          0.0005,
          0.001,
          0.002,
          0.005,
          0.01,
          0.02,
          0.05,
          0.1,
      ],
  )

  _time_per_request = Histogram(
      name="jetstream_time_per_request",
      documentation="End to end request latency in seconds",
      labelnames=["id"],
      buckets=[1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0],
  )

  _wait_time_per_request = Histogram(
      name="jetstream_wait_time_per_request",
      documentation="Time each request is not being prefilled or decoded",
      labelnames=["id"],
      buckets=[
          0.01,
          0.02,
          0.05,
          0.1,
          0.2,
          0.5,
          1.0,
          2.0,
          5.0,
          10.0,
          20.0,
          50.0,
          100.0,
      ],
  )

  def get_prefill_backlog_metric(self):
    return self._prefill_backlog.labels(id=self._id)

  def get_transfer_backlog_metric(self, idx: int):
    return self._transfer_backlog.labels(id=self._id, idx=idx)

  def get_generate_backlog_metric(self, idx: int):
    return self._generate_backlog.labels(id=self._id, idx=idx)

  def get_queue_duration(self):
    return self._queue_duration.labels(id=self._id)

  def get_slots_used_percentage_metric(self, idx: int):
    return self._slots_used_percentage.labels(id=self._id, idx=idx)

  def get_server_startup_latency_metric(self):
    return self._server_startup_latency.labels(id=self._id)

  def get_time_to_first_token(self):
    return self._time_to_first_token.labels(id=self._id)

  def get_time_per_output_token(self):
    return self._time_per_output_token.labels(id=self._id)

  def get_time_per_prefill_token(self):
    return self._time_per_prefill_token.labels(id=self._id)

  def get_time_per_request(self):
    return self._time_per_request.labels(id=self._id)

  def get_wait_time_per_request(self):
    return self._wait_time_per_request.labels(id=self._id)

  def get_request_input_length(self):
    return self._request_input_length.labels(id=self._id)

  def get_request_output_length(self):
    return self._request_output_length.labels(id=self._id)

  def get_request_success_count_metric(self):
    return self._request_success_count.labels(id=self._id)


-----------------------

/jetstream/core/orchestrator.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Orchestrates the engines with performance optimization for inference.

1. A client sends a DecodeRequest via gRPC to the server, an 'LLMOrchestrator'.
2. This gets wrapped as an 'ActiveRequest' inside the orchestrator, with a
    'return_channel' queue as a place that output tokens can be placed.
    - The ActiveRequest is placed on the 'prefill_queue'.
    - A while loop runs continuously, yielding any tokens placed on the return
      channel until an end condition is met (EOS token or max tokens).
3. There is a prefill_thread per prefill_engine, each of which runs on a
    distinct prefill_slice.
4. There is a generate_thread per generate_engine, each of which runs on a
    distinct generate_slice.
5. Within a prefill thread:
    - It attempts to pop ActiveRequests off the prefill_queue.
    - It tokenizes the request.
    - When successful, it performs a prefill operation, transfers the kv cache
      to the generation slice and pops this information (still wrapped in the
      same ActiveRequest) onto the generation queue.
6. Within a generation thread:
   - There is a queue of integers representing 'available slots'.
   - It checks if there is something on both the slots_queue and generation_
     queue.
   - If so, the kv_cache associated with that request into the decoding state
    of the generation loop at the relevant slot.
   - Regardless, it performs a step.
  - It takes the sampled tokens, and places them on a 'detokenizing_queue'.
7. Within the detokenizing thread:
  - Tokens are detokenized for every 'slot' in a given set of sampled tokens.
  - When an end condition is met, the 'slot' integer is returned to the
    respective generation queue.
  - This does mean that a single generation step may run after detokenizing
    indicates that row is no longer valid (if the detokenizing is running behind
    generation steps), this is fine as it avoids detokenizing being blocking of
    the generate thread.

If you haven't worked with concurrency in python before - queues are thread-safe
by default, so we can happily use them to transfer pointers to data between
different processes. The structure of this server is simple as a result - a
thread for each thing we might want to do (prefill, transfer, generate,
detokenize), and corresponding queues that an active request is passed between.
The same goes for the 'return_channel' of the request itself, where we can just
pop tokens once they are done and try to pop them back to transmit them over
grpc.
It is literally queues all the way down! :)
The primary concern is GIL contention between threads, which is why we block
on queues that don't have an ongoing activity (i.e. everything but the
generation queue) because we don't control to go back to those queues until
necessary. Blocking means that the GIL doesn't switch back to that thread,
wheras continual queue get operations 'chop' control and mean that we do not
achieve good throughput. This is okay on the prefill/transfer/detokenization
threads because we don't need to do anything other than react to the presence
of items on these queues, wheras the generation thread needs to also run a
step - so it cannot block until it has new things to insert.

## Testing
This server is intended to be easy to locally test.

Either use :orchestrator test, which tests the multi-threading components,
:server_test, which extends this to test grpc_components, or run it locally
to debug hangs due to bugs in threads (it is easier to debug with live logs).
"""

import dataclasses
import functools
import itertools
import logging
import os
import queue
import signal
import sys
import threading
import time
import traceback
from typing import Any, AsyncIterator, Optional, Tuple, cast

import grpc
import jax
from jetstream.core.proto import jetstream_pb2
from jetstream.core.proto import jetstream_pb2_grpc
from jetstream.core.utils import async_multifuture
from jetstream.core.utils.return_sample import ReturnSample
from jetstream.engine import engine_api, tokenizer_api, token_utils
from jetstream.core.metrics.prometheus import JetstreamMetricsCollector
import numpy as np

root = logging.getLogger()
root.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
handler.setFormatter(formatter)
root.addHandler(handler)


@dataclasses.dataclass
class ActiveRequestMetadata:
  """Inference request metadata."""

  start_time: Optional[float] = None

  prefill_enqueue_time: Optional[float] = None
  prefill_dequeue_time: Optional[float] = None

  transfer_enqueue_time: Optional[float] = None
  transfer_dequeue_time: Optional[float] = None

  generate_enqueue_time: Optional[float] = None
  generate_dequeue_time: Optional[float] = None

  complete_time: Optional[float] = None


@dataclasses.dataclass
class ActiveRequest:
  """Current state of the driver."""

  #################### Information relevant for generation #####################
  max_tokens: int
  # We keep prefill and decode information together in the same object so that
  # there is less indirection about where this return channel is.
  # The return channel returns a list of strings, one per sample for that query.
  return_channel: async_multifuture.AsyncMultifuture[list[ReturnSample]]
  # [num_samples,] which corresponds to whether each sample is complete for the
  # requests.
  complete: Optional[np.ndarray] = None
  prefill_result: Any = None
  #################### Information relevant for prefill ########################
  prefill_content: Optional[str | list[int]] = None
  ################## Information relevant for detokenization ###################
  # Which generate step this was added at.
  generate_timestep_added: Optional[int] = None
  is_client_side_tokenization: Optional[bool] = False
  ################## Information relevant for metrics ###################
  metadata: ActiveRequestMetadata = ActiveRequestMetadata()

  def enqueue_samples(self, generated_samples: list[ReturnSample]):
    """Adds the generated sample(s) to return channel for current step.

    Args:
      generated_samples: The generated sample(s) for current step.

    This should be called only from within the Drivers background thread.
    """
    self.return_channel.add_result(generated_samples)


class JetThread(threading.Thread):
  """Thread that kills the program if it fails.

  If a driver thread goes down, we can't operate.
  """

  def run(self):
    try:
      super().run()
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"Thread {self.name} encountered an error: {e}")
      traceback.print_exc()
      os.kill(os.getpid(), signal.SIGKILL)


async def _abort_or_raise(
    context: grpc.aio.ServicerContext | None,
    code: grpc.StatusCode,
    details: str,
):
  """Safely aborts a gRPC context if available, or raises an Exception."""
  if context is None:
    raise RuntimeError(details)

  await context.abort(code, details)


class Driver:
  """Drives the engines."""

  _prefill_engines: list[engine_api.Engine]
  _generate_engines: list[engine_api.Engine]
  # Allows us to pre-load the params, primarily so that we can iterate quickly
  # on the driver in colab without reloading weights.
  _prefill_params: list[Any]
  _generate_params: list[Any]
  # Stage 1
  _prefill_backlog: queue.Queue[ActiveRequest | None]
  # Stage 2
  _transfer_backlogs: list[queue.Queue[ActiveRequest]] = []
  # Stage 3
  # We keep this as a dict to avoid a possibly expensive object comparison
  # when logging the index of the generate engine we send a prefill result
  # to, it allows us to natively have the index from the min operation, rather
  # than have to call .index()
  _generate_backlogs: dict[int, queue.Queue[ActiveRequest]] = {}
  # Stage 4
  # This can be a list because we can pass it as an arg to generate and
  # detokenize threads. It is a list of tokens to be detokenized.
  _detokenize_backlogs: list[queue.Queue[engine_api.ResultTokens]] = []
  _generate_slots: list[queue.Queue[int]] = []
  _active_requests: list[queue.Queue[tuple[int, ActiveRequest]]] = []

  # For interleaved_mode, only generate if all slots are full
  # or corresponding prefill queue is empty.
  _interleaved_mode: bool = False

  # todo: remove jax_padding after all then engine migrate to np padding
  _jax_padding = True

  # All metrics we want to monitor should be collected with this
  _metrics_collector: JetstreamMetricsCollector | None = None

  def __init__(
      self,
      prefill_engines: Optional[list[engine_api.Engine]] = None,
      generate_engines: Optional[list[engine_api.Engine]] = None,
      prefill_params: Optional[list[Any]] = None,
      generate_params: Optional[list[Any]] = None,
      interleaved_mode: bool = False,
      jax_padding: bool = True,
      metrics_collector: JetstreamMetricsCollector | None = None,
      is_ray_backend: bool = False,
  ):
    if prefill_engines is None:
      prefill_engines = []
    if generate_engines is None:
      generate_engines = []
    if prefill_params is None:
      prefill_params = []
    if generate_params is None:
      generate_params = []

    logging.info(
        "Initialising driver with %d prefill engines and %d generate engines.",
        len(prefill_engines),
        len(generate_engines),
    )
    self._prefill_engines = prefill_engines
    self._generate_engines = generate_engines
    self._prefill_params = prefill_params
    self._generate_params = generate_params
    self._interleaved_mode = interleaved_mode
    self._metrics_collector = metrics_collector

    # Stages 1-4 represent the life cycle of a request.
    # Stage 1
    # At first, a request is placed here in order to get prefilled.
    self._prefill_backlog = queue.Queue()
    if self._metrics_collector:
      self._metrics_collector.get_prefill_backlog_metric().set_function(
          lambda: float(self._prefill_backlog.qsize())
      )

    # Stage 2
    # After prefilling, it is placed here in order to get transferred to
    # one of the generate backlogs.
    # Interleaved Mode: Max size is 1 to increase the HBM utilization
    # during generate.
    # Disaggregated Mode: Max size is 4 to allow for 2 prefills to be enqueued
    # while 1 transfer is enqueued while 1 is being transferred.
    # TODO: Make queue size configurable.
    self._transfer_backlogs = [
        queue.Queue(1 if self._interleaved_mode else 4)
        for i in range(len(self._prefill_engines))
    ]
    if self._metrics_collector:
      for idx, backlog in enumerate(self._transfer_backlogs):
        self._metrics_collector.get_transfer_backlog_metric(idx).set_function(
            functools.partial(float, backlog.qsize())
        )
    # Stage 3
    # Each generate engine accesses its own generate backlog.
    # Interleaved Mode: Max size is 1 to increase the HBM utilization
    # during generate.
    # Disaggregated Mode: Set as 1/3 the number of concurrent decodes.
    # TODO: Calculate the backlog to saturate the generate engine while
    # minimizing the memory usage for disaggregated mode.
    # TODO: Make queue size configurable.
    self._generate_backlogs = {
        idx: queue.Queue(
            1 if self._interleaved_mode else engine.max_concurrent_decodes // 3
        )
        for idx, engine in enumerate(self._generate_engines)
    }
    if self._metrics_collector:
      for idx, backlog in self._generate_backlogs.items():
        self._metrics_collector.get_generate_backlog_metric(idx).set_function(
            functools.partial(float, backlog.qsize())
        )
    # Stage 4
    # After generation, ActiveRequests are placed on the detokenization backlog
    # for tokens to be sent into each ActiveRequest's return channel.
    # We have one of these per generate engine to simplify the logic keeping
    # track of which generation engine to replace slots on.
    # This is a queue of either - tuple[int, ActiveRequest] which represents our
    # active requests, or tuple[int, sample_tokens]. We combine these into one
    # queue because it allows us to be somewhat clever with how we do
    # detokenization.
    # If the detokenization receives an (int, ActiveRequest) this signifies
    # that slot int should from now be placing tokens in the return channel of
    # the ActiveRequest.
    # If it receives (int, sample_tokens) then it actually
    # does a detokenization for any slots which have previously been set active
    # via the previous kind of object, and the int is used to log which step
    # the tokens were created at. By having them in one queue we prevent
    # the possibility of race conditions where a slot is made live before the
    # tokens are ready and it receives tokens from a different sequence,
    # or tokens detokenized before the relevant slot is live.
    self._detokenize_backlogs = [
        # We don't let detokenization accumulate more than 8 steps to avoid
        # synchronization issues.
        queue.Queue(8)
        for _ in self._generate_engines
    ]

    # A queue of integers representing available 'slots' in the decode
    # operation. I.e. potentially available rows in the batch and/or microbatch.
    # When we want to insert a prefill result, we pop an integer to insert at.
    # When this is empty, it means all slots are full.
    self._generate_slots = [
        queue.Queue(engine.max_concurrent_decodes)
        for engine in self._generate_engines
    ]
    _ = [
        [
            self._generate_slots[idx].put(i)
            for i in range(engine.max_concurrent_decodes)
        ]
        for idx, engine in enumerate(self._generate_engines)
    ]

    self._jax_padding = jax_padding

    # Create all threads
    self._prefill_threads = [
        JetThread(
            target=functools.partial(self._prefill_thread, idx),
            name=f"prefill-{idx}",
            daemon=True,
        )
        for idx in range(len(self._prefill_engines))
    ]
    self._transfer_threads = [
        JetThread(
            target=functools.partial(
                self._transfer_thread,
                idx,
            ),
            name=f"transfer-{idx}",
            daemon=True,
        )
        for idx in range(len(self._prefill_engines))
    ]
    self._generate_threads = [
        JetThread(
            target=functools.partial(
                self._generate_thread,
                idx,
            ),
            name=f"generate-{idx}",
            daemon=True,
        )
        for idx in range(len(self._generate_engines))
    ]
    self.detokenize_threads = [
        JetThread(
            target=functools.partial(
                self._detokenize_thread,
                idx,
            ),
            name=f"detokenize-{idx}",
        )
        for idx in range(len(self._generate_engines))
    ]
    self._all_threads = list(
        itertools.chain(
            self._prefill_threads,
            self._transfer_threads,
            self._generate_threads,
            self.detokenize_threads,
        )
    )
    self.live = True
    self._is_ray_backend = is_ray_backend
    # Start all threads
    for t in self._all_threads:
      t.start()

  def stop(self):
    """Stops the driver and all background threads."""
    # Signal to all threads that they should stop.
    self.live = False

    all_backlogs = list(
        itertools.chain(
            [self._prefill_backlog],
            self._transfer_backlogs,
            self._generate_backlogs.values(),
            self._detokenize_backlogs,
        )
    )

    while any(t.is_alive() for t in self._all_threads):
      # Empty all backlogs and mark any remaining requests as cancelled.
      for q in all_backlogs:
        while True:
          try:
            r = q.get_nowait()
            if r is None:
              continue
            elif isinstance(r, ActiveRequest):
              r.return_channel = None
            else:  # detokenize backlog
              _, r = r
              if isinstance(r, ActiveRequest):
                r.return_channel = None
          except queue.Empty:
            break

      # Put sentinels to unblock threads.
      for q in all_backlogs:
        try:
          q.put_nowait(None)
        except queue.Full:
          pass

    # Wait for all threads to stop.
    for t in self._all_threads:
      t.join()

  def get_total_concurrent_requests(self) -> int:
    """Gets the total number of concurrent requests the driver can handle."""
    # We don't support filling all backlogs at once because it can cause GIL
    # contention.
    total_max_concurrent_decodes = sum(
        [e.max_concurrent_decodes for e in self._generate_engines]
    )
    return total_max_concurrent_decodes

  def place_request_on_prefill_queue(self, request: ActiveRequest):
    """Used to place new requests for prefilling and generation."""
    # Don't block so we can fail and shed load when the queue is full.
    self._prefill_backlog.put(request, block=False)

  def _process_prefill_content(
      self,
      request: ActiveRequest,
      tokenizer: tokenizer_api.Tokenizer,
      is_bos: bool,
      max_prefill_length: int,
  ) -> Tuple[jax.Array | np.ndarray, int]:
    content = request.prefill_content
    if isinstance(content, str):
      # If it's text input, tokenize and pad the input.
      return tokenizer.encode(
          content,
          is_bos=is_bos,
          max_prefill_length=max_prefill_length,
          jax_padding=self._jax_padding,
      )
    else:
      # If it's token input, pad the input.
      return token_utils.pad_tokens(
          content,
          tokenizer.bos_id,
          tokenizer.pad_id,
          is_bos=is_bos,
          max_prefill_length=max_prefill_length,
          jax_padding=self._jax_padding,
      )

  def _prefill_thread(self, idx: int):
    """Thread which runs in the background performing prefills."""
    logging.info("---------Spinning up prefill thread %d.---------", idx)
    prefill_engine = self._prefill_engines[idx]
    prefill_params = self._prefill_params[idx]
    metadata = prefill_engine.get_tokenizer()
    tokenizer = prefill_engine.build_tokenizer(metadata)
    logging.info("---------Prefill params %d loaded.---------", idx)

    while self.live:
      my_transfer_backlog = self._transfer_backlogs[idx]
      # The prefill thread can just sleep until it has work to do.
      request = self._prefill_backlog.get(block=True)

      if request is None:
        break
      request.metadata.prefill_dequeue_time = time.perf_counter()
      is_bos = True
      logging.info(
          "Prefilling on prefill engine %d : prefill queue size, %d,"
          " is_bos: %s",
          idx,
          self._prefill_backlog.qsize(),
          is_bos,
      )
      # Tokenize and padding the text or token input.
      padded_tokens, true_length = self._process_prefill_content(
          request, tokenizer, is_bos, prefill_engine.max_prefill_length
      )

      # Compute new kv cache for the prefill_content.
      prefill_result, first_token = prefill_engine.prefill(
          params=prefill_params,
          padded_tokens=padded_tokens,
          true_length=true_length,
      )
      request.prefill_result = prefill_result

      # put first token to detokenize queue
      request.complete = np.zeros((prefill_engine.samples_per_slot,), np.bool_)
      my_detokenize_backlog = self._detokenize_backlogs[idx]
      request.metadata.transfer_enqueue_time = time.perf_counter()
      my_detokenize_backlog.put(
          (first_token, request, request.metadata.prefill_dequeue_time),
          block=True,
      )

      # Once prefill is complete, place it on the generation queue and block if
      # full.
      my_transfer_backlog.put(request, block=True)
      logging.info(
          "Placed request on transfer queue %d, %d queued requests.",
          idx,
          my_transfer_backlog.qsize(),
      )
      if self._metrics_collector:
        self._metrics_collector.get_request_input_length().observe(true_length)

      if self._metrics_collector:
        self._metrics_collector.get_time_per_prefill_token().observe(
            (
                request.metadata.transfer_enqueue_time
                - request.metadata.prefill_dequeue_time
            )
            / true_length
        )

      del prefill_result
      del request

  def _jax_transfer_prefill_result(
      self, new_request: ActiveRequest, target_idx: int
  ):
    new_request.prefill_result = jax.device_put(
        new_request.prefill_result,
        self._generate_engines[target_idx].get_prefix_destination_sharding(),
    )
    # Block here so we don't block on the generate thread that steps.
    jax.block_until_ready(new_request.prefill_result)

  def _ray_transfer_prefill_result(
      self, new_request: ActiveRequest, target_idx: int
  ):
    self._generate_engines[target_idx].transfer(new_request.prefill_result)

  def _transfer_prefill_result(
      self, new_request: ActiveRequest, target_idx: int
  ):
    if self._is_ray_backend:
      self._ray_transfer_prefill_result(new_request, target_idx)
    else:
      self._jax_transfer_prefill_result(new_request, target_idx)

  def _transfer_thread(self, idx: int):
    """Transfers the kv cache on an active request to the least full
    generate backlog."""
    transfer_backlog = self._transfer_backlogs[idx]

    while self.live:
      # The transfer thread can just sleep until it has work to do.
      new_request = transfer_backlog.get(block=True)
      if new_request is None:
        break
      new_request.metadata.transfer_dequeue_time = time.perf_counter()
      target_idx = min(
          self._generate_backlogs.items(), key=lambda q: q[1].qsize()
      )[0]
      # Only transfer the KVCache for the disaggregated serving.
      # TODO: Remove the conditional after fixing the compatibility.
      if not self._interleaved_mode:
        logging.info(
            "Transferring prefill from prefill engine %d "
            "to generate engine %d.",
            idx,
            target_idx,
        )
        # Transfer the info to the relevant generate slice.
        self._transfer_prefill_result(new_request, target_idx)
      # Place the request on the correct generate backlog and block if full.
      new_request.metadata.generate_enqueue_time = time.perf_counter()
      self._generate_backlogs[target_idx].put(new_request, block=True)
      logging.info(
          "Successfully transferred prefill "
          "from prefill engine %d to generate engine %d "
          "(%d requests now in backlog).",
          idx,
          target_idx,
          self._generate_backlogs[target_idx].qsize(),
      )

  def _generate_thread(self, idx: int):
    """Step token generation and insert prefills from backlog."""
    logging.info("---------Spinning up generate thread %d.---------", idx)
    generate_engine = self._generate_engines[idx]
    my_slots = self._generate_slots[idx]
    my_generate_backlog = self._generate_backlogs[idx]
    my_detokenize_backlog = self._detokenize_backlogs[idx]

    # Keep track of what step tokens were generated at.
    generate_timestep = 0
    # State to store things like running kv cache in.
    decode_state = generate_engine.init_decode_state()

    generate_params = self._generate_params[idx]
    logging.info("---------Generate params %d loaded.---------", idx)
    time_of_last_generate = time.time()
    time_of_last_print = time.time()
    while self.live:
      if (time.time() - time_of_last_print) > 1:
        logging.info(
            "Generate thread making a decision with:"
            " prefill_backlog=%d"
            " generate_free_slots=%d",
            self._prefill_backlog.qsize(),
            my_slots.qsize(),
        )
        time_of_last_print = time.time()

      max_concurrent_decodes = generate_engine.max_concurrent_decodes

      if self._metrics_collector:
        self._metrics_collector.get_slots_used_percentage_metric(
            idx
        ).set_function(
            lambda: float(1 - (my_slots.qsize() / max_concurrent_decodes))
        )

      # Check if there are any free my_slots. We don't want to block here since
      # we can still generate if we can't insert. We do this in a while loop to
      # insert as many sequences as possible.
      while True:
        my_slots_size = my_slots.qsize()

        try:
          slot = my_slots.get(block=False)
          # Found a slot, now see if we can fill it.
        except queue.Empty:
          # Exit this while loop as we have no free slots to insert into.
          break

        # We block when the decode slots are all free since we need to get a
        # prefilled request to insert. We add timeout for the block to handle
        # the case when the prefill backlog is cancelled and we end up with no
        # more useful prefill work to do.
        block = my_slots_size == max_concurrent_decodes
        if self._interleaved_mode:
          # For interleaved mode, we also blocks when prefill backlog
          # is not empty or there are transfer work to do.
          block |= not self._prefill_backlog.empty()
          block |= not self._transfer_backlogs[idx].empty()
        try:
          new_request = my_generate_backlog.get(block=block, timeout=1.0)
          if new_request is None:
            break
          new_request.metadata.generate_dequeue_time = time.perf_counter()
          if (
              self._metrics_collector
              and new_request.metadata.start_time is not None
          ):
            self._metrics_collector.get_queue_duration().observe(
                # Time in prefill queue
                new_request.metadata.prefill_dequeue_time
                - new_request.metadata.prefill_enqueue_time
                # Time in transfer queue
                + new_request.metadata.transfer_dequeue_time
                - new_request.metadata.transfer_enqueue_time
                # Time in generate queue
                + new_request.metadata.generate_dequeue_time
                - new_request.metadata.generate_enqueue_time
            )
          # Got free slot and new request, use them.
        except queue.Empty:
          # No new requests, we can't insert, so put back slot.
          my_slots.put(slot, block=False)
          # If we were blocking and hit the timeout, then retry the loop.
          # Otherwise, we can exit and proceed to generation.
          if block:
            continue
          else:
            break

        # Signal to kill the thread.
        if new_request is None:
          return

        logging.info(
            "Generate slice %d filling slot %d at step %d.",
            idx,
            slot,
            generate_timestep,
        )

        decode_state = generate_engine.insert(
            new_request.prefill_result, decode_state, slot=slot
        )
        del new_request.prefill_result
        new_request.generate_timestep_added = generate_timestep
        new_request.complete = np.zeros(
            (generate_engine.samples_per_slot,), dtype=np.bool_
        )
        # Respond to detokenization backpressure.
        my_detokenize_backlog.put((slot, new_request), block=True)

      # At this point, we know that we have at least some slots filled.
      assert (
          my_slots.qsize() < max_concurrent_decodes
      ), "At this point we must have some requests inserted into the slots."

      # Now we actually take a generate step on requests in the slots.
      decode_state, sampled_tokens = generate_engine.generate(
          generate_params, decode_state
      )
      sampled_tokens.copy_to_host_async()
      # Respond to detokenization backpressure.
      my_detokenize_backlog.put((generate_timestep, sampled_tokens), block=True)
      generate_timestep += 1
      logging.info(
          "Generate engine %d step %d - slots free : %d / %d, took %.2fms",
          idx,
          generate_timestep,
          my_slots_size,
          max_concurrent_decodes,
          (time.time() - time_of_last_generate) * 10**3,
      )
      time_of_last_generate = time.time()

  def _detokenize_thread(self, idx: int):
    """Detokenize sampled tokens and returns them to the user."""
    # One of these per generate engine.
    # For all filled my_slots, pop the sampled token onto the relevant
    # requests return channel. If it done, place it back onto free slots.
    my_detokenize_backlog = self._detokenize_backlogs[idx]
    my_generate_engine = self._generate_engines[idx]
    my_slots = self._generate_slots[idx]

    metadata = my_generate_engine.get_tokenizer()
    tokenizer = my_generate_engine.build_tokenizer(metadata)
    my_live_requests = {
        i: None for i in range(my_generate_engine.max_concurrent_decodes)
    }
    while self.live:
      data = my_detokenize_backlog.get(block=True)
      if data is None:
        break
      start_detokenize_time = time.time()
      # prefill first token
      if isinstance(data[0], engine_api.ResultTokens):
        request_first_token, request, _ = data
        request_first_token = request_first_token.convert_to_numpy()

        results, complete = token_utils.process_result_tokens(
            tokenizer=tokenizer,
            slot=0,  # always 0 as prefill only run 1 sample
            slot_max_length=request.max_tokens,
            result_tokens=request_first_token,
            is_client_side_tokenization=request.is_client_side_tokenization,
            complete=request.complete,
        )
        request.complete = complete
        # Return some output samples.
        request.enqueue_samples(results)

        first_token_return_time = time.perf_counter()
        if self._metrics_collector:
          self._metrics_collector.get_time_to_first_token().observe(
              first_token_return_time - request.metadata.prefill_dequeue_time
          )
        logging.info(
            "TTFT duration: %fms",
            (first_token_return_time - request.metadata.prefill_dequeue_time)
            * 1000,
        )
      # generate step tokens
      elif isinstance(data[1], engine_api.ResultTokens):
        # We want to detokenize them.
        generate_timestep_added, result_tokens = data
        # Disable attribute error because pytype doesn't know this
        # is a result tokens, and we can't annotate the tuple.
        result_tokens = result_tokens.convert_to_numpy()

        for slot, request in my_live_requests.items():
          if request is not None:
            results, complete = token_utils.process_result_tokens(
                tokenizer=tokenizer,
                slot=slot,
                slot_max_length=request.max_tokens,
                result_tokens=result_tokens,
                is_client_side_tokenization=request.is_client_side_tokenization,
                complete=request.complete,
            )
            request.complete = complete
            # Return some output samples.
            request.enqueue_samples(results)
            if request.complete.all():
              request.metadata.complete_time = time.perf_counter()
              request.return_channel.close()
              if self._metrics_collector:
                self._metrics_collector.get_request_output_length().observe(
                    result_tokens.get_result_at_slot(slot).lengths
                )
                self._metrics_collector.get_request_success_count_metric().inc()
                self._metrics_collector.get_time_per_output_token().observe(
                    (
                        request.metadata.complete_time
                        - request.metadata.transfer_enqueue_time
                    )
                    / result_tokens.get_result_at_slot(slot).lengths
                )
                self._metrics_collector.get_time_per_request().observe(
                    request.metadata.complete_time
                    - request.metadata.transfer_enqueue_time
                )

                if request.metadata.start_time:
                  total_time = (
                      request.metadata.complete_time
                      - request.metadata.start_time
                  )
                  prefill_time = (
                      request.metadata.transfer_enqueue_time
                      - request.metadata.prefill_dequeue_time
                  )
                  generate_time = (
                      request.metadata.complete_time
                      - request.metadata.generate_dequeue_time
                  )
                  self._metrics_collector.get_wait_time_per_request().observe(
                      total_time - prefill_time - generate_time
                  )
              # Place the slot back on the free queue.
              my_live_requests[slot] = None
              my_slots.put(slot, block=False)  # This should always have space.
              my_generate_engine.free_resource(slot)
        logging.info(
            "Detokenizing generate step %d took %.2fms",
            generate_timestep_added,
            (time.time() - start_detokenize_time) * 10**3,
        )
      else:
        # We want to update a slot with the new channel.
        slot, active_request = data
        my_live_requests[slot] = active_request


class LLMOrchestrator(jetstream_pb2_grpc.OrchestratorServicer):
  """Coordinates a set of prefill and generate slices for LLM decoding."""

  _driver: Driver

  def __init__(self, driver: Driver):
    self._driver = driver

  def _get_prefill_content(
      self, request: jetstream_pb2.DecodeRequest
  ) -> Tuple[str | list[int], bool]:
    which_content = request.WhichOneof("content")
    content = getattr(request, which_content)
    if which_content == "text_content":
      return cast(jetstream_pb2.DecodeRequest.TextContent, content).text, False
    else:
      return (
          list(
              cast(jetstream_pb2.DecodeRequest.TokenContent, content).token_ids
          ),
          True,
      )

  def process_client_side_tokenization_response(self, response: Any):
    samples = []
    for sample in response:
      samples.append(
          jetstream_pb2.DecodeResponse.StreamContent.Sample(
              token_ids=sample.token_ids,
          )
      )
    return jetstream_pb2.DecodeResponse(
        stream_content=jetstream_pb2.DecodeResponse.StreamContent(
            samples=samples
        )
    )

  def should_buffer_response(self, response: Any) -> bool:
    for item in response:
      if item.text and token_utils.is_byte_token(item.text[-1]):
        # If any sample ends in bytes, this means we might still need to
        # decode more bytes to compose the string.
        return True

  def process_server_side_tokenization_response(
      self, response: Any, buffered_response_list
  ):
    # Flush the buffered responses to each sample of current response.
    current_response_with_flushed_buffer = list(
        zip(*buffered_response_list, response)
    )
    # Empty buffer: [[s0_cur], [s1_cur], ...]
    # Has buffer:
    # [[s0_b0, s0_b1, ..., s0_cur], [s1_b0, s1_b1, ..., s1_cur], ...]
    current_response_with_flushed_buffer = cast(
        list[list[ReturnSample]], current_response_with_flushed_buffer
    )
    # Form correct sample(s) and return as StreamContent for this iteration.
    samples = []
    for sample in current_response_with_flushed_buffer:
      text = []
      token_ids = []
      for resp in sample:
        text.extend(resp.text)
        token_ids.extend(resp.token_ids)
      samples.append(
          jetstream_pb2.DecodeResponse.StreamContent.Sample(
              text=token_utils.text_tokens_to_str(text),
              token_ids=token_ids,
          )
      )
    return jetstream_pb2.DecodeResponse(
        stream_content=jetstream_pb2.DecodeResponse.StreamContent(
            samples=samples
        )
    )

  async def Decode(  # pylint: disable=invalid-overridden-method
      self,
      request: jetstream_pb2.DecodeRequest,
      context: Optional[grpc.aio.ServicerContext] = None,
  ) -> AsyncIterator[jetstream_pb2.DecodeResponse]:
    """Decode."""
    if context is None:
      logging.warning(
          "LLM orchestrator is being used in offline test mode, and will not"
          " respond to gRPC queries - only direct function calls."
      )
    is_client_side_tokenization = False
    return_channel = async_multifuture.AsyncMultifuture()
    if context:
      context.add_done_callback(return_channel.cancel)
    prefill_content, is_client_side_tokenization = self._get_prefill_content(
        request
    )
    # Wrap request as an ActiveRequest.
    active_request = ActiveRequest(
        max_tokens=request.max_tokens,
        prefill_content=prefill_content,
        is_client_side_tokenization=is_client_side_tokenization,
        return_channel=return_channel,
        metadata=ActiveRequestMetadata(
            start_time=request.metadata.start_time,
            prefill_enqueue_time=time.perf_counter(),
        ),
    )
    # The first stage is being prefilled, all other stages are handled
    # inside the driver (transfer, generate*N, detokenize).
    try:
      self._driver.place_request_on_prefill_queue(active_request)
    except queue.Full:
      # Safely abort the gRPC server thread with a retriable error.
      await _abort_or_raise(
          context=context,
          code=grpc.StatusCode.RESOURCE_EXHAUSTED,
          details=(
              "The driver prefill queue is full and more requests cannot be"
              " handled. You may retry this request."
          ),
      )
    logging.info(
        "Placed request on the prefill queue.",
    )
    # When an active request is created a queue is instantiated. New tokens
    # are placed there during the decoding loop, we pop from that queue by
    # using the .next method on the active request.
    # Yielding allows for the response to be a streaming grpc call - which
    # can be called via iterating over a for loop on the client side.
    # The DecodeResponse stream should consume all generated tokens in
    # return_channel when complete signal is received (AsyncMultifuture
    # promises this).
    buffered_response_list = []
    async for response in active_request.return_channel:
      response = cast(list[ReturnSample], response)
      if is_client_side_tokenization:
        # If is_client_side_tokenization, the client should request with token
        # ids, and the JetStream server will return token ids as response.
        # The client should take care of tokenization and detokenization.
        yield self.process_client_side_tokenization_response(response)
      else:
        # Buffer response mechanism is used to handle streaming
        # detokenization with special character (For some edge cases with
        # SentencePiece tokenizer, it requires to decode a complete sequence
        # instead of a single token).
        if self.should_buffer_response(response):
          buffered_response_list.append(response)
          continue
        yield self.process_server_side_tokenization_response(
            response, buffered_response_list
        )
        # Reset buffer after flushed.
        buffered_response_list = []

  async def HealthCheck(  # pylint: disable=invalid-overridden-method
      self,
      request: jetstream_pb2.HealthCheckRequest,
      context: Optional[grpc.aio.ServicerContext] = None,
  ) -> jetstream_pb2.HealthCheckResponse:
    """HealthCheck."""
    if context is None:
      logging.warning(
          "LLM orchestrator is being used in offline test mode, and will not"
          " respond to gRPC queries - only direct function calls."
      )
    is_live = self._driver.live
    return jetstream_pb2.HealthCheckResponse(is_live=is_live)


-----------------------

/jetstream/core/proto/__init__.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


-----------------------

/jetstream/core/proto/jetstream.proto:
-----------------------

// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// NOTICE: run `make generate-protos` if making changes to this file

syntax = "proto3";

package jetstream_proto;

// TODO: Merge this with main JetStream core once we settle on an API.

service Orchestrator {
  // Query LLM to generate text or tokens.
  rpc Decode(DecodeRequest) returns (stream DecodeResponse) {}
  // Checks if the model server is live.
  rpc HealthCheck(HealthCheckRequest) returns (HealthCheckResponse) {}
}

message DecodeRequest {
  // The maximum output length of a sequence. It's used in JetStream to control
  // the output/decode length of a sequence. It would not be used in the engine.
  // We should always set max_tokens <= (max_target_length -
  // max_prefill_predict_length). max_target_length is the maximum length of a
  // sequence; max_prefill_predict_length is the maximum length of the
  // input/prefill of a sequence.
  int32 max_tokens = 4;

  message TextContent {
    string text = 1;
  }
  message TokenContent {
    repeated int32 token_ids = 1;
  }

  // The client can pass the inputs either as a string, in which case the server will
  // tokenize it, or as tokens, in which case it's the client's responsibility to
  // ensure they tokenize its input strings with the correct tokenizer.
  oneof content {
    TextContent text_content = 5;
    TokenContent token_content = 6;
  }

  message Metadata {
    float start_time = 1;
  }

  oneof metadata_optional {
    Metadata metadata = 7;
  }

  reserved 1, 2, 3;
  // Next ID: 8
}

message DecodeResponse {
  // InitialContent supports returning initial one-off response data from the
  // stream. It's a placeholder for future features such as history cache.
  message InitialContent {}
  message StreamContent {
    message Sample {
      // The text string decoded from token id(s).
      string text = 1;
      // List of token ids, one list per sample. When speculative decoding is disabled, the list size should be 1; When speculative decoding is enabled, the list size should be >= 1.
      repeated int32 token_ids = 2;
    }
    // Supports multiple samples in the StreamContent. The Sample list size depends on text generation strategy the engine used.
    repeated Sample samples = 1;
  }

  oneof content {
    InitialContent initial_content = 2;
    StreamContent stream_content = 3;
  }
  reserved 1;
  // Next ID: 4
}

message HealthCheckRequest {}

message HealthCheckResponse {
  // Denotes whether the model server is live
  bool is_live = 1;
}

-----------------------

/jetstream/core/proto/jetstream_pb2.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: jetstream/core/proto/jetstream.proto
# Protobuf Python Version: 4.25.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n$jetstream/core/proto/jetstream.proto\x12\x0fjetstream_proto"\xfc\x02\n\rDecodeRequest\x12\x12\n\nmax_tokens\x18\x04 \x01(\x05\x12\x42\n\x0ctext_content\x18\x05 \x01(\x0b\x32*.jetstream_proto.DecodeRequest.TextContentH\x00\x12\x44\n\rtoken_content\x18\x06 \x01(\x0b\x32+.jetstream_proto.DecodeRequest.TokenContentH\x00\x12;\n\x08metadata\x18\x07 \x01(\x0b\x32\'.jetstream_proto.DecodeRequest.MetadataH\x01\x1a\x1b\n\x0bTextContent\x12\x0c\n\x04text\x18\x01 \x01(\t\x1a!\n\x0cTokenContent\x12\x11\n\ttoken_ids\x18\x01 \x03(\x05\x1a\x1e\n\x08Metadata\x12\x12\n\nstart_time\x18\x01 \x01(\x02\x42\t\n\x07\x63ontentB\x13\n\x11metadata_optionalJ\x04\x08\x01\x10\x02J\x04\x08\x02\x10\x03J\x04\x08\x03\x10\x04"\xcb\x02\n\x0e\x44\x65\x63odeResponse\x12I\n\x0finitial_content\x18\x02 \x01(\x0b\x32..jetstream_proto.DecodeResponse.InitialContentH\x00\x12G\n\x0estream_content\x18\x03 \x01(\x0b\x32-.jetstream_proto.DecodeResponse.StreamContentH\x00\x1a\x10\n\x0eInitialContent\x1a\x81\x01\n\rStreamContent\x12\x45\n\x07samples\x18\x01 \x03(\x0b\x32\x34.jetstream_proto.DecodeResponse.StreamContent.Sample\x1a)\n\x06Sample\x12\x0c\n\x04text\x18\x01 \x01(\t\x12\x11\n\ttoken_ids\x18\x02 \x03(\x05\x42\t\n\x07\x63ontentJ\x04\x08\x01\x10\x02"\x14\n\x12HealthCheckRequest"&\n\x13HealthCheckResponse\x12\x0f\n\x07is_live\x18\x01 \x01(\x08\x32\xb9\x01\n\x0cOrchestrator\x12M\n\x06\x44\x65\x63ode\x12\x1e.jetstream_proto.DecodeRequest\x1a\x1f.jetstream_proto.DecodeResponse"\x00\x30\x01\x12Z\n\x0bHealthCheck\x12#.jetstream_proto.HealthCheckRequest\x1a$.jetstream_proto.HealthCheckResponse"\x00\x62\x06proto3'
)

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(
    DESCRIPTOR, "jetstream.core.proto.jetstream_pb2", _globals
)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals["_DECODEREQUEST"]._serialized_start = 58
  _globals["_DECODEREQUEST"]._serialized_end = 438
  _globals["_DECODEREQUEST_TEXTCONTENT"]._serialized_start = 294
  _globals["_DECODEREQUEST_TEXTCONTENT"]._serialized_end = 321
  _globals["_DECODEREQUEST_TOKENCONTENT"]._serialized_start = 323
  _globals["_DECODEREQUEST_TOKENCONTENT"]._serialized_end = 356
  _globals["_DECODEREQUEST_METADATA"]._serialized_start = 358
  _globals["_DECODEREQUEST_METADATA"]._serialized_end = 388
  _globals["_DECODERESPONSE"]._serialized_start = 441
  _globals["_DECODERESPONSE"]._serialized_end = 772
  _globals["_DECODERESPONSE_INITIALCONTENT"]._serialized_start = 607
  _globals["_DECODERESPONSE_INITIALCONTENT"]._serialized_end = 623
  _globals["_DECODERESPONSE_STREAMCONTENT"]._serialized_start = 626
  _globals["_DECODERESPONSE_STREAMCONTENT"]._serialized_end = 755
  _globals["_DECODERESPONSE_STREAMCONTENT_SAMPLE"]._serialized_start = 714
  _globals["_DECODERESPONSE_STREAMCONTENT_SAMPLE"]._serialized_end = 755
  _globals["_HEALTHCHECKREQUEST"]._serialized_start = 774
  _globals["_HEALTHCHECKREQUEST"]._serialized_end = 794
  _globals["_HEALTHCHECKRESPONSE"]._serialized_start = 796
  _globals["_HEALTHCHECKRESPONSE"]._serialized_end = 834
  _globals["_ORCHESTRATOR"]._serialized_start = 837
  _globals["_ORCHESTRATOR"]._serialized_end = 1022
# @@protoc_insertion_point(module_scope)


-----------------------

/jetstream/core/proto/jetstream_pb2_grpc.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from jetstream.core.proto import jetstream_pb2 as jetstream_dot_core_dot_proto_dot_jetstream__pb2


class OrchestratorStub(object):
  """TODO: Merge this with main JetStream core once we settle on an API."""

  def __init__(self, channel):
    """Constructor.

    Args:
        channel: A grpc.Channel.
    """
    self.Decode = channel.unary_stream(
        "/jetstream_proto.Orchestrator/Decode",
        request_serializer=jetstream_dot_core_dot_proto_dot_jetstream__pb2.DecodeRequest.SerializeToString,
        response_deserializer=jetstream_dot_core_dot_proto_dot_jetstream__pb2.DecodeResponse.FromString,
    )
    self.HealthCheck = channel.unary_unary(
        "/jetstream_proto.Orchestrator/HealthCheck",
        request_serializer=jetstream_dot_core_dot_proto_dot_jetstream__pb2.HealthCheckRequest.SerializeToString,
        response_deserializer=jetstream_dot_core_dot_proto_dot_jetstream__pb2.HealthCheckResponse.FromString,
    )


class OrchestratorServicer(object):
  """TODO: Merge this with main JetStream core once we settle on an API."""

  def Decode(self, request, context):
    """Query LLM to generate text or tokens."""
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details("Method not implemented!")
    raise NotImplementedError("Method not implemented!")

  def HealthCheck(self, request, context):
    """Checks if the model server is live."""
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details("Method not implemented!")
    raise NotImplementedError("Method not implemented!")


def add_OrchestratorServicer_to_server(servicer, server):
  rpc_method_handlers = {
      "Decode": grpc.unary_stream_rpc_method_handler(
          servicer.Decode,
          request_deserializer=jetstream_dot_core_dot_proto_dot_jetstream__pb2.DecodeRequest.FromString,
          response_serializer=jetstream_dot_core_dot_proto_dot_jetstream__pb2.DecodeResponse.SerializeToString,
      ),
      "HealthCheck": grpc.unary_unary_rpc_method_handler(
          servicer.HealthCheck,
          request_deserializer=jetstream_dot_core_dot_proto_dot_jetstream__pb2.HealthCheckRequest.FromString,
          response_serializer=jetstream_dot_core_dot_proto_dot_jetstream__pb2.HealthCheckResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      "jetstream_proto.Orchestrator", rpc_method_handlers
  )
  server.add_generic_rpc_handlers((generic_handler,))


# This class is part of an EXPERIMENTAL API.
class Orchestrator(object):
  """TODO: Merge this with main JetStream core once we settle on an API."""

  @staticmethod
  def Decode(
      request,
      target,
      options=(),
      channel_credentials=None,
      call_credentials=None,
      insecure=False,
      compression=None,
      wait_for_ready=None,
      timeout=None,
      metadata=None,
  ):
    return grpc.experimental.unary_stream(
        request,
        target,
        "/jetstream_proto.Orchestrator/Decode",
        jetstream_dot_core_dot_proto_dot_jetstream__pb2.DecodeRequest.SerializeToString,
        jetstream_dot_core_dot_proto_dot_jetstream__pb2.DecodeResponse.FromString,
        options,
        channel_credentials,
        insecure,
        call_credentials,
        compression,
        wait_for_ready,
        timeout,
        metadata,
    )

  @staticmethod
  def HealthCheck(
      request,
      target,
      options=(),
      channel_credentials=None,
      call_credentials=None,
      insecure=False,
      compression=None,
      wait_for_ready=None,
      timeout=None,
      metadata=None,
  ):
    return grpc.experimental.unary_unary(
        request,
        target,
        "/jetstream_proto.Orchestrator/HealthCheck",
        jetstream_dot_core_dot_proto_dot_jetstream__pb2.HealthCheckRequest.SerializeToString,
        jetstream_dot_core_dot_proto_dot_jetstream__pb2.HealthCheckResponse.FromString,
        options,
        channel_credentials,
        insecure,
        call_credentials,
        compression,
        wait_for_ready,
        timeout,
        metadata,
    )


-----------------------

/jetstream/core/server_lib.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains common functions for a JetStream core server.

See implementations/*/sever.py for examples.
"""

import asyncio
from concurrent import futures
import logging
import os
import signal
import threading
import time
import traceback
from typing import Any, Type


import grpc
import jax
from jetstream.core import config_lib
from jetstream.core import orchestrator
from jetstream.core.metrics.prometheus import JetstreamMetricsCollector
from jetstream.core.proto import jetstream_pb2_grpc
from jetstream.engine import warmup_utils, engine_api

from prometheus_client import start_http_server

_HOST = "[::]"


class JetStreamServer:
  """JetStream grpc server."""

  def __init__(
      self, driver: orchestrator.Driver, threads: int, port, credentials
  ):
    self._executor = futures.ThreadPoolExecutor(max_workers=threads)

    self._loop = asyncio.new_event_loop()
    self._loop.set_default_executor(self._executor)
    self._loop_thread = threading.Thread(target=self._loop.run_forever)
    self._loop_thread.start()

    async def do_init():
      self._grpc_server = grpc.aio.server(
          self._executor,
      )

    asyncio.run_coroutine_threadsafe(do_init(), loop=self._loop).result()
    self._driver = driver
    jetstream_pb2_grpc.add_OrchestratorServicer_to_server(
        orchestrator.LLMOrchestrator(driver=self._driver), self._grpc_server
    )
    self._grpc_server.add_secure_port(f"{_HOST}:{port}", credentials)

  async def _async_start(self) -> None:
    await self._grpc_server.start()

  def start(self) -> None:
    asyncio.run_coroutine_threadsafe(
        self._async_start(), loop=self._loop
    ).result()

  async def _async_stop(self) -> None:
    await self._grpc_server.stop(grace=10)

  def stop(self) -> None:
    # Gracefully clean up threads in the orchestrator.
    self._driver.stop()
    asyncio.run_coroutine_threadsafe(self._async_stop(), self._loop).result()
    self._loop.call_soon_threadsafe(self._loop.stop)
    self._loop_thread.join()

  def wait_for_termination(self) -> None:
    try:
      asyncio.run_coroutine_threadsafe(
          self._grpc_server.wait_for_termination(), self._loop
      ).result()
    finally:
      self.stop()


def create_driver(
    config: Type[config_lib.ServerConfig],
    devices: Any,
    jax_padding: bool = True,
    metrics_collector: JetstreamMetricsCollector | None = None,
    enable_model_warmup: bool = False,
):
  """Creates a driver with a specified config.

  Args:
    config: A ServerConfig to config engine, model, device slices, etc.
    devices: Device objects, will be used to get engine with proper slicing.
    jax_padding: The flag to enable JAX padding during tokenization.
    metrics_collector: The JetStream Promethus metric collector.
    enable_model_warmup: The flag to enable model server warmup.

  Returns:
    An orchestrator driver.
  """
  engines = config_lib.get_engines(config, devices=devices)
  prefill_params = [pe.load_params() for pe in engines.prefill_engines]
  generate_params = [ge.load_params() for ge in engines.generate_engines]
  shared_params = [ie.load_params() for ie in engines.interleaved_engines]
  logging.info("Loaded all weights.")
  interleaved_mode = (
      len(config.prefill_slices) + len(config.generate_slices) == 0
  )

  prefill_engines = engines.prefill_engines + engines.interleaved_engines
  generate_engines = engines.generate_engines + engines.interleaved_engines
  prefill_params = prefill_params + shared_params
  generate_params = generate_params + shared_params

  if prefill_engines is None:
    prefill_engines = []
  if generate_engines is None:
    generate_engines = []
  if prefill_params is None:
    prefill_params = []
  if generate_params is None:
    generate_params = []

  if enable_model_warmup:
    prefill_engines = [engine_api.JetStreamEngine(pe) for pe in prefill_engines]
    generate_engines = [
        engine_api.JetStreamEngine(ge) for ge in generate_engines
    ]

    try:
      _ = warmup_utils.layout_params_and_compile_executables(
          prefill_engines,  # pylint: disable=protected-access
          generate_engines,  # pylint: disable=protected-access
          prefill_params,  # pylint: disable=protected-access
          generate_params,  # pylint: disable=protected-access
      )

    except ValueError as e:
      print(f"Model warmup encountered an error: {e}")
      traceback.print_exc()
      os.kill(os.getpid(), signal.SIGKILL)

  return orchestrator.Driver(
      prefill_engines=prefill_engines,
      generate_engines=generate_engines,
      prefill_params=prefill_params,
      generate_params=generate_params,
      interleaved_mode=interleaved_mode,
      jax_padding=jax_padding,
      metrics_collector=metrics_collector,
      is_ray_backend=config.is_ray_backend,
  )


def run(
    port: int,
    config: Type[config_lib.ServerConfig],
    devices: Any,
    credentials: Any = grpc.insecure_server_credentials(),
    threads: int | None = None,
    jax_padding: bool = True,
    metrics_server_config: config_lib.MetricsServerConfig | None = None,
    enable_jax_profiler: bool = False,
    jax_profiler_port: int = 9999,
    enable_model_warmup: bool = False,
) -> JetStreamServer:
  """Runs a server with a specified config.

  Args:
    port: Port on which the server will be made available.
    config: A ServerConfig to config engine, model, device slices, etc.
    devices: Device objects, will be used to get engine with proper slicing.
    credentials: Should use grpc credentials by default.
    threads: Number of RPC handlers worker threads. This should be at least
      equal to the decoding batch size to fully saturate the decoding queue.
    jax_padding: The flag to enable JAX padding during tokenization.
    metrics_server_config: The config to enable Promethus metric server.
    enable_jax_profiler: The flag to enable JAX profiler server.
    jax_profiler_port: The port JAX profiler server (default to 9999).
    enable_model_warmup: The flag to enable model server warmup.

  Returns:
    JetStreamServer that wraps the grpc server and orchestrator driver.
  """
  server_start_time = time.time()
  logging.info("Kicking off gRPC server.")
  # Setup Prometheus server
  metrics_collector: JetstreamMetricsCollector = None
  if metrics_server_config and metrics_server_config.port:
    logging.info(
        "Starting Prometheus server on port %d", metrics_server_config.port
    )
    start_http_server(metrics_server_config.port)
    metrics_collector = JetstreamMetricsCollector()
  else:
    logging.info(
        "Not starting Prometheus server: --prometheus_port flag not set"
    )

  driver = create_driver(
      config, devices, jax_padding, metrics_collector, enable_model_warmup
  )
  # We default threads to the total number of concurrent allowed decodes,
  # to make sure we can fully saturate the model. Set default minimum to 64.
  threads = threads or max(driver.get_total_concurrent_requests(), 64)
  jetstream_server = JetStreamServer(driver, threads, port, credentials)
  logging.info("Starting server on port %d with %d threads", port, threads)

  jetstream_server.start()

  if metrics_collector:
    metrics_collector.get_server_startup_latency_metric().set(
        time.time() - server_start_time
    )

  # Setup Jax Profiler
  if enable_jax_profiler:
    logging.info("Starting JAX profiler server on port %s", jax_profiler_port)
    jax.profiler.start_server(jax_profiler_port)
  else:
    logging.info("Not starting JAX profiler server: %s", enable_jax_profiler)

  # Start profiling server by default for proxy backend.
  if jax.config.jax_platforms and "proxy" in jax.config.jax_platforms:
    from jetstream.core.utils import proxy_util  # pylint: disable=import-outside-toplevel

    thread = threading.Thread(
        target=proxy_util.start_profiling_server, args=(jax_profiler_port,)
    )
    thread.run()

  return jetstream_server


def get_devices() -> Any:
  """Gets devices."""
  # TODO: Add more logs for the devices.
  devices = jax.devices()
  logging.info("Using devices: %d", len(devices))
  return devices


-----------------------

/jetstream/core/utils/__init__.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


-----------------------

/jetstream/core/utils/async_multifuture.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AsyncMultifuture is a data structure utility.

It is a data structure that returns multiple futures asynchronously.
"""

import asyncio
from concurrent import futures
import threading
from typing import Any, Generic, TypeVar

V = TypeVar("V")


class _Exception:
  """A class for propagating exceptions through a queue.

  By wrapping them with a custom private class we ensure that any type
  (including Exception) can be used as a V.
  """

  def __init__(self, exception: Exception) -> None:
    self.exception = exception


class AsyncMultifuture(Generic[V]):
  """AsyncMultifuture is like concurrent.futures.Future but supports returning

  multiple results. It provides an unidirectional stream with buffering and
  exception propagation.

  Supports delivering results to an async Python event loop. Must be
  constructed inside of the event loop.
  """

  def __init__(self) -> None:
    self._cancelled = threading.Event()
    self._done = threading.Event()
    self._loop = asyncio.get_running_loop()
    self._queue = asyncio.Queue[V | _Exception]()

  def cancel(self, unused: Any = None) -> None:
    """Cancels the asyncmultifuture."""
    # Needed for compatibility with grpc.aio.ServicerContext.add_done_callback.
    del unused
    self._cancelled.set()
    self.set_exception(futures.CancelledError())

  def cancelled(self) -> bool:
    """Returns whether the asyncmultifuture has been cancelled."""
    return self._cancelled.is_set()

  def done(self) -> bool:
    """AsyncMultifuture is done when it is finalized with close() or

    set_exception().
    """
    return self._done.is_set()

  def set_exception(self, exception: Exception) -> None:
    """Stores the given exception in the asyncmultifuture.

    The exception would be delivered after all previously added results are
    yielded. set_exception can be called multiple times, however subsequent
    calls will be ignored.

    Args:
      exception: The exception to set.
    """
    self._loop.call_soon_threadsafe(
        self._queue.put_nowait, _Exception(exception)
    )
    self._loop.call_soon_threadsafe(self._done.set)

  def add_result(self, result: V) -> None:
    """Adds the result to the asyncmultifuture.

    Caller must call .close() once all results are added.

    Args:
      result: The result to add.
    """
    self._loop.call_soon_threadsafe(self._queue.put_nowait, result)

  def close(self) -> None:
    """Notifies the receiver that no more results would be added."""
    self.set_exception(StopAsyncIteration())

  def __aiter__(self) -> "AsyncMultifuture":
    return self

  async def __anext__(self) -> V:
    """Returns the next value."""
    value = await self._queue.get()
    if isinstance(value, _Exception):
      raise value.exception
    return value


-----------------------

/jetstream/core/utils/proxy_util.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Proxy util functions."""

import dataclasses
import logging
import jax
import time
from fastapi import FastAPI
import uvicorn


# TODO: add a manner way to terminate.
def start_profiling_server(port: int):

  logging.info("Starting JAX profiler server on port %s", port)
  app = FastAPI()

  @dataclasses.dataclass
  class ProfilingConfig:
    seconds: int
    output_dir: str

  @app.post("/profiling")
  async def profiling(pc: ProfilingConfig):
    jax.profiler.start_trace(pc.output_dir)
    logging.info("Capturing the profiling data for next %s seconds", pc.seconds)
    time.sleep(pc.seconds)
    logging.info("Writing profiling data to %s", pc.output_dir)
    jax.profiler.stop_trace()
    return {"response": "profiling completed"}

  @app.get("/")
  async def root():
    return {"message": "Hello from proxy profiling server"}

  uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


-----------------------

/jetstream/core/utils/return_sample.py:
-----------------------

"""ReturnSample is a data structure utility.

It is a data structure that stores the return samples.
"""

import dataclasses


@dataclasses.dataclass
class ReturnSample:
  """Both the token ids, their string representation, and other data.

  Attributes:
    text: Text piece(s) detokenized from token id(s).
    token_ids: Raw result token id(s).
  """

  text: list[str]
  token_ids: list[int]


-----------------------

/jetstream/engine/README.md:
-----------------------

# Inference Engine Subpackage - Defines Engine API Interface

The Engine class in `engine_api.py` is the Engine API Interface.

mock_engine.py is the mock implementation of the Engine API Interface.


-----------------------

/jetstream/engine/__init__.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Initialization for any Engine implementation."""

import jax

try:
  import pathwaysutils
except ImportError as e:
  print("Proxy backend support is not added")
  pass


-----------------------

/jetstream/engine/engine_api.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines the JetStream API.

These functions are the accelerator functions which an outer sampling loop
could want to call, enabling interleaved (continuous batching) inference.
"""

import abc
from typing import Any, Optional, Tuple, Union, Callable

from flax import struct
import jax
import numpy as np

from jetstream.engine import tokenizer_pb2
from jetstream.engine import token_utils


# The model parameters - their partitioning will be unique for different prefill
# and decode topoologies.
Params = Any
# The result of a prefill operation, often a batch size 1 KVCache.
Prefix = Any
# The inputs into a generation step, often a prefill and generate cache tuple.
DecodeState = Any
# Accelerator representation of tokens.
DeviceTokens = Any
# Cpus asscociated with the mesh.
CpuDevices = Any
# Tokenkizer used by the engine
Tokenizer = Any


@struct.dataclass
class SlotData:
  """Class to store slot data."""

  tokens: Union[jax.Array, np.ndarray]
  valid: Union[jax.Array, np.ndarray]
  lengths: Union[jax.Array, np.ndarray]


# pylint: disable=g-doc-args
@struct.dataclass
class ResultTokens(abc.ABC):
  """Class to store returned tokens in.

  We store everything in one array, and keep indexes - because copying
  a single array to host is much faster.
  Each tuple represents the indices of the relevant data.
  """

  # Shape: [batch, tokens.shape[1] + validity.shape[1] + lengths.shape[1]]
  data: Union[jax.Array, np.ndarray]
  # The range of indices which contain tokens.
  tokens_idx: tuple[int, int] = struct.field(
      pytree_node=False,
  )
  # The range of indices which contain the validity of
  # the tokens.
  valid_idx: tuple[int, int] = struct.field(
      pytree_node=False,
  )
  # The range of indices which contain the lengths up till now of the lengths
  # of each generated sequence.
  length_idx: tuple[int, int] = struct.field(
      pytree_node=False,
  )
  samples_per_slot: int = struct.field(
      pytree_node=False,
  )

  def copy_to_host_async(self: "ResultTokens") -> None:
    """Copy to host asynchronously."""
    # Do nothing for np array
    if isinstance(self.data, np.ndarray):
      return
    self.data.copy_to_host_async()

  def convert_to_numpy(self: "ResultTokens") -> "ResultTokens":
    """Converts to numpy."""
    return ResultTokens(
        np.array(self.data),
        self.tokens_idx,
        self.valid_idx,
        self.length_idx,
        self.samples_per_slot,
    )

  def get_result_at_slot(self, slot: int) -> SlotData:
    """Returns the token at a given slot.

    Args:
      slot: An integer from [0, n) representing an index into the batch.

    Note: implementations of this method must correctly handle
    microbatches, if microbatches are used.
    """
    # Potentially get multiple beams for given slot.
    start_idx = slot * self.samples_per_slot
    end_idx = (slot + 1) * self.samples_per_slot
    # Mask out any non valid tokens.
    return SlotData(
        tokens=self.data[
            start_idx:end_idx, self.tokens_idx[0] : self.tokens_idx[1]
        ],
        valid=self.data[
            start_idx:end_idx, self.valid_idx[0] : self.valid_idx[1]
        ],
        # Only get a 1D representation here
        lengths=self.data[
            start_idx:end_idx, self.length_idx[0] : self.length_idx[1]
        ][:, 0],
    )


class Engine(abc.ABC):
  """The computational core of the generative model server.

  Engine defines an API that models must adhere to as they plug into the
  JetStream efficient serving infrastructure.
  """

  @abc.abstractmethod
  def prefill(
      self,
      *,
      params: Params,
      existing_prefix: Optional[Prefix] = None,
      padded_tokens: jax.Array,
      true_length: int,
      sampler: Optional[Callable[[Any], Any]] = None,
  ) -> Tuple[Prefix, ResultTokens]:
    """Computes a kv-cache for a set of tokens conditional on existing cache.

    existing_prefix (if provided) represents a prefix that has already been
    processed by the underlying model. tokens is logically appended
    to the text represented by `existing_prefix`. This method returns a new
    kv_cache (typically) for the resulting text.

    If sampler is passed, then the engine should use it do sample next token.
    """

  @abc.abstractmethod
  def generate(
      self,
      params: Params,
      decode_state: DecodeState,
      sampler: Optional[Callable[[Any], Any]] = None,
  ) -> Tuple[DecodeState, ResultTokens]:
    """Generates tokens for each sequence being decoded in parallel.

    Generate takes a batch of pre-computed kv-caches, and computes:
      - the predicted next token for each of the sequences
      - an updated set of kv-caches

    In the case of pipelining, this will handle N cycles (where each cycle
    consists of each microbatch progressing through every stage), in
    non-pipelined code this is a full forward pass. In both cases, this accounts
    for a full embed-layerstack-unembed-sample operation.

    If sampler is passed, then the engine should use it do sample next token.
    """

  @abc.abstractmethod
  def insert(
      self,
      prefix: Prefix,
      decode_state: DecodeState,
      slot: int,
  ) -> DecodeState:
    """Adds `new_request` into `caches` at 'slot'.

    When decoding multiple requests in parallel, when one request finishes, a
    new request must be slotted into the recently vacated spot: `insert`!

    This can occur in between and async to generate calls, and takes a lock over
    that row of the cache.

    The slot may represent a tuple of positions (e.g. microbatch, pipeline stage
    and batch), but at the engine interface level all of these are exposed as
    a [0, n) range of slots and converted internally.
    """

  def free_resource(
      self,
      slot: int,  # pylint: disable=unused-argument
  ) -> Any:
    """Free cache and other decode resource for the slot.

    This function is needed for advanced attetnion kenel like PageAttetion.
    After finishing one request, the engine need to free all used page block
    resource and reuse for coming requests.
    """
    return None

  @abc.abstractmethod
  def load_params(self, *args, **kwargs) -> Params:
    """Loads parameters.

    May not be used in full production form, where weights are part of the saved
    model.
    """

  @abc.abstractmethod
  def get_prefix_destination_sharding(self) -> Any:
    """Returns the shardings necessary to transfer data between engines."""

  @abc.abstractmethod
  def get_tokenizer(
      self,
  ) -> tokenizer_pb2.TokenizerParameters:
    """Returns the info to construct a tokenizer in py/c++."""

  def build_tokenizer(
      self,
      metadata: tokenizer_pb2.TokenizerParameters,
  ) -> Tokenizer:
    """Builds a new tokenizer object and returns it."""
    return token_utils.SentencePieceTokenizer(metadata)

  @abc.abstractmethod
  def init_decode_state(self, *args, **kwargs) -> DecodeState:
    """Initialises any state which a generation step transforms."""

  @property
  @abc.abstractmethod
  def max_concurrent_decodes(self) -> int:
    """Total capacity."""

  @property
  @abc.abstractmethod
  def samples_per_slot(self) -> int:
    """Total samples per slot."""

  @property
  @abc.abstractmethod
  def max_prefill_length(self) -> int:
    """Maximum prefill length."""

  @property
  @abc.abstractmethod
  def mesh(self) -> jax.sharding.Mesh:
    """Mesh which the engine is running on."""

  @property
  @abc.abstractmethod
  def colocated_cpus(self) -> Union[list[CpuDevices], None]:
    """CPU devices colocated with the engine's accelerators."""


class JetStreamEngine(Engine):
  """A wrapper engine of the Engine class.

  JetStreamEngine defines the warmed up model server engine.
  """

  def __init__(self, downstream_engine: Engine):
    self._downstream_engine = downstream_engine

    self.prefill_buckets = None
    self.warm = False

  def prefill(
      self,
      *,
      params: Params,
      existing_prefix: Optional[Prefix] = None,
      padded_tokens: jax.Array,
      true_length: int,
  ) -> Tuple[Prefix, ResultTokens]:

    prefill_result, first_token = self._downstream_engine.prefill(
        params=params,
        padded_tokens=padded_tokens,
        true_length=true_length,
    )
    return prefill_result, first_token

  def insert(
      self,
      prefix: Prefix,
      decode_state: DecodeState,
      slot: int,
  ) -> DecodeState:

    decode_state = self._downstream_engine.insert(
        prefix=prefix,
        decode_state=decode_state,
        slot=slot,
    )
    return decode_state

  def generate(
      self, params: Params, decode_state: DecodeState
  ) -> Tuple[DecodeState, ResultTokens]:
    decode_state, sampled_tokens = self._downstream_engine.generate(
        params=params, decode_state=decode_state
    )
    return decode_state, sampled_tokens

  def load_params(self, *args, **kwargs) -> Params:
    return self._downstream_engine.load_params(*args, **kwargs)

  def get_prefix_destination_sharding(self) -> Any:
    return self._downstream_engine.get_prefix_destination_sharding()

  def get_tokenizer(
      self,
  ) -> tokenizer_pb2.TokenizerParameters:
    return self._downstream_engine.get_tokenizer()

  def build_tokenizer(
      self,
      metadata: tokenizer_pb2.TokenizerParameters,
  ) -> Tokenizer:
    """Builds a new tokenizer object and returns it."""
    return self._downstream_engine.build_tokenizer(metadata)

  def init_decode_state(self, *args, **kwargs) -> DecodeState:
    return self._downstream_engine.init_decode_state(*args, **kwargs)

  @property
  def max_concurrent_decodes(self) -> int:
    return self._downstream_engine.max_concurrent_decodes

  @property
  def samples_per_slot(self) -> int:
    return self._downstream_engine.samples_per_slot

  @property
  def max_prefill_length(self) -> int:
    return self._downstream_engine.max_prefill_length

  @property
  def mesh(self) -> jax.sharding.Mesh:
    return self._downstream_engine.mesh

  @property
  def colocated_cpus(self) -> Union[list[CpuDevices], None]:
    return self._downstream_engine.colocated_cpus


-----------------------

/jetstream/engine/mock_engine.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Simple test engine for the JetStream API described.

Contains simple functions that we can hand calculate the desired outcome of.

Prefill: Doubles the sequence by multiplying it with an integer weight.
Insert: Writes this sequence into a cache row.
Generate step: Return sum(prefill_cache) + sum(generate_cache)/weight.

I.e. if we prefill [2, 65, 66] (i.e. <BOS>, 'A', 'B') using an ACII vocab,
we should get [4, 130, 132].

If we then insert that and run three generation steps, we should see
266+0 / 2 = 266
266 + [266] /2  = 399
266 + [266, 399] /2 = 598
I.e. ['Ċ', 'Ə', 'ɖ'] when converted back with chr()
"""

import functools
from typing import Any, Optional, Tuple

from flax import struct
import jax
from jax.experimental import mesh_utils
import jax.numpy as jnp

from jetstream.engine import engine_api
from jetstream.engine import tokenizer_pb2


Params = jax.Array  # [1,].
Prefix = jax.Array  # [batch,] of strings with different lengths.


@struct.dataclass
class DecodeState:
  """The inputs into a generation step."""

  prefill_cache: jax.Array
  generate_cache: jax.Array
  generate_cache_index: int
  generate_lengths: jax.Array
  generate_tokens: jax.Array


class TestEngine(engine_api.Engine):
  """The computational core of the generative model server.

  Engine defines an API that models must adhere to as they plug into the
  JetStream efficient serving infrastructure.
  """

  def __init__(self, batch_size: int, cache_length: int, weight: float):
    self.prefill_cache_batch = batch_size
    self.generate_cache_batch = batch_size
    self.cache_length = cache_length
    self.weight = weight
    self._mesh = jax.sharding.Mesh(
        mesh_utils.create_device_mesh((1, 1, 1), jax.devices()), ("x", "y", "z")
    )

  def load_params(self) -> Params:
    """Loads model weights."""
    # An integer, used to multiply inputs.
    return jnp.array([self.weight], dtype=jnp.float32)

  @functools.partial(jax.jit, static_argnums=(0,))
  def prefill(
      self,
      *,
      params: Params,
      existing_prefix: Optional[jax.Array] = None,
      padded_tokens: jax.Array,
      true_length: int,
  ) -> Tuple[Prefix, engine_api.ResultTokens]:
    """Computes a kv-cache for a new generate request.

    Args:
      params: Scalar multiplier.
      existing_prefix: If provided, represents a prefix that has already been
        processed by the underlying model.
      padded_tokens: Logically appended tokens to any existing prefix, this is
        what we compute prefill on.
      true_length: The real length of the tokens, pre-pad.
    Returns:
      kv_cache: For the resulting text.
    """
    if existing_prefix is not None:
      raise NotImplementedError
    del true_length
    assert padded_tokens.ndim == 1
    # Wait to simulate model step time.
    fake_size = 4096
    fake_work = jnp.ones((fake_size, fake_size)) @ jnp.ones(
        (fake_size, fake_size)
    )
    # Do some fake work that isn't eliminated by dead code elimination (DCE).
    params = params + fake_work.mean() - fake_work.mean()
    prefill_cache = padded_tokens[None, :] * params

    # get dummy first token
    first_step = (prefill_cache.sum(axis=-1))[:, jnp.newaxis]
    first_token_data = jnp.concatenate(
        [first_step, jnp.ones_like(first_step), jnp.ones_like(first_step)],
        axis=-1,
    )
    speculations = first_step.shape[1]
    first_token = engine_api.ResultTokens(
        data=first_token_data.astype(jnp.int32),
        tokens_idx=(0, speculations),
        # Validity occupies the same amount of space, but next in line.
        valid_idx=(speculations, 2 * speculations),
        # And lengths is rank 1.
        length_idx=(2 * speculations, 2 * speculations + 1),
        samples_per_slot=self.generate_cache_batch // self.prefill_cache_batch,
    )

    return (prefill_cache, first_step), first_token

  @functools.partial(jax.jit, static_argnums=(0,))
  def generate(
      self, params: Params, decode_state: DecodeState
  ) -> Tuple[DecodeState, engine_api.ResultTokens]:
    """Generates tokens for each sequence being decoded in parallel."""
    (
        prefill_cache,
        generate_cache,
        generate_cache_index,
        generate_lengths,
        previous_timestep,
    ) = (
        decode_state.prefill_cache,
        decode_state.generate_cache,
        decode_state.generate_cache_index,
        decode_state.generate_lengths,
        decode_state.generate_tokens,
    )

    # Update generate cache
    generate_cache = jax.lax.dynamic_update_slice_in_dim(
        generate_cache,
        previous_timestep,
        start_index=generate_cache_index,
        axis=1,
    )
    generate_cache_index = (generate_cache_index + 1) % self.cache_length

    # Sum each row of prefill cache and generate cache to produce new timestep,
    # multiply by params.
    l_iota = jax.lax.broadcasted_iota(
        jnp.int32,
        (self.generate_cache_batch, self.cache_length),
        dimension=1,
    )

    # The generate cache should be circular and right aligned.
    # TODO: Do we need a left aligned one to test spec sampling?
    # Don't need the + 1 you normally would, because we don't provide a
    # token from prefill in the dummy.
    # This iota and masking is to allow for a cicular cache.
    length_mask = (
        -(l_iota - generate_cache_index) % self.cache_length
    ) <= generate_lengths[:, None]
    length_masked_gen_cache = generate_cache * length_mask
    new_timestep = (
        prefill_cache.sum(axis=-1)
        + (length_masked_gen_cache.sum(axis=-1) / params)
    )[:, jnp.newaxis]
    # Wait to simulate model step time.
    fake_size = 4096
    fake_work = jnp.ones((fake_size, fake_size)) @ jnp.ones(
        (fake_size, fake_size)
    )
    # Do some fake work that isn't eliminated by dead code elimination (DCE).
    generate_cache = generate_cache + fake_work.mean() - fake_work.mean()
    new_lengths = generate_lengths + 1
    speculations = new_timestep.shape[1]
    # Concatenates the tokens, their validity and the lengths of each sequence
    # into one tensor so that copy operations are faster on Cloud TPU
    # infrastructure.
    token_data = jnp.concatenate(
        [new_timestep, jnp.ones_like(new_timestep), new_lengths[:, None]],
        axis=-1,
    )
    return DecodeState(
        prefill_cache=prefill_cache,
        generate_cache=generate_cache,
        generate_cache_index=generate_cache_index,
        generate_lengths=new_lengths,
        generate_tokens=new_timestep,
    ), engine_api.ResultTokens(
        data=token_data.astype(jnp.int32),
        # Tokens are shape [batch, speculations], so when we concatenate
        # tokens, validity and length along their index 1 dimension then they
        # occupy 0:speculations.
        tokens_idx=(0, speculations),
        # Validity occupies the same amount of space, but next in line.
        valid_idx=(speculations, 2 * speculations),
        # And lengths is rank 1.
        length_idx=(2 * speculations, 2 * speculations + 1),
        samples_per_slot=self.generate_cache_batch // self.prefill_cache_batch,
    )

  @functools.partial(jax.jit, static_argnums=(0,), donate_argnums=(2,))
  def insert(
      self,
      prefix: Prefix,
      decode_state: DecodeState,
      slot: int,
  ) -> DecodeState:
    """Adds `prefix` into `decode_state` at `slot`."""
    # [B, T], [T,] -> [B, T]
    prefill_cache, previous_timestep = prefix
    prefill_cache = jax.lax.dynamic_update_slice_in_dim(
        decode_state.prefill_cache, prefill_cache, slot, axis=0
    )
    generate_cache = jax.lax.dynamic_update_slice_in_dim(
        decode_state.generate_cache,
        jnp.zeros((1, self.cache_length)),
        slot,
        axis=0,
    )
    samples_per_slot = self.generate_cache_batch // self.prefill_cache_batch
    generate_lengths = jax.lax.dynamic_update_slice_in_dim(
        decode_state.generate_lengths,
        jnp.ones((samples_per_slot), dtype=jnp.int32),
        slot * samples_per_slot,
        axis=0,
    )
    generate_tokens = jax.lax.dynamic_update_slice_in_dim(
        decode_state.generate_tokens,
        previous_timestep,
        slot * samples_per_slot,
        axis=0,
    )
    return decode_state.replace(
        prefill_cache=prefill_cache,
        generate_cache=generate_cache,
        generate_lengths=generate_lengths,
        generate_tokens=generate_tokens,
    )

  def get_prefix_destination_sharding(self) -> Any:
    return jax.sharding.NamedSharding(
        mesh=self.mesh, spec=jax.sharding.PartitionSpec()
    )

  def get_tokenizer(self) -> tokenizer_pb2.TokenizerParameters:
    """Return a protobuf of tokenizer info, callable from Py or C++."""
    return tokenizer_pb2.TokenizerParameters(path="test", extra_ids=0)

  def init_decode_state(self) -> DecodeState:
    """Initialises any state which a generation step transforms."""
    return DecodeState(
        prefill_cache=jnp.zeros(
            (self.prefill_cache_batch, self.cache_length), dtype=jnp.float32
        ),
        generate_cache=jnp.zeros(
            (self.generate_cache_batch, self.cache_length), dtype=jnp.float32
        ),
        generate_cache_index=0,
        generate_lengths=jnp.zeros(
            (self.generate_cache_batch), dtype=jnp.int32
        ),
        generate_tokens=jnp.zeros(
            (self.generate_cache_batch, 1), dtype=jnp.float32
        ),
    )

  @property
  def max_concurrent_decodes(self) -> int:
    """Free slots."""
    return self.prefill_cache_batch

  @property
  def max_prefill_length(self) -> int:
    """Maximum prefill length."""
    return self.cache_length

  @property
  def samples_per_slot(self) -> int:
    """Number of samples per slot."""
    return self.generate_cache_batch // self.max_concurrent_decodes

  @property
  def mesh(self) -> jax.sharding.Mesh:
    """Mesh which the engine is running on."""
    return self._mesh

  @property
  def colocated_cpus(self) -> None:
    """CPU devices colocated with the engine's accelerators."""
    raise NotImplementedError


-----------------------

/jetstream/engine/mock_utils.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utils for engine operation."""

from typing import List, Sequence
from flax import struct
import numpy as np
from seqio.vocabularies import Vocabulary


@struct.dataclass
class TestTokenizer:
  """Tokenizer used for testing purposes."""

  def IdToPiece(self, integer: int) -> str:  # pylint: disable=invalid-name
    """In the real version, unlike encode_tf/decode_tf, doesn't strip trailing

    whitespace.
    """
    return chr(integer)

  def decode(self, tokens: np.ndarray):  # pylint: disable=invalid-name
    """Converts a numpy array into a string.

    Uses tokens[0] as we are doing streaming decode now
    """
    return chr(tokens[0])


@struct.dataclass
class TestVocab(Vocabulary):
  """Mock vocabulary used for tests.

  These methods are duplicative on the test vocab, but required to fit
  the seqio.Vocabulary interface.
  """

  pad_id = 0
  eos_id = 1
  bos_id = 2
  unk_id = 3
  stop_tokens = {pad_id, eos_id}
  _base_vocab_size = 2**16
  tokenizer: TestTokenizer = TestTokenizer()

  def _encode(self, s: str) -> Sequence[int]:
    """Converts a string into a integer sequenc."""
    # 'We use array methods, not python iterables so we don't
    # implement this method in the mock vocab.
    raise NotImplementedError

  def _decode(self, ids: np.ndarray):
    """Converts a numpy array into a string."""
    return "".join([chr(r) for r in list(ids) if r not in self.stop_tokens])

  def _encode_tf(self, s: str) -> np.ndarray:
    """Converts a string into a numpy array."""
    # We mock using numpy to avoid propagating tf dependencies.
    chars = np.array([ord(c) for c in s]).astype(np.int32)
    return chars

  def _decode_tf(self, ids: np.ndarray) -> List[str]:
    """Converts a numpy array into a string."""
    # We mock using numpy to avoid propagating tf dependencies.
    results = np.split(ids, ids.shape[0])
    return ["".join([chr(r) for r in list(line[0])]) for line in results]

  def decode(self, ids: np.ndarray, is_streaming=True):
    """Converts a numpy array into a string."""
    return is_streaming and self._decode(ids)

  def encode_tf(self, s: str) -> np.ndarray:
    """Converts a string into a numpy array."""
    return self._encode_tf(s)

  def decode_tf(self, ids: np.ndarray) -> List[str]:
    """Converts a numpy array into a string."""
    return self._decode_tf(ids)


-----------------------

/jetstream/engine/sampling_utils.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=bare-except, consider-using-generator
""" Inference sampling utilities.

    Inspired by an Google-internal implementation, Global Vision Transformer.
"""

import jax
import jax.numpy as jnp

NEG_INF = -1.0e7  # Masking purpose


def sampling(logits, rng, algorithm, topk=0, nucleus_topp=0, temperature=1.0):
  """
  logits: unnormalized logits to sample, shaped [YOUR_LEADING_DIMS, Vocab],
  before logit
  rng: rng key to use
  algorithm: string representing supported algorithms
  topk: restricting to topk logits before sampling
  nucleus_topp: restricting to p probability mass before sampling
  temperature: temperature parameter for scaling probability
  """
  if algorithm == "greedy":
    return jnp.argmax(logits, axis=-1)
  elif algorithm == "weighted":
    return jax.random.categorical(rng, logits / temperature)
  elif algorithm == "nucleus":
    return sample_nucleus_topp_logits(logits, nucleus_topp, temperature, rng)
  elif algorithm == "topk":
    return sample_topk_logits(logits, topk, temperature, rng)
  else:
    raise ValueError(f"Sampling {algorithm=} not supported!")


def sample_nucleus_topp_logits(logits, nucleus_topp, temperature, rng):
  """Restrict sampling to the top logits with cumulative probability >=
  nucleus_topp.

  The nucleus sampling method is proposed in the paper `The Curious Case of
  Neural Text Degeneration (https://arxiv.org/pdf/1904.09751.pdf)`

  """
  if nucleus_topp < 0:
    raise ValueError(
        "Can't apply nucleus with parameter {nucleus_topp=} less zero"
    )
  logits_sorted = jnp.sort(logits, axis=-1)[..., ::-1]  # sort descending
  sorted_cum_probs = jnp.cumsum(
      jax.nn.softmax(logits_sorted, axis=-1), axis=-1
  )  # get cumsum probs
  cutoff_index = jnp.sum(
      sorted_cum_probs < nucleus_topp, axis=-1, keepdims=True
  )  # find cutoff index
  cutoff_logit = jnp.take_along_axis(logits_sorted, cutoff_index, axis=-1)
  logits = jnp.where(
      logits < cutoff_logit, jnp.full_like(logits, NEG_INF), logits
  )
  return jax.random.categorical(rng, logits / temperature)


def sample_topk_logits(logits, topk, temperature, rng):
  """Restricting sampling to the best k logits."""
  if topk <= 0:
    raise ValueError("Can't apply algorithm topk with parameter {topk=} <= 0")
  topk_logits, topk_idxs = jax.lax.top_k(logits, topk)
  topk_token = jnp.expand_dims(
      jax.random.categorical(rng, topk_logits / temperature).astype(jnp.int32),
      axis=-1,
  )
  sampled_tokens = jnp.squeeze(
      jnp.take_along_axis(topk_idxs, topk_token, axis=-1), axis=-1
  ).astype(jnp.int32)
  return sampled_tokens


-----------------------

/jetstream/engine/token_utils.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Token manipulation utilities."""

from bisect import bisect_left
import logging
from typing import Any, Iterable, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from seqio.vocabularies import SentencePieceVocabulary
from seqio.vocabularies import Vocabulary

from jetstream.core.utils.return_sample import ReturnSample
from jetstream.engine import mock_utils
from jetstream.engine import tokenizer_api
from jetstream.engine import tokenizer_pb2
from jetstream.third_party.llama3 import llama3_tokenizer

# ResultToken class to store tokens ids.
ResultTokens = Any

DEFAULT_PREFILL_BUCKETS = [
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
]


def take_nearest_length(lengths: list[int], length: int) -> int:
  """Gets the nearest length to the right in a set of lengths."""
  pos = bisect_left(lengths, length)
  if pos == len(lengths):
    return lengths[-1]
  return lengths[pos]


def tokenize_and_pad(
    s: str,
    vocab: Vocabulary,
    is_bos: bool = True,
    prefill_lengths: Optional[List[int]] = None,
    max_prefill_length: Optional[int] = None,
    jax_padding: bool = True,
) -> Tuple[Union[jax.Array, np.ndarray], int]:
  """Tokenize and pads a string.

  Args:
    s: String to tokenize.
    vocab: Vocabulary to tokenize with.
    is_bos: Whether or not this is the beginning of a sequence. Default to yes
      as prefill is typically used when beginning sequences.
    prefill_lengths: Buckets to pad the sequence to for static compilation.
    max_prefill_length: Maximum bucket to use.
    jax_padding: convert to JAX padded tokens if True.

  Returns:
    tokens: Tokenized into integers.
    true_length: Actual length of the non-padded sequence.
  """

  tokens = np.array(vocab.encode_tf(s))  # [Length]
  bos_id = vocab.bos_id
  pad_id = vocab.pad_id
  assert pad_id == 0, "Further logic required if pad_id not 0."

  padded_tokens, true_length = pad_tokens(
      tokens=tokens,
      bos_id=bos_id,
      pad_id=pad_id,
      is_bos=is_bos,
      prefill_lengths=prefill_lengths,
      max_prefill_length=max_prefill_length,
      jax_padding=jax_padding,
  )
  return padded_tokens, true_length


def pad_tokens(
    tokens: np.ndarray,
    bos_id: int,
    pad_id: int,
    is_bos: bool = True,
    prefill_lengths: Optional[List[int]] = None,
    max_prefill_length: Optional[int] = None,
    jax_padding: bool = True,
) -> Tuple[Union[jax.Array, np.ndarray], int]:
  """Pads tokens to the nearest prefill length that is equal to or greater
     than the token length.

  Args:
    tokens: Tokens.
    bos_id: Bos ID.
    pad_id: Pad ID.
    is_bos: Add a beginning of sequence token if this is ture.
    prefill_lengths: Buckets to pad the sequence to for static compilation.
    max_prefill_length: Maximum bucket to use.
    jax_padding: convert to JAX padded tokens if True.

  Returns:
    tokens: Tokenized into integers.
    true_length: Actual length of the non-padded sequence.
  """
  if prefill_lengths is None:
    prefill_lengths = DEFAULT_PREFILL_BUCKETS
  if max_prefill_length is not None:
    prefill_lengths = prefill_lengths[
        : prefill_lengths.index(max_prefill_length)
    ] + [
        max_prefill_length,
    ]
  # Add a beginning of sequence token if this is the beginning.
  if is_bos:
    tokens = np.concatenate(
        [
            np.array(
                [
                    bos_id,
                ]
            ),
            tokens,
        ],
        axis=-1,
    )
  true_length = tokens.shape[-1]
  padded_length = take_nearest_length(prefill_lengths, true_length)
  padding = padded_length - true_length
  if padding < 0:
    logging.warning("Provided sequence longer than available.")
    # Take the last N tokens if we have too many.
    padded_tokens = tokens[-padded_length:]
  else:
    padded_tokens = np.pad(tokens, (0, padding), constant_values=(pad_id,))
  if jax_padding:
    padded_tokens = jnp.array(padded_tokens)
  return padded_tokens, true_length


def process_result_tokens(
    tokenizer: tokenizer_api.Tokenizer,
    slot: int,
    slot_max_length: int,
    result_tokens: ResultTokens,
    complete: np.ndarray,
    is_client_side_tokenization: bool = False,
    debug: bool = False,
) -> Tuple[List[ReturnSample], np.ndarray]:
  """Processes a result tokens into a list of strings, handling multiple
    samples.

  Args:
    slot: The slot at which to draw tokens from.
    slot_max_length: Max length for a sample in the slot.
    result_tokens: The tokens to access by slot.
    complete: Array representing the completion status of each sample in the
      slot.
    is_client_side_tokenization: Whether to detokenize on client side.
    debug: Whether to log step by step detokenisation.

  Returns:
    return_samples: List of ReturnSample.
    complete: Updated complete.
  """
  # tokens: [samples, speculations]
  slot_data = result_tokens.get_result_at_slot(slot)
  slot_tokens = slot_data.tokens
  slot_valid = slot_data.valid
  slot_lengths = slot_data.lengths
  samples, speculations = slot_tokens.shape
  stop_tokens = tokenizer.stop_tokens
  # Stop anything which has reached it's max length.
  complete = complete | (slot_lengths > slot_max_length)
  if debug:
    logging.info(
        "Complete %s, slot_tokens: %s, slot_lengths: %s",
        str(complete),
        str(slot_tokens),
        str(slot_lengths),
    )
  return_samples = []
  for idx in range(samples):
    text_so_far = []
    tok_id_so_far = []
    if not complete[idx].item():
      for spec_idx in range(speculations):
        tok_id = slot_tokens[idx, spec_idx].item()
        valid = slot_valid[idx, spec_idx].item()
        if debug:
          logging.info(
              "Sample idx: %d Speculation idx: %d Token: %d",
              idx,
              spec_idx,
              tok_id,
          )
        if tok_id in stop_tokens or not valid:
          complete[idx] = True
          tok_id_so_far.append(tok_id)
          break
        else:
          if not is_client_side_tokenization:
            if isinstance(tokenizer, SentencePieceTokenizer):
              text_so_far.append(tokenizer.decode([tok_id], is_streaming=True))
            else:
              text_so_far.append(tokenizer.decode([tok_id]))
          tok_id_so_far.append(tok_id)
    return_samples.append(
        ReturnSample(text=text_so_far, token_ids=tok_id_so_far)
    )
    if debug:
      logging.info("Return samples %s", str(return_samples))
  return return_samples, complete


def load_vocab(path: str, extra_ids: int = 0) -> Vocabulary:
  """Eagerly loads a vocabulary.

  Args:
    path: Vocabulary file path.
    extra_ids: Number of extra IDs.

  Returns:
    A seqio Vocabulary.
  """
  if path == "test":
    return mock_utils.TestVocab()
  else:
    vocab = SentencePieceVocabulary(
        path,
        extra_ids=extra_ids,
    )
    # SentencePieceVocabulary uses lazy loading. Request access to a property,
    # forcing the lazy loading to happen now.
    sp_model = vocab.sp_model
    del sp_model
    return vocab


def is_byte_token(s: str) -> bool:
  """Returns True if s is a byte string like "<0xAB>"."""
  # Bytes look like "<0xAB>".
  if len(s) != 6 or s[0:3] != "<0x" or s[-1] != ">":
    return False
  return True


def text_tokens_to_str(text_tokens: Iterable[str]) -> str:
  """Converts an iterable of token text to a single string, collapsing bytes.

  e.g. ['你', '好', '<0xE5>', '<0x90>', '<0x97>', 'hello'] -> '你好吗hello'
  """
  bytes_so_far = []
  for text_token in text_tokens:
    if is_byte_token(text_token):
      bytes_so_far.append(bytes([int(text_token[1:-1], 16)]))
    else:
      bytes_so_far.append(bytes(text_token, "utf-8"))
  return b"".join(bytes_so_far).decode("utf-8", "replace")


class SentencePieceTokenizer(tokenizer_api.Tokenizer):
  """Tokenizer to convert strings to token ids and vice-versa."""

  def __init__(self, metadata: tokenizer_pb2.TokenizerParameters):
    self.vocab = load_vocab(metadata.path, metadata.extra_ids)

  def encode(
      self, s: str, **kwargs
  ) -> Tuple[Union[jax.Array, np.ndarray], int]:
    """Tokenize a string.
    Args:
        s: String to tokenize.
        **kwargs: Additional keyword arguments.
    Returns:
        tokens: Tokenized into integers.
        true_length: Actual length of the non-padded sequence
          if padding is used.
    """
    is_bos = kwargs.pop("is_bos", True)
    prefill_lengths = kwargs.pop("prefill_lengths", None)
    max_prefill_length = kwargs.pop("max_prefill_length", None)
    jax_padding = kwargs.pop("jax_padding", True)

    tokens = np.array(self.vocab.encode_tf(s))

    tokens, true_length = pad_tokens(
        tokens,
        self.bos_id,
        self.pad_id,
        is_bos=is_bos,
        prefill_lengths=prefill_lengths,
        max_prefill_length=max_prefill_length,
        jax_padding=jax_padding,
    )
    return tokens, true_length

  def decode(self, token_ids: list[int], **kwargs) -> str:
    """Processess input token ids to generate a string.
    Args:
      token_ids: List of token ids.
      **kwargs: Additional keyword arguments.
    Returns:
      str: String generated from the token ids.
    """
    # If is_streaming, we need to decode a token id to a piece.
    is_streaming = kwargs.pop("is_streaming", False)
    if is_streaming:
      # The piece could be a byte token or a text token. It requires further
      # processing for the byte tokens. For JetStream, it's handled in
      # LLMOrchestrator.
      piece = self.vocab.tokenizer.IdToPiece(token_ids[0])
      # SentencePiece escapes the whitespace with a meta symbol "▁" (U+2581)
      return piece.replace("▁", " ")
    else:
      # If it's not streaming decoding, we can directly decode the full list
      # of token ids to a complete sequence.
      return self.vocab.tokenizer.decode(token_ids)

  @property
  def pad_id(self) -> int:
    """ID of the pad token."""
    return self.vocab.pad_id

  @property
  def eos_id(self) -> int:
    """ID of EOS token."""
    return self.vocab.eos_id

  @property
  def bos_id(self) -> int:
    """ID of the BOS token."""
    return self.vocab.bos_id


class TikToken(tokenizer_api.Tokenizer):
  """Tokenizer to convert strings to token ids and vice-versa."""

  def __init__(self, metadata: tokenizer_pb2.TokenizerParameters):
    self.tokenizer = llama3_tokenizer.Tokenizer(metadata.path)

  def encode(
      self, s: str, **kwargs
  ) -> Tuple[Union[jax.Array, np.ndarray], int]:
    """Tokenize a string.
    Args:
        s: String to tokenize.
        **kwargs: Additional keyword arguments
    Returns:
        tokens: Tokenized into integers.
        true_length: Actual length of the non-padded sequence
          if padding is used.
    """
    is_bos = kwargs.pop("is_bos", True)
    prefill_lengths = kwargs.pop("prefill_lengths", None)
    max_prefill_length = kwargs.pop("max_prefill_length", None)
    jax_padding = kwargs.pop("jax_padding", True)

    tokens = np.array(self.tokenizer.encode(s, bos=False, eos=False))

    tokens, true_length = pad_tokens(
        tokens,
        self.bos_id,
        self.pad_id,
        is_bos=is_bos,
        prefill_lengths=prefill_lengths,
        max_prefill_length=max_prefill_length,
        jax_padding=jax_padding,
    )
    return tokens, true_length

  def decode(self, token_ids: list[int]) -> str:
    """Processess input token ids to generate a string.
    Args:
      token_ids: List of token ids.
    Returns:
      str: String generated from the token ids.
    """
    return self.tokenizer.decode(token_ids)

  @property
  def stop_tokens(self) -> set[int]:
    """ID of the stop token."""
    return self.tokenizer.stop_tokens

  @property
  def pad_id(self) -> int:
    """ID of the pad token."""
    return self.tokenizer.pad_id

  @property
  def eos_id(self) -> int:
    """ID of EOS token."""
    return self.tokenizer.eos_id

  @property
  def bos_id(self) -> int:
    """ID of the BOS token."""
    return self.tokenizer.bos_id


-----------------------

/jetstream/engine/tokenizer.proto:
-----------------------

// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

syntax = "proto3";

package engine;

option java_multiple_files = true;

message TokenizerParameters {
  string path = 1;
  int32 extra_ids = 2;
}


-----------------------

/jetstream/engine/tokenizer_api.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines the JetStream Tokenizer API."""

import abc
from typing import Any, Tuple, Union

import numpy as np
import jax

# Class to store token ids.
ResultTokens = Any


class Tokenizer(abc.ABC):
  """Tokenizer to convert strings to token ids and vice-versa."""

  @abc.abstractmethod
  def encode(
      self, s: str, **kwargs
  ) -> Tuple[Union[jax.Array, np.ndarray], int]:
    """Tokenize a string.
    Args:
        s: String to tokenize.
        **kwargs: Additional keyword arguments.
    Returns:
        tokens: Tokenized into integers.
        true_length: Actual length of the non-padded sequence
          if padding is used.
    """

  @abc.abstractmethod
  def decode(self, token_ids: list[int], **kwargs) -> str:
    """Processess input token ids to generate a string.
    Args:
      token_ids: List of token ids.
      **kwargs: Additional keyword arguments.
    Returns:
      str: String generated from the token ids.
    """

  @property
  @abc.abstractmethod
  def pad_id(self) -> int:
    """ID of the pad token."""

  @property
  @abc.abstractmethod
  def eos_id(self) -> int:
    """ID of EOS token."""

  @property
  @abc.abstractmethod
  def bos_id(self) -> int:
    """ID of BOS token."""

  @property
  def stop_tokens(self) -> set[int]:
    """ID of the stop token."""
    return {self.eos_id, self.pad_id}


-----------------------

/jetstream/engine/tokenizer_pb2.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: jetstream/engine/tokenizer.proto
# Protobuf Python Version: 4.25.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n jetstream/engine/tokenizer.proto\x12\x06\x65ngine"6\n\x13TokenizerParameters\x12\x0c\n\x04path\x18\x01 \x01(\t\x12\x11\n\textra_ids\x18\x02 \x01(\x05\x42\x02P\x01\x62\x06proto3'
)

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(
    DESCRIPTOR, "jetstream.engine.tokenizer_pb2", _globals
)
if _descriptor._USE_C_DESCRIPTORS == False:
  _globals["DESCRIPTOR"]._options = None
  _globals["DESCRIPTOR"]._serialized_options = b"P\001"
  _globals["_TOKENIZERPARAMETERS"]._serialized_start = 44
  _globals["_TOKENIZERPARAMETERS"]._serialized_end = 98
# @@protoc_insertion_point(module_scope)


-----------------------

/jetstream/engine/tokenizer_pb2_grpc.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc


-----------------------

/jetstream/engine/warmup_utils.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model server warmup utils."""

import jax.numpy as jnp
import concurrent.futures
from typing import Any, Optional
import logging
from jetstream.engine import engine_api, token_utils


def layout_params_and_compile_executables(
    prefill_engines: Optional[list[engine_api.JetStreamEngine]] = None,
    generate_engines: Optional[list[engine_api.JetStreamEngine]] = None,
    prefill_params: Optional[list[Any]] = None,
    generate_params: Optional[list[Any]] = None,
) -> bool:
  """Organizes the engines and executables.

  Args:
      prefill_engines: Prefill only engines.
      generate_engines: Generate only engines.
      prefill_params: Prefill only params.
      generate_params: Generate only params.
  """
  prefill_engines = prefill_engines if prefill_engines else []
  generate_engines = generate_engines if generate_engines else []
  prefill_params = prefill_params if prefill_params else []
  generate_params = generate_params if generate_params else []

  any_prefill_engine = None
  any_prefill_params = None

  prefills_compiled = []
  inserts_generate_compiled = []

  for i, pe in enumerate(prefill_engines):
    any_prefill_engine = pe
    any_prefill_params = prefill_params[i]
    prefill_compiled = initialize_prefill_jit_cache(
        prefill_engine=pe,
        prefill_params=prefill_params[i],
        prefill_idx=i,
    )
    prefills_compiled.append(prefill_compiled)

  for i, ge in enumerate(generate_engines):
    insert_generate_compiled = initialize_insert_generate_jit_cache(
        prefill_engine=any_prefill_engine,
        generate_engine=ge,
        prefill_params=any_prefill_params,
        generate_params=generate_params[i],
        generate_idx=i,
    )
    inserts_generate_compiled.append([insert_generate_compiled])

  if all(prefills_compiled) and all(inserts_generate_compiled):
    return True
  return False


def initialize_prefill_jit_cache(
    *,
    prefill_engine: engine_api.JetStreamEngine,
    prefill_params: Any,
    prefill_idx: int,
):
  """Precompile all prefill functions in parallel.
  If we don't do this, then when a new request triggers a new prefill bucket it
  will take a very long time for that query to come back.

  Args:
      prefill_engine: A prefill engine to be compiled for.
      prefill_params: The associated prefill parameters.
      prefill_idx: Which prefill engine it is.
  """
  prefill_buckets = token_utils.DEFAULT_PREFILL_BUCKETS
  prefill_buckets = [
      bucket
      for bucket in prefill_buckets
      if bucket <= prefill_engine.max_prefill_length
  ]
  prefill_engine.prefill_buckets = prefill_buckets
  if prefill_engine.max_prefill_length not in prefill_buckets:
    prefill_buckets.append(prefill_engine.max_prefill_length)

  def compile_prefill(length):
    padded_tokens, true_length = jnp.ones((length), dtype="int32"), length

    _, _ = prefill_engine._downstream_engine.prefill(  # pylint: disable=protected-access
        params=prefill_params,
        padded_tokens=padded_tokens,
        true_length=true_length,
    )

    logging.info(
        "---------Prefill engine %d compiled for prefill length %d.---------",
        prefill_idx,
        length,
    )

  logging.info("---------Prefill compilation %d begun.---------", prefill_idx)

  with concurrent.futures.ThreadPoolExecutor(
      max_workers=len(prefill_buckets)
  ) as executor:
    _ = executor.map(compile_prefill, prefill_buckets)

  prefill_engine.warm = True

  logging.info(
      "---------Prefill compilation %d complete.---------", prefill_idx
  )

  return prefill_engine.warm


def initialize_insert_generate_jit_cache(
    *,
    prefill_engine: engine_api.JetStreamEngine,
    generate_engine: engine_api.JetStreamEngine,
    prefill_params: Any,
    generate_params: Any,
    generate_idx: int,
):
  """Initialiszes jit cache for insert and generate.

  Args:
      generate_engine: A generate engine to be compiled for.
      generate_params: The associated parameters.
      generate_idx: Which generate engine it is.
  """

  prefill_buckets = token_utils.DEFAULT_PREFILL_BUCKETS
  prefill_buckets = [
      bucket
      for bucket in prefill_buckets
      if bucket <= generate_engine.max_prefill_length
  ]
  generate_engine.prefill_buckets = prefill_buckets
  if generate_engine.max_prefill_length not in prefill_buckets:
    prefill_buckets.append(generate_engine.max_prefill_length)

  decode_state = generate_engine.init_decode_state()

  def compile_insert(length):
    padded_tokens, true_length = jnp.ones((length), dtype="int32"), length

    prefill, _ = prefill_engine._downstream_engine.prefill(  # pylint: disable=protected-access
        params=prefill_params,
        padded_tokens=padded_tokens,
        true_length=true_length,
    )

    generate_engine.insert(prefix=prefill, decode_state=decode_state, slot=0)

    logging.info(
        "---------Generate engine %d compiled for insert length %d.---------",
        generate_idx,
        length,
    )

  def compile_generate():

    logging.info(
        "---------Generate compilation %d begun.---------", generate_idx
    )

    generate_engine._downstream_engine.generate(  # pylint: disable=protected-access
        params=generate_params,
        decode_state=decode_state,
    )

    logging.info(
        "---------Generate engine %d compiled.---------",
        generate_idx,
    )

    logging.info(
        "---------Generate compilation %d complete.---------", generate_idx
    )

  logging.info(
      "---------Insertion generation compilation %d begun.---------",
      generate_idx,
  )

  compile_generate()

  logging.info(
      "---------Generate engine %d compiled generation step.---------",
      generate_idx,
  )

  with concurrent.futures.ThreadPoolExecutor(
      max_workers=len(prefill_buckets)
  ) as executor:
    _ = executor.map(compile_insert, prefill_buckets)

  generate_engine.warm = True

  logging.info(
      "---------Insertion generation compilation %d complete.---------",
      generate_idx,
  )

  return generate_engine.warm


-----------------------

/jetstream/entrypoints/__init__.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


-----------------------

/jetstream/entrypoints/config.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Config for JetStream Server (including engine init)."""

from typing import Type

from jetstream.core import config_lib


def get_server_config(
    config_str: str,
) -> config_lib.ServerConfig | Type[config_lib.ServerConfig]:
  match config_str:
    case "InterleavedCPUTestServer":
      server_config = config_lib.InterleavedCPUTestServer
    case "CPUTestServer":
      server_config = config_lib.CPUTestServer
    case _:
      raise NotImplementedError
  return server_config


-----------------------

/jetstream/entrypoints/http/__init__.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


-----------------------

/jetstream/entrypoints/http/api_server.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""JetStream Http API server."""

import json
import logging
import time
from typing import Sequence
from absl import app as abslapp
from absl import flags
from fastapi import APIRouter, Response
import fastapi
from fastapi.responses import StreamingResponse
from prometheus_client import start_http_server
import uvicorn
from google.protobuf.json_format import Parse

from jetstream.core import config_lib, orchestrator, server_lib
from jetstream.core.metrics.prometheus import JetstreamMetricsCollector
from jetstream.core.proto import jetstream_pb2
from jetstream.entrypoints.config import get_server_config
from jetstream.entrypoints.http.protocol import DecodeRequest
from jetstream.entrypoints.http.utils import proto_to_json_generator

flags.DEFINE_string("host", "0.0.0.0", "server host address")
flags.DEFINE_integer("port", 8080, "http server port")
flags.DEFINE_string(
    "config",
    "InterleavedCPUTestServer",
    "available servers",
)
flags.DEFINE_integer(
    "prometheus_port",
    9988,
    "prometheus_port",
)

llm_orchestrator: orchestrator.LLMOrchestrator

# Define Fast API endpoints (use llm_orchestrator to handle).
router = APIRouter()


@router.get("/")
def root():
  """Root path for Jetstream HTTP Server."""
  return Response(
      content=json.dumps({"message": "JetStream HTTP Server"}, indent=4),
      media_type="application/json",
  )


@router.post("/v1/generate")
async def generate(request: DecodeRequest):
  start_time = time.perf_counter()
  proto_request = Parse(request.json(), jetstream_pb2.DecodeRequest())
  metadata = jetstream_pb2.DecodeRequest.Metadata()
  metadata.start_time = start_time
  proto_request.metadata.CopyFrom(metadata)
  generator = llm_orchestrator.Decode(proto_request)
  return StreamingResponse(
      content=proto_to_json_generator(generator), media_type="text/event-stream"
  )


@router.get("/v1/health")
async def health() -> Response:
  """Health check."""
  response = await llm_orchestrator.HealthCheck(
      jetstream_pb2.HealthCheckRequest()
  )
  return Response(
      content=json.dumps({"is_live": str(response.is_live)}, indent=4),
      media_type="application/json",
      status_code=200,
  )


def server(argv: Sequence[str]):
  # Init Fast API.
  app = fastapi.FastAPI()
  app.include_router(router)

  # Init LLMOrchestrator which would be the main handler in the api endpoints.
  devices = server_lib.get_devices()
  print(f"devices: {devices}")
  server_config = get_server_config(flags.FLAGS.config)
  print(f"server_config: {server_config}")
  del argv

  metrics_server_config: config_lib.MetricsServerConfig | None = None
  # Setup Prometheus server
  metrics_collector: JetstreamMetricsCollector = None
  if flags.FLAGS.prometheus_port != 0:
    metrics_server_config = config_lib.MetricsServerConfig(
        port=flags.FLAGS.prometheus_port
    )
    logging.info(
        "Starting Prometheus server on port %d", metrics_server_config.port
    )
    start_http_server(metrics_server_config.port)
    metrics_collector = JetstreamMetricsCollector()
  else:
    logging.info(
        "Not starting Prometheus server: --prometheus_port flag not set"
    )

  global llm_orchestrator
  llm_orchestrator = orchestrator.LLMOrchestrator(
      driver=server_lib.create_driver(
          config=server_config,
          devices=devices,
          metrics_collector=metrics_collector,
      )
  )

  # Start uvicorn http server.
  uvicorn.run(
      app, host=flags.FLAGS.host, port=flags.FLAGS.port, log_level="info"
  )


if __name__ == "__main__":
  # Run Abseil app w flags parser.
  abslapp.run(server)


-----------------------

/jetstream/entrypoints/http/protocol.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Http API server protocol."""

from pydantic import BaseModel  # type: ignore


class TextContent(BaseModel):
  text: str


class TokenContent(BaseModel):
  token_ids: list[int]


class Metadata(BaseModel):
  start_time: float


class DecodeRequest(BaseModel):
  max_tokens: int
  text_content: TextContent | None = None
  token_content: TokenContent | None = None
  metadata: Metadata | None = None

  # Config to enforce the oneof behavior at runtime.
  class Config:
    extra = "forbid"  # Prevent extra fields.
    anystr_strip_whitespace = True


-----------------------

/jetstream/entrypoints/http/utils.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Http API server utilities."""

from google.protobuf.json_format import MessageToJson


async def proto_to_json_generator(proto_generator):
  """Wraps a generator yielding Protocol Buffer messages into a generator

  yielding JSON messages.
  """
  async for proto_message in proto_generator:
    json_string = MessageToJson(proto_message)
    yield json_string


-----------------------

/jetstream/tests/__init__.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


-----------------------

/jetstream/tests/core/__init__.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


-----------------------

/jetstream/tests/core/test_config_lib.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit test for config_lib.py."""

import unittest
from parameterized import parameterized
from jetstream.core import config_lib


class TestConfigLib(unittest.TestCase):

  @parameterized.expand([("tpu=8", 8), ("v5e-8", 8), ("v5e=4", 4), ("v4-8", 4)])
  def test_slice_to_num_chips(self, accelerator_slice, expected_num_devices):
    got = config_lib.slice_to_num_chips(accelerator_slice)
    self.assertEqual(got, expected_num_devices)

  def test_get_engines_invalid(self):
    with self.assertRaises(ValueError):
      config_lib.get_engines(config_lib.InterleavedCPUTestServer, [])


-----------------------

/jetstream/tests/core/test_orchestrator.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Integration test of the orchestrator.

This test tests the multi-htreaded orchestrator, where a prefill request is
popped onto a prefill queue, prefilled, sent to a generation queue and run for
a number of decoding steps.

In operation, it will use gRPC so we can 'yield' from the function to get return
values in the same way that they would be streamed back.

Similar to 'mock_engine_test' we can use known token values and a singleton
weight to test our operation.

Let the prefill engine have a weight of [2] and the generate engine have a
weight of [3].

I.e. if we prefill [2, 65, 66] (i.e. <BOS>, 'A', 'B') using an ACII vocab,
we should get [4, 130, 132].

If we then insert that and run three generation steps, we should see
266+0 / 2 = 266
266 + [266] / 4  = 332
266 + [266, 332] / 4 = 415
I.e. ['Ċ', 'Ō', 'Ɵ'] when converted back with chr().

Therefore we should get back the character sequence '$lǔ' if we request 3 tokens
decoded (these are the ascii chars at those indices which is what the test
tokenizer returns).
"""

import unittest
from jetstream.core import orchestrator
from jetstream.core.proto import jetstream_pb2
from jetstream.core.utils.return_sample import ReturnSample
from jetstream.engine import mock_engine


class OrchestratorTest(unittest.IsolatedAsyncioTestCase):

  def _setup_driver_interleaved_mode(self):
    prefill_engine = mock_engine.TestEngine(
        batch_size=32, cache_length=256, weight=2.0
    )
    # Create a generate engine with a different set of weights
    # so that we can test that the right one is in use at a given time.
    generate_engine = mock_engine.TestEngine(
        batch_size=4, cache_length=32, weight=4.0
    )
    driver = orchestrator.Driver(
        prefill_engines=[prefill_engine],
        generate_engines=[generate_engine],
        prefill_params=[prefill_engine.load_params()],
        generate_params=[generate_engine.load_params()],
        interleaved_mode=True,
    )
    return driver

  async def test_orchestrator_interleaved_mode(self):
    """Test the multithreaded orchestration."""
    driver = self._setup_driver_interleaved_mode()
    client = orchestrator.LLMOrchestrator(driver=driver)

    # The string representation of np.array([[65, 66]]), [2] will be prepend
    # as BOS.
    text = "AB"

    request = jetstream_pb2.DecodeRequest(
        text_content=jetstream_pb2.DecodeRequest.TextContent(text=text),
        max_tokens=3,
    )
    iterator = client.Decode(request)
    # chr of [266, 332, 415].
    expected_text = ["Ċ", "Ō", "Ɵ", ""]
    expected_token_ids = [266, 332, 415, None]
    counter = 0
    async for resp in iterator:
      output_text = resp.stream_content.samples[0].text
      token_ids = resp.stream_content.samples[0].token_ids
      output_token_id = token_ids[0] if len(token_ids) > 0 else None
      print(f"actual output: {output_text=} {output_token_id=}")
      assert output_text == expected_text[counter]
      assert output_token_id == expected_token_ids[counter]
      counter += 1
    driver.stop()
    print("Orchestrator driver stopped.")

  async def test_orchestrator_interleaved_mode_client_tokenization(self):
    """Test the multithreaded orchestration."""
    driver = self._setup_driver_interleaved_mode()
    client = orchestrator.LLMOrchestrator(driver=driver)

    # The token ids of  string "AB", [2] will be prepend
    # as BOS.
    token_ids = [65, 66]

    request = jetstream_pb2.DecodeRequest(
        token_content=jetstream_pb2.DecodeRequest.TokenContent(
            token_ids=token_ids
        ),
        max_tokens=3,
    )
    iterator = client.Decode(request)
    # Return token ids only when in client side tokenization mode.
    expected_text = ["", "", "", ""]
    expected_token_ids = [266, 332, 415, None]
    counter = 0
    async for resp in iterator:
      output_text = resp.stream_content.samples[0].text
      token_ids = resp.stream_content.samples[0].token_ids
      output_token_id = token_ids[0] if len(token_ids) > 0 else None
      print(f"actual output: {output_text=} {output_token_id=}")
      assert output_text == expected_text[counter]
      assert output_token_id == expected_token_ids[counter]
      counter += 1
    driver.stop()
    print("Orchestrator driver stopped.")

  def test_should_buffer_response(self):
    driver = self._setup_driver_interleaved_mode()
    client = orchestrator.LLMOrchestrator(driver=driver)
    self.assertTrue(
        client.should_buffer_response(
            [ReturnSample(text=["<0xAB>"], token_ids=[13])]
        )
    )
    driver.stop()
    print("Orchestrator driver stopped.")


-----------------------

/jetstream/tests/core/test_server.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests gRPC server end-to-end.

See orchestrator test for why these characters specifically will be the
response.
"""

from typing import Any, Type
import unittest


import requests
from parameterized import parameterized
import grpc
from jetstream.core import config_lib
from jetstream.core import server_lib
from jetstream.core.proto import jetstream_pb2
from jetstream.core.proto import jetstream_pb2_grpc
from jetstream.engine import engine_api
import portpicker


class ServerTest(unittest.IsolatedAsyncioTestCase):

  @parameterized.expand(
      [
          # Uses weight 2 for prefill, 4 for decode.
          (
              config_lib.CPUTestServer,
              True,
              ["Ċ", "Ō", "Ɵ", ""],
              [266, 332, 415, None],
              [None, None],
          ),
          # Uses the same prefill / generate weights (2).
          (
              config_lib.InterleavedCPUTestServer,
              True,
              ["Ċ", "Ə", "ɖ", ""],
              [266, 399, 598, None],
              [None],
          ),
          # Disable the metrics server.
          (
              config_lib.InterleavedCPUTestServer,
              False,
              ["Ċ", "Ə", "ɖ", ""],
              [266, 399, 598, None],
              [None],
          ),
      ]
  )
  async def test_server(
      self,
      config: Type[config_lib.ServerConfig],
      metrics_enabled: bool,
      expected_text: list[str],
      expected_token_ids: list[int | None],
      devices: list[Any],
  ):
    """Sets up a server and requests token responses."""
    ######################### Server side ######################################
    port = portpicker.pick_unused_port()
    metrics_port = portpicker.pick_unused_port()

    print("port: " + str(port))
    credentials = grpc.local_server_credentials()

    server = server_lib.run(
        port=port,
        config=config,
        devices=devices,
        credentials=credentials,
        metrics_server_config=config_lib.MetricsServerConfig(port=metrics_port)
        if metrics_enabled is True
        else None,
    )
    ###################### Requester side ######################################

    # if prometheus not configured, assert no metrics collector on Driver
    if metrics_enabled is not True:
      assert server._driver._metrics_collector is None  # pylint: disable=protected-access

    async with grpc.aio.secure_channel(
        f"localhost:{port}", grpc.local_channel_credentials()
    ) as channel:
      stub = jetstream_pb2_grpc.OrchestratorStub(channel)

      healthcheck_request = jetstream_pb2.HealthCheckRequest()
      healthcheck_response = stub.HealthCheck(healthcheck_request)
      healthcheck_response = await healthcheck_response

      assert healthcheck_response.is_live is True

      # The string representation of np.array([[65, 66]]), [2] will be prepended
      # as BOS
      text = "AB"
      request = jetstream_pb2.DecodeRequest(
          text_content=jetstream_pb2.DecodeRequest.TextContent(text=text),
          max_tokens=3,
      )
      iterator = stub.Decode(request)
      counter = 0
      async for resp in iterator:
        output_text = resp.stream_content.samples[0].text
        token_ids = resp.stream_content.samples[0].token_ids
        output_token_id = token_ids[0] if len(token_ids) > 0 else None
        print(f"actual output: {output_text=} {output_token_id=}")
        assert output_text == expected_text[counter]
        assert output_token_id == expected_token_ids[counter]
        counter += 1
      # assert prometheus server is running and responding
      if metrics_enabled is True:
        assert server._driver._metrics_collector is not None  # pylint: disable=protected-access
        assert (
            requests.get(
                f"http://localhost:{metrics_port}", timeout=5
            ).status_code
            == requests.status_codes.codes["ok"]
        )
      server.stop()

  def test_jax_profiler_server(self):
    port = portpicker.pick_unused_port()
    print("port: " + str(port))
    credentials = grpc.local_server_credentials()
    # Now test server with prometheus config
    server = server_lib.run(
        port=port,
        config=config_lib.InterleavedCPUTestServer,
        devices=[None],
        credentials=credentials,
        enable_jax_profiler=True,
    )
    assert server
    server.stop()

  def test_get_devices(self):
    assert len(server_lib.get_devices()) == 1

  async def test_model_warmup(self):
    port = portpicker.pick_unused_port()

    print("port: " + str(port))
    credentials = grpc.local_server_credentials()

    server = server_lib.run(
        port=port,
        config=config_lib.InterleavedCPUTestServer,
        devices=[None],
        credentials=credentials,
        enable_model_warmup=True,
    )

    async with grpc.aio.secure_channel(
        f"localhost:{port}", grpc.local_channel_credentials()
    ) as channel:
      stub = jetstream_pb2_grpc.OrchestratorStub(channel)

      healthcheck_request = jetstream_pb2.HealthCheckRequest()
      healthcheck_response = stub.HealthCheck(healthcheck_request)
      healthcheck_response = await healthcheck_response

      assert healthcheck_response.is_live is True

      for pe in server._driver._prefill_engines:  # pylint: disable=protected-access
        assert isinstance(pe, engine_api.JetStreamEngine)
        assert pe.warm is True

      for ge in server._driver._generate_engines:  # pylint: disable=protected-access
        assert isinstance(ge, engine_api.JetStreamEngine)
        assert ge.warm is True

      server.stop()


-----------------------

/jetstream/tests/engine/__init__.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


-----------------------

/jetstream/tests/engine/test_mock_engine.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for a mock version of the engine API.

What should we expect?

Prefill: Doubles the sequence by multiplying it with a weight [2].
Insert: Writes this sequence into a cache row
Generate step: Return sum(prefill_cache) + sum(generate_cache)/weight

I.e. if we prefill [2, 65, 66] (i.e. <BOS>, 'A', 'B') using an ACII vocab,
we should get [4, 130, 132].

If we then insert that and run three generation steps, we should see
266+0 / 2 = 266
266 + [266] /2  = 399
266 + [266, 399] /2 = 598
I.e. ['Ċ', 'Ə', 'ɖ'] when converted back with chr()
"""

import unittest
import jax.numpy as jnp
import numpy as np

from jetstream.engine import mock_engine
from jetstream.engine import token_utils


class EngineTest(unittest.TestCase):

  def _setup(self):
    """Initialises a test engine."""
    engine = mock_engine.TestEngine(batch_size=32, cache_length=256, weight=2.0)
    params = engine.load_params()
    return engine, params

  def _prefill(self):
    """Performs prefill and returns a kv cache."""
    engine, params = self._setup()
    # A 2 will be pre-pended as 'bos' token from the vocab.
    text = "AB"
    metadata = engine.get_tokenizer()
    tokenizer = engine.build_tokenizer(metadata)
    tokens, true_length = tokenizer.encode(text, is_bos=True)
    prefill_result, first_token = engine.prefill(
        params=params, padded_tokens=tokens, true_length=3
    )
    return engine, params, prefill_result, true_length, first_token

  def _prefill_np(self):
    """Performs prefill and returns a kv cache."""
    engine, params = self._setup()
    # A 2 will be pre-pended as 'bos' token from the vocab.
    text = "AB"
    metadata = engine.get_tokenizer()
    tokenizer = engine.build_tokenizer(metadata)
    tokens, true_length = tokenizer.encode(text, is_bos=True, jax_padding=False)
    prefill_result, first_token = engine.prefill(
        params=params, padded_tokens=tokens, true_length=3
    )
    return engine, params, prefill_result, true_length, first_token

  def _generate(self, slot=1):
    """Performs a single generation step."""
    engine, params, prefill_result, _, _ = self._prefill()
    decode_state = engine.init_decode_state()
    decode_state = engine.insert(
        prefix=prefill_result, decode_state=decode_state, slot=slot
    )
    decode_state, sampled_tokens = engine.generate(
        params=params, decode_state=decode_state
    )
    return engine, params, decode_state, sampled_tokens

  def test_load_params(self):
    """Just loads params."""
    _, params = self._setup()
    assert params == jnp.array([2.0])

  def test_prefill(self):
    """Tests prefill with weight = 2."""
    engine, _, prefill_result, true_length, first_token = self._prefill()
    prefill_cache, _ = prefill_result
    np.testing.assert_array_equal(
        prefill_cache[:, :true_length], np.array([[4.0, 130.0, 132.0]])
    )

    # test first token
    token_data = first_token.get_result_at_slot(0)
    tok = token_data.tokens

    metadata = engine.get_tokenizer()
    tokenizer = token_utils.load_vocab(
        metadata.path, metadata.extra_ids
    ).tokenizer
    assert tokenizer.IdToPiece(int(tok.item())) == "Ċ"

  def test_prefill_np(self):
    """Tests prefill with weight = 2."""
    _, _, prefill_result, true_length, _ = self._prefill_np()
    prefill_cache, _ = prefill_result
    np.testing.assert_array_equal(
        prefill_cache[:, :true_length], np.array([[4.0, 130.0, 132.0]])
    )

  def test_generate(self, slot=1):
    """Tests multiple generation steps."""
    engine, params, decode_state, sampled_tokens = self._generate(slot=slot)
    metadata = engine.get_tokenizer()
    tokenizer = token_utils.load_vocab(
        metadata.path, metadata.extra_ids
    ).tokenizer

    # Char for 399
    token_data = sampled_tokens.get_result_at_slot(slot)
    tok = token_data.tokens
    assert tokenizer.IdToPiece(int(tok.item())) == "Ə"
    _, sampled_tokens = engine.generate(
        params=params, decode_state=decode_state
    )
    # Char for 598
    token_data = sampled_tokens.get_result_at_slot(slot)
    tok = token_data.tokens
    assert tokenizer.IdToPiece(int(tok.item())) == "ɖ"


-----------------------

/jetstream/tests/engine/test_sampling_utils.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests functionality of inference sampling utils."""

import jax
import jax.numpy as jnp
import unittest
from jetstream.engine import sampling_utils


class SamplingUtilsTest(unittest.TestCase):

  def setUp(self):
    self.rng = jax.random.PRNGKey(0)
    self.logits = jnp.array([[-0.5, 1.2, 0.8], [-1.0, 0.3, 0.7]])

  def test_greedy_sampling(self):
    token = sampling_utils.sampling(self.logits, self.rng, "greedy")
    expected_token = jnp.array([1, 2])
    self.assertTrue(jnp.array_equal(token, expected_token))

  def test_weighted_sampling(self):
    # Multiple samples to increase the chance of catching errors
    for _ in range(10):
      result = sampling_utils.sampling(self.logits, self.rng, "weighted")
      self.assertTrue(
          jnp.all(jnp.isin(result, jnp.array([0, 1, 2])))
      )  # Check if sampled from valid indices

  def test_nucleus_sampling(self):
    for _ in range(10):
      result = sampling_utils.sampling(
          self.logits, self.rng, "nucleus", nucleus_topp=0.8
      )
      self.assertTrue(jnp.all(jnp.isin(result, jnp.array([0, 1, 2]))))
    invalid_topp = -0.1
    with self.assertRaises(ValueError) as context:
      sampling_utils.sampling(
          self.logits, self.rng, "nucleus", nucleus_topp=invalid_topp
      )
      self.assertIn(
          f"Can't apply nucleus with parameter {invalid_topp=} less zero",
          str(context.exception),
      )

  def test_topk_sampling(self):
    for _ in range(10):
      result = sampling_utils.sampling(self.logits, self.rng, "topk", topk=2)
      self.assertTrue(
          jnp.all(jnp.isin(result, jnp.array([1, 2])))
      )  # Only top 2 logits should be sampled
    invalid_topk = 0
    with self.assertRaises(ValueError) as context:
      sampling_utils.sampling(self.logits, self.rng, "topk", topk=invalid_topk)
      self.assertIn(
          f"Can't apply algorithm topk with parameter {invalid_topk=} <= 0",
          str(context.exception),
      )

  def test_unsupported_algorithm(self):
    with self.assertRaises(ValueError):
      sampling_utils.sampling(self.logits, self.rng, "unsupported_algorithm")


-----------------------

/jetstream/tests/engine/test_token_utils.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests functionality of the tokenizer with supported models."""

import os
import unittest
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
from sentencepiece import SentencePieceProcessor
from jetstream.engine import tokenizer_pb2, token_utils
from jetstream.engine import engine_api


class SPTokenizer:
  """Tokenier used in original llama2 git"""

  def __init__(self, tokenizer_path: str):
    self.tokenizer = SentencePieceProcessor()
    self.tokenizer.Load(model_file=tokenizer_path)
    assert self.tokenizer.vocab_size() == self.tokenizer.get_piece_size()

  def decode(self, t: List[int]) -> str:
    token = self.tokenizer.decode(t)
    return token


class JetStreamTokenizer:
  """Tokenier used in JetStream before mix_token"""

  def __init__(self, tokenizer_path: str):
    metadata = tokenizer_pb2.TokenizerParameters(path=tokenizer_path)
    self.vocab = token_utils.load_vocab(metadata.path, metadata.extra_ids)

  def decode(self, t: int) -> str:
    token = self.vocab.tokenizer.IdToPiece(t)
    token = token.replace("▁", " ")
    return token


class TokenUtilsTest(unittest.TestCase):

  def setup_sentencepiece(self):
    self.tokenizer_path = "third_party/llama2/tokenizer.model"
    current_dir = os.path.dirname(__file__)
    self.tokenizer_path = os.path.join(current_dir, self.tokenizer_path)
    print(f"model_path: {self.tokenizer_path}")
    assert os.path.isfile(
        self.tokenizer_path
    ), f"file not found tokenizer_path: {self.tokenizer_path}"
    self.sp_tokenizer = SPTokenizer(self.tokenizer_path)
    self.jt_tokenizer = JetStreamTokenizer(self.tokenizer_path)

  def setup_tiktoken(self):
    self.tokenizer_path = "third_party/llama3/tokenizer.model"
    current_dir = os.path.dirname(__file__)
    self.tokenizer_path = os.path.join(current_dir, self.tokenizer_path)
    print(f"model_path: {self.tokenizer_path}")
    assert os.path.isfile(
        self.tokenizer_path
    ), f"file not found tokenizer_path: {self.tokenizer_path}"

  def test_decode_vs_piece(self):
    self.setup_sentencepiece()
    tokens = [304, 13, 2266, 526, 777, 9590, 2020, 29901]
    expeted_sp_output = []
    jt_output = []
    for t in tokens:
      expeted_sp_output.append(self.sp_tokenizer.decode([t]))
      jt_output.append(self.jt_tokenizer.decode(t))

    self.assertNotEqual(jt_output, expeted_sp_output)

  def test_sp_vs_seqio(self):
    self.setup_sentencepiece()
    for n in range(0, self.sp_tokenizer.tokenizer.vocab_size()):
      sp_t = self.sp_tokenizer.decode([n])
      seqio_t = self.jt_tokenizer.vocab.tokenizer.decode([n])
      self.assertEqual(sp_t, seqio_t)

  def test_tokenize_and_pad_jax(self):
    jax.config.update("jax_platform_name", "cpu")
    self.setup_sentencepiece()
    s = "I believe the meaning of life is"
    vocab = self.jt_tokenizer.vocab
    max_prefill_length = 1024
    tokens = vocab.encode_tf(s)
    padded_tokens, true_length = token_utils.pad_tokens(
        tokens,
        bos_id=vocab.bos_id,
        pad_id=vocab.pad_id,
        max_prefill_length=max_prefill_length,
    )
    expected_padded_tokens = jnp.array(
        [1, 306, 4658, 278, 6593, 310, 2834, 338, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    expected_true_length = 8
    self.assertTrue(
        jnp.allclose(padded_tokens, expected_padded_tokens, atol=1e-7)
    )
    self.assertEqual(true_length, expected_true_length)

  def test_tokenize_and_pad_np(self):
    self.setup_sentencepiece()
    s = "I believe the meaning of life is"
    vocab = self.jt_tokenizer.vocab
    max_prefill_length = 1024
    tokens = vocab.encode_tf(s)
    padded_tokens, true_length = token_utils.pad_tokens(
        tokens,
        bos_id=vocab.bos_id,
        pad_id=vocab.pad_id,
        max_prefill_length=max_prefill_length,
        jax_padding=False,
    )
    expected_padded_tokens = np.array(
        [1, 306, 4658, 278, 6593, 310, 2834, 338, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    expected_true_length = 8
    self.assertTrue(
        np.allclose(padded_tokens, expected_padded_tokens, atol=1e-7)
    )
    self.assertEqual(true_length, expected_true_length)

  def test_tokenize_and_pad(self):
    jax.config.update("jax_platform_name", "cpu")
    self.setup_sentencepiece()
    s = "I believe the meaning of life is"
    vocab = self.jt_tokenizer.vocab
    max_prefill_length = 1024
    padded_tokens, true_length = token_utils.tokenize_and_pad(
        s,
        vocab,
        max_prefill_length=max_prefill_length,
    )
    expected_padded_tokens = jnp.array(
        [1, 306, 4658, 278, 6593, 310, 2834, 338, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    expected_true_length = 8
    self.assertTrue(
        jnp.allclose(padded_tokens, expected_padded_tokens, atol=1e-7)
    )
    self.assertEqual(true_length, expected_true_length)

  def test_pad_token_padding_less_than_zero(self):
    jax.config.update("jax_platform_name", "cpu")
    self.setup_sentencepiece()
    s = "I believe the meaning of life is having different experiences and "
    s += "enjoy everyday of my life."
    vocab = self.jt_tokenizer.vocab
    max_prefill_length = 16
    tokens = vocab.encode_tf(s)
    padded_tokens, true_length = token_utils.pad_tokens(
        tokens,
        bos_id=vocab.bos_id,
        pad_id=vocab.pad_id,
        max_prefill_length=max_prefill_length,
    )
    # Take the last N tokens if we have too many.
    expected_padded_tokens = jnp.array(
        [
            278,
            6593,
            310,
            2834,
            338,
            2534,
            1422,
            27482,
            322,
            13389,
            1432,
            3250,
            310,
            590,
            2834,
            29889,
        ]
    )
    expected_true_length = 19
    self.assertTrue(
        jnp.allclose(padded_tokens, expected_padded_tokens, atol=1e-7)
    )
    self.assertEqual(true_length, expected_true_length)

  def test_sentencepiece_tokenizer_encode(self):
    self.setup_sentencepiece()
    s = "I believe the meaning of life is"
    metadata = tokenizer_pb2.TokenizerParameters(path=self.tokenizer_path)
    tokenizer = token_utils.SentencePieceTokenizer(metadata)
    tokens, true_length = tokenizer.encode(s)
    expected_padded_tokens = np.array(
        [1, 306, 4658, 278, 6593, 310, 2834, 338, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    expected_true_length = 8
    self.assertTrue(np.allclose(tokens, expected_padded_tokens, atol=1e-7))
    self.assertEqual(true_length, expected_true_length)

  def test_sentencepiece_tokenizer_encode_no_bos(self):
    self.setup_sentencepiece()
    s = "I believe the meaning of life is"
    metadata = tokenizer_pb2.TokenizerParameters(path=self.tokenizer_path)
    tokenizer = token_utils.SentencePieceTokenizer(metadata)
    tokens, true_length = tokenizer.encode(s, is_bos=False)
    expected_padded_tokens = np.array(
        [306, 4658, 278, 6593, 310, 2834, 338, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    expected_true_length = 7
    self.assertTrue(np.allclose(tokens, expected_padded_tokens, atol=1e-7))
    self.assertEqual(true_length, expected_true_length)

  def test_sentencepiece_tokenizer_encode_prefill_lengths(self):
    self.setup_sentencepiece()
    s = "I believe the meaning of life is"
    metadata = tokenizer_pb2.TokenizerParameters(path=self.tokenizer_path)
    tokenizer = token_utils.SentencePieceTokenizer(metadata)
    tokens, true_length = tokenizer.encode(s, prefill_lengths=[12, 24, 36])
    expected_padded_tokens = np.array(
        [1, 306, 4658, 278, 6593, 310, 2834, 338, 0, 0, 0, 0]
    )
    expected_true_length = 8
    self.assertTrue(np.allclose(tokens, expected_padded_tokens, atol=1e-7))
    self.assertEqual(true_length, expected_true_length)

  def test_sentencepiece_tokenizer_encode_jax(self):
    jax.config.update("jax_platform_name", "cpu")
    self.setup_sentencepiece()
    s = "I believe the meaning of life is"
    metadata = tokenizer_pb2.TokenizerParameters(path=self.tokenizer_path)
    tokenizer = token_utils.SentencePieceTokenizer(metadata)
    padded_tokens, true_length = tokenizer.encode(s, jax_padding=True)
    expected_padded_tokens = jnp.array(
        [1, 306, 4658, 278, 6593, 310, 2834, 338, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    expected_true_length = 8
    self.assertTrue(
        jnp.allclose(padded_tokens, expected_padded_tokens, atol=1e-7)
    )
    self.assertEqual(true_length, expected_true_length)

  def test_process_result_with_sentencepiece_tokenizer_decode(self):
    self.setup_sentencepiece()
    metadata = tokenizer_pb2.TokenizerParameters(path=self.tokenizer_path)
    tokenizer = token_utils.SentencePieceTokenizer(metadata)
    complete = np.zeros((1,), dtype=np.bool_)

    length = 7
    result_tokens = engine_api.ResultTokens(
        data=np.array(
            [
                [
                    306,
                    4658,
                    278,
                    6593,
                    310,
                    2834,
                    338,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    length,
                ]
            ]
        ),
        tokens_idx=(0, length),
        valid_idx=(length, 2 * length),
        length_idx=(2 * length, 2 * length + 1),
        samples_per_slot=1,
    )
    samples, complete = token_utils.process_result_tokens(
        tokenizer, 0, 16, result_tokens, complete, False
    )
    # Note: the expected_tokens list is for the output token(s) for 1 decode
    # step. Currently, JetStream only output 1 token (1 text piece) for 1
    # decode step.
    expected_tokens = np.array([[306, 4658, 278, 6593, 310, 2834, 338]])
    self.assertTrue(
        np.allclose(
            [sample.token_ids for sample in samples], expected_tokens, atol=1e-7
        )
    )
    self.assertTrue(
        samples[0].text
        == [" I", " believe", " the", " meaning", " of", " life", " is"]
    )
    self.assertTrue(np.allclose(complete, np.zeros((1,), dtype=np.bool_)))

  def test_process_result_with_sentencepiece_tokenizer_client_decode(self):
    self.setup_sentencepiece()
    metadata = tokenizer_pb2.TokenizerParameters(path=self.tokenizer_path)
    tokenizer = token_utils.SentencePieceTokenizer(metadata)
    complete = np.zeros((1,), dtype=np.bool_)

    length = 7
    result_tokens = engine_api.ResultTokens(
        data=np.array(
            [
                [
                    306,
                    4658,
                    278,
                    6593,
                    310,
                    2834,
                    338,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    length,
                ]
            ]
        ),
        tokens_idx=(0, length),
        valid_idx=(length, 2 * length),
        length_idx=(2 * length, 2 * length + 1),
        samples_per_slot=1,
    )
    samples, complete = token_utils.process_result_tokens(
        tokenizer, 0, 16, result_tokens, complete, True
    )
    # Note: the expected_tokens list is for the output token(s) for 1 decode
    # step. Currently, JetStream only output 1 token (1 text piece) for 1
    # decode step.
    expected_tokens = np.array([[306, 4658, 278, 6593, 310, 2834, 338]])
    self.assertTrue(
        np.allclose(
            [sample.token_ids for sample in samples], expected_tokens, atol=1e-7
        )
    )
    # Return token ids only when in client side tokenization mode.
    self.assertTrue(samples[0].text == [])
    self.assertTrue(np.allclose(complete, np.zeros((1,), dtype=np.bool_)))

  def test_sentencepiece_tokenizer_decode(self):
    self.setup_sentencepiece()
    metadata = tokenizer_pb2.TokenizerParameters(path=self.tokenizer_path)
    tokenizer = token_utils.SentencePieceTokenizer(metadata)
    result = tokenizer.decode([306, 4658, 278, 6593, 310, 2834, 338])
    self.assertTrue(result == "I believe the meaning of life is")

  def test_tiktoken_tokenizer_encode(self):
    self.setup_tiktoken()
    s = "I believe the meaning of life is"
    metadata = tokenizer_pb2.TokenizerParameters(path=self.tokenizer_path)
    tokenizer = token_utils.TikToken(metadata)
    tokens, true_length = tokenizer.encode(s)
    expected_padded_tokens = np.array(
        [
            128000,
            40,
            4510,
            279,
            7438,
            315,
            2324,
            374,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
        ]
    )
    expected_true_length = 8
    self.assertTrue(np.allclose(tokens, expected_padded_tokens, atol=1e-7))
    self.assertEqual(true_length, expected_true_length)

  def test_tiktoken_encode_no_bos(self):
    self.setup_tiktoken()
    s = "I believe the meaning of life is"
    metadata = tokenizer_pb2.TokenizerParameters(path=self.tokenizer_path)
    tokenizer = token_utils.TikToken(metadata)
    tokens, true_length = tokenizer.encode(s, is_bos=False)
    expected_padded_tokens = np.array(
        [
            40,
            4510,
            279,
            7438,
            315,
            2324,
            374,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
        ]
    )
    expected_true_length = 7
    self.assertTrue(np.allclose(tokens, expected_padded_tokens, atol=1e-7))
    self.assertEqual(true_length, expected_true_length)

  def test_tiktoken_encode_prefill_lengths(self):
    self.setup_tiktoken()
    s = "I believe the meaning of life is"
    metadata = tokenizer_pb2.TokenizerParameters(path=self.tokenizer_path)
    tokenizer = token_utils.TikToken(metadata)
    tokens, true_length = tokenizer.encode(s, prefill_lengths=[12, 24, 36])
    expected_padded_tokens = np.array(
        [128000, 40, 4510, 279, 7438, 315, 2324, 374, -1, -1, -1, -1]
    )
    expected_true_length = 8
    self.assertTrue(np.allclose(tokens, expected_padded_tokens, atol=1e-7))
    self.assertEqual(true_length, expected_true_length)

  def test_tiktoken_encode_jax(self):
    jax.config.update("jax_platform_name", "cpu")
    self.setup_tiktoken()
    s = "I believe the meaning of life is"
    metadata = tokenizer_pb2.TokenizerParameters(path=self.tokenizer_path)
    tokenizer = token_utils.TikToken(metadata)
    padded_tokens, true_length = tokenizer.encode(s, jax_padding=True)
    expected_padded_tokens = jnp.array(
        [
            128000,
            40,
            4510,
            279,
            7438,
            315,
            2324,
            374,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
        ]
    )
    expected_true_length = 8
    self.assertTrue(
        jnp.allclose(padded_tokens, expected_padded_tokens, atol=1e-7)
    )
    self.assertEqual(true_length, expected_true_length)

  def test_process_result_with_tiktoken_decode(self):
    self.setup_tiktoken()
    metadata = tokenizer_pb2.TokenizerParameters(path=self.tokenizer_path)
    tokenizer = token_utils.TikToken(metadata)
    complete = np.zeros((1,), dtype=np.bool_)

    length = 7
    result_tokens = engine_api.ResultTokens(
        data=np.array(
            [[40, 4510, 279, 7438, 315, 2324, 374, 1, 1, 1, 1, 1, 1, 1, length]]
        ),
        tokens_idx=(0, length),
        valid_idx=(length, 2 * length),
        length_idx=(2 * length, 2 * length + 1),
        samples_per_slot=1,
    )
    samples, complete = token_utils.process_result_tokens(
        tokenizer, 0, 16, result_tokens, complete, False
    )
    # Note: the expected_tokens list is for the output token(s) for 1 decode
    # step. Currently, JetStream only output 1 token (1 text piece) for 1
    # decode step.
    expected_tokens = np.array([[40, 4510, 279, 7438, 315, 2324, 374]])
    self.assertTrue(
        np.allclose(
            [sample.token_ids for sample in samples], expected_tokens, atol=1e-7
        )
    )
    self.assertTrue(
        samples[0].text
        == ["I", " believe", " the", " meaning", " of", " life", " is"]
    )
    self.assertTrue(np.allclose(complete, np.zeros((1,), dtype=np.bool_)))

  def test_process_result_with_tiktoken_client_decode(self):
    self.setup_tiktoken()
    metadata = tokenizer_pb2.TokenizerParameters(path=self.tokenizer_path)
    tokenizer = token_utils.TikToken(metadata)
    complete = np.zeros((1,), dtype=np.bool_)

    length = 7
    result_tokens = engine_api.ResultTokens(
        data=np.array(
            [[40, 4510, 279, 7438, 315, 2324, 374, 1, 1, 1, 1, 1, 1, 1, length]]
        ),
        tokens_idx=(0, length),
        valid_idx=(length, 2 * length),
        length_idx=(2 * length, 2 * length + 1),
        samples_per_slot=1,
    )
    samples, complete = token_utils.process_result_tokens(
        tokenizer, 0, 16, result_tokens, complete, True
    )
    # Note: the expected_tokens list is for the output token(s) for 1 decode
    # step. Currently, JetStream only output 1 token (1 text piece) for 1
    # decode step.
    expected_tokens = np.array([[40, 4510, 279, 7438, 315, 2324, 374]])
    self.assertTrue(
        np.allclose(
            [sample.token_ids for sample in samples], expected_tokens, atol=1e-7
        )
    )
    # Return token ids only when in client side tokenization mode.
    self.assertTrue(samples[0].text == [])
    self.assertTrue(np.allclose(complete, np.zeros((1,), dtype=np.bool_)))

  def test_tiktoken_decode(self):
    self.setup_tiktoken()
    metadata = tokenizer_pb2.TokenizerParameters(path=self.tokenizer_path)
    tokenizer = token_utils.TikToken(metadata)
    result = tokenizer.decode([40, 4510, 279, 7438, 315, 2324, 374])
    self.assertTrue(result == "I believe the meaning of life is")

  def test_text_tokens_to_str(self):
    # Start with text token
    self.assertTrue(
        token_utils.text_tokens_to_str(
            ["你", "好", "<0xE5>", "<0x90>", "<0x97>", "hello"]
        )
        == "你好吗hello"
    )
    self.assertTrue(
        token_utils.text_tokens_to_str(
            ["你", "好", "<0xE5>", "<0x90>", "<0x97>", "<0x0A>", "hello"]
        )
        == "你好吗\nhello"
    )
    self.assertTrue(
        token_utils.text_tokens_to_str(
            [
                "你",
                "好",
                "<0xE5>",
                "<0x90>",
                "<0x97>",
                "<0x0A>",
                "<0x0A>",
                "hello",
            ]
        )
        == "你好吗\n\nhello"
    )
    self.assertTrue(
        token_utils.text_tokens_to_str(
            ["你", "好", "<0xE5>", "<0x90>", "<0x97>", "hello", "<0x0A>"]
        )
        == "你好吗hello\n"
    )
    # Start with byte token
    self.assertTrue(
        token_utils.text_tokens_to_str(
            ["<0x0A>", "你", "好", "<0xE5>", "<0x90>", "<0x97>", "hello"]
        )
        == "\n你好吗hello"
    )
    self.assertTrue(
        token_utils.text_tokens_to_str(
            [
                "<0x0A>",
                "<0x0A>",
                "你",
                "好",
                "<0xE5>",
                "<0x90>",
                "<0x97>",
                "hello",
            ]
        )
        == "\n\n你好吗hello"
    )
    self.assertTrue(
        token_utils.text_tokens_to_str(["<0xE5>", "<0x90>", "<0x97>", "hello"])
        == "吗hello"
    )
    self.assertTrue(
        token_utils.text_tokens_to_str(
            ["<0x0A>", "<0x0A>", "<0xE5>", "<0x90>", "<0x97>", "hello"]
        )
        == "\n\n吗hello"
    )
    self.assertTrue(
        token_utils.text_tokens_to_str(
            ["<0x0A>", "<0x0A>", "<0xE5>", "<0x90>", "<0x97>"]
        )
        == "\n\n吗"
    )
    # Invalid byte token sequence
    self.assertTrue(
        token_utils.text_tokens_to_str(
            ["你", "好", "<0xE5>", "<0x90>", "<0x0A>", "<0x97>", "hello"]
        )
        == "你好�\n�hello"
    )


-----------------------

/jetstream/tests/engine/test_utils.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests functionality of the token processing utils using mock engine vocab."""

import numpy as np
import unittest
from jetstream.engine import engine_api
from jetstream.engine import mock_utils
from jetstream.engine import token_utils


class UtilsTest(unittest.TestCase):

  def test_speculations_with_multi_sample_slots(self, samples_per_slot=2):
    # [4, 1]
    mock_tokens = np.array(
        [
            [0, ord("A")],
            [ord("A"), ord("D")],
            [ord("T"), ord("3")],
            [ord("A"), 1],
        ]
    ).astype(np.int32)
    mock_valid_tokens = np.ones_like(mock_tokens, dtype=np.int32)
    mock_lengths = np.ones(mock_tokens.shape[0], dtype=np.int32) * 2
    # completion is 'per slot' because we track it for a given request.
    mock_complete = np.zeros(
        (mock_tokens.shape[0] // samples_per_slot), dtype=np.int32
    )
    data = np.concatenate(
        [
            mock_tokens,
            mock_valid_tokens,
            mock_lengths[:, None],
        ],
        axis=-1,
    )
    speculations = mock_tokens.shape[1]
    result_tokens = engine_api.ResultTokens(
        data=data.astype(np.int32),
        tokens_idx=(0, speculations),
        valid_idx=(speculations, 2 * speculations),
        length_idx=(2 * speculations, 2 * speculations + 1),
        samples_per_slot=2,
    )
    vocab = mock_utils.TestVocab()
    per_channel, complete = token_utils.process_result_tokens(
        tokenizer=vocab,
        slot=0,
        slot_max_length=4,
        result_tokens=result_tokens,
        complete=mock_complete,
        is_client_side_tokenization=False,
    )
    np.testing.assert_equal(complete, np.array([1, 0]))

    text_output = [
        mock_utils.TestVocab().decode(row.token_ids) for row in per_channel
    ]
    assert not text_output[0]  # i.e. == '', because of the pad.
    assert text_output[1] == "AD"
    mock_complete = np.zeros(
        (mock_tokens.shape[0] // samples_per_slot), dtype=np.int32
    )
    per_channel, complete = token_utils.process_result_tokens(
        tokenizer=vocab,
        slot=1,
        slot_max_length=4,
        result_tokens=result_tokens,
        complete=mock_complete,
        is_client_side_tokenization=False,
    )
    text_output = [
        mock_utils.TestVocab().decode(row.token_ids) for row in per_channel
    ]
    assert text_output[0] == "T3"
    assert text_output[1] == "A"  # second token is padded.
    np.testing.assert_equal(complete, np.array([0, 1]))

  def test_mock_utils(self):
    vocab = mock_utils.TestVocab()
    # test encode()
    with self.assertRaises(NotImplementedError):
      vocab.encode("AB")
    # test encode_tf()
    token_ids = vocab.encode_tf("AB")
    np.testing.assert_equal(token_ids, np.array([65, 66]))
    # test decode()
    ids = np.array([ord("A")])
    expected = "A"
    result = vocab.decode(ids)
    self.assertEqual(result, expected)
    # test decode_tf()
    ids = np.array([[ord("A")]])
    expected = ["A"]
    result_tf = vocab.decode_tf(ids)
    self.assertEqual(result_tf, expected)


-----------------------

/jetstream/tests/engine/third_party/llama2/tokenizer.model:
-----------------------



-----------------------

/jetstream/tests/engine/third_party/llama3/tokenizer.model:
-----------------------

IQ== 0
Ig== 1
Iw== 2
JA== 3
JQ== 4
...

-----------------------

/jetstream/tests/entrypoints/__init__.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


-----------------------

/jetstream/tests/entrypoints/http/__init__.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


-----------------------

/jetstream/tests/entrypoints/http/test_api_server.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests http server end-to-end."""

import subprocess
import sys
import time
import unittest


import requests


class HTTPServerTest(unittest.IsolatedAsyncioTestCase):

  @classmethod
  def setUpClass(cls):
    """Sets up a JetStream http server for unit tests."""
    cls.base_url = "http://localhost:8080"
    cls.server = subprocess.Popen(
        [
            "python",
            "-m",
            "jetstream.entrypoints.http.api_server",
            "--config=InterleavedCPUTestServer",
        ],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    time.sleep(10)

  @classmethod
  def tearDownClass(cls):
    """Stop the server gracefully."""
    cls.server.terminate()

  async def test_root_endpoint(self):
    response = requests.get(self.base_url + "/", timeout=5)
    assert response.status_code == 200
    expected_data = {"message": "JetStream HTTP Server"}
    assert response.json() == expected_data

  async def test_health_endpoint(self):
    response = requests.get(self.base_url + "/v1/health", timeout=5)
    assert response.status_code == 200
    data = response.json()
    assert "is_live" in data
    assert data["is_live"] == "True"

  async def test_generate_endpoint(self):
    # Prepare a sample request (replace with actual data)
    sample_request_data = {
        "max_tokens": 10,
        "text_content": {"text": "translate this to french: hello world"},
    }

    response = requests.post(
        self.base_url + "/v1/generate",
        json=sample_request_data,
        stream=True,
        timeout=5,
    )
    assert response.status_code == 200
    full_response = []
    for chunk in response.iter_content(
        chunk_size=None
    ):  # chunk_size=None for complete lines
      if chunk:
        stream_response = chunk.decode("utf-8")
        print(f"{stream_response=}")
        full_response.append(stream_response)
    assert len(full_response) == 11  # 10 tokens + eos token


-----------------------

/jetstream/third_party/__init__.py:
-----------------------




-----------------------

/jetstream/third_party/llama3/__init__.py:
-----------------------



-----------------------

/jetstream/third_party/llama3/llama3_tokenizer.py:
-----------------------

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the
# Llama 3 Community License Agreement.

"""Tiktoken tokenizer used for llama3."""

import os
from logging import getLogger
from pathlib import Path
from typing import (
    AbstractSet,
    cast,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Sequence,
    TypedDict,
    Union,
)

import tiktoken
from tiktoken.load import load_tiktoken_bpe


logger = getLogger(__name__)


Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
  role: Role
  content: str


Dialog = Sequence[Message]


class Tokenizer:
  """
  Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
  """

  special_tokens: Dict[str, int]

  num_reserved_special_tokens = 256

  # pylint: disable=line-too-long
  pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

  def __init__(self, model_path: str):
    """
    Initializes the Tokenizer with a Tiktoken model.
    Args:
        model_path (str): The path to the Tiktoken model file.
    """
    assert os.path.isfile(model_path), model_path

    mergeable_ranks = load_tiktoken_bpe(model_path)
    num_base_tokens = len(mergeable_ranks)
    special_tokens = [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|reserved_special_token_0|>",
        "<|reserved_special_token_1|>",
        "<|reserved_special_token_2|>",
        "<|reserved_special_token_3|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|reserved_special_token_4|>",
        "<|eot_id|>",  # end of turn
    ] + [
        f"<|reserved_special_token_{i}|>"
        for i in range(5, self.num_reserved_special_tokens - 5)
    ]
    self.special_tokens = {
        token: num_base_tokens + i for i, token in enumerate(special_tokens)
    }
    self.model = tiktoken.Encoding(
        name=Path(model_path).name,
        pat_str=self.pat_str,
        mergeable_ranks=mergeable_ranks,
        special_tokens=self.special_tokens,
    )
    logger.info("Reloaded tiktoken model from %s", model_path)

    self.n_words: int = self.model.n_vocab
    # BOS / EOS token IDs
    self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
    self.eos_id: int = self.special_tokens["<|end_of_text|>"]
    self.pad_id: int = -1
    self.stop_tokens = {
        self.special_tokens["<|end_of_text|>"],
        self.special_tokens["<|eot_id|>"],
    }

    logger.info(
        "#words: %d - BOS ID: %d - EOS ID: %d",
        self.n_words,
        self.bos_id,
        self.eos_id,
    )

  def encode(
      self,
      s: str,
      *,
      bos: bool = False,
      eos: bool = False,
      allowed_special: Union[Literal["all"], AbstractSet[str]] | None = None,
      disallowed_special: Union[Literal["all"], Collection[str]] = (),
  ) -> List[int]:
    """
    Encodes a string into a list of token IDs.
    Args:
        s (str): The input string to be encoded.
        bos (bool): Whether to prepend the beginning-of-sequence token.
        eos (bool): Whether to append the end-of-sequence token.
        allowed_tokens ("all"|set[str]): allowed special tokens in string
        disallowed_tokens ("all"|set[str]): special tokens that raise an error when in string
    Returns:
        list[int]: A list of token IDs.
    By default, setting disallowed_special=() encodes a string by ignoring
    special tokens. Specifically:
    - Setting `disallowed_special` to () will cause all text corresponding
      to special tokens to be encoded as natural text (insteading of raising
      an error).
    - Setting `allowed_special` to "all" will treat all text corresponding
      to special tokens to be encoded as special tokens.
    """
    assert isinstance(s, str)

    # The tiktoken tokenizer can handle <=400k chars without
    # pyo3_runtime.PanicException.
    tiktoken_max_encode_chars = 400_000

    # https://github.com/openai/tiktoken/issues/195
    # Here we iterate over subsequences and split if we exceed the limit
    # of max consecutive non-whitespace or whitespace characters.
    max_no_whitespaces_chars = 25_000

    substrs = (
        substr
        for i in range(0, len(s), tiktoken_max_encode_chars)
        for substr in self._split_whitespaces_or_nonwhitespaces(
            s[i : i + tiktoken_max_encode_chars], max_no_whitespaces_chars
        )
    )
    t: List[int] = []
    if allowed_special is None:
      allowed_special = set()
    for substr in substrs:
      t.extend(
          self.model.encode(
              substr,
              allowed_special=allowed_special,
              disallowed_special=disallowed_special,
          )
      )
    if bos:
      t.insert(0, self.bos_id)
    if eos:
      t.append(self.eos_id)
    return t

  def decode(self, t: Sequence[int]) -> str:
    """
    Decodes a list of token IDs into a string.
    Args:
        t (List[int]): The list of token IDs to be decoded.
    Returns:
        str: The decoded string.
    """
    # Typecast is safe here. Tiktoken doesn't do anything list-related with the sequence.
    return self.model.decode(cast(List[int], t))

  @staticmethod
  def _split_whitespaces_or_nonwhitespaces(
      s: str, max_consecutive_slice_len: int
  ) -> Iterator[str]:
    """
    Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
    consecutive whitespaces or consecutive non-whitespaces.
    """
    current_slice_len = 0
    current_slice_is_space = s[0].isspace() if len(s) > 0 else False
    slice_start = 0

    for i in range(len(s)):
      is_now_space = s[i].isspace()

      if current_slice_is_space ^ is_now_space:
        current_slice_len = 1
        current_slice_is_space = is_now_space
      else:
        current_slice_len += 1
        if current_slice_len > max_consecutive_slice_len:
          yield s[slice_start:i]
          slice_start = i
          current_slice_len = 1
    yield s[slice_start:]


class ChatFormat:
  """Helper class to encode/decode messages in chat format."""

  def __init__(self, tokenizer: Tokenizer):
    self.tokenizer = tokenizer

  def encode_header(self, message: Message) -> List[int]:
    tokens = []
    tokens.append(self.tokenizer.special_tokens["<|start_header_id|>"])
    tokens.extend(self.tokenizer.encode(message["role"], bos=False, eos=False))
    tokens.append(self.tokenizer.special_tokens["<|end_header_id|>"])
    tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))
    return tokens

  def encode_message(self, message: Message) -> List[int]:
    tokens = self.encode_header(message)
    tokens.extend(
        self.tokenizer.encode(message["content"].strip(), bos=False, eos=False)
    )
    tokens.append(self.tokenizer.special_tokens["<|eot_id|>"])
    return tokens

  def encode_dialog_prompt(self, dialog: Dialog) -> List[int]:
    tokens = []
    tokens.append(self.tokenizer.special_tokens["<|begin_of_text|>"])
    for message in dialog:
      tokens.extend(self.encode_message(message))
    # Add the start of an assistant message for the model to complete.
    tokens.extend(self.encode_header({"role": "assistant", "content": ""}))
    return tokens


-----------------------

/jetstream/tools/load_tester.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Miniature load test of the mock server."""

import concurrent.futures
import functools
import time
from typing import Iterator, Sequence

from absl import app
from absl import flags
import grpc
from jetstream.core.proto import jetstream_pb2
from jetstream.core.proto import jetstream_pb2_grpc


_SERVER = flags.DEFINE_string("server", "0.0.0.0", "server address")
_PORT = flags.DEFINE_string("port", "9000", "port to ping")
_TEXT = flags.DEFINE_string("text", "AB", "The message")
_MAX_TOKENS = flags.DEFINE_integer(
    "max_tokens", 100, "Maximum number of output/decode tokens of a sequence"
)


def collect_tokens(
    response: Iterator[jetstream_pb2.DecodeResponse], print_interim: bool
) -> list[str]:
  tokens = []
  for resp in response:
    text_pieces = resp.stream_content.samples[0].text
    if print_interim:
      print(text_pieces, end="", flush=True)
    tokens.extend(text_pieces)
  return tokens


def api_call(
    stub: jetstream_pb2_grpc.OrchestratorStub,
    text: str,
    max_tokens: int,
    print_interim: bool = True,
) -> str:
  """Sends a request to server and returns text."""
  request = jetstream_pb2.DecodeRequest(
      text_content=jetstream_pb2.DecodeRequest.TextContent(text=text),
      max_tokens=max_tokens,
  )
  response = stub.Decode(request)
  print("---------------------- Sent!!!----------------------")
  tokens = collect_tokens(response, print_interim=print_interim)

  return "".join(tokens)


def ping(
    stub: jetstream_pb2_grpc.OrchestratorStub, text: str, number: int
) -> str:
  response = api_call(stub, text, _MAX_TOKENS.value, print_interim=False)
  print(f"Completed {number}")
  return response


def load_test(
    stub: jetstream_pb2_grpc.OrchestratorStub,
    text: list[str],
    queries: int = 64,
) -> list[str]:
  """Sends many queries to the server."""
  assert queries % len(text) == 0
  # repeat out
  text = text * (queries // len(text))
  number = list(range(len(text)))
  start = time.time()
  ping_partial = functools.partial(ping, stub)
  with concurrent.futures.ThreadPoolExecutor(max_workers=queries) as executor:
    responses = list(executor.map(ping_partial, text, number))
  time_taken = time.time() - start
  print(f"Time taken: {time_taken}")
  print(f"QPS: {queries/time_taken}")
  return responses


def main(argv: Sequence[str]):
  del argv
  address = f"{_SERVER.value}:{_PORT.value}"
  # Note: Uses insecure_channel only for local testing. Please add grpc
  # credentials for Production.
  with grpc.insecure_channel(address) as channel:
    grpc.channel_ready_future(channel).result()
    stub = jetstream_pb2_grpc.OrchestratorStub(channel)
    _ = load_test(stub, text=[_TEXT.value], queries=64)


if __name__ == "__main__":
  app.run(main)


-----------------------

/jetstream/tools/maxtext/model_ckpt_conversion.sh:
-----------------------

#!/bin/bash
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# This script will do the following:
# - Create GCS buckets to store model artifacts for the JetStream Maxtext Inference demo.
# - Convert the downloaded checkpoints to MaxText compatible checkpoints.
# - Convert the MaxText compatible checkpoints to unscanned checkpoints for inference.
# Device requirements:
# - Both checkpoints conversion only requires CPU (with JAX CPU mode).
set -ex

idx=$(date +%Y-%m-%d-%H-%M)
# Modify the `MODEL` and `MODEL_VARIATION` based on the model you use.
export MODEL=$1
export MODEL_VARIATION=$2
export MODEL_NAME=${MODEL}-${MODEL_VARIATION}

# After downloading checkpoints, copy them to GCS bucket at $CHKPT_BUCKET
# Please use separate GCS paths for uploading open source model weights ($CHKPT_BUCKET) and MaxText compatible weights ($MODEL_BUCKET).
# Point these variables to a GCS bucket that you created.
# An example of CHKPT_BUCKET could be: gs://${USER}-maxtext/chkpt/${MODEL}/${MODEL_VARIATION}
export CHKPT_BUCKET=$3
export MODEL_BUCKET=$4

# Point `BASE_OUTPUT_DIRECTORY` to a GCS bucket that you created, this bucket will store all the files generated by MaxText during a run, specifically the unscanned checkpoint.
export BASE_OUTPUT_DIRECTORY=$5

export BUCKET_LOCATION=US

# Create three GCS buckets for the demo.
gcloud storage buckets create ${MODEL_BUCKET} --location=${BUCKET_LOCATION} || true
gcloud storage buckets create ${BASE_OUTPUT_DIRECTORY} --location=${BUCKET_LOCATION} || true

# Convert model checkpoints to MaxText compatible checkpoints.
if [ "$MODEL" == "gemma" ]; then
    CONVERT_CKPT_SCRIPT="convert_gemma_chkpt.py"
    JAX_PLATFORMS=cpu python MaxText/${CONVERT_CKPT_SCRIPT} \
    --base_model_path ${CHKPT_BUCKET} \
    --maxtext_model_path ${MODEL_BUCKET}/${MODEL}/${MODEL_VARIATION}/${idx} \
    --model_size ${MODEL_VARIATION}
else
    # We install torch CPU because the checkpoint conversion script MaxText/llama_or_mistral_ckpt.py does not need a TPU/GPU.
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    # llama_or_mistral_ckpt.py requires local path, so we need to copy the checkpoint from CHKPT_BUCKET to local.
    tmp_ckpt_path="/tmp/"
    gcloud storage cp -r ${CHKPT_BUCKET} ${tmp_ckpt_path}
    path_parts=(${CHKPT_BUCKET//\// })
    directory_substring=${path_parts[-1]}
    CONVERT_CKPT_SCRIPT="llama_or_mistral_ckpt.py"
    JAX_PLATFORMS=cpu python MaxText/${CONVERT_CKPT_SCRIPT} \
    --base-model-path ${tmp_ckpt_path}${directory_substring} \
    --maxtext-model-path ${MODEL_BUCKET}/${MODEL}/${MODEL_VARIATION}/${idx} \
    --model-size ${MODEL_NAME}
fi
echo "Written MaxText compatible checkpoint to ${MODEL_BUCKET}/${MODEL}/${MODEL_VARIATION}/${idx}"

# We define `SCANNED_CKPT_PATH` to refer to the checkpoint subdirectory.
export SCANNED_CKPT_PATH=${MODEL_BUCKET}/${MODEL}/${MODEL_VARIATION}/${idx}/0/items

# Convert MaxText compatible checkpoints to unscanned checkpoints.
# Note that the `SCANNED_CKPT_PATH` is in a `scanned` format which is great for training but for efficient decoding performance we want the checkpoint in an `unscanned` format.
export RUN_NAME=${MODEL_NAME}_unscanned_chkpt_${idx}

JAX_PLATFORMS=cpu python MaxText/generate_param_only_checkpoint.py \
MaxText/configs/base.yml \
base_output_directory=${BASE_OUTPUT_DIRECTORY} \
load_parameters_path=${SCANNED_CKPT_PATH} \
run_name=${RUN_NAME} \
model_name=${MODEL_NAME} \
force_unroll=true
echo "Written MaxText unscanned checkpoint to ${BASE_OUTPUT_DIRECTORY}/${RUN_NAME}/checkpoints"

# We will use the unscanned checkpoints by passing `UNSCANNED_CKPT_PATH` into `LOAD_PARAMETERS_PATH` in the following sections.
export UNSCANNED_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/${RUN_NAME}/checkpoints/0/items


-----------------------

/jetstream/tools/maxtext/model_ckpt_finetune_with_aqt.sh:
-----------------------

#!/bin/bash
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# This script will do the following:
# - Finetuning the MaxText compatible checkpoint (converted from original checkpoints) with AQT
# - Convert the AQT-finetuned checkpoints to unscanned checkpoints for inference
# TPU device requirements:
# - For llama2-7b, it requires at least a v5e-8 TPU VM.
# - For llama2-13B/70b, it requires a v4-128 TPU VM.
set -ex

idx=$(date +%Y-%m-%d-%H-%M)
# Modify the `MODEL` and `MODEL_VARIATION` based on the model you use.
export MODEL=$1
export MODEL_VARIATION=$2
export MODEL_NAME=${MODEL}-${MODEL_VARIATION}

# After downloading checkpoints, copy them to GCS bucket at $CHKPT_BUCKET \
# Please use separate GCS paths for uploading open source model weights ($CHKPT_BUCKET) and MaxText compatible weights ($MODEL_BUCKET).
# Point these variables to a GCS bucket that you created.
# An example of CHKPT_BUCKET could be: gs://${USER}-maxtext/chkpt/${MODEL}/${MODEL_VARIATION}
export CHKPT_BUCKET=$3
export MODEL_BUCKET=gs://${USER}-maxtext

# Point `BASE_OUTPUT_DIRECTORY` to a GCS bucket that you created, this bucket will store all the files generated by MaxText during a run.
export BASE_OUTPUT_DIRECTORY=gs://${USER}-runner-maxtext-logs

# Point `DATASET_PATH` to the GCS bucket where you have your training data
export DATASET_PATH=gs://${USER}-maxtext-dataset

# Prepare C4 dataset for fine tuning: https://github.com/allenai/allennlp/discussions/5056
sudo gsutil -u $4 -m cp 'gs://allennlp-tensorflow-datasets/c4/en/3.0.1/*' ${DATASET_PATH}/c4/en/3.0.1/

# We define `CONVERTED_CHECKPOINT` to refer to the checkpoint subdirectory.
export CONVERTED_CHECKPOINT=${MODEL_BUCKET}/${MODEL}/${MODEL_VARIATION}/${idx}/0/items

# Fine tune the converted model checkpoints with AQT.
export RUN_NAME=finetune_aqt_${idx}

python3 MaxText/train.py \
MaxText/configs/base.yml \
run_name=${RUN_NAME} \
base_output_directory=${BASE_OUTPUT_DIRECTORY} \
dataset_path=${DATASET_PATH} \
steps=501 \
enable_checkpointing=True \
load_parameters_path=${CONVERTED_CHECKPOINT} \
model_name=${MODEL_NAME} \
per_device_batch_size=1 \
quantization=int8 \
checkpoint_period=100

# We will convert the `AQT_CKPT` to unscanned checkpoint in the next step.
export AQT_CKPT=${BASE_OUTPUT_DIRECTORY}/${RUN_NAME}/checkpoints/100/items

# Convert MaxText compatible AQT-fine-tuned checkpoints to unscanned checkpoints.
# Note that the `AQT_CKPT` is in a `scanned` format which is great for training but for efficient decoding performance we want the checkpoint in an `unscanned` format.
export RUN_NAME=${MODEL_NAME}_unscanned_chkpt_${idx}

JAX_PLATFORMS=cpu python MaxText/generate_param_only_checkpoint.py \
MaxText/configs/base.yml \
base_output_directory=${BASE_OUTPUT_DIRECTORY} \
load_parameters_path=${AQT_CKPT} \
run_name=${RUN_NAME} \
model_name=${MODEL_NAME} \
force_unroll=true
echo "Written MaxText unscanned checkpoint to ${BASE_OUTPUT_DIRECTORY}/${RUN_NAME}/checkpoints"

# We will use the unscanned checkpoints by passing `UNSCANNED_CKPT_PATH` into `LOAD_PARAMETERS_PATH` in the following sections.
export UNSCANNED_CKPT_PATH=${BASE_OUTPUT_DIRECTORY}/${RUN_NAME}/checkpoints/0/items


-----------------------

/jetstream/tools/proxy_dev/base.Dockerfile:
-----------------------

# Ubuntu:22.04
# Use Ubuntu 22.04 from Docker Hub.
# https://hub.docker.com/_/ubuntu/tags\?page\=1\&name\=22.04
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt -y update && apt install -y --no-install-recommends apt-transport-https ca-certificates gnupg git python3.10 python3-pip curl nano vim

RUN update-alternatives --install     /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && apt-get update -y && apt-get install google-cloud-sdk -y


# Copy all files from local workspace into docker container
COPY  JetStream ./JetStream
COPY  maxtext ./maxtext

RUN cd maxtext/ && \
pip install -r requirements.txt

RUN pip install setuptools==58 fastapi==0.103.2 uvicorn

RUN pip install ./JetStream

RUN apt -y update && apt-get -y install python3-dev && apt-get -y install build-essential
RUN pip install \
    transformers==4.31.0 \
    nltk==3.8.1 \
    evaluate==0.4.0 \
    absl-py==1.4.0 \
    rouge-score==0.1.2 \
    sentencepiece==0.1.99 \
    accelerate==0.21.0

ENTRYPOINT ["bash"]


-----------------------

/jetstream/tools/proxy_dev/dev.Dockerfile:
-----------------------

# Ubuntu:22.04
# Use Ubuntu 22.04 from Docker Hub.
# https://hub.docker.com/_/ubuntu/tags\?page\=1\&name\=22.04
FROM base_image

ENV DEBIAN_FRONTEND=noninteractive

ENV JAX_PLATFORMS=proxy
ENV JAX_BACKEND_TARGET=grpc://localhost:38681

# Copy all files from local workspace into docker container
COPY  JetStream ./JetStream
COPY  maxtext ./maxtext

RUN pip install ./JetStream
RUN pip install -r ./maxtext/requirements.txt

ENTRYPOINT ["bash"]


-----------------------

/jetstream/tools/requester.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A test request."""

from typing import Sequence

from absl import app
from absl import flags
import grpc
from jetstream.core.proto import jetstream_pb2
from jetstream.core.proto import jetstream_pb2_grpc
from jetstream.engine.token_utils import load_vocab


_SERVER = flags.DEFINE_string("server", "0.0.0.0", "server address")
_PORT = flags.DEFINE_string("port", "9000", "port to ping")
_TEXT = flags.DEFINE_string("text", "My dog is cute", "The message")
_MAX_TOKENS = flags.DEFINE_integer(
    "max_tokens", 3, "Maximum number of output/decode tokens of a sequence"
)
_TOKENIZER = flags.DEFINE_string(
    "tokenizer",
    None,
    "Name or path of the tokenizer (matched to the model)",
    required=True,
)
_CLIENT_SIDE_TOKENIZATION = flags.DEFINE_bool(
    "client_side_tokenization",
    False,
    "Enable client side tokenization with tokenizer.",
)


def _GetResponseAsync(
    stub: jetstream_pb2_grpc.OrchestratorStub,
    request: jetstream_pb2.DecodeRequest,
) -> None:
  """Gets an async response."""

  response = stub.Decode(request)
  output = []
  for resp in response:
    if _CLIENT_SIDE_TOKENIZATION.value:
      output.extend(resp.stream_content.samples[0].token_ids)
    else:
      output.extend(resp.stream_content.samples[0].text)
  if _CLIENT_SIDE_TOKENIZATION.value:
    vocab = load_vocab(_TOKENIZER.value)
    text_output = vocab.tokenizer.decode(output)
  else:
    text_output = "".join(output)
  print(f"Prompt: {_TEXT.value}")
  print(f"Response: {text_output}")


def main(argv: Sequence[str]) -> None:
  del argv
  # Note: Uses insecure_channel only for local testing. Please add grpc
  # credentials for Production.
  address = f"{_SERVER.value}:{_PORT.value}"
  with grpc.insecure_channel(address) as channel:
    grpc.channel_ready_future(channel).result()
    stub = jetstream_pb2_grpc.OrchestratorStub(channel)
    print(f"Sending request to: {address}")
    if _CLIENT_SIDE_TOKENIZATION.value:
      vocab = load_vocab(_TOKENIZER.value)
      token_ids = vocab.tokenizer.encode(_TEXT.value)
      request = jetstream_pb2.DecodeRequest(
          token_content=jetstream_pb2.DecodeRequest.TokenContent(
              token_ids=token_ids
          ),
          max_tokens=_MAX_TOKENS.value,
      )
    else:
      request = jetstream_pb2.DecodeRequest(
          text_content=jetstream_pb2.DecodeRequest.TextContent(
              text=_TEXT.value
          ),
          max_tokens=_MAX_TOKENS.value,
      )
    return _GetResponseAsync(stub, request)


if __name__ == "__main__":
  app.run(main)


-----------------------

/license_preamble.txt:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


-----------------------

/pylintrc:
-----------------------


# This Pylint rcfile contains a best-effort configuration to uphold the
# best-practices and style described in the Google Python style guide:
#   https://google.github.io/styleguide/pyguide.html
#
# Its canonical open-source location is:
#   https://google.github.io/styleguide/pylintrc

[MAIN]

# Files or directories to be skipped. They should be base names, not paths.
ignore=third_party

# Files or directories matching the regex patterns are skipped. The regex
# matches against base names, not paths.
ignore-patterns=

# Pickle collected data for later comparisons.
persistent=no

# List of plugins (as comma separated values of python modules names) to load,
# usually to register additional checkers.
load-plugins=

# Use multiple processes to speed up Pylint.
jobs=4

# Allow loading of arbitrary C extensions. Extensions are imported into the
# active Python interpreter and may run arbitrary code.
unsafe-load-any-extension=no


[MESSAGES CONTROL]

# Only show warnings with the listed confidence levels. Leave empty to show
# all. Valid levels: HIGH, INFERENCE, INFERENCE_FAILURE, UNDEFINED
confidence=

# Enable the message, report, category or checker with the given id(s). You can
# either give multiple identifier separated by comma (,) or put this option
# multiple time (only on the command line, not in the configuration file where
# it should appear only once). See also the "--disable" option for examples.
#enable=

# Disable the message, report, category or checker with the given id(s). You
# can either give multiple identifiers separated by comma (,) or put this
# option multiple times (only on the command line, not in the configuration
# file where it should appear only once).You can also use "--disable=all" to
# disable everything first and then reenable specific checks. For example, if
# you want to run only the similarities checker, you can use "--disable=all
# --enable=similarities". If you want to run only the classes checker, but have
# no Warning level messages displayed, use"--disable=all --enable=classes
# --disable=W"
disable=R,
        abstract-method,
        apply-builtin,
        arguments-differ,
        attribute-defined-outside-init,
        backtick,
        bad-option-value,
        basestring-builtin,
        buffer-builtin,
        c-extension-no-member,
        consider-using-enumerate,
        cmp-builtin,
        cmp-method,
        coerce-builtin,
        coerce-method,
        delslice-method,
        div-method,
        eq-without-hash,
        execfile-builtin,
        file-builtin,
        filter-builtin-not-iterating,
        fixme,
        getslice-method,
        global-statement,
        hex-method,
        idiv-method,
        implicit-str-concat,
        import-error,
        import-self,
        import-star-module-level,
        input-builtin,
        intern-builtin,
        invalid-str-codec,
        locally-disabled,
        long-builtin,
        long-suffix,
        map-builtin-not-iterating,
        misplaced-comparison-constant,
        missing-function-docstring,
        metaclass-assignment,
        next-method-called,
        next-method-defined,
        no-absolute-import,
        no-init,  # added
        no-member,
        no-name-in-module,
        no-self-use,
        nonzero-method,
        oct-method,
        old-division,
        old-ne-operator,
        old-octal-literal,
        old-raise-syntax,
        parameter-unpacking,
        print-statement,
        raising-string,
        range-builtin-not-iterating,
        raw_input-builtin,
        rdiv-method,
        reduce-builtin,
        relative-import,
        reload-builtin,
        round-builtin,
        setslice-method,
        signature-differs,
        standarderror-builtin,
        suppressed-message,
        sys-max-int,
        trailing-newlines,
        unichr-builtin,
        unicode-builtin,
        unnecessary-pass,
        unpacking-in-except,
        useless-else-on-loop,
        useless-suppression,
        using-cmp-argument,
        wrong-import-order,
        xrange-builtin,
        zip-builtin-not-iterating,


[REPORTS]

# Set the output format. Available formats are text, parseable, colorized, msvs
# (visual studio) and html. You can also give a reporter class, eg
# mypackage.mymodule.MyReporterClass.
output-format=text

# Tells whether to display a full report or only the messages
reports=no

# Python expression which should return a note less than 10 (10 is the highest
# note). You have access to the variables errors warning, statement which
# respectively contain the number of errors / warnings messages and the total
# number of statements analyzed. This is used by the global evaluation report
# (RP0004).
evaluation=10.0 - ((float(5 * error + warning + refactor + convention) / statement) * 10)

# Template used to display messages. This is a python new-style format string
# used to format the message information. See doc for all details
#msg-template=


[BASIC]

# Good variable names which should always be accepted, separated by a comma
good-names=main,_

# Bad variable names which should always be refused, separated by a comma
bad-names=

# Colon-delimited sets of names that determine each other's naming style when
# the name regexes allow several styles.
name-group=

# Include a hint for the correct naming format with invalid-name
include-naming-hint=no

# List of decorators that produce properties, such as abc.abstractproperty. Add
# to this list to register other decorators that produce valid properties.
property-classes=abc.abstractproperty,cached_property.cached_property,cached_property.threaded_cached_property,cached_property.cached_property_with_ttl,cached_property.threaded_cached_property_with_ttl

# Regular expression matching correct function names
function-rgx=^(?:(?P<exempt>setUp|tearDown|setUpModule|tearDownModule)|(?P<camel_case>_?[A-Z][a-zA-Z0-9]*)|(?P<snake_case>_?[a-z][a-z0-9_]*))$

# Regular expression matching correct variable names
variable-rgx=^[a-z][a-z0-9_]*$

# Regular expression matching correct constant names
const-rgx=^(_?[A-Z][A-Z0-9_]*|__[a-z0-9_]+__|_?[a-z][a-z0-9_]*)$

# Regular expression matching correct attribute names
attr-rgx=^_{0,2}[a-z][a-z0-9_]*$

# Regular expression matching correct argument names
argument-rgx=^[a-z][a-z0-9_]*$

# Regular expression matching correct class attribute names
class-attribute-rgx=^(_?[A-Z][A-Z0-9_]*|__[a-z0-9_]+__|_?[a-z][a-z0-9_]*)$

# Regular expression matching correct inline iteration names
inlinevar-rgx=^[a-z][a-z0-9_]*$

# Regular expression matching correct class names
class-rgx=^_?[A-Z][a-zA-Z0-9]*$

# Regular expression matching correct module names
module-rgx=^(_?[a-z][a-z0-9_]*|__init__)$

# Regular expression matching correct method names
method-rgx=(?x)^(?:(?P<exempt>_[a-z0-9_]+__|runTest|setUp|tearDown|setUpTestCase|tearDownTestCase|setupSelf|tearDownClass|setUpClass|(test|assert)_*[A-Z0-9][a-zA-Z0-9_]*|next)|(?P<camel_case>_{0,2}[A-Z][a-zA-Z0-9_]*)|(?P<snake_case>_{0,2}[a-z][a-z0-9_]*))$

# Regular expression which should only match function or class names that do
# not require a docstring.
no-docstring-rgx=(__.*__|main|test.*|.*test|.*Test)$

# Minimum line length for functions/classes that require docstrings, shorter
# ones are exempt.
docstring-min-length=12


[TYPECHECK]

# List of decorators that produce context managers, such as
# contextlib.contextmanager. Add to this list to register other decorators that
# produce valid context managers.
contextmanager-decorators=contextlib.contextmanager,contextlib2.contextmanager

# List of module names for which member attributes should not be checked
# (useful for modules/projects where namespaces are manipulated during runtime
# and thus existing member attributes cannot be deduced by static analysis. It
# supports qualified module names, as well as Unix pattern matching.
ignored-modules=

# List of class names for which member attributes should not be checked (useful
# for classes with dynamically set attributes). This supports the use of
# qualified names.
ignored-classes=optparse.Values,thread._local,_thread._local

# List of members which are set dynamically and missed by pylint inference
# system, and so shouldn't trigger E1101 when accessed. Python regular
# expressions are accepted.
generated-members=


[FORMAT]

# Maximum number of characters on a single line.
max-line-length=80

# TODO(https://github.com/pylint-dev/pylint/issues/3352): Direct pylint to exempt
# lines made too long by directives to pytype.

# Regexp for a line that is allowed to be longer than the limit.
ignore-long-lines=(?x)(
  ^\s*(\#\ )?<?https?://\S+>?$|
  ^\s*(from\s+\S+\s+)?import\s+.+$)

# Allow the body of an if to be on the same line as the test if there is no
# else.
single-line-if-stmt=yes

# Maximum number of lines in a module
max-module-lines=99999

# String used as indentation unit.  The internal Google style guide mandates 2
# spaces.  Google's externaly-published style guide says 4, consistent with
# PEP 8.  Here, we use 2 spaces, for conformity with many open-sourced Google
# projects (like TensorFlow).
indent-string='  '

# Number of spaces of indent required inside a hanging  or continued line.
indent-after-paren=4

# Expected format of line ending, e.g. empty (any line ending), LF or CRLF.
expected-line-ending-format=


[MISCELLANEOUS]

# List of note tags to take in consideration, separated by a comma.
notes=TODO


[STRING]

# This flag controls whether inconsistent-quotes generates a warning when the
# character used as a quote delimiter is used inconsistently within a module.
check-quote-consistency=yes


[VARIABLES]

# Tells whether we should check for unused import in __init__ files.
init-import=no

# A regular expression matching the name of dummy variables (i.e. expectedly
# not used).
dummy-variables-rgx=^\*{0,2}(_$|unused_|dummy_)

# List of additional names supposed to be defined in builtins. Remember that
# you should avoid to define new builtins when possible.
additional-builtins=

# List of strings which can identify a callback function by name. A callback
# name must start or end with one of those strings.
callbacks=cb_,_cb

# List of qualified module names which can have objects that can redefine
# builtins.
redefining-builtins-modules=six,six.moves,past.builtins,future.builtins,functools


[LOGGING]

# Logging modules to check that the string format arguments are in logging
# function parameter format
logging-modules=logging,absl.logging,tensorflow.io.logging


[SIMILARITIES]

# Minimum lines number of a similarity.
min-similarity-lines=4

# Ignore comments when computing similarities.
ignore-comments=yes

# Ignore docstrings when computing similarities.
ignore-docstrings=yes

# Ignore imports when computing similarities.
ignore-imports=no


[SPELLING]

# Spelling dictionary name. Available dictionaries: none. To make it working
# install python-enchant package.
spelling-dict=

# List of comma separated words that should not be checked.
spelling-ignore-words=

# A path to a file that contains private dictionary; one word per line.
spelling-private-dict-file=

# Tells whether to store unknown words to indicated private dictionary in
# --spelling-private-dict-file option instead of raising a message.
spelling-store-unknown-words=no


[IMPORTS]

# Deprecated modules which should not be used, separated by a comma
deprecated-modules=regsub,
                   TERMIOS,
                   Bastion,
                   rexec,
                   sets

# Create a graph of every (i.e. internal and external) dependencies in the
# given file (report RP0402 must not be disabled)
import-graph=

# Create a graph of external dependencies in the given file (report RP0402 must
# not be disabled)
ext-import-graph=

# Create a graph of internal dependencies in the given file (report RP0402 must
# not be disabled)
int-import-graph=

# Force import order to recognize a module as part of the standard
# compatibility libraries.
known-standard-library=

# Force import order to recognize a module as part of a third party library.
known-third-party=enchant, absl

# Analyse import fallback blocks. This can be used to support both Python 2 and
# 3 compatible code, which means that the block might have code that exists
# only in one or another interpreter, leading to false positives when analysed.
analyse-fallback-blocks=no


[CLASSES]

# List of method names used to declare (i.e. assign) instance attributes.
defining-attr-methods=__init__,
                      __new__,
                      setUp

# List of member names, which should be excluded from the protected access
# warning.
exclude-protected=_asdict,
                  _fields,
                  _replace,
                  _source,
                  _make

# List of valid names for the first argument in a class method.
valid-classmethod-first-arg=cls,
                            class_

# List of valid names for the first argument in a metaclass class method.
valid-metaclass-classmethod-first-arg=mcs


-----------------------

/requirements.txt:
-----------------------

absl-py
coverage
flax
grpcio
jax
jaxlib
numpy
portpicker
prometheus-client
pytest
seqio
tiktoken
blobfile
parameterized
shortuuid
fastapi
uvicorn
# For profiling
tensorboard-plugin-profile

-----------------------

/setup.py:
-----------------------

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import find_packages, setup


def parse_requirements(filename):
  """load requirements from a pip requirements file."""
  with open(filename) as f:
    lineiter = (line.strip() for line in f)
    return [line for line in lineiter if line and not line.startswith("#")]


setup(
    name="google-jetstream",
    version="0.2.2",
    description=(
        "JetStream is a throughput and memory optimized engine for LLM inference on XLA devices, starting with TPUs (and GPUs in future -- PRs welcome)."
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Google LLC",
    url="https://github.com/google/JetStream",
    packages=find_packages(exclude="benchmarks"),
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=parse_requirements("requirements.txt"),
)