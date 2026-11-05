#!/usr/bin/env bash
set -euo pipefail
cd /workspace/cluster
export UV_CACHE_DIR=/workspace/.cache/uv
export PIP_CACHE_DIR=/workspace/.cache/pip
uv python install 3.10.22 --install-dir /workspace/toolchains --no-progress
if [ ! -x .venv/bin/python ]; then
  uv venv --python /workspace/toolchains/cpython-3.10.22-linux-x86_64-gnu/bin/python3.10 .venv
fi
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  'numpy==1.23.5' 'torch==2.0.0+cpu' 'setuptools==68.2.2' pip wheel
# Build the pinned graph extensions from PyPI if no compatible local wheel exists.
# data.pyg.org is not required. TLS/package verification remains enabled.
MAX_JOBS=2 .venv/bin/python -m pip install --no-build-isolation \
  --extra-index-url https://download.pytorch.org/whl/cpu -r hyperbolic/requirements-lock.txt
.venv/bin/python -m pip check
.venv/bin/python -c 'import torch, torch_scatter, torch_cluster; print(torch.__version__, torch_scatter.__version__, torch_cluster.__version__)'
