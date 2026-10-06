#!/usr/bin/env bash
# 创建 defense 环境：先按 environment.yml 安装，再补装 opendelta / flash-attn。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3}"
OPENDELTA_VERSION="${OPENDELTA_VERSION:-0.3.2}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
MAX_JOBS="${MAX_JOBS:-$(nproc)}"

if ! command -v conda >/dev/null 2>&1; then
  for candidate in \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh" \
    "/root/miniconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [[ -f "$candidate" ]]; then
      # shellcheck disable=SC1090
      source "$candidate"
      break
    fi
  done
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "error: conda not found; source conda.sh first or add conda to PATH" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx defense; then
  echo "==> removing existing conda env: defense"
  conda env remove -n defense -y
fi

echo "==> creating conda env from environment.yml"
conda env create -f environment.yml

conda activate defense

echo "==> installing opendelta==${OPENDELTA_VERSION} (--no-deps; avoids broken PyPI 'sklearn')"
pip install "opendelta==${OPENDELTA_VERSION}" --no-deps

# opendelta 0.3.2 imports transformers.deepspeed，该路径在新版 transformers 中已迁走
OPENDELTA_BASEMODEL="$(python -c 'import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec("opendelta").origin).with_name("basemodel.py"))')"
if grep -q 'from transformers.deepspeed import' "${OPENDELTA_BASEMODEL}"; then
  echo "==> patching opendelta for transformers>=4.3x deepspeed import"
  sed -i 's/from transformers\.deepspeed import/from transformers.integrations.deepspeed import/' "${OPENDELTA_BASEMODEL}"
fi

if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "error: nvcc not found at ${CUDA_HOME}/bin/nvcc (flash-attn needs CUDA toolkit)" >&2
  exit 1
fi

export CUDA_HOME
export MAX_JOBS
if python -c 'import flash_attn' >/dev/null 2>&1; then
  echo "==> flash-attn already installed, skipping build"
else
  echo "==> installing flash-attn==${FLASH_ATTN_VERSION} (CUDA_HOME=${CUDA_HOME}, MAX_JOBS=${MAX_JOBS})"
  pip install "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation
fi

echo "==> verifying imports"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
import flash_attn
print("flash_attn", flash_attn.__version__)
from opendelta.basemodel import DeltaBase
from opendelta.utils.decorate import decorate
print("opendelta ok", DeltaBase, decorate)
PY

echo "==> done. Activate with: conda activate defense"
