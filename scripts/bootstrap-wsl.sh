#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MINIFORGE_ROOT="${HOME}/miniforge3"

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  export PATH="${CONDA_BASE}/bin:${PATH}"
elif [[ -x "${MINIFORGE_ROOT}/bin/conda" ]]; then
  export PATH="${MINIFORGE_ROOT}/bin:${PATH}"
else
  INSTALLER="$(mktemp --suffix=.sh)"
  trap 'rm -f "${INSTALLER}"' EXIT
  curl -fsSL -o "${INSTALLER}" \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash "${INSTALLER}" -b -p "${MINIFORGE_ROOT}"
  export PATH="${MINIFORGE_ROOT}/bin:${PATH}"
fi

cd "${PROJECT_ROOT}"
if conda env list | grep -Eq '^3dgs[[:space:]]'; then
  conda env update --name 3dgs --file environment.yml --prune
else
  conda install --name base --yes --channel conda-forge conda-lock=4.0.2
  conda-lock install --conda "$(command -v conda)" --name 3dgs conda-lock.yml
fi
conda run -n 3dgs python -m pip install --editable '.[dev]'
conda run -n 3dgs python scripts/apply_nerfstudio_patch.py
conda run -n 3dgs gsdb doctor
