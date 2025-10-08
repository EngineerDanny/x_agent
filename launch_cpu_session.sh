#!/usr/bin/env bash
# Helper to land on a CPU compute node with the llama-cpu Conda env active.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV_NAME="llama-cpu"
CONDA_BASE="${CONDA_BASE_OVERRIDE:-/packages/anaconda3/2024.02}"

# Redirect cache directories away from $HOME.
CACHE_ROOT="${CACHE_ROOT_OVERRIDE:-${SCRIPT_DIR}/.cache}"
mkdir -p "${CACHE_ROOT}"
export XDG_CACHE_HOME="${CACHE_ROOT}"
export HF_HOME="${CACHE_ROOT}/huggingface"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/huggingface"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/huggingface/transformers"
export HF_DATASETS_CACHE="${CACHE_ROOT}/hf-datasets"
export TORCH_HOME="${CACHE_ROOT}/torch"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"

mkdir -p \
  "${HF_HOME}" \
  "${TRANSFORMERS_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${TORCH_HOME}" \
  "${PIP_CACHE_DIR}"

# Activate the Conda environment on the login node if needed so conda commands are available.
if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV_NAME}" ]]; then
  if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    echo "Cannot find conda.sh under ${CONDA_BASE}. Set CONDA_BASE_OVERRIDE to the conda installation root." >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
fi

# Allow resource tweaks via environment variables.
SRUN_PARTITION="${SRUN_PARTITION:-core}"
SRUN_CPUS="${SRUN_CPUS:-16}"
SRUN_MEM="${SRUN_MEM:-20G}"
SRUN_TIME="${SRUN_TIME:-04:00:00}"

echo "Requesting compute node (partition=${SRUN_PARTITION}, cpus=${SRUN_CPUS}, mem=${SRUN_MEM}, time=${SRUN_TIME})..."

srun \
  --partition="${SRUN_PARTITION}" \
  --nodes=1 \
  --cpus-per-task="${SRUN_CPUS}" \
  --mem="${SRUN_MEM}" \
  --time="${SRUN_TIME}" \
  --pty bash -lc "
    source \"${CONDA_BASE}/etc/profile.d/conda.sh\"
    conda activate \"${CONDA_ENV_NAME}\"
    cd \"${SCRIPT_DIR}\"
    echo 'Conda environment '${CONDA_ENV_NAME}' activated on node:' \"\$(hostname)\"
    echo 'Run your inference, e.g.:'
    echo '  python scripts/news_summarize.py --limit 10 --dry-run'
    echo '  python scripts/news_summarize.py --country us --category technology --tweet'
    exec bash
  "
