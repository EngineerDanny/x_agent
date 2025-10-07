#!/usr/bin/env bash
# Helper to land on a CPU compute node with the llama-cpu Conda env active.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV_NAME="llama-cpu"
CONDA_BASE="${CONDA_BASE_OVERRIDE:-/packages/anaconda3/2024.02}"

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
SRUN_CPUS="${SRUN_CPUS:-32}"
SRUN_MEM="${SRUN_MEM:-180G}"
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
    echo '  python run_cpu_llm.py --prompt \"Say hello in one sentence.\" --n-predict 32'
    exec bash
  "
