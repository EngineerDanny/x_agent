#!/bin/bash
set -euo pipefail
SCRIPT_DIR="/projects/genomic-ml/da2343/x_agent"
sbatch "${SCRIPT_DIR}/scripts/run_news_summary.sbatch"
