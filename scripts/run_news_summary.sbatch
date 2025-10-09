#!/bin/bash
#SBATCH --job-name=news-summary
#SBATCH --partition=core
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --output=/projects/genomic-ml/da2343/x_agent/logs/news_%j.out
#SBATCH --error=/projects/genomic-ml/da2343/x_agent/logs/news_%j.err

set -euo pipefail

SCRIPT_DIR="/projects/genomic-ml/da2343/x_agent"
cd "${SCRIPT_DIR}"

# activate conda env
source /packages/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate llama-cpu

mkdir -p logs

./llama-cpu/bin/python scripts/news_summarize.py \
  --limit 50 \
  --country us \
  --category technology \
  --tweet
