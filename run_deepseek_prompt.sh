#!/usr/bin/env bash
set -euo pipefail

MODEL="/projects/genomic-ml/da2343/x_agent/llm/llama.cpp/models/deepseek-r1-llama8b-unsloth/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"
PROMPT_FILE="/projects/genomic-ml/da2343/x_agent/prompts/boiling_water.json"
THREADS="${THREADS:-16}"
PREDICT="${PREDICT:-200}"

python /projects/genomic-ml/da2343/x_agent/run_cpu_llm.py \
  --model "$MODEL" \
  --threads "$THREADS" \
  --prompt-file "$PROMPT_FILE" \
  --n-predict "$PREDICT" "$@"
