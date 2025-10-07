# CPU-Only LLM Quickstart

This guide documents how to stand up a local large-language-model inference stack on the cluster **without** requesting GPUs. It targets the `core` partition CPU nodes exposed through Slurm (`srun`, `sbatch`) such as the 64-core, 500 GB RAM machines (e.g., `cn3`, `cn4`, …).

## 1. Request an interactive CPU node

Use Slurm to land on an idle CPU node. Adjust time, memory, or cores as needed for your workload.

```bash
srun \
  --partition=core \
  --nodes=1 \
  --cpus-per-task=32 \
  --mem=180G \
  --time=04:00:00 \
  --pty bash
```

Tips:
- `--cpus-per-task` controls the threading budget for inference; 24–48 threads work well for 7B–13B models.
- Increase `--mem` if you plan to load larger quantized checkpoints (≈80 GB for 34B Q4_K).
- For batch runs, convert this to an `sbatch` script (see §5).

## 2. Prepare a software environment

Inside the interactive shell, ensure compilers and Python tooling are available. Example using Conda:

```bash
module load gcc/12.2 || true  # optional; ignore failures if module tree differs
source /packages/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate llama-cpu || (conda create -n llama-cpu -y python=3.11 && conda activate llama-cpu)
conda install -y cmake ninja git huggingface_hub
```

The environment now resides at `/projects/genomic-ml/da2343/x_agent/llama-cpu` (a compatibility symlink remains at `/projects/genomic-ml/da2343/llama-cpu`, so `conda activate llama-cpu` continues to work).

## 3. Build `llama.cpp`

`llama.cpp` delivers highly optimized CPU inference and supports GGUF quantized checkpoints.

```bash
mkdir -p /projects/genomic-ml/da2343/x_agent/llm
cd /projects/genomic-ml/da2343/x_agent/llm
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -G Ninja -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build -j $SLURM_CPUS_PER_TASK
```

Key binaries (including `llama-cli`) sit in `build/bin/`.

## 4. Download a quantized model

Pick a GGUF model that fits CPU inference. Popular options:
- `TheBloke/Llama-3-8B-Instruct-GGUF` (`Q4_K_M` ≈4 GB RAM, good chat quality)
- `TheBloke/Mistral-7B-Instruct-v0.2-GGUF`
- `bartowski/Qwen2-7B-Instruct-GGUF`

Download with the Hugging Face CLI (requires a token for gated repos):

```bash
hf download TheBloke/Mistral-7B-Instruct-v0.2-GGUF \
  --local-dir models/mistral-7b-instruct \
  --include "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
```

If the CLI cannot reach the repo, fall back to `wget` using the Hugging Face `resolve/main/<filename>` URL. Keep checkpoints under `llama.cpp/models/` so paths match the commands below.

## 5. Run inference

```bash
cd /projects/genomic-ml/da2343/x_agent/llm/llama.cpp
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
build/bin/llama-cli \
  --model models/mistral-7b-instruct/mistral-7b-instruct-v0.2.Q4_K_M.gguf \
  --threads $SLURM_CPUS_PER_TASK \
  --batch-size 64 \
  --n-predict 256 \
  --no-conversation \
  --prompt "You are a helpful assistant. Who are you?"
```

Key flags:
- `--threads`: total CPU threads to use (≤ `cpus-per-task`).
- `--batch-size`: increase (128–256) for longer prompts and better throughput.
- `--ctx-size`: raise (e.g., 4096) for longer context windows if RAM allows.
- `--no-conversation`: bypass chat templates when providing a raw prompt.

## 6. Automate with Slurm (`sbatch`)

Create `run_llm_cpu.sbatch`:

```bash
#!/bin/bash
#SBATCH --partition=core
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --job-name=llama_cpu

module load gcc/12.2 || true
source /packages/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate llama-cpu
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /projects/genomic-ml/da2343/x_agent/llm/llama.cpp
build/bin/llama-cli \
  --model models/mistral-7b-instruct/mistral-7b-instruct-v0.2.Q4_K_M.gguf \
  --threads $SLURM_CPUS_PER_TASK \
  --prompt-file prompts/input.txt \
  --n-predict 512 \
  --ctx-size 4096 \
  --no-conversation \
  --log-disable
```

Submit with:

```bash
sbatch run_llm_cpu.sbatch
```

Monitor progress via `squeue -u $USER` and review output in `slurm-<jobid>.out`.

## 7. Troubleshooting & optimization

- **OOM / abort**: increase `--mem` or use a smaller quantization tier (`Q4_0` < `Q5_0` < `Q8_0`).
- **Slow tokens**: verify `OMP_NUM_THREADS` matches the Slurm allocation; consider `--batch-size 128` and `--threads` equal to physical cores (not hyperthreads).
- **Tokenizer mismatch**: ensure you downloaded the matching tokenizer files (`tokenizer.model`, `tokenizer.json`) when using other frontends.
- **Persisted cache**: llama.cpp supports `--mmproj` and `--prompt-cache`. Store caches on node-local SSD (`/tmp`) for repeated prompts.

## 8. Alternatives

- `text-generation-inference --model-id <gguf-dir> --max-input-length 4096 --max-total-tokens 6000 --num-shard 1 --device cpu`
- `exllamaV2` CPU path for 4-bit quantized LLaMA-family models (experimental).
- `ollama serve --models /path/to/models` for a managed API wrapper if you prefer HTTP endpoints (install via static binary).

With these steps you can provision, run, and automate CPU-only inference for quantized LLMs entirely within the cluster, no GPUs required.

## 9. Convenience helper (`run_cpu_llm.py`)

After activating the `llama-cpu` environment you can launch a quick test without wrestling with the full CLI:

```bash
python run_cpu_llm.py --prompt "Say hello in one sentence." --n-predict 32
```

By default the helper wraps `build/bin/llama-cli` to avoid hardware-specific wheels that can crash with `Illegal instruction`. If you omit `--n-predict`, the script now lets the model run up to the (default) context window, which is typically enough for “long” answers without hanging. Key options:
- `--threads`, `--batch-size`, `--ctx-size`, `--model` mirror the CLI flags.
- `--prompt-file` reads a UTF-8 file instead of inline text.
- `--use-python` opts into the Python `llama_cpp` bindings (requires `pip install llama-cpp-python==0.2.90`).
