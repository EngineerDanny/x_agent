# News Summarizer (CPU-only GGUF)

Automates twice-daily summaries of top headlines from NewsAPI.org using a local GGUF model (CPU inference via `llama.cpp`). Each run picks the first unseen article, generates a social-style blurb, and (optionally) tweets the result. The repository includes helpers for manual testing, Slurm job submission, and cron-based scheduling.

## Features

- **CPU-only inference**: wraps `run_cpu_llm.py` around `llama.cpp` binaries built in the project (`llm/llama.cpp`).
- **Headline sourcing**: calls NewsAPI.org (`/v2/top-headlines`) with configurable `--limit`, `--country`, and `--category` filters (currently limited to `sports` or `technology`).
- **JSON prompt / fallback**: prompts the local model for `{summary, tweet}` JSON; if the model drifts, the script retries and falls back to deterministic text so every article yields output.
- **Tweet de-duplication**: hashes tweet text to avoid reposts; caching lives in `cache/news_tweets.json`.
- **Automation**: Slurm batch script (`scripts/run_news_summary.sbatch`) + helper wrapper (`scripts/submit_news_summary.sh`) + cron entry to fire at 09:05 and 17:05 daily.
- **Debug-friendly**: `tweet_summary` currently prints the tweet and skips posting—remove the `return False` when ready to go live.

## Requirements

- Access to a NewsAPI.org API key (`NEWS_API_KEY` stored in `.env`).
- CPU node with ≥16 cores, 20 GB RAM (per Slurm settings in `run_news_summary.sbatch`).
- Python environment `llama-cpu` containing project dependencies (`requests`, `tweepy`, `trafilatura`, `python-dotenv`).
- GGUF checkpoint (defaults to **Meta Llama 3.1 8B Instruct Q4_K_M** unless you swap the path in `scripts/news_summarize.py`).
- `llama.cpp` built under `llm/llama.cpp` with the `llama-cli` binary in `build/bin/`.

## Setup

1. **Clone / copy the repo** onto the cluster under `<repo-root>` (paths are baked into scripts; adjust if you relocate).
2. **Activate environment**:
   ```bash
   cd <repo-root>
   source <anaconda-root>/etc/profile.d/conda.sh
   conda activate llama-cpu
   ```
3. **Install dependencies** (if missing):
   ```bash
   pip install requests tweepy trafilatura python-dotenv
   ```
4. **Build llama.cpp** (if not already compiled):
   ```bash
   cd llm/llama.cpp
   cmake -B build -DLLAMA_BUILD_EXAMPLES=ON
   cmake --build build -j16
   ```
   The repo uses `llama-cli` from `build/bin/`.
5. **Download a GGUF model** (default: Llama 3.1 8B Instruct Q4_K_M):
   ```bash
   mkdir -p <repo-root>/llm/llama.cpp/models/llama3_8b_instruct
   huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
     Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
     --local-dir <repo-root>/llm/llama.cpp/models/llama3_8b_instruct
   ```
   Update `DEFAULT_MODEL` in `scripts/news_summarize.py` if you switch checkpoints (e.g., Qwen 2.5 14B later).
6. **Configure environment variables**: create or edit `.env` with at least:
   ```ini
   NEWS_API_KEY=YOUR_KEY
   TWITTER_CONSUMER_KEY=...
   TWITTER_CONSUMER_SECRET=...
   TWITTER_ACCESS_TOKEN=...
   TWITTER_ACCESS_SECRET=...
   ```
   (The Twitter keys are optional during testing while `tweet_summary` returns early.)

## Running Manually

```bash
cd <repo-root>
<repo-root>/llama-cpu/bin/python scripts/news_summarize.py \
  --limit 50 --country us --category technology --dry-run
```

- `--dry-run` prints the selected article/summary but skips tweets.
- Remove `--dry-run` and the debug `return False` in `tweet_summary` to post.
- Use `--tweet` to exercise the tweet path (still printing due to debug stub).

## Automation (Slurm + Cron)

1. **One-off job**:
   ```bash
   ./scripts/submit_news_summary.sh
   squeue -u "$USER"   # optional monitor
   tail -n 40 <repo-root>/logs/news_<jobid>.out
   ```
   Logs land in `logs/news_<jobid>.out|err`. The tweet cache at `cache/news_tweets.json` prevents duplicates.

2. **Schedule twice daily**:
   - Edit crontab (`EDITOR=nano crontab -e`) and add:
     ```
     5 9,17 * * * <repo-root>/scripts/submit_news_summary.sh \
       ><repo-root>/logs/cron_submit.log 2>&1
     ```
   - Cron runs on the login node and invokes the Slurm job at 09:05 and 17:05.
   - Review `logs/cron_submit.log` plus the usual job logs for output.

## Customizing

- **Article filters**: adjust `--limit`, `--country`, `--category` on the CLI or edit defaults inside `scripts/news_summarize.py`.
- **Model prompt**: `NEWS_PROMPT_TEMPLATE` and `NEWS_PROMPT_RETRY_TEMPLATE` define the JSON instructions and retry message.
- **Fallback behavior**: After two failed JSON attempts the script synthesizes summary/tweet text from the article excerpt/title (ensures length and hashtag).
- **Tweet posting**: Remove the debug `return False` block in `tweet_summary` to actually call Tweepy’s `create_tweet`.
- **Logging**: tweak output paths in `scripts/run_news_summary.sbatch` (`<repo-root>/logs/news_%j.out|err`).

## Debugging Tips

- **NewsAPI errors**: any HTTP failure or invalid JSON is surfaced to stderr and logged; typically indicates a quota or credential issue.
- **Model drift**: if the model keeps ignoring JSON prompts, consider upgrading to a larger instruction-tuned checkpoint or reinstalling the grammar forcing later.
- **Tweet duplicates**: hashes live in `cache/news_tweets.json`; delete or edit to reset posting history.
- **Cron not firing**: check `crontab -l`, review `logs/cron_submit.log`, and ensure the login node can submit Slurm jobs.

## Directory Overview

```
x_agent/
├── scripts/
│   ├── news_summarize.py        # main summarizer
│   ├── fetch_news_top.py        # fetch-only helper (optional)
│   ├── run_news_summary.sbatch  # Slurm batch script
│   └── submit_news_summary.sh   # wrapper invoked by cron
├── llm/llama.cpp/               # llama.cpp checkout + build
├── llama-cpu/                   # conda environment
├── cache/news_tweets.json       # tweet hash cache
├── logs/                        # Slurm + cron logs
├── run_cpu_llm.py               # llama.cpp wrapper
├── launch_cpu_session.sh        # helper to request interactive node
├── README.md                    # this file
└── .env                         # NewsAPI + Twitter credentials
```

## License

Specify your project license here (e.g., MIT, Apache-2.0). Update this section before pushing to GitHub.
