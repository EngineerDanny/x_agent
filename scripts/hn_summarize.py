#!/usr/bin/env python3
"""End-to-end Hacker News summarizer.

1. Fetch the current front-page hits via Algolia.
2. Pick the first unseen story above a score threshold.
3. Fetch the linked article and extract readable text (using trafilatura if available).
4. Ask the local GGUF model (via run_cpu_llm.py) for a ≤240-char summary.
5. Print the formatted result and update the cache.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import textwrap
import tempfile
from typing import List

import requests
from html import unescape
from html.parser import HTMLParser
import re

from fetch_hn_top import (
    DEFAULT_CACHE,
    DEFAULT_LIMIT,
    DEFAULT_SCORE_MIN,
    fetch_front_page,
    format_story,
    load_cache,
    save_cache,
    select_story,
)

HN_PROMPT_TEMPLATE = textwrap.dedent(
    """
    You are a concise editor. Using the headline and article extract below, write a single-sentence social post of at most 240 characters. Keep it informative, neutral, and do not echo these instructions. Reply with the sentence only.

    Headline: {title}
    URL: {url}

    Article excerpt:
    {excerpt}
    """
).strip()

DEFAULT_MODEL = "/projects/genomic-ml/da2343/x_agent/llm/llama.cpp/models/deepseek-r1-llama8b-unsloth/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"
DEFAULT_THREADS = 16
SUMMARY_TOKENS = 160
ARTICLE_CHAR_LIMIT = 2000
SUMMARY_MAX_CHARS = 240


def _select_summary_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("summary:") or stripped.lower().startswith("body:"):
            continue
        return stripped
    return text.strip()


def _truncate_summary(text: str) -> str:
    if len(text) <= SUMMARY_MAX_CHARS:
        return text
    truncated = text[: SUMMARY_MAX_CHARS - 1]
    truncated = truncated.rsplit(" ", 1)[0]
    return truncated + "…"


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize top Hacker News articles with local LLM")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Number of HN hits to fetch (default: 10)")
    parser.add_argument(
        "--score-min",
        type=int,
        default=DEFAULT_SCORE_MIN,
        help="Minimum HN score to consider (default: 0)",
    )
    parser.add_argument(
        "--cache",
        type=pathlib.Path,
        default=DEFAULT_CACHE,
        help="Cache file for seen story IDs",
    )
    parser.add_argument(
        "--model",
        type=pathlib.Path,
        default=pathlib.Path(DEFAULT_MODEL),
        help="Path to GGUF model for summarization",
    )
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS, help="LLM threads (default: 16)")
    parser.add_argument("--reset-cache", action="store_true", help="Clear cache before processing")
    parser.add_argument("--dry-run", action="store_true", help="Skip summarization; just show selected story")
    return parser.parse_args(argv)


class _MetaDescriptionParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.descriptions: list[str] = []

    def handle_starttag(self, tag: str, attrs):
        if tag.lower() != "meta":
            return
        attr_dict = {k.lower(): v for k, v in attrs}
        name = attr_dict.get("name") or attr_dict.get("property")
        if name and name.lower() in {"description", "og:description", "twitter:description"}:
            content = attr_dict.get("content")
            if content:
                self.descriptions.append(content.strip())


def _fallback_meta_description(html: str) -> str:
    parser = _MetaDescriptionParser()
    try:
        parser.feed(html)
    except Exception:
        return ""
    for desc in parser.descriptions:
        if desc:
            return desc
    return ""


def extract_article(url: str) -> str:
    """Return readable text from the article URL."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; HackerNewsBot/1.0)"}
    try:
        response = requests.get(url, timeout=15, headers=headers)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Warning: failed to fetch article ({exc})", file=sys.stderr)
        return ""

    html = response.text
    try:
        import trafilatura

        text = trafilatura.extract(html, url=url) or ""
    except Exception as exc:  # broad: trafilatura optional
        print(f"Warning: trafilatura extract failed ({exc}); falling back to raw snippet", file=sys.stderr)
        text = ""

    if not text:
        meta_desc = _fallback_meta_description(html)
        if meta_desc:
            return meta_desc[:ARTICLE_CHAR_LIMIT]
        paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", html, flags=re.IGNORECASE | re.DOTALL)
        cleaned: list[str] = []
        for para in paragraphs[:5]:
            text = re.sub(r"<[^>]+>", "", para)
            text = unescape(" ".join(text.split()))
            if text:
                cleaned.append(text)
        if cleaned:
            return " ".join(cleaned)[:ARTICLE_CHAR_LIMIT]
        snippet = " ".join(html.split())
        return unescape(snippet[:ARTICLE_CHAR_LIMIT])

    return text[:ARTICLE_CHAR_LIMIT]


def summarize_with_llm(model: pathlib.Path, threads: int, prompt: str) -> str:
    """Call run_cpu_llm.py and capture its output."""
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
        tmp.write(prompt)
        tmp_path = tmp.name

    try:
        cmd = [
            sys.executable,
            "/projects/genomic-ml/da2343/x_agent/run_cpu_llm.py",
            "--model",
            str(model),
            "--threads",
            str(threads),
            "--prompt-file",
            tmp_path,
            "--n-predict",
            str(SUMMARY_TOKENS),
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    finally:
        os.unlink(tmp_path)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    hits = fetch_front_page(args.limit)
    cached_ids = load_cache(args.cache, args.reset_cache)
    hit, updated_cache = select_story(hits, cached_ids, args.score_min)
    if hit is None:
        print("No unseen stories met the criteria.", file=sys.stderr)
        return 1

    save_cache(args.cache, updated_cache)

    summary_line, url = format_story(hit)
    print(summary_line)
    print(url)

    if args.dry_run:
        return 0

    article_text = extract_article(url)
    if not article_text:
        print("(No article text extracted; skipping summarization)", file=sys.stderr)
        return 1

    prompt = HN_PROMPT_TEMPLATE.format(
        title=hit.get("title") or hit.get("story_title") or "Untitled",
        url=url,
        excerpt=article_text,
    )
    raw_summary = summarize_with_llm(args.model, args.threads, prompt)
    summary = _truncate_summary(_select_summary_line(raw_summary))
    print("\nSummary:")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
