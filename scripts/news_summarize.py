#!/usr/bin/env python3
"""End-to-end top-headline summarizer powered by NewsAPI.org.

1. Fetch the current top headlines from NewsAPI.org.
2. Pick the first unseen story that matches the provided filters.
3. Use the NewsAPI description as the article excerpt.
4. Ask the local GGUF model (via run_cpu_llm.py) for a ≤240-char summary.
5. Print the formatted result and update the cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import textwrap
from typing import Iterable, List

import requests
import tweepy
from dotenv import load_dotenv

NEWS_API_URL = "https://newsapi.org/v2/top-headlines"
NEWS_API_PAGE_SIZE_MAX = 100  # documented hard limit for pageSize
DEFAULT_LIMIT = 10
DEFAULT_COUNTRY = "us"
DEFAULT_CACHE = pathlib.Path("/projects/genomic-ml/da2343/x_agent/cache/news_seen.json")

NEWS_PROMPT_TEMPLATE = textwrap.dedent(
    """
    You are a JSON-only assistant. Return a single JSON object on one line with exactly the keys "summary" and "tweet".
    Format strictly as:
    {{"summary": "...", "tweet": "..."}}

    Requirements:
    - summary: one short paragraph (<=320 characters) recapping the article.
    - tweet: conversational (<=200 characters), include exactly one relevant hashtag or mention, no raw URLs.
    - Do NOT include any extra text, commentary, explanations, or Markdown outside the JSON braces.

    Headline: {title}
    URL: {url}

    Article excerpt:
    {excerpt}
    """
).strip()

DEFAULT_MODEL = "/projects/genomic-ml/da2343/x_agent/llm/llama.cpp/models/llama3_8b_instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
DEFAULT_THREADS = 16
SUMMARY_TOKENS = 160
ARTICLE_CHAR_LIMIT = 2000
TWEET_CACHE = pathlib.Path("/projects/genomic-ml/da2343/x_agent/cache/news_tweets.json")
MIN_SUMMARY_LENGTH = 10
MAX_SUMMARY_LENGTH = 320
MAX_TWEET_LENGTH = 200


def _parse_model_output(raw_text: str) -> tuple[str, str]:
    """Parse the LLM response, which should be a JSON object with summary/tweet."""
    raw_text = raw_text.strip()
    if not raw_text:
        raise ValueError("Empty model response")

    if not raw_text.lstrip().startswith("{"):
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and start < end:
            raw_text = raw_text[start : end + 1]

    try:
        data = json.loads(raw_text)
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object")
    except json.JSONDecodeError as exc:
        preview = raw_text[:400].replace("\n", "\\n")
        summary_match = re.search(r'"summary"\s*:\s*"(.*?)"', raw_text, flags=re.DOTALL)
        tweet_match = re.search(r'"tweet"\s*:\s*"(.*?)"', raw_text, flags=re.DOTALL)
        if summary_match and tweet_match:
            summary = summary_match.group(1).strip()
            tweet = tweet_match.group(1).strip()
            summary = summary.replace('\\"', '"')
            tweet = tweet.replace('\\"', '"')
            return summary, tweet
        raise ValueError(f"Model response was not valid JSON: {exc}; preview={preview!r}")

    summary = (data.get("summary") or "").strip()
    tweet = (data.get("tweet") or "").strip()

    if not summary:
        raise ValueError("JSON payload missing non-empty 'summary' field")
    if not tweet:
        raise ValueError("JSON payload missing non-empty 'tweet' field")

    return summary, tweet


def _load_tweeted(cache_path: pathlib.Path) -> set[str]:
    if cache_path.exists():
        try:
            return set(json.loads(cache_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            pass
    return set()


def _save_tweeted(cache_path: pathlib.Path, ids: set[str]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(sorted(ids)), encoding="utf-8")


def _fingerprint_summary(summary: str) -> str:
    return hashlib.sha256(summary.strip().encode("utf-8")).hexdigest()


def tweet_summary(tweet_text: str, seen: set[str]) -> set[str]:
    """Post tweet text to Twitter if we haven't published it yet."""
    tweet_text = tweet_text.strip()
    unique_id = _fingerprint_summary(tweet_text)
    if unique_id in seen:
        print("Summary already tweeted; skipping.", file=sys.stderr)
        return seen

    required_env = [
        "TWITTER_CONSUMER_KEY",
        "TWITTER_CONSUMER_SECRET",
        "TWITTER_ACCESS_TOKEN",
        "TWITTER_ACCESS_SECRET",
    ]
    missing = [env for env in required_env if env not in os.environ]
    if missing:
        raise RuntimeError(
            "Missing Twitter credentials; set environment variables: "
            + ", ".join(missing)
        )

    client = tweepy.Client(
        consumer_key=os.environ["TWITTER_CONSUMER_KEY"],
        consumer_secret=os.environ["TWITTER_CONSUMER_SECRET"],
        access_token=os.environ["TWITTER_ACCESS_TOKEN"],
        access_token_secret=os.environ["TWITTER_ACCESS_SECRET"],
    )

    try:
        response = client.create_tweet(text=tweet_text)
        tweet_id = response.data.get("id")
        if tweet_id:
            print(f"Tweet posted: https://twitter.com/i/web/status/{tweet_id}")
        else:
            print("Tweet posted.", file=sys.stderr)
        seen.add(unique_id)
        _save_tweeted(TWEET_CACHE, seen)
    except tweepy.TweepyException as exc:
        print(f"Tweet failed ({exc.__class__.__name__}): {exc}", file=sys.stderr)
    return seen


def parse_args(argv: List[str]) -> argparse.Namespace:
    load_dotenv(pathlib.Path(__file__).resolve().parent.parent / ".env")
    parser = argparse.ArgumentParser(description="Summarize top NewsAPI headlines with a local LLM")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Number of headlines to fetch (NewsAPI caps pageSize at 100; default: 10)",
    )
    parser.add_argument(
        "--country",
        type=str,
        default=DEFAULT_COUNTRY,
        help="2-letter country code for NewsAPI filtering (e.g., us, gb, ca)",
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=("sports", "technology"),
        default=None,
        help="Optional NewsAPI category filter (sports or technology)",
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
    parser.add_argument("--tweet", action="store_true", help="Post the summary to Twitter")
    return parser.parse_args(argv)


def load_cache(cache_path: pathlib.Path, reset: bool) -> set[str]:
    """Load cached article IDs from disk."""
    if reset and cache_path.exists():
        cache_path.unlink()
    if not cache_path.exists():
        return set()
    try:
        return set(json.loads(cache_path.read_text(encoding="utf-8")))
    except json.JSONDecodeError:
        return set()


def save_cache(cache_path: pathlib.Path, ids: Iterable[str]) -> None:
    """Persist cached article IDs."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(sorted(set(ids))), encoding="utf-8")


def fetch_top_headlines(
    api_key: str,
    limit: int,
    country: str | None = None,
    category: str | None = None,
) -> list[dict]:
    """Call NewsAPI's top-headlines endpoint and return article payloads."""
    if not api_key:
        raise RuntimeError("NEWS_API_KEY environment variable is required to call NewsAPI.org.")

    page_size = max(1, min(limit, NEWS_API_PAGE_SIZE_MAX))
    params: dict[str, str] = {"pageSize": str(page_size)}
    if country:
        params["country"] = country
    if category:
        params["category"] = category

    headers = {"X-Api-Key": api_key}
    response = requests.get(NEWS_API_URL, headers=headers, params=params, timeout=10)
    response.raise_for_status()
    payload = response.json()

    # NewsAPI signals business errors via status != "ok" even when HTTP status is 200.
    if payload.get("status") != "ok":
        message = payload.get("message", "Unknown NewsAPI error")
        code = payload.get("code", "unknown_error")
        raise RuntimeError(f"NewsAPI request failed ({code}): {message}")

    return payload.get("articles") or []


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
    api_key = os.environ.get("NEWS_API_KEY")
    try:
        articles = fetch_top_headlines(
            api_key=api_key,
            limit=args.limit,
            country=args.country,
            category=args.category,
        )
    except (requests.RequestException, RuntimeError) as exc:
        print(f"Error: failed to retrieve headlines ({exc})", file=sys.stderr)
        return 1

    cached_ids = load_cache(args.cache, args.reset_cache)
    seen_ids = set(cached_ids)
    tweeted_ids = _load_tweeted(TWEET_CACHE)

    for article in articles:
        url = article.get("url")
        title = article.get("title") or "Untitled"
        unique_key = url or hashlib.sha256(title.encode("utf-8")).hexdigest()

        if unique_key in seen_ids:
            continue

        source_name = ((article.get("source") or {}).get("name")) or ""
        published_at = article.get("publishedAt") or ""
        summary_parts = [title]
        if source_name:
            summary_parts.append(f"via {source_name}")
        if published_at:
            summary_parts.append(published_at)
        summary_line = " — ".join(part for part in summary_parts if part)
        print(summary_line)

        if args.dry_run:
            seen_ids.add(unique_key)
            save_cache(args.cache, seen_ids)
            return 0

        if not url:
            print("Skipping: article lacks a URL to summarize.", file=sys.stderr)
            seen_ids.add(unique_key)
            continue

        article_text = article.get("description") or ""
        if not article_text:
            print("Skipping: article lacks a usable description.", file=sys.stderr)
            seen_ids.add(unique_key)
            continue
        article_text = textwrap.shorten(article_text.strip(), width=ARTICLE_CHAR_LIMIT, placeholder="")

        prompt = NEWS_PROMPT_TEMPLATE.format(
            title=title,
            url=url,
            excerpt=article_text,
        )
        raw_summary = summarize_with_llm(args.model, args.threads, prompt)
        try:
            summary, tweet_text = _parse_model_output(raw_summary)
        except ValueError as exc:
            print(f"Skipping: model output invalid ({exc}).", file=sys.stderr)
            seen_ids.add(unique_key)
            continue

        if len(summary) < MIN_SUMMARY_LENGTH:
            print("Skipping: summary is too short to be informative.", file=sys.stderr)
            seen_ids.add(unique_key)
            continue

        if len(summary) > MAX_SUMMARY_LENGTH:
            summary = summary[:MAX_SUMMARY_LENGTH].rstrip()

        if len(tweet_text) > MAX_TWEET_LENGTH:
            print(
                f"Skipping: tweet text exceeds {MAX_TWEET_LENGTH} characters (got {len(tweet_text)}).",
                file=sys.stderr,
            )
            seen_ids.add(unique_key)
            continue

        tweet_hash = _fingerprint_summary(tweet_text)
        if tweet_hash in tweeted_ids:
            print("Skipping: summary text already used in a previous tweet.", file=sys.stderr)
            seen_ids.add(unique_key)
            continue

        print("\nSummary:")
        print(summary)
        print("\nTweet preview:")
        print(tweet_text)
        if args.tweet:
            tweeted_ids = tweet_summary(tweet_text, tweeted_ids)
        elif not args.dry_run:
            tweeted_ids.add(tweet_hash)
            _save_tweeted(TWEET_CACHE, tweeted_ids)

        seen_ids.add(unique_key)
        save_cache(args.cache, seen_ids)
        return 0

    save_cache(args.cache, seen_ids)
    print("No unseen stories produced a usable summary.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
