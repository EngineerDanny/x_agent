#!/usr/bin/env python3
"""
Fetch Hacker News front-page stories, filter out entries that were already
posted, and print a short status line for the next unseen candidate.

Usage:
  python scripts/fetch_hn_top.py [--limit 5] [--score-min 300]

State is tracked in ~/.cache/hackernews_posted.json by default; override with
HN_POSTED_CACHE env var if you want per-project separation.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Iterable, List
import urllib.error
import urllib.request

API_URL = "https://hn.algolia.com/api/v1/search?tags=front_page&hitsPerPage={limit}"
DEFAULT_LIMIT = 10
DEFAULT_SCORE_MIN = 0
CACHE_ENV = "HN_POSTED_CACHE"
DEFAULT_CACHE = pathlib.Path("/projects/genomic-ml/da2343/x_agent/cache/hn_posted.json")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Hacker News top stories.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Number of stories to fetch (default: 10)")
    parser.add_argument(
        "--score-min",
        type=int,
        default=DEFAULT_SCORE_MIN,
        help="Minimum points required to consider a story (default: 0)",
    )
    parser.add_argument(
        "--cache",
        type=pathlib.Path,
        default=pathlib.Path(os.environ.get(CACHE_ENV, DEFAULT_CACHE)),
        help="Path to posted-id cache (default: ~/.cache/hackernews_posted.json)",
    )
    parser.add_argument(
        "--reset-cache",
        action="store_true",
        help="Ignore and clear the existing cache before running",
    )
    return parser.parse_args(argv)


def load_cache(cache_path: pathlib.Path, reset: bool) -> set[str]:
    if reset and cache_path.exists():
        cache_path.unlink()
    if not cache_path.exists():
        return set()
    try:
        with cache_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            return set(map(str, data))
    except (OSError, json.JSONDecodeError):
        return set()


def save_cache(cache_path: pathlib.Path, ids: Iterable[str]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as fh:
        json.dump(sorted(set(ids)), fh)


def fetch_front_page(limit: int) -> List[dict]:
    url = API_URL.format(limit=limit)
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.load(resp)
    hits = data.get("hits") or []
    return hits


def format_story(hit: dict) -> tuple[str, str]:
    title = hit.get("title") or hit.get("story_title") or "Untitled"
    url = hit.get("url") or hit.get("story_url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
    points = hit.get("points")
    author = hit.get("author")

    summary_parts = [title]
    if points is not None:
        summary_parts.append(f"({points} points)")
    if author:
        summary_parts.append(f"by {author}")

    summary = " — ".join(summary_parts)
    return summary, url


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    try:
        hits = fetch_front_page(args.limit)
    except urllib.error.URLError as exc:
        print(f"Error: failed to reach Hacker News API ({exc})", file=sys.stderr)
        return 1

    cached_ids = load_cache(args.cache, args.reset_cache)
    new_cache = set(cached_ids)

    for hit in hits:
        object_id = str(hit.get("objectID"))
        if args.score_min and (hit.get("points") or 0) < args.score_min:
            continue
        if object_id in cached_ids:
            continue

        summary, url = format_story(hit)
        print(summary)
        print(url)
        new_cache.add(object_id)
        save_cache(args.cache, new_cache)
        return 0

    print("No unseen stories met the criteria.", file=sys.stderr)
    save_cache(args.cache, new_cache)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
