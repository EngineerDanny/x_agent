#!/usr/bin/env python3
"""NewsAPI helper: fetch top headlines and emit the next unseen article."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
from typing import Iterable, List

import requests

NEWS_API_URL = "https://newsapi.org/v2/top-headlines"
NEWS_API_PAGE_SIZE_MAX = 100
DEFAULT_LIMIT = 10
DEFAULT_COUNTRY = "us"
CACHE_ENV = "NEWS_POSTED_CACHE"
DEFAULT_CACHE = pathlib.Path("/projects/genomic-ml/da2343/x_agent/cache/news_posted.json")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch NewsAPI.org top headlines.")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Number of articles to request (NewsAPI caps pageSize at 100; default: 10)",
    )
    parser.add_argument(
        "--country",
        type=str,
        default=DEFAULT_COUNTRY,
        help="Optional 2-letter country code (omit when specifying --sources)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Optional NewsAPI category filter (business, entertainment, general, health, science, sports, technology)",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default=None,
        help="Comma-separated NewsAPI source identifiers (mutually exclusive with --country/--category)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Optional keyword filter applied via the 'q' parameter",
    )
    parser.add_argument(
        "--cache",
        type=pathlib.Path,
        default=pathlib.Path(os.environ.get(CACHE_ENV, DEFAULT_CACHE)),
        help="Path to posted-article cache",
    )
    parser.add_argument("--reset-cache", action="store_true", help="Clear cached article IDs before running")
    return parser.parse_args(argv)


def load_cache(cache_path: pathlib.Path, reset: bool) -> set[str]:
    if reset and cache_path.exists():
        cache_path.unlink()
    if not cache_path.exists():
        return set()
    try:
        return set(json.loads(cache_path.read_text(encoding="utf-8")))
    except json.JSONDecodeError:
        return set()


def save_cache(cache_path: pathlib.Path, ids: Iterable[str]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(sorted(set(ids))), encoding="utf-8")


def fetch_headlines(
    api_key: str,
    limit: int,
    country: str | None,
    category: str | None,
    sources: str | None,
    query: str | None,
) -> list[dict]:
    if not api_key:
        raise RuntimeError("Missing NEWS_API_KEY environment variable.")

    if sources and (country or category):
        raise ValueError("NewsAPI prohibits combining sources with country or category filters.")

    page_size = max(1, min(limit, NEWS_API_PAGE_SIZE_MAX))
    params: dict[str, str] = {"pageSize": str(page_size)}
    if sources:
        params["sources"] = sources
    else:
        if country:
            params["country"] = country
        if category:
            params["category"] = category
    if query:
        params["q"] = query

    headers = {"X-Api-Key": api_key}
    response = requests.get(NEWS_API_URL, headers=headers, params=params, timeout=10)
    response.raise_for_status()
    payload = response.json()

    if payload.get("status") != "ok":
        code = payload.get("code", "unknown_error")
        message = payload.get("message", "Unknown NewsAPI error")
        raise RuntimeError(f"NewsAPI request failed ({code}): {message}")

    return payload.get("articles") or []


def format_article(article: dict) -> tuple[str, str]:
    title = article.get("title") or "Untitled"
    source_name = ((article.get("source") or {}).get("name")) or ""
    published_at = article.get("publishedAt") or ""
    parts = [title]
    if source_name:
        parts.append(f"via {source_name}")
    if published_at:
        parts.append(published_at)
    return " — ".join(part for part in parts if part), article.get("url") or ""


def select_article(articles: List[dict], cached_ids: set[str]) -> tuple[dict | None, set[str]]:
    updated_cache = set(cached_ids)
    for article in articles:
        url = article.get("url")
        unique_id = url or hashlib.sha256((article.get("title") or "").encode("utf-8")).hexdigest()
        if unique_id in cached_ids:
            continue
        updated_cache.add(unique_id)
        return article, updated_cache
    return None, updated_cache


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    api_key = os.environ.get("NEWS_API_KEY")
    try:
        articles = fetch_headlines(
            api_key=api_key,
            limit=args.limit,
            country=args.country,
            category=args.category,
            sources=args.sources,
            query=args.query,
        )
    except (requests.RequestException, RuntimeError, ValueError) as exc:
        print(f"Error: failed to retrieve headlines ({exc})", file=sys.stderr)
        return 1

    cached_ids = load_cache(args.cache, args.reset_cache)
    article, updated_cache = select_article(articles, cached_ids)
    save_cache(args.cache, updated_cache)

    if article is None:
        print("No unseen articles met the criteria.", file=sys.stderr)
        return 1

    summary, url = format_article(article)
    print(summary)
    print(url or "(No URL provided by NewsAPI response)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
