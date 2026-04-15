#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.compare import run_comparison, to_markdown
from src.search import HierarchicalSemanticSearcher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HS Code semantic classifier (chapters 07-11)")
    parser.add_argument("--query", type=str, help="Natural language product description")
    parser.add_argument("--compare", action="store_true", help="Run semantic vs TF-IDF comparison report")
    parser.add_argument("--hs6-csv", type=Path, default=Path("data/hs6_global_chapters_07_11.csv"))
    parser.add_argument("--country-csv", type=Path, default=Path("data/hs_country_us_chapters_07_11.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--write-report", type=Path, default=Path("comparison_report.md"))
    return parser


def run_query(args: argparse.Namespace) -> None:
    searcher = HierarchicalSemanticSearcher.from_csv(
        hs6_csv=args.hs6_csv,
        country_csv=args.country_csv,
        cache_dir=args.cache_dir,
    )
    result = searcher.search(args.query)

    print(f'Query: "{result.query}"')
    print("\nHS-6 Match:")
    print(f"  Code:        {result.hs6.code}")
    print(f"  Description: {result.hs6.description}")
    print(f"  Score:       {result.hs6.score:.3f}")
    print(f"  Match %:     {result.match_percent:.1f}%")
    print(f"  Confidence:  {result.confidence}")

    print("\nCountry Code Match:")
    if result.country is None:
        print("  No country-specific child code found under matched HS-6")
    else:
        print(f"  Code:        {result.country.code}")
        print(f"  Description: {result.country.description}")
        print(f"  Score:       {result.country.score:.3f}")

    print("\nWhy this match:")
    print(f"  {result.explanation}")

    if result.notes:
        print("\nScope Notes:")
        for note in result.notes:
            print(f"  - {note}")

    if result.top_hs6:
        print("\nTop HS-6 Candidates:")
        for cand in result.top_hs6[:3]:
            print(f"  - {cand.code} | {cand.description} | score={cand.score:.3f}")

    if result.top_country:
        print("\nTop Country Candidates:")
        for cand in result.top_country[:3]:
            print(f"  - {cand.code} | {cand.description} | score={cand.score:.3f}")


def run_compare(args: argparse.Namespace) -> None:
    rows = run_comparison(
        hs6_csv=args.hs6_csv,
        country_csv=args.country_csv,
        cache_dir=args.cache_dir,
    )
    report = to_markdown(rows)
    args.write_report.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nSaved comparison report to {args.write_report}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.compare:
        run_compare(args)
        return

    if not args.query:
        parser.error("Provide --query for single search, or use --compare")

    run_query(args)


if __name__ == "__main__":
    main()
