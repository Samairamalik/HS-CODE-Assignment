from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .baseline import TfidfHierarchicalBaseline
from .data_loader import load_dataset
from .search import HierarchicalSemanticSearcher


@dataclass
class ComparisonRow:
    query: str
    expected_hs6: str
    semantic_hs6: str
    tfidf_hs6: str
    semantic_country: str
    tfidf_country: str
    semantic_better: bool


DEFAULT_QUERIES = [
    ("Fresh tomatoes", "070200"),
    ("Garbanzo beans", "071320"),
    ("Dried chickpeas in bulk bags", "071320"),
    ("Seed potatoes for planting", "070110"),
    ("Aubergine slices", "070930"),
    ("Frozen green peas", "071022"),
    ("Fresh vine tomatoes", "070200"),
    ("Ground coffee", "090121"),
]


def run_comparison(
    hs6_csv: Path,
    country_csv: Path,
    cache_dir: Path,
    query_pairs: Iterable[tuple[str, str]] = DEFAULT_QUERIES,
) -> list[ComparisonRow]:
    semantic = HierarchicalSemanticSearcher.from_csv(hs6_csv, country_csv, cache_dir)
    dataset = load_dataset(hs6_csv, country_csv)
    tfidf = TfidfHierarchicalBaseline(dataset)

    rows: list[ComparisonRow] = []
    for query, expected in query_pairs:
        s = semantic.search(query)
        b = tfidf.search(query)

        s_country = s.country.code if s.country else ""
        b_country = b.country.code if b.country else ""

        semantic_ok = s.hs6.code == expected
        baseline_ok = b.hs6.code == expected
        semantic_better = semantic_ok and not baseline_ok

        rows.append(
            ComparisonRow(
                query=query,
                expected_hs6=expected,
                semantic_hs6=s.hs6.code,
                tfidf_hs6=b.hs6.code,
                semantic_country=s_country,
                tfidf_country=b_country,
                semantic_better=semantic_better,
            )
        )

    return rows


def to_markdown(rows: list[ComparisonRow]) -> str:
    lines = []
    lines.append("| Query | Expected HS-6 | Semantic HS-6 | TF-IDF HS-6 | Semantic Country | TF-IDF Country | Semantic Better? |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in rows:
        lines.append(
            f"| {row.query} | {row.expected_hs6} | {row.semantic_hs6} | {row.tfidf_hs6} | {row.semantic_country} | {row.tfidf_country} | {'Yes' if row.semantic_better else 'No'} |"
        )

    wins = sum(1 for r in rows if r.semantic_better)
    lines.append("")
    lines.append(f"Semantic wins: {wins}/{len(rows)}")
    return "\n".join(lines)
