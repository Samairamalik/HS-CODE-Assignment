#!/usr/bin/env python3
"""Convert HS PDF files into CSV datasets for chapters 07-11."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
HS_DIR = ROOT / "HS Code"
OUT_DIR = ROOT / "data"

CHAPTER_MIN = 7
CHAPTER_MAX = 11


@dataclass
class HS6Row:
    chapter: str
    hs6_code: str
    description: str
    source_file: str


@dataclass
class CountryRow:
    chapter: str
    full_code: str
    normalized_code: str
    hs6_prefix: str
    description: str
    source_file: str


def _extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        pages.append((page.extract_text() or "").replace("\x00", ""))
    return "\n".join(pages)


def _normalize_spaces(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .:-\t")


def _chapter_ok(ch: str) -> bool:
    try:
        value = int(ch)
    except ValueError:
        return False
    return CHAPTER_MIN <= value <= CHAPTER_MAX


def parse_hs6_pdf(pdf_path: Path) -> list[HS6Row]:
    text = _extract_text(pdf_path)
    rows: list[HS6Row] = []

    # Pattern 1: heading + full hs6 on same line, e.g. "07.02 0702.00 Tomatoes..."
    p_direct = re.compile(r"\b(\d{2})\.(\d{2})\s+(\d{4})\.(\d{2})\s+([^\n]+)")
    # Pattern 2: indented hs6 bullet, e.g. "0701.10 - Seed"
    p_bullet = re.compile(r"(?m)^\s*(\d{4})\.(\d{2})\s*-\s*([^\n]+)")

    for m in p_direct.finditer(text):
        chapter = m.group(1)
        if not _chapter_ok(chapter):
            continue
        hs6 = f"{m.group(3)}{m.group(4)}"
        desc = _normalize_spaces(m.group(5))
        if desc:
            rows.append(HS6Row(chapter, hs6, desc, pdf_path.name))

    for m in p_bullet.finditer(text):
        chapter = m.group(1)[:2]
        if not _chapter_ok(chapter):
            continue
        hs6 = f"{m.group(1)}{m.group(2)}"
        desc = _normalize_spaces(m.group(3))
        if desc:
            rows.append(HS6Row(chapter, hs6, desc, pdf_path.name))

    dedup: dict[str, HS6Row] = {}
    for row in rows:
        current = dedup.get(row.hs6_code)
        if current is None or len(row.description) > len(current.description):
            dedup[row.hs6_code] = row

    return sorted(dedup.values(), key=lambda r: r.hs6_code)


def _extract_country_candidates(lines: Iterable[str]) -> list[CountryRow]:
    """Extract country-specific codes with descriptions.
    
    Strategy: For each tariff code, use the most recent main heading description.
    The heading is in format: "Product Name: XXXX.XX"
    Everything between codes is tariff conditions/duties—we skip it.
    """
    code_re = re.compile(r"(?:\b00)?(\d{4}\.\d{2}\.\d{2}(?:\.\d{2})?)\b")
    heading_re = re.compile(r"^([^:]+):\s*(\d{4}\.\d{2})\s*$")
    
    rows: list[CountryRow] = []
    current_heading = ""
    
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        
        # Check for main heading (e.g., "Tomatoes, fresh or chilled: 0702.00")
        heading_match = heading_re.match(line)
        if heading_match:
            current_heading = _normalize_spaces(heading_match.group(1))
            continue
        
        # Check for tariff code
        m = code_re.search(line)
        if not m:
            continue
        
        code = m.group(1)
        chapter = code[:2]
        if not _chapter_ok(chapter):
            continue
        
        normalized = re.sub(r"\D", "", code)
        if len(normalized) not in (8, 10):
            continue
        
        hs6 = normalized[:6]
        if not hs6.isdigit():
            continue
        
        # Use only the main heading as description
        desc = current_heading
        if not desc:
            continue
        
        rows.append(
            CountryRow(
                chapter=chapter,
                full_code=code,
                normalized_code=normalized,
                hs6_prefix=hs6,
                description=desc,
                source_file="",
            )
        )
    
    return rows


def parse_country_pdf(pdf_path: Path) -> list[CountryRow]:
    text = _extract_text(pdf_path)
    lines = text.splitlines()
    rows = _extract_country_candidates(lines)

    dedup: dict[str, CountryRow] = {}
    for row in rows:
        row.source_file = pdf_path.name
        current = dedup.get(row.normalized_code)
        if current is None or len(row.description) > len(current.description):
            dedup[row.normalized_code] = row

    return sorted(dedup.values(), key=lambda r: r.normalized_code)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    six_dir = HS_DIR / "6 digit"
    country_dir = HS_DIR / "country specific"

    hs6_rows: list[HS6Row] = []
    for pdf in sorted(six_dir.glob("*.pdf")):
        hs6_rows.extend(parse_hs6_pdf(pdf))

    hs6_dedup: dict[str, HS6Row] = {}
    for row in hs6_rows:
        current = hs6_dedup.get(row.hs6_code)
        if current is None or len(row.description) > len(current.description):
            hs6_dedup[row.hs6_code] = row

    hs6_out = sorted(hs6_dedup.values(), key=lambda r: r.hs6_code)

    country_rows: list[CountryRow] = []
    for pdf in sorted(country_dir.glob("*.pdf")):
        country_rows.extend(parse_country_pdf(pdf))

    country_dedup: dict[str, CountryRow] = {}
    for row in country_rows:
        current = country_dedup.get(row.normalized_code)
        if current is None or len(row.description) > len(current.description):
            country_dedup[row.normalized_code] = row

    country_out = sorted(country_dedup.values(), key=lambda r: r.normalized_code)

    hs6_csv = OUT_DIR / "hs6_global_chapters_07_11.csv"
    country_csv = OUT_DIR / "hs_country_us_chapters_07_11.csv"

    _write_csv(
        hs6_csv,
        ["chapter", "hs6_code", "description", "source_file"],
        [
            {
                "chapter": r.chapter,
                "hs6_code": r.hs6_code,
                "description": r.description,
                "source_file": r.source_file,
            }
            for r in hs6_out
        ],
    )

    _write_csv(
        country_csv,
        [
            "chapter",
            "full_code",
            "normalized_code",
            "hs6_prefix",
            "description",
            "source_file",
        ],
        [
            {
                "chapter": r.chapter,
                "full_code": r.full_code,
                "normalized_code": r.normalized_code,
                "hs6_prefix": r.hs6_prefix,
                "description": r.description,
                "source_file": r.source_file,
            }
            for r in country_out
        ],
    )

    print(f"Wrote {hs6_csv} ({len(hs6_out)} rows)")
    print(f"Wrote {country_csv} ({len(country_out)} rows)")


if __name__ == "__main__":
    main()
