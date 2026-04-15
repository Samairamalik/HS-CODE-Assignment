from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass
class HS6Entry:
    chapter: str
    code: str
    description: str


@dataclass
class CountryEntry:
    chapter: str
    code: str
    normalized_code: str
    hs6_prefix: str
    description: str


@dataclass
class Dataset:
    hs6: List[HS6Entry]
    country: List[CountryEntry]
    by_hs6: Dict[str, List[CountryEntry]]


def _clean_text(value: str) -> str:
    value = re.sub(r"\s+", " ", str(value or "")).strip()
    return value.strip(" .:-")


def _clean_digits(value: str) -> str:
    return re.sub(r"\D", "", str(value or ""))


def load_dataset(hs6_csv: Path, country_csv: Path) -> Dataset:
    hs6_df = pd.read_csv(hs6_csv, dtype=str).fillna("")
    country_df = pd.read_csv(country_csv, dtype=str).fillna("")

    hs6_df["hs6_code"] = hs6_df["hs6_code"].map(_clean_digits)
    hs6_df["description"] = hs6_df["description"].map(_clean_text)
    hs6_df = hs6_df[(hs6_df["hs6_code"].str.len() == 6) & (hs6_df["description"] != "")]
    hs6_df = hs6_df.drop_duplicates(subset=["hs6_code"], keep="first")

    country_df["normalized_code"] = country_df["normalized_code"].map(_clean_digits)
    country_df["hs6_prefix"] = country_df["normalized_code"].str[:6]
    country_df["description"] = country_df["description"].map(_clean_text)
    country_df = country_df[
        country_df["normalized_code"].str.len().isin([8, 10]) & (country_df["description"] != "")
    ]
    country_df = country_df.drop_duplicates(subset=["normalized_code"], keep="first")

    hs6_codes = set(hs6_df["hs6_code"].tolist())
    orphan = country_df[~country_df["hs6_prefix"].isin(hs6_codes)]
    if not orphan.empty:
        LOGGER.warning("Found %d orphan country codes without HS-6 parent", len(orphan))

    country_df = country_df[country_df["hs6_prefix"].isin(hs6_codes)]

    hs6_rows = [
        HS6Entry(chapter=row.get("chapter", ""), code=row["hs6_code"], description=row["description"])
        for _, row in hs6_df.iterrows()
    ]

    country_rows = [
        CountryEntry(
            chapter=row.get("chapter", ""),
            code=row.get("full_code", ""),
            normalized_code=row["normalized_code"],
            hs6_prefix=row["hs6_prefix"],
            description=row["description"],
        )
        for _, row in country_df.iterrows()
    ]

    by_hs6: Dict[str, List[CountryEntry]] = {}
    for row in country_rows:
        by_hs6.setdefault(row.hs6_prefix, []).append(row)

    return Dataset(hs6=hs6_rows, country=country_rows, by_hs6=by_hs6)
