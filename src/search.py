from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from sentence_transformers import SentenceTransformer

from .data_loader import CountryEntry, Dataset, HS6Entry, load_dataset
from .embeddings import get_or_create_embeddings, load_model

SYNONYMS: Dict[str, str] = {
    "garbanzo": "chickpea",
    "garbanzos": "chickpeas",
    "aubergine": "eggplant",
    "aubergines": "eggplants",
    "sowing": "planting",
}


@dataclass
class MatchResult:
    code: str
    description: str
    score: float


@dataclass
class SearchResult:
    query: str
    hs6: MatchResult
    country: MatchResult | None
    explanation: str


def _normalize_query(query: str) -> str:
    text = re.sub(r"\s+", " ", query.strip().lower())
    for source, target in SYNONYMS.items():
        text = re.sub(rf"\b{re.escape(source)}\b", target, text)
    return text


class HierarchicalSemanticSearcher:
    def __init__(
        self,
        dataset: Dataset,
        model: SentenceTransformer,
        hs6_embeddings: np.ndarray,
        country_embeddings: np.ndarray,
    ) -> None:
        self.dataset = dataset
        self.model = model
        self.hs6_embeddings = hs6_embeddings
        self.country_embeddings = country_embeddings

    @classmethod
    def from_csv(
        cls,
        hs6_csv: Path,
        country_csv: Path,
        cache_dir: Path,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> "HierarchicalSemanticSearcher":
        dataset = load_dataset(hs6_csv=hs6_csv, country_csv=country_csv)
        hs6_texts = [x.description for x in dataset.hs6]
        country_texts = [x.description for x in dataset.country]

        hs6_cache = cache_dir / "hs6"
        country_cache = cache_dir / "country"

        hs6_embeddings = get_or_create_embeddings(
            cache_dir=hs6_cache,
            matrix_name="hs6_embeddings",
            texts=hs6_texts,
            source_files=[hs6_csv, country_csv],
            model_name=model_name,
        )
        country_embeddings = get_or_create_embeddings(
            cache_dir=country_cache,
            matrix_name="country_embeddings",
            texts=country_texts,
            source_files=[hs6_csv, country_csv],
            model_name=model_name,
        )

        model = load_model(model_name)
        return cls(dataset=dataset, model=model, hs6_embeddings=hs6_embeddings, country_embeddings=country_embeddings)

    def _encode_query(self, query: str) -> np.ndarray:
        expanded = _normalize_query(query)
        emb = self.model.encode([expanded], convert_to_numpy=True, normalize_embeddings=True)
        return emb

    def _search_hs6(self, query_emb: np.ndarray) -> tuple[HS6Entry, float]:
        scores = cosine_similarity(query_emb, self.hs6_embeddings)[0]
        idx = int(np.argmax(scores))
        return self.dataset.hs6[idx], float(scores[idx])

    def _search_country(self, query_emb: np.ndarray, hs6_code: str) -> tuple[CountryEntry, float] | None:
        candidates = self.dataset.by_hs6.get(hs6_code, [])
        if not candidates:
            return None

        all_index_by_code = {row.normalized_code: i for i, row in enumerate(self.dataset.country)}
        indices = [all_index_by_code[row.normalized_code] for row in candidates]
        sub_embeddings = self.country_embeddings[indices]

        scores = cosine_similarity(query_emb, sub_embeddings)[0]
        local_idx = int(np.argmax(scores))
        row = candidates[local_idx]
        return row, float(scores[local_idx])

    def search(self, query: str) -> SearchResult:
        raw_query = query.strip()
        if not raw_query:
            raise ValueError("Query is empty")

        query_emb = self._encode_query(raw_query)

        hs6_row, hs6_score = self._search_hs6(query_emb)
        country_pair = self._search_country(query_emb, hs6_row.code)

        country_match = None
        if country_pair is not None:
            country_row, country_score = country_pair
            country_match = MatchResult(
                code=country_row.code or country_row.normalized_code,
                description=country_row.description,
                score=country_score,
            )

        explanation = (
            f"Matched HS-6 {hs6_row.code} using semantic similarity, then searched only within that HS-6 branch "
            f"to select the most relevant country-specific code."
        )

        return SearchResult(
            query=raw_query,
            hs6=MatchResult(code=hs6_row.code, description=hs6_row.description, score=hs6_score),
            country=country_match,
            explanation=explanation,
        )
