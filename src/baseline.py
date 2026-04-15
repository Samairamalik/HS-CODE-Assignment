from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data_loader import CountryEntry, Dataset, HS6Entry


@dataclass
class BaselineMatch:
    code: str
    description: str
    score: float


@dataclass
class BaselineResult:
    hs6: BaselineMatch
    country: BaselineMatch | None


class TfidfHierarchicalBaseline:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

        self.hs6_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.hs6_matrix = self.hs6_vectorizer.fit_transform([x.description for x in dataset.hs6])

        self.country_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.country_matrix = self.country_vectorizer.fit_transform([x.description for x in dataset.country])

        self.country_index_by_code = {row.normalized_code: i for i, row in enumerate(self.dataset.country)}

    def _match_hs6(self, query: str) -> tuple[HS6Entry, float]:
        q = self.hs6_vectorizer.transform([query])
        scores = cosine_similarity(q, self.hs6_matrix)[0]
        idx = int(np.argmax(scores))
        return self.dataset.hs6[idx], float(scores[idx])

    def _match_country(self, query: str, hs6_code: str) -> tuple[CountryEntry, float] | None:
        candidates = self.dataset.by_hs6.get(hs6_code, [])
        if not candidates:
            return None

        idxs = [self.country_index_by_code[c.normalized_code] for c in candidates]
        q = self.country_vectorizer.transform([query])
        scores = cosine_similarity(q, self.country_matrix[idxs])[0]
        local = int(np.argmax(scores))
        return candidates[local], float(scores[local])

    def search(self, query: str) -> BaselineResult:
        text = query.strip()
        if not text:
            raise ValueError("Query is empty")

        hs6_row, hs6_score = self._match_hs6(text)
        country_pair = self._match_country(text, hs6_row.code)

        country_match = None
        if country_pair is not None:
            c_row, c_score = country_pair
            country_match = BaselineMatch(
                code=c_row.code or c_row.normalized_code,
                description=c_row.description,
                score=c_score,
            )

        return BaselineResult(
            hs6=BaselineMatch(code=hs6_row.code, description=hs6_row.description, score=hs6_score),
            country=country_match,
        )
