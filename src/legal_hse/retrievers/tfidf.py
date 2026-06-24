from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from legal_hse.retrievers.base import SearchResult
from legal_hse.text import lexical_tokenize


@dataclass(frozen=True)
class TfidfConfig:
    analyzer: str = "word"
    ngram_range: tuple[int, int] = (1, 1)
    min_df: int | float = 1
    max_df: int | float = 1.0
    min_len: int = 3
    lemmatize: bool = False
    stop_words: bool = True
    preserve_legal_refs: bool = False
    legal_stop_words: bool = False
    add_bigrams: bool = False


class TfidfRetriever:
    def __init__(self, name: str, config: TfidfConfig | None = None) -> None:
        self.name = name
        self.config = config or TfidfConfig()
        self.units: pd.DataFrame | None = None
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None

    def fit(self, units: pd.DataFrame) -> "TfidfRetriever":
        required = {"unit_id", "doc_id", "text"}
        missing = required.difference(units.columns)
        if missing:
            raise ValueError(f"TF-IDF units are missing columns: {sorted(missing)}")
        self.units = units.reset_index(drop=True).copy()
        if self.config.analyzer == "word":
            self.vectorizer = TfidfVectorizer(
                tokenizer=lambda text: lexical_tokenize(
                    text,
                    min_len=self.config.min_len,
                    stop_words=self.config.stop_words,
                    lemmatize=self.config.lemmatize,
                    preserve_legal_refs=self.config.preserve_legal_refs,
                    legal_stop_words=self.config.legal_stop_words,
                    add_bigrams=self.config.add_bigrams,
                ),
                lowercase=False,
                token_pattern=None,
                ngram_range=self.config.ngram_range,
                min_df=self.config.min_df,
                max_df=self.config.max_df,
            )
        else:
            self.vectorizer = TfidfVectorizer(
                analyzer=self.config.analyzer,
                lowercase=True,
                ngram_range=self.config.ngram_range,
                min_df=self.config.min_df,
                max_df=self.config.max_df,
            )
        self.matrix = self.vectorizer.fit_transform(self.units["text"].fillna(""))
        return self

    def search(self, queries: list[str], top_k: int = 5) -> list[list[SearchResult]]:
        if self.units is None or self.vectorizer is None or self.matrix is None:
            raise RuntimeError("TfidfRetriever must be fitted before search")
        q_matrix = self.vectorizer.transform(queries)
        sims = linear_kernel(q_matrix, self.matrix)
        rankings: list[list[SearchResult]] = []
        for row in sims:
            top_idx = np.argsort(-row)[:top_k]
            results: list[SearchResult] = []
            for idx in top_idx:
                unit = self.units.iloc[int(idx)]
                results.append(
                    SearchResult(
                        doc_id=str(unit.doc_id),
                        unit_id=str(unit.unit_id),
                        score=float(row[idx]),
                        source=self.name,
                        text=str(unit.text),
                    )
                )
            rankings.append(results)
        return rankings
