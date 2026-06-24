from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log

import numpy as np
import pandas as pd

from legal_hse.retrievers.base import SearchResult
from legal_hse.text import lexical_tokenize


@dataclass(frozen=True)
class BM25Config:
    k1: float = 1.5
    b: float = 0.75
    min_len: int = 3
    stop_words: bool = True
    lemmatize: bool = False
    preserve_legal_refs: bool = False
    legal_stop_words: bool = False
    add_bigrams: bool = False


class BM25Retriever:
    def __init__(self, name: str, config: BM25Config | None = None) -> None:
        self.name = name
        self.config = config or BM25Config()
        self.units: pd.DataFrame | None = None
        self.doc_lens: np.ndarray | None = None
        self.avgdl = 0.0
        self.idf: dict[str, float] = {}
        self.postings: dict[str, list[tuple[int, int]]] = {}

    def fit(self, units: pd.DataFrame) -> "BM25Retriever":
        required = {"unit_id", "doc_id", "text"}
        missing = required.difference(units.columns)
        if missing:
            raise ValueError(f"BM25 units are missing columns: {sorted(missing)}")

        self.units = units.reset_index(drop=True).copy()
        token_counts: list[Counter[str]] = []
        doc_lens: list[int] = []
        df: Counter[str] = Counter()
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for idx, text in enumerate(self.units["text"].fillna("")):
            tokens = lexical_tokenize(
                text,
                min_len=self.config.min_len,
                stop_words=self.config.stop_words,
                lemmatize=self.config.lemmatize,
                preserve_legal_refs=self.config.preserve_legal_refs,
                legal_stop_words=self.config.legal_stop_words,
                add_bigrams=self.config.add_bigrams,
            )
            counts = Counter(tokens)
            token_counts.append(counts)
            doc_lens.append(sum(counts.values()))
            for term, tf in counts.items():
                df[term] += 1
                postings[term].append((idx, tf))

        n_docs = len(token_counts)
        self.doc_lens = np.asarray(doc_lens, dtype=np.float32)
        self.avgdl = float(self.doc_lens.mean()) if n_docs else 0.0
        self.idf = {
            term: log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }
        self.postings = dict(postings)
        return self

    def search(self, queries: list[str], top_k: int = 5) -> list[list[SearchResult]]:
        if self.units is None or self.doc_lens is None:
            raise RuntimeError("BM25Retriever must be fitted before search")
        return [self._search_one(query, top_k) for query in queries]

    def _search_one(self, query: str, top_k: int) -> list[SearchResult]:
        assert self.units is not None and self.doc_lens is not None
        scores = np.zeros(len(self.units), dtype=np.float32)
        query_terms = set(
            lexical_tokenize(
                query,
                min_len=self.config.min_len,
                stop_words=self.config.stop_words,
                lemmatize=self.config.lemmatize,
                preserve_legal_refs=self.config.preserve_legal_refs,
                legal_stop_words=self.config.legal_stop_words,
                add_bigrams=self.config.add_bigrams,
            )
        )
        if query_terms:
            denom_const = self.config.k1 * (
                1.0 - self.config.b + self.config.b * self.doc_lens / max(self.avgdl, 1e-9)
            )
            for term in query_terms:
                posting = self.postings.get(term)
                if not posting:
                    continue
                idf = self.idf[term]
                for idx, tf in posting:
                    scores[idx] += idf * (tf * (self.config.k1 + 1.0)) / (tf + denom_const[idx])

        top_idx = np.argsort(-scores)[:top_k]
        results: list[SearchResult] = []
        for idx in top_idx:
            row = self.units.iloc[int(idx)]
            results.append(
                SearchResult(
                    doc_id=str(row.doc_id),
                    unit_id=str(row.unit_id),
                    score=float(scores[idx]),
                    source=self.name,
                    text=str(row.text),
                )
            )
        return results
