from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from legal_hse.retrievers.base import SearchResult


@dataclass(frozen=True)
class DenseConfig:
    model_name: str = "intfloat/multilingual-e5-base"
    batch_size: int = 32
    normalize_embeddings: bool = True
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    device: str | None = None


class DenseRetriever:
    """Sentence-Transformers based retriever, imported lazily for cheap baseline runs."""

    def __init__(self, name: str, config: DenseConfig | None = None) -> None:
        self.name = name
        self.config = config or DenseConfig()
        self.units: pd.DataFrame | None = None
        self.model = None
        self.embeddings: np.ndarray | None = None

    def fit(self, units: pd.DataFrame) -> "DenseRetriever":
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "DenseRetriever requires sentence-transformers. "
                "Install `pip install -e .[dense]` or `pip install -r requirements-optional.txt`."
            ) from exc

        required = {"unit_id", "doc_id", "text"}
        missing = required.difference(units.columns)
        if missing:
            raise ValueError(f"Dense units are missing columns: {sorted(missing)}")
        self.units = units.reset_index(drop=True).copy()
        self.model = SentenceTransformer(self.config.model_name, device=self.config.device)
        passages = [self.config.passage_prefix + str(text) for text in self.units["text"].fillna("")]
        self.embeddings = self.model.encode(
            passages,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=True,
        ).astype("float32")
        return self

    def search(self, queries: list[str], top_k: int = 5) -> list[list[SearchResult]]:
        if self.units is None or self.model is None or self.embeddings is None:
            raise RuntimeError("DenseRetriever must be fitted before search")
        encoded = self.model.encode(
            [self.config.query_prefix + str(query) for query in queries],
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=True,
        ).astype("float32")
        sims = encoded @ self.embeddings.T
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
