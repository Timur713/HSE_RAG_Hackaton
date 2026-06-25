from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy import sparse

from legal_hse.retrievers.base import SearchResult


@dataclass(frozen=True)
class BgeM3Config:
    model_name: str = "BAAI/bge-m3"
    batch_size: int = 4
    max_length: int = 2048
    use_fp16: bool | None = None
    normalize_dense: bool = True
    use_sparse: bool = True
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    device: str | None = None


class BgeM3Retriever:
    """Native FlagEmbedding BGE-M3 retriever.

    This uses BGE-M3 dense vectors plus its lexical sparse weights. ColBERT
    scoring is intentionally left out of first-stage retrieval because it is
    substantially more expensive over all chunks and is better used as a
    bounded rerank step after candidate generation.
    """

    def __init__(self, name: str, config: BgeM3Config | None = None) -> None:
        self.name = name
        self.config = config or BgeM3Config()
        self.units: pd.DataFrame | None = None
        self.model = None
        self.dense_embeddings: np.ndarray | None = None
        self.sparse_embeddings: sparse.csr_matrix | None = None
        self._token_to_col: dict[str, int] = {}

    def fit(self, units: pd.DataFrame) -> "BgeM3Retriever":
        required = {"unit_id", "doc_id", "text"}
        missing = required.difference(units.columns)
        if missing:
            raise ValueError(f"BGE-M3 units are missing columns: {sorted(missing)}")

        self.units = units.reset_index(drop=True).copy()
        self.model = self._load_model()
        output = self._encode(self.units["text"].fillna("").astype(str).tolist())
        self.dense_embeddings = _as_float32_matrix(output["dense_vecs"])
        if self.config.normalize_dense:
            self.dense_embeddings = _l2_normalize(self.dense_embeddings)

        if self.config.use_sparse:
            self.sparse_embeddings = _lexical_weights_to_csr(
                _require_lexical_weights(output),
                token_to_col=self._token_to_col,
                allow_new=True,
            )
        return self

    def search(self, queries: list[str], top_k: int = 5) -> list[list[SearchResult]]:
        if self.units is None or self.model is None or self.dense_embeddings is None:
            raise RuntimeError("BgeM3Retriever must be fitted before search")

        output = self._encode([str(query) for query in queries])
        query_dense = _as_float32_matrix(output["dense_vecs"])
        if self.config.normalize_dense:
            query_dense = _l2_normalize(query_dense)

        scores = self.config.dense_weight * (query_dense @ self.dense_embeddings.T)
        if self.config.use_sparse and self.sparse_embeddings is not None:
            query_sparse = _lexical_weights_to_csr(
                _require_lexical_weights(output),
                token_to_col=self._token_to_col,
                allow_new=False,
                n_cols=self.sparse_embeddings.shape[1],
            )
            sparse_scores = query_sparse @ self.sparse_embeddings.T
            scores = scores + self.config.sparse_weight * sparse_scores.toarray().astype("float32")

        rankings: list[list[SearchResult]] = []
        for row in np.asarray(scores, dtype="float32"):
            top_idx = _top_indices(row, top_k)
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

    def _load_model(self) -> Any:
        try:
            from FlagEmbedding import BGEM3FlagModel  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "BgeM3Retriever requires FlagEmbedding. "
                "Install `pip install -r requirements-optional.txt`."
            ) from exc

        kwargs: dict[str, Any] = {
            "use_fp16": self._use_fp16(),
            "normalize_embeddings": self.config.normalize_dense,
        }
        if self.config.device is not None:
            kwargs["device"] = self.config.device

        for optional_key in ("normalize_embeddings", "device"):
            try:
                return BGEM3FlagModel(self.config.model_name, **kwargs)
            except TypeError:
                kwargs.pop(optional_key, None)
        return BGEM3FlagModel(self.config.model_name, **kwargs)

    def _encode(self, texts: list[str]) -> dict[str, Any]:
        assert self.model is not None
        return self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            max_length=self.config.max_length,
            return_dense=True,
            return_sparse=self.config.use_sparse,
            return_colbert_vecs=False,
        )

    def _use_fp16(self) -> bool:
        if self.config.use_fp16 is not None:
            return bool(self.config.use_fp16)
        if self.config.device is not None and str(self.config.device).startswith("cpu"):
            return False
        try:
            import torch  # type: ignore

            return bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001 - optional dependency only used for device probing.
            return False


def _as_float32_matrix(values: Any) -> np.ndarray:
    matrix = np.asarray(values, dtype="float32")
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    return matrix


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _require_lexical_weights(output: Mapping[str, Any]) -> list[Mapping[Any, float]]:
    weights = output.get("lexical_weights")
    if weights is None:
        raise RuntimeError("BGE-M3 encode output does not contain lexical_weights")
    return list(weights)


def _lexical_weights_to_csr(
    rows: list[Mapping[Any, float]],
    *,
    token_to_col: dict[str, int],
    allow_new: bool,
    n_cols: int | None = None,
) -> sparse.csr_matrix:
    data: list[float] = []
    indices: list[int] = []
    indptr: list[int] = [0]

    for weights in rows:
        for token_id, weight in dict(weights).items():
            token_key = str(token_id)
            col = token_to_col.get(token_key)
            if col is None:
                if not allow_new:
                    continue
                col = len(token_to_col)
                token_to_col[token_key] = col
            indices.append(col)
            data.append(float(weight))
        indptr.append(len(indices))

    width = n_cols if n_cols is not None else len(token_to_col)
    return sparse.csr_matrix(
        (
            np.asarray(data, dtype="float32"),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int32),
        ),
        shape=(len(rows), width),
        dtype="float32",
    )


def _top_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    if len(scores) == 0:
        return np.asarray([], dtype=np.int64)
    keep = min(top_k, len(scores))
    unordered = np.argpartition(-scores, keep - 1)[:keep]
    return unordered[np.argsort(-scores[unordered])]
