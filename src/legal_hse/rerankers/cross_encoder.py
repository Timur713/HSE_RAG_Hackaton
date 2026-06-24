from __future__ import annotations

from dataclasses import dataclass

from legal_hse.retrievers.base import SearchResult


@dataclass(frozen=True)
class CrossEncoderConfig:
    model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    batch_size: int = 16
    max_length: int = 512
    device: str | None = None


class CrossEncoderReranker:
    def __init__(self, config: CrossEncoderConfig | None = None) -> None:
        self.config = config or CrossEncoderConfig()
        self.model = None

    def load(self) -> "CrossEncoderReranker":
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "CrossEncoderReranker requires sentence-transformers. "
                "Install `pip install -e .[dense]` or `pip install -r requirements-optional.txt`."
            ) from exc

        self.model = CrossEncoder(
            self.config.model_name,
            max_length=self.config.max_length,
            device=self.config.device,
        )
        return self

    def rerank(self, query: str, candidates: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
        if self.model is None:
            self.load()
        assert self.model is not None
        pairs = [(query, candidate.text or "") for candidate in candidates]
        scores = self.model.predict(pairs, batch_size=self.config.batch_size, show_progress_bar=False)
        reranked = [
            SearchResult(
                doc_id=candidate.doc_id,
                unit_id=candidate.unit_id,
                score=float(score),
                source=f"cross_encoder:{candidate.source}",
                text=candidate.text,
            )
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        return sorted(reranked, key=lambda item: item.score, reverse=True)[:top_k]
