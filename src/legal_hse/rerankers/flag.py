from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FlagRerankerConfig:
    model_name: str = "BAAI/bge-reranker-v2-m3"
    batch_size: int = 16
    use_fp16: bool = True
    normalize: bool = False


class FlagEmbeddingReranker:
    def __init__(self, config: FlagRerankerConfig | None = None) -> None:
        self.config = config or FlagRerankerConfig()
        self.model = None

    def load(self) -> "FlagEmbeddingReranker":
        try:
            from FlagEmbedding import FlagReranker  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "FlagEmbeddingReranker requires FlagEmbedding. "
                "Install `pip install -e .[dense]` or `pip install FlagEmbedding`."
            ) from exc

        self.model = FlagReranker(self.config.model_name, use_fp16=self.config.use_fp16)
        return self

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        if self.model is None:
            self.load()
        assert self.model is not None
        pair_list = [[query, passage] for query, passage in pairs]
        try:
            scores = self.model.compute_score(
                pair_list,
                batch_size=self.config.batch_size,
                normalize=self.config.normalize,
            )
        except TypeError:
            scores = self.model.compute_score(pair_list, batch_size=self.config.batch_size)
        if isinstance(scores, float):
            return [float(scores)]
        return [float(score) for score in scores]
