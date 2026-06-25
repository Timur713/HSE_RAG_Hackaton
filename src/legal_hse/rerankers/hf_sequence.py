from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HfSequenceRerankerConfig:
    model_name: str = "BAAI/bge-reranker-v2-m3"
    batch_size: int = 16
    max_length: int = 1024
    device: str | None = None
    trust_remote_code: bool = False


class HfSequenceReranker:
    def __init__(self, config: HfSequenceRerankerConfig | None = None) -> None:
        self.config = config or HfSequenceRerankerConfig()
        self.tokenizer = None
        self.model = None
        self.device = None

    def load(self) -> "HfSequenceReranker":
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "HfSequenceReranker requires torch and transformers. "
                "Install `pip install -e .[dense]` or `pip install transformers torch`."
            ) from exc

        if self.config.device is not None:
            self.device = torch.device(self.config.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )
        self.model.to(self.device)
        self.model.eval()
        return self

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        if self.model is None or self.tokenizer is None or self.device is None:
            self.load()
        assert self.model is not None and self.tokenizer is not None and self.device is not None

        import torch

        scores: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(pairs), self.config.batch_size):
                batch = pairs[start : start + self.config.batch_size]
                encoded = self.tokenizer(
                    [query for query, _ in batch],
                    [passage for _, passage in batch],
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                logits = self.model(**encoded, return_dict=True).logits
                scores.extend(logits.reshape(-1).detach().float().cpu().tolist())
        return scores
