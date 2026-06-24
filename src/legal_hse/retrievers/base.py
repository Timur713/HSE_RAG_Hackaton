from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SearchResult:
    doc_id: str
    score: float
    source: str
    unit_id: str | None = None
    text: str | None = None


class Retriever(Protocol):
    name: str

    def fit(self, units) -> "Retriever":
        ...

    def search(self, queries: list[str], top_k: int) -> list[list[SearchResult]]:
        ...
