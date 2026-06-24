from __future__ import annotations

from collections import defaultdict
from math import exp

from legal_hse.retrievers.base import SearchResult


def dedupe_ranked_docs(results: list[SearchResult], top_k: int | None = None) -> list[SearchResult]:
    best: dict[str, SearchResult] = {}
    for result in results:
        existing = best.get(result.doc_id)
        if existing is None or result.score > existing.score:
            best[result.doc_id] = result
    ranked = sorted(best.values(), key=lambda item: item.score, reverse=True)
    return ranked[:top_k] if top_k is not None else ranked


def aggregate_chunk_results(
    results: list[SearchResult],
    *,
    method: str = "max",
    top_k: int | None = None,
    source: str | None = None,
) -> list[SearchResult]:
    grouped: dict[str, list[SearchResult]] = defaultdict(list)
    for result in results:
        grouped[result.doc_id].append(result)

    aggregated: list[SearchResult] = []
    for doc_id, doc_results in grouped.items():
        doc_results = sorted(doc_results, key=lambda item: item.score, reverse=True)
        scores = [item.score for item in doc_results]
        if method == "max":
            score = scores[0]
        elif method == "top2_mean":
            score = sum(scores[:2]) / min(2, len(scores))
        elif method == "max_plus_second":
            score = scores[0] + 0.2 * scores[1] if len(scores) > 1 else scores[0]
        elif method == "softmax_top3":
            top_scores = scores[:3]
            weights = [exp(score - max(top_scores)) for score in top_scores]
            norm = sum(weights)
            score = sum(s * w for s, w in zip(top_scores, weights, strict=True)) / norm
        else:
            raise ValueError(f"Unknown chunk aggregation method: {method}")

        best = doc_results[0]
        aggregated.append(
            SearchResult(
                doc_id=doc_id,
                unit_id=best.unit_id,
                score=float(score),
                source=source or f"{best.source}:{method}",
                text=best.text,
            )
        )

    ranked = sorted(aggregated, key=lambda item: item.score, reverse=True)
    return ranked[:top_k] if top_k is not None else ranked


def rrf_fusion(
    rankings: list[list[SearchResult]],
    *,
    k: int = 60,
    top_k: int = 5,
    source: str = "rrf",
) -> list[SearchResult]:
    scores: dict[str, float] = defaultdict(float)
    exemplars: dict[str, SearchResult] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, item in enumerate(ranking, start=1):
            if item.doc_id in seen:
                continue
            seen.add(item.doc_id)
            scores[item.doc_id] += 1.0 / (k + rank)
            if item.doc_id not in exemplars or item.score > exemplars[item.doc_id].score:
                exemplars[item.doc_id] = item

    fused = [
        SearchResult(
            doc_id=doc_id,
            unit_id=exemplars[doc_id].unit_id,
            score=score,
            source=source,
            text=exemplars[doc_id].text,
        )
        for doc_id, score in scores.items()
    ]
    return sorted(fused, key=lambda item: item.score, reverse=True)[:top_k]
