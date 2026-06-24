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


def quota_union_fusion(
    rankings: list[list[SearchResult]],
    *,
    quota: int = 8,
    per_ranking_quota: list[int] | None = None,
    k: int = 60,
    top_k: int = 50,
    source: str = "quota_rrf",
) -> list[SearchResult]:
    """Diversify fused candidates by reserving slots from each input ranking.

    RRF is still used as the fallback ordering, but the first pass takes a
    round-robin quota from every branch. This is intentionally recall-oriented
    and is useful for candidate generation before reranking.
    """

    unique_rankings = [_unique_by_doc_id(ranking) for ranking in rankings]
    quotas = per_ranking_quota or [quota] * len(unique_rankings)
    if len(quotas) != len(unique_rankings):
        raise ValueError("per_ranking_quota must match the number of rankings")

    scores: dict[str, float] = defaultdict(float)
    exemplars: dict[str, SearchResult] = {}
    for ranking in unique_rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item.doc_id] += 1.0 / (k + rank)
            if item.doc_id not in exemplars or item.score > exemplars[item.doc_id].score:
                exemplars[item.doc_id] = item

    selected: list[str] = []
    selected_set: set[str] = set()
    pointers = [0] * len(unique_rankings)
    for round_idx in range(max(quotas, default=0)):
        for idx, ranking in enumerate(unique_rankings):
            if round_idx >= quotas[idx]:
                continue
            while pointers[idx] < len(ranking) and ranking[pointers[idx]].doc_id in selected_set:
                pointers[idx] += 1
            if pointers[idx] >= len(ranking):
                continue
            doc_id = ranking[pointers[idx]].doc_id
            selected.append(doc_id)
            selected_set.add(doc_id)
            pointers[idx] += 1
            if len(selected) >= top_k:
                break
        if len(selected) >= top_k:
            break

    for doc_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        if len(selected) >= top_k:
            break
        if doc_id not in selected_set:
            selected.append(doc_id)
            selected_set.add(doc_id)

    return [
        SearchResult(
            doc_id=doc_id,
            unit_id=exemplars[doc_id].unit_id,
            score=float(scores[doc_id]),
            source=source,
            text=exemplars[doc_id].text,
        )
        for doc_id in selected[:top_k]
    ]


def _unique_by_doc_id(ranking: list[SearchResult]) -> list[SearchResult]:
    seen: set[str] = set()
    unique: list[SearchResult] = []
    for item in ranking:
        if item.doc_id in seen:
            continue
        seen.add(item.doc_id)
        unique.append(item)
    return unique
