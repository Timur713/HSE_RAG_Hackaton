from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def dedupe_topk(doc_ids: Sequence[str], k: int) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for doc_id in doc_ids:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        result.append(str(doc_id))
        if len(result) >= k:
            break
    return result


def recall_at_k(gold: Sequence[str], predictions: Sequence[Sequence[str]], k: int = 5) -> float:
    if len(gold) != len(predictions):
        raise ValueError("gold and predictions must have the same length")
    if not gold:
        return 0.0
    hits = 0
    for expected, predicted in zip(gold, predictions, strict=True):
        hits += str(expected) in dedupe_topk(predicted, k)
    return hits / len(gold)


def mrr_at_k(gold: Sequence[str], predictions: Sequence[Sequence[str]], k: int = 10) -> float:
    if len(gold) != len(predictions):
        raise ValueError("gold and predictions must have the same length")
    if not gold:
        return 0.0
    score = 0.0
    for expected, predicted in zip(gold, predictions, strict=True):
        for rank, doc_id in enumerate(dedupe_topk(predicted, k), start=1):
            if str(expected) == doc_id:
                score += 1.0 / rank
                break
    return score / len(gold)


def evaluate_predictions(
    gold: Sequence[str],
    predictions: Sequence[Sequence[str]],
    *,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"recall@{k}"] = recall_at_k(gold, predictions, k)
    metrics["mrr@10"] = mrr_at_k(gold, predictions, 10)
    return metrics


def prediction_frame_to_lists(frame: pd.DataFrame, qids: Sequence[str], top_k: int = 5) -> list[list[str]]:
    required = {"qid", "doc_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"prediction frame is missing columns: {sorted(missing)}")
    grouped = frame.groupby("qid")["doc_id"].apply(list).to_dict()
    return [dedupe_topk(grouped.get(qid, []), top_k) for qid in qids]
