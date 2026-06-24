from __future__ import annotations

from pathlib import Path

import pandas as pd

from legal_hse.metrics import dedupe_topk


def make_submission_frame(qids: list[str], predictions: list[list[str]], *, top_k: int = 5) -> pd.DataFrame:
    rows: list[tuple[str, str]] = []
    for qid, predicted in zip(qids, predictions, strict=True):
        for doc_id in dedupe_topk(predicted, top_k):
            rows.append((qid, doc_id))
    return pd.DataFrame(rows, columns=["qid", "doc_id"])


def validate_submission(
    submission: pd.DataFrame,
    test: pd.DataFrame,
    documents: pd.DataFrame,
    *,
    top_k: int = 5,
) -> None:
    if list(submission.columns) != ["qid", "doc_id"]:
        raise ValueError("submission columns must be exactly: qid, doc_id")

    expected_qids = list(test["qid"])
    actual_qids = set(submission["qid"])
    missing = set(expected_qids).difference(actual_qids)
    extra = actual_qids.difference(expected_qids)
    if missing:
        raise ValueError(f"submission is missing qid values: {sorted(missing)[:5]}")
    if extra:
        raise ValueError(f"submission has unknown qid values: {sorted(extra)[:5]}")

    counts = submission.groupby("qid")["doc_id"].size()
    too_many = counts[counts > top_k]
    if not too_many.empty:
        raise ValueError(f"some qids have more than {top_k} rows: {too_many.head().to_dict()}")

    duplicates = submission.duplicated(["qid", "doc_id"])
    if duplicates.any():
        raise ValueError("submission contains duplicate doc_id values inside a qid")

    unknown_docs = set(submission["doc_id"]).difference(set(documents["doc_id"]))
    if unknown_docs:
        raise ValueError(f"submission contains unknown doc_id values: {sorted(unknown_docs)[:5]}")


def write_submission(
    qids: list[str],
    predictions: list[list[str]],
    path: str | Path,
    *,
    test: pd.DataFrame,
    documents: pd.DataFrame,
    top_k: int = 5,
) -> Path:
    frame = make_submission_frame(qids, predictions, top_k=top_k)
    validate_submission(frame, test, documents, top_k=top_k)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path
