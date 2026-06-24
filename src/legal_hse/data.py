from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from legal_hse.config import PathConfig


@dataclass(frozen=True)
class DataBundle:
    documents: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame
    sample_submission: pd.DataFrame


def load_data(root: str | Path) -> DataBundle:
    paths = PathConfig.from_root(root)
    paths.require_input_files()
    documents = pd.read_csv(paths.documents)
    train = pd.read_csv(paths.train)
    test = pd.read_csv(paths.test)
    sample_submission = pd.read_csv(paths.sample_submission)
    validate_dataframes(documents, train, test, sample_submission)
    return DataBundle(
        documents=documents,
        train=train,
        test=test,
        sample_submission=sample_submission,
    )


def validate_dataframes(
    documents: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> None:
    required = {
        "documents": {"doc_id", "text"},
        "train": {
            "qid",
            "question",
            "gold_doc_id",
            "ideal_answer",
            "gold_evidence_text",
            "gold_evidence_char_start",
            "gold_evidence_char_end",
            "topic",
        },
        "test": {"qid", "question"},
        "sample_submission": {"qid", "doc_id"},
    }
    frames = {
        "documents": documents,
        "train": train,
        "test": test,
        "sample_submission": sample_submission,
    }
    for name, cols in required.items():
        missing = cols.difference(frames[name].columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")

    if documents["doc_id"].duplicated().any():
        dupes = documents.loc[documents["doc_id"].duplicated(), "doc_id"].head().tolist()
        raise ValueError(f"documents.doc_id must be unique; examples: {dupes}")

    unknown_gold = set(train["gold_doc_id"]).difference(set(documents["doc_id"]))
    if unknown_gold:
        raise ValueError(f"train.gold_doc_id contains unknown doc_id values: {sorted(unknown_gold)[:5]}")


def check_evidence_alignment(train: pd.DataFrame, documents: pd.DataFrame, max_errors: int = 5) -> list[dict]:
    """Return evidence span mismatches without failing the whole run."""

    docs_by_id = documents.set_index("doc_id")["text"].to_dict()
    errors: list[dict] = []
    for row in train.itertuples(index=False):
        text = docs_by_id[row.gold_doc_id]
        start = int(row.gold_evidence_char_start)
        end = int(row.gold_evidence_char_end)
        if text[start:end] != row.gold_evidence_text:
            errors.append({"qid": row.qid, "doc_id": row.gold_doc_id, "start": start, "end": end})
            if len(errors) >= max_errors:
                break
    return errors
