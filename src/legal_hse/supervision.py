from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class EvidenceWindow:
    qid: str
    question: str
    doc_id: str
    text: str
    start: int
    end: int
    topic: str


def build_evidence_windows(
    train: pd.DataFrame,
    documents: pd.DataFrame,
    *,
    window_chars: int = 1200,
) -> pd.DataFrame:
    """Build supervised query->passage positives around gold evidence spans."""

    docs = documents.set_index("doc_id")["text"].to_dict()
    rows: list[EvidenceWindow] = []
    half = max(0, window_chars // 2)
    for item in train.itertuples(index=False):
        doc_text = docs[item.gold_doc_id]
        ev_start = int(item.gold_evidence_char_start)
        ev_end = int(item.gold_evidence_char_end)
        center = (ev_start + ev_end) // 2
        start = max(0, center - half)
        end = min(len(doc_text), center + half)
        if ev_start < start:
            start = ev_start
        if ev_end > end:
            end = ev_end
        rows.append(
            EvidenceWindow(
                qid=str(item.qid),
                question=str(item.question),
                doc_id=str(item.gold_doc_id),
                text=doc_text[start:end].strip(),
                start=start,
                end=end,
                topic=str(item.topic),
            )
        )
    return pd.DataFrame([row.__dict__ for row in rows])


def build_pairwise_training_frame(
    train: pd.DataFrame,
    documents: pd.DataFrame,
    *,
    window_chars: int = 1200,
) -> pd.DataFrame:
    positives = build_evidence_windows(train, documents, window_chars=window_chars)
    positives = positives.rename(columns={"text": "positive_text"})
    return positives[["qid", "question", "doc_id", "positive_text", "start", "end", "topic"]]


def mine_hard_negatives(
    train: pd.DataFrame,
    candidate_rankings: list[list[str]],
    *,
    negatives_per_query: int = 5,
) -> pd.DataFrame:
    rows: list[dict] = []
    for item, ranking in zip(train.itertuples(index=False), candidate_rankings, strict=True):
        negatives = [doc_id for doc_id in ranking if doc_id != item.gold_doc_id][:negatives_per_query]
        for rank, doc_id in enumerate(negatives, start=1):
            rows.append(
                {
                    "qid": item.qid,
                    "question": item.question,
                    "gold_doc_id": item.gold_doc_id,
                    "negative_doc_id": doc_id,
                    "negative_rank": rank,
                    "topic": item.topic,
                }
            )
    return pd.DataFrame(rows)
