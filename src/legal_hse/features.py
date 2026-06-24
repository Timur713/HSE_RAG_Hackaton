from __future__ import annotations

import re

import pandas as pd

ARTICLE_RE = re.compile(r"(?:ст\.?|статья|статьи)\s*\d+(?:\.\d+)?(?:-\d+)?", re.IGNORECASE)
DATE_RE = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b")
MONEY_RE = re.compile(r"\b\d[\d\s]*(?:руб|рублей|р\.)\b", re.IGNORECASE)


def document_kind(text: str) -> str:
    head = str(text)[:1200].upper().replace(" ", "")
    if "АПЕЛЛЯЦИОННОЕОПРЕДЕЛЕНИЕ" in head:
        return "апелляционное определение"
    if "ОПРЕДЕЛИЛ" in head:
        return "определение"
    if "РЕШИЛ" in head:
        return "решение"
    return "судебный акт"


def extract_document_features(documents: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for row in documents[["doc_id", "text"]].itertuples(index=False):
        text = str(row.text)
        articles = sorted({item.lower().replace("статья", "ст.") for item in ARTICLE_RE.findall(text)})
        rows.append(
            {
                "doc_id": row.doc_id,
                "doc_kind": document_kind(text),
                "articles": " ".join(articles),
                "date_count": len(DATE_RE.findall(text)),
                "money_count": len(MONEY_RE.findall(text)),
                "has_ustanovil": int("УСТАНОВИЛ" in text.upper()),
                "has_opredelil": int("ОПРЕДЕЛИЛ" in text.upper()),
                "has_reshil": int("РЕШИЛ" in text.upper()),
            }
        )
    return pd.DataFrame(rows)


def add_field_aware_text(documents: pd.DataFrame) -> pd.DataFrame:
    features = extract_document_features(documents)
    merged = documents.merge(features, on="doc_id", how="left")
    structural_text = (
        "тип_акта "
        + merged["doc_kind"].fillna("")
        + " статьи "
        + merged["articles"].fillna("")
        + " признаки "
        + merged[["has_ustanovil", "has_opredelil", "has_reshil"]].astype(str).agg(" ".join, axis=1)
    )
    result = documents.copy()
    result["text"] = documents["text"].fillna("").astype(str) + "\n\n" + structural_text
    return result
