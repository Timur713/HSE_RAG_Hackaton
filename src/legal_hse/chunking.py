from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class ChunkConfig:
    unit: str = "line"
    size: int = 10
    overlap: int = 5


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    start: int
    end: int
    position: int


def build_chunk_index(documents: pd.DataFrame, config: ChunkConfig) -> pd.DataFrame:
    chunks: list[Chunk] = []
    for row in documents[["doc_id", "text"]].itertuples(index=False):
        if config.unit == "line":
            doc_chunks = line_window_chunks(row.doc_id, row.text, config.size, config.overlap)
        elif config.unit == "char":
            doc_chunks = char_window_chunks(row.doc_id, row.text, config.size, config.overlap)
        elif config.unit == "paragraph":
            doc_chunks = paragraph_window_chunks(row.doc_id, row.text, config.size, config.overlap)
        else:
            raise ValueError(f"Unknown chunk unit: {config.unit}")
        chunks.extend(doc_chunks)
    return pd.DataFrame([chunk.__dict__ for chunk in chunks])


def line_window_chunks(doc_id: str, text: str, size: int, overlap: int) -> list[Chunk]:
    spans = _non_empty_line_spans(text)
    if not spans:
        return [Chunk(f"{doc_id}::line::0", doc_id, text, 0, len(text), 0)]
    windows = _windows(len(spans), size, overlap)
    chunks = []
    for position, (left, right) in enumerate(windows):
        start = spans[left][1]
        end = spans[right - 1][2]
        chunk_text = text[start:end].strip()
        chunks.append(Chunk(f"{doc_id}::line::{position}", doc_id, chunk_text, start, end, position))
    return chunks


def char_window_chunks(doc_id: str, text: str, size: int, overlap: int) -> list[Chunk]:
    if size <= 0:
        raise ValueError("char chunk size must be positive")
    step = max(1, size - overlap)
    chunks: list[Chunk] = []
    for position, start in enumerate(range(0, max(1, len(text)), step)):
        end = min(len(text), start + size)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(f"{doc_id}::char::{position}", doc_id, chunk_text, start, end, position))
        if end >= len(text):
            break
    return chunks or [Chunk(f"{doc_id}::char::0", doc_id, text, 0, len(text), 0)]


def paragraph_window_chunks(doc_id: str, text: str, size: int, overlap: int) -> list[Chunk]:
    spans = _paragraph_spans(text)
    if not spans:
        return [Chunk(f"{doc_id}::paragraph::0", doc_id, text, 0, len(text), 0)]
    windows = _windows(len(spans), size, overlap)
    chunks: list[Chunk] = []
    for position, (left, right) in enumerate(windows):
        start = spans[left][1]
        end = spans[right - 1][2]
        chunk_text = text[start:end].strip()
        chunks.append(Chunk(f"{doc_id}::paragraph::{position}", doc_id, chunk_text, start, end, position))
    return chunks


def _windows(n_items: int, size: int, overlap: int) -> Iterable[tuple[int, int]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    step = max(1, size - overlap)
    start = 0
    while start < n_items:
        end = min(n_items, start + size)
        yield start, end
        if end == n_items:
            break
        start += step


def _non_empty_line_spans(text: str) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    offset = 0
    for line in str(text).splitlines(keepends=True):
        start = offset
        end = offset + len(line)
        if line.strip():
            spans.append((line, start, end))
        offset = end
    return spans


def _paragraph_spans(text: str) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    for match in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|$)", str(text), flags=re.DOTALL):
        paragraph = match.group(0).strip()
        if paragraph:
            spans.append((paragraph, match.start(), match.end()))
    return spans
