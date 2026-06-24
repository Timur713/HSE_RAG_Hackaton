from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from legal_hse.chunking import ChunkConfig, build_chunk_index
from legal_hse.config import PathConfig
from legal_hse.data import load_data
from legal_hse.fusion import aggregate_chunk_results, dedupe_ranked_docs, quota_union_fusion, rrf_fusion
from legal_hse.features import add_field_aware_text
from legal_hse.metrics import dedupe_topk, evaluate_predictions
from legal_hse.retrievers.base import SearchResult
from legal_hse.retrievers.bm25 import BM25Config, BM25Retriever
from legal_hse.retrievers.dense import DenseConfig, DenseRetriever
from legal_hse.retrievers.tfidf import TfidfConfig, TfidfRetriever
from legal_hse.splits import Split, make_group_holdout, make_group_kfold
from legal_hse.submission import write_submission

EVAL_KS = (5, 10, 20, 50)
DEFAULT_EVAL_DEPTH = max(EVAL_KS)
METRIC_COLUMNS = [f"recall@{k}" for k in EVAL_KS]
EVAL_PART_ORDER = ("train", "valid", "holdout")
DEFAULT_COMPARISON_BASELINE = "bm25_doc"


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    kind: str
    params: dict[str, Any]
    priority: str = "P0"
    description: str = ""


def default_experiments(*, include_optional: bool = False) -> list[ExperimentSpec]:
    lemma_config = {"lemmatize": True, "legal_stop_words": True}
    legal_lemma_config = {
        "lemmatize": True,
        "preserve_legal_refs": True,
        "legal_stop_words": True,
        "min_len": 2,
    }
    legal_phrase_config = {**legal_lemma_config, "add_bigrams": True}
    specs = [
        ExperimentSpec(
            name="tfidf_word_doc",
            kind="tfidf_doc",
            params={"config": {"analyzer": "word", "ngram_range": (1, 1)}},
            priority="P0",
            description="Baseline-style word TF-IDF over full documents.",
        ),
        ExperimentSpec(
            name="tfidf_char_doc_3_5",
            kind="tfidf_doc",
            params={"config": {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 1}},
            priority="P1",
            description="Character n-gram TF-IDF control for Russian morphology and typos.",
        ),
        ExperimentSpec(
            name="tfidf_word_lemma_doc",
            kind="tfidf_doc",
            params={"config": {"analyzer": "word", "ngram_range": (1, 1), **lemma_config}},
            priority="P1",
            description="Word TF-IDF over full documents with pymorphy lemmatization and anonymization-token cleanup.",
        ),
        ExperimentSpec(
            name="tfidf_word_legal_lemma_doc",
            kind="tfidf_doc",
            params={"config": {"analyzer": "word", "ngram_range": (1, 1), **legal_lemma_config}},
            priority="P1",
            description="Word TF-IDF with Russian lemmatization plus legal references and short legal abbreviations.",
        ),
        ExperimentSpec(
            name="bm25_doc",
            kind="bm25_doc",
            params={"config": {"k1": 1.5, "b": 0.75}},
            priority="P0",
            description="BM25 over full documents.",
        ),
        ExperimentSpec(
            name="bm25_lemma_doc",
            kind="bm25_doc",
            params={"config": {"k1": 1.5, "b": 0.75, **lemma_config}},
            priority="P0",
            description="BM25 over full documents with pymorphy lemmatization and anonymization-token cleanup.",
        ),
        ExperimentSpec(
            name="bm25_legal_lemma_doc",
            kind="bm25_doc",
            params={"config": {"k1": 1.5, "b": 0.75, **legal_lemma_config}},
            priority="P0",
            description="BM25 over full documents with legal-aware tokenization, short-code preservation, and lemmatization.",
        ),
        ExperimentSpec(
            name="bm25_legal_phrase_doc",
            kind="bm25_doc",
            params={"config": {"k1": 1.5, "b": 0.75, **legal_phrase_config}},
            priority="P1",
            description="BM25 over full documents with legal-aware lemmas and adjacent lemma bigrams for stable legal phrases.",
        ),
        ExperimentSpec(
            name="bm25_field_aware_doc",
            kind="bm25_field_doc",
            params={"config": {"k1": 1.5, "b": 0.75}},
            priority="P1",
            description="BM25 over full documents enriched with extracted structural legal fields.",
        ),
        ExperimentSpec(
            name="bm25_legal_lemma_field_doc",
            kind="bm25_field_doc",
            params={"config": {"k1": 1.5, "b": 0.75, **legal_lemma_config}},
            priority="P1",
            description="Field-aware BM25 with legal-aware lemmatization.",
        ),
        ExperimentSpec(
            name="bm25_chunk_line_10_5_max",
            kind="bm25_chunk",
            params={
                "config": {"k1": 1.5, "b": 0.75},
                "chunk": {"unit": "line", "size": 10, "overlap": 5},
                "aggregation": "max",
                "rank_depth": 120,
            },
            priority="P0",
            description="BM25 over line-window chunks with max chunk-to-doc aggregation.",
        ),
        ExperimentSpec(
            name="bm25_legal_lemma_chunk_line_10_5_max",
            kind="bm25_chunk",
            params={
                "config": {"k1": 1.5, "b": 0.75, **legal_lemma_config},
                "chunk": {"unit": "line", "size": 10, "overlap": 5},
                "aggregation": "max",
                "rank_depth": 120,
            },
            priority="P0",
            description="Line-window BM25 chunks with legal-aware lemmatization and max aggregation.",
        ),
        ExperimentSpec(
            name="bm25_legal_phrase_chunk_line_10_5_max",
            kind="bm25_chunk",
            params={
                "config": {"k1": 1.5, "b": 0.75, **legal_phrase_config},
                "chunk": {"unit": "line", "size": 10, "overlap": 5},
                "aggregation": "max",
                "rank_depth": 120,
            },
            priority="P1",
            description="Line-window BM25 chunks with legal lemmas and adjacent lemma bigrams.",
        ),
        ExperimentSpec(
            name="bm25_chunk_line_8_4_top2",
            kind="bm25_chunk",
            params={
                "config": {"k1": 1.5, "b": 0.75},
                "chunk": {"unit": "line", "size": 8, "overlap": 4},
                "aggregation": "top2_mean",
                "rank_depth": 120,
            },
            priority="P0",
            description="Alternative line-window chunks with top-2 mean aggregation.",
        ),
        ExperimentSpec(
            name="rrf_bm25_doc_chunk",
            kind="rrf",
            params={
                "members": ["bm25_doc", "bm25_chunk_line_10_5_max"],
                "rrf_k": 60,
                "rank_depth": 100,
            },
            priority="P0",
            description="RRF fusion of document BM25 and chunk BM25.",
        ),
        ExperimentSpec(
            name="rrf_legal_lemma_doc_chunk",
            kind="rrf",
            params={
                "members": ["bm25_legal_lemma_doc", "bm25_legal_lemma_chunk_line_10_5_max"],
                "rrf_k": 60,
                "rank_depth": 100,
            },
            priority="P0",
            description="RRF fusion of legal-aware lemmatized full-document and chunk BM25.",
        ),
        ExperimentSpec(
            name="rrf_legal_phrase_doc_chunk",
            kind="rrf",
            params={
                "members": ["bm25_legal_phrase_doc", "bm25_legal_phrase_chunk_line_10_5_max"],
                "rrf_k": 60,
                "rank_depth": 100,
            },
            priority="P1",
            description="RRF fusion of legal-aware phrase BM25 over documents and chunks.",
        ),
        ExperimentSpec(
            name="rrf_sparse_doc_chunk_char",
            kind="rrf",
            params={
                "members": ["bm25_doc", "bm25_chunk_line_10_5_max", "tfidf_char_doc_3_5"],
                "rrf_k": 60,
                "rank_depth": 100,
            },
            priority="P1",
            description="Sparse fusion: BM25 full-doc + BM25 chunks + char TF-IDF.",
        ),
        ExperimentSpec(
            name="rrf_sparse_legal_lemma_char",
            kind="rrf",
            params={
                "members": [
                    "bm25_legal_lemma_doc",
                    "bm25_legal_lemma_chunk_line_10_5_max",
                    "tfidf_char_doc_3_5",
                ],
                "rrf_k": 60,
                "rank_depth": 100,
            },
            priority="P0",
            description="Sparse fusion: legal-aware lemmatized BM25 doc/chunk plus char TF-IDF.",
        ),
        ExperimentSpec(
            name="rrf_sparse_legal_lemma_word_char",
            kind="rrf",
            params={
                "members": [
                    "bm25_legal_lemma_doc",
                    "bm25_legal_lemma_chunk_line_10_5_max",
                    "tfidf_word_legal_lemma_doc",
                    "tfidf_char_doc_3_5",
                ],
                "rrf_k": 60,
                "rank_depth": 100,
            },
            priority="P1",
            description="Sparse fusion with legal-aware BM25, legal-aware word TF-IDF, and char TF-IDF.",
        ),
        ExperimentSpec(
            name="rrf_sparse_doc_chunk_char_field",
            kind="rrf",
            params={
                "members": ["bm25_doc", "bm25_chunk_line_10_5_max", "tfidf_char_doc_3_5", "bm25_field_aware_doc"],
                "rrf_k": 60,
                "rank_depth": 100,
            },
            priority="P1",
            description="Sparse fusion with an additional field-aware BM25 branch.",
        ),
    ]
    if include_optional:
        specs.extend(
            [
                ExperimentSpec(
                    name="dense_e5_chunk_line_10_5",
                    kind="dense_chunk",
                    params={
                        "config": {"model_name": "intfloat/multilingual-e5-base", "batch_size": 32},
                        "chunk": {"unit": "line", "size": 10, "overlap": 5},
                        "aggregation": "max",
                        "rank_depth": 120,
                    },
                    priority="P2",
                    description="Optional multilingual E5 dense chunk retriever.",
                ),
                ExperimentSpec(
                    name="rrf_bm25_dense",
                    kind="rrf",
                    params={
                        "members": ["bm25_doc", "bm25_chunk_line_10_5_max", "dense_e5_chunk_line_10_5"],
                        "rrf_k": 60,
                        "rank_depth": 100,
                    },
                    priority="P2",
                    description="Optional hybrid sparse+dense RRF.",
                ),
            ]
        )
    return specs


def recall_candidate_experiments(
    *,
    include_optional: bool = False,
    include_bge_m3: bool = False,
) -> list[ExperimentSpec]:
    """Recall@20/50-oriented candidate-generation experiments.

    These are intentionally separate from `default_experiments` so older
    notebooks and reports keep their exact experiment suite. Pass the returned
    specs to `run_suite(..., extra_experiments=...)`.
    """

    specs: list[ExperimentSpec] = []

    for depth in (300, 600, 1000):
        specs.append(
            _bm25_chunk_spec(
                name=f"bm25_legal_lemma_chunk_line_10_5_max_rd{depth}",
                unit="line",
                size=10,
                overlap=5,
                aggregation="max",
                rank_depth=depth,
                priority="P0",
                description=f"Current legal line 10/5 chunk BM25 with deeper chunk rank_depth={depth}.",
            )
        )

    chunk_views = [
        ("line", 6, 3, "line_6_3"),
        ("line", 8, 4, "line_8_4"),
        ("line", 12, 6, "line_12_6"),
        ("line", 16, 8, "line_16_8"),
        ("char", 1200, 600, "char_1200_600"),
        ("char", 1600, 800, "char_1600_800"),
        ("char", 2000, 1000, "char_2000_1000"),
        ("paragraph", 1, 0, "paragraph_1_0"),
        ("paragraph", 2, 1, "paragraph_2_1"),
    ]
    aggregations = ("max", "top2_mean", "max_plus_second", "softmax_top3")
    for unit, size, overlap, label in chunk_views:
        for aggregation in aggregations:
            specs.append(
                _bm25_chunk_spec(
                    name=f"bm25_legal_lemma_chunk_{label}_{_short_aggregation_name(aggregation)}_rd600",
                    unit=unit,
                    size=size,
                    overlap=overlap,
                    aggregation=aggregation,
                    rank_depth=600,
                    priority="P0" if unit == "line" and aggregation == "max" else "P1",
                )
            )

    doc_grid = [(k1, b) for k1 in (1.2, 1.5, 1.8) for b in (0.5, 0.75, 0.9)]
    chunk_grid = [(k1, b) for k1 in (0.9, 1.2, 1.5) for b in (0.2, 0.5, 0.75)]
    for k1, b in doc_grid:
        if (k1, b) == (1.5, 0.75):
            continue
        specs.append(
            ExperimentSpec(
                name=f"bm25_legal_lemma_doc_k1{_float_label(k1)}_b{_float_label(b)}",
                kind="bm25_doc",
                params={"config": _legal_bm25_config(k1=k1, b=b)},
                priority="P1",
                description=f"Legal-aware BM25 over full documents with k1={k1}, b={b}.",
            )
        )
    for k1, b in chunk_grid:
        if (k1, b) == (1.5, 0.75):
            continue
        specs.append(
            _bm25_chunk_spec(
                name=f"bm25_legal_lemma_chunk_line_10_5_k1{_float_label(k1)}_b{_float_label(b)}_rd600",
                unit="line",
                size=10,
                overlap=5,
                aggregation="max",
                rank_depth=600,
                k1=k1,
                b=b,
                priority="P1",
            )
        )

    specs.extend(
        [
            ExperimentSpec(
                name="rrf_sparse_deep_legal_lemma_char",
                kind="rrf",
                params={
                    "members": [
                        "bm25_legal_lemma_doc",
                        "bm25_legal_lemma_chunk_line_10_5_max_rd600",
                        "tfidf_char_doc_3_5",
                    ],
                    "rrf_k": 60,
                    "rank_depth": 100,
                },
                priority="P0",
                description="RRF with deeper legal line chunk BM25 and char TF-IDF.",
            ),
            ExperimentSpec(
                name="rrf_sparse_multichunk_legal_char",
                kind="rrf",
                params={
                    "members": [
                        "bm25_legal_lemma_doc",
                        "bm25_legal_lemma_chunk_line_6_3_max_rd600",
                        "bm25_legal_lemma_chunk_line_10_5_max_rd600",
                        "bm25_legal_lemma_chunk_line_16_8_max_rd600",
                        "bm25_legal_lemma_chunk_char_1600_800_max_rd600",
                        "bm25_legal_lemma_chunk_paragraph_2_1_max_rd600",
                        "tfidf_char_doc_3_5",
                    ],
                    "rrf_k": 60,
                    "rank_depth": 100,
                },
                priority="P0",
                description="RRF across several sparse document and chunk views for broader candidate recall.",
            ),
            ExperimentSpec(
                name="quota_sparse_legal_lemma_char_q5",
                kind="quota_rrf",
                params={
                    "members": [
                        "bm25_legal_lemma_doc",
                        "bm25_legal_lemma_chunk_line_10_5_max_rd600",
                        "tfidf_char_doc_3_5",
                    ],
                    "quota": 5,
                    "rrf_k": 60,
                    "rank_depth": 100,
                    "member_rank_depth": 100,
                },
                priority="P0",
                description="Quota/union fusion over the current strong sparse branches.",
            ),
            ExperimentSpec(
                name="quota_sparse_legal_lemma_char_q10",
                kind="quota_rrf",
                params={
                    "members": [
                        "bm25_legal_lemma_doc",
                        "bm25_legal_lemma_chunk_line_10_5_max_rd600",
                        "tfidf_char_doc_3_5",
                    ],
                    "quota": 10,
                    "rrf_k": 60,
                    "rank_depth": 100,
                    "member_rank_depth": 100,
                },
                priority="P0",
                description="Wider quota/union fusion over the current strong sparse branches.",
            ),
            ExperimentSpec(
                name="quota_sparse_multichunk_q8",
                kind="quota_rrf",
                params={
                    "members": [
                        "bm25_legal_lemma_doc",
                        "bm25_legal_lemma_chunk_line_6_3_max_rd600",
                        "bm25_legal_lemma_chunk_line_10_5_max_rd600",
                        "bm25_legal_lemma_chunk_line_16_8_max_rd600",
                        "bm25_legal_lemma_chunk_char_1600_800_max_rd600",
                        "bm25_legal_lemma_chunk_paragraph_2_1_max_rd600",
                        "tfidf_char_doc_3_5",
                    ],
                    "quota": 8,
                    "rrf_k": 60,
                    "rank_depth": 100,
                    "member_rank_depth": 100,
                },
                priority="P0",
                description="Quota/union fusion across several sparse chunk views.",
            ),
        ]
    )

    if include_optional:
        specs.extend(
            [
                ExperimentSpec(
                    name="dense_e5_chunk_line_10_5_rd600",
                    kind="dense_chunk",
                    params={
                        "config": {"model_name": "intfloat/multilingual-e5-base", "batch_size": 32},
                        "chunk": {"unit": "line", "size": 10, "overlap": 5},
                        "aggregation": "max",
                        "rank_depth": 600,
                    },
                    priority="P1",
                    description="E5-base dense retrieval over line chunks for candidate recall.",
                ),
                ExperimentSpec(
                    name="dense_e5_chunk_char_1600_800_rd600",
                    kind="dense_chunk",
                    params={
                        "config": {"model_name": "intfloat/multilingual-e5-base", "batch_size": 32},
                        "chunk": {"unit": "char", "size": 1600, "overlap": 800},
                        "aggregation": "max",
                        "rank_depth": 600,
                    },
                    priority="P1",
                    description="E5-base dense retrieval over char chunks for candidate recall.",
                ),
                ExperimentSpec(
                    name="rrf_sparse_e5_line",
                    kind="rrf",
                    params={
                        "members": [
                            "bm25_legal_lemma_doc",
                            "bm25_legal_lemma_chunk_line_10_5_max_rd600",
                            "tfidf_char_doc_3_5",
                            "dense_e5_chunk_line_10_5_rd600",
                        ],
                        "rrf_k": 60,
                        "rank_depth": 100,
                    },
                    priority="P1",
                    description="Hybrid sparse + E5 dense RRF candidate generator.",
                ),
                ExperimentSpec(
                    name="quota_sparse_e5_line_q8",
                    kind="quota_rrf",
                    params={
                        "members": [
                            "bm25_legal_lemma_doc",
                            "bm25_legal_lemma_chunk_line_10_5_max_rd600",
                            "tfidf_char_doc_3_5",
                            "dense_e5_chunk_line_10_5_rd600",
                        ],
                        "quota": 8,
                        "rrf_k": 60,
                        "rank_depth": 100,
                        "member_rank_depth": 100,
                    },
                    priority="P1",
                    description="Quota/union hybrid sparse + E5 candidate generator.",
                ),
            ]
        )
        if include_bge_m3:
            specs.extend(
                [
                    ExperimentSpec(
                        name="dense_bge_m3_chunk_line_10_5_rd600",
                        kind="dense_chunk",
                        params={
                            "config": {"model_name": "BAAI/bge-m3", "batch_size": 16},
                            "chunk": {"unit": "line", "size": 10, "overlap": 5},
                            "aggregation": "max",
                            "rank_depth": 600,
                        },
                        priority="P2",
                        description="BGE-M3 dense retrieval over line chunks.",
                    ),
                    ExperimentSpec(
                        name="rrf_sparse_bge_m3_line",
                        kind="rrf",
                        params={
                            "members": [
                                "bm25_legal_lemma_doc",
                                "bm25_legal_lemma_chunk_line_10_5_max_rd600",
                                "tfidf_char_doc_3_5",
                                "dense_bge_m3_chunk_line_10_5_rd600",
                            ],
                            "rrf_k": 60,
                            "rank_depth": 100,
                        },
                        priority="P2",
                        description="Hybrid sparse + BGE-M3 dense RRF candidate generator.",
                    ),
                ]
            )

    return _unique_specs(specs)


def _legal_bm25_config(k1: float = 1.5, b: float = 0.75, *, add_bigrams: bool = False) -> dict[str, Any]:
    config: dict[str, Any] = {
        "k1": k1,
        "b": b,
        "lemmatize": True,
        "preserve_legal_refs": True,
        "legal_stop_words": True,
        "min_len": 2,
    }
    if add_bigrams:
        config["add_bigrams"] = True
    return config


def _bm25_chunk_spec(
    *,
    name: str,
    unit: str,
    size: int,
    overlap: int,
    aggregation: str = "max",
    rank_depth: int = 600,
    k1: float = 1.5,
    b: float = 0.75,
    priority: str = "P0",
    description: str | None = None,
) -> ExperimentSpec:
    return ExperimentSpec(
        name=name,
        kind="bm25_chunk",
        params={
            "config": _legal_bm25_config(k1=k1, b=b),
            "chunk": {"unit": unit, "size": size, "overlap": overlap},
            "aggregation": aggregation,
            "rank_depth": rank_depth,
        },
        priority=priority,
        description=description
        or f"Legal-aware BM25 over {unit} chunks {size}/{overlap}, {aggregation}, rank_depth={rank_depth}.",
    )


def _float_label(value: float) -> str:
    return str(value).replace(".", "p")


def _short_aggregation_name(aggregation: str) -> str:
    return {
        "max": "max",
        "top2_mean": "top2",
        "max_plus_second": "max2",
        "softmax_top3": "softmax3",
    }[aggregation]


def _unique_specs(specs: list[ExperimentSpec]) -> list[ExperimentSpec]:
    seen: set[str] = set()
    result: list[ExperimentSpec] = []
    for spec in specs:
        if spec.name in seen:
            continue
        seen.add(spec.name)
        result.append(spec)
    return result


def _merge_specs(base: list[ExperimentSpec], extra: list[ExperimentSpec] | None = None) -> list[ExperimentSpec]:
    by_name: dict[str, ExperimentSpec] = {}
    for spec in [*base, *(extra or [])]:
        by_name.setdefault(spec.name, spec)
    return list(by_name.values())


def run_suite(
    *,
    data_dir: str | Path,
    output_dir: str | Path | None = None,
    experiment_names: list[str] | None = None,
    extra_experiments: list[ExperimentSpec] | None = None,
    mode: str = "holdout",
    include_optional: bool = False,
    seed: int = 42,
    n_splits: int = 5,
    eval_depth: int = DEFAULT_EVAL_DEPTH,
    eval_ks: tuple[int, ...] | None = None,
    run_id: str | None = None,
    comparison_baseline: str = DEFAULT_COMPARISON_BASELINE,
) -> pd.DataFrame:
    paths = PathConfig.from_root(data_dir)
    paths.ensure_dirs()
    data = load_data(paths.root)
    output = Path(output_dir) if output_dir is not None else paths.reports_dir
    output.mkdir(parents=True, exist_ok=True)
    metrics_dir = output / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    eval_ks = tuple(eval_ks or EVAL_KS)
    eval_depth = max(eval_depth, max(eval_ks))

    all_specs = _merge_specs(default_experiments(include_optional=include_optional), extra_experiments)
    specs = all_specs
    if experiment_names:
        selected = set(experiment_names)
        specs = [spec for spec in all_specs if spec.name in selected]
        missing = selected.difference({spec.name for spec in all_specs})
        if missing:
            raise ValueError(f"Unknown experiment names: {sorted(missing)}")

    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    records: list[dict[str, Any]] = []
    query_records: list[dict[str, Any]] = []
    metrics_path = metrics_dir / f"{run_id}.jsonl"

    for split, eval_frames in _make_eval_folds(data.train, mode=mode, seed=seed, n_splits=n_splits):
        combined_eval = pd.concat(
            [
                frame.assign(_eval_part=eval_part, _eval_order=range(len(frame)))
                for eval_part, frame in eval_frames
            ],
            ignore_index=True,
        )
        queries = combined_eval["question"].astype(str).tolist()
        cache: dict[str, list[list[SearchResult]]] = {}
        for spec in specs:
            started = datetime.now(timezone.utc)
            base_record = {
                "run_id": run_id,
                "timestamp_utc": started.isoformat(),
                "split": split.name,
                "mode": mode,
                "experiment": spec.name,
                "priority": spec.priority,
                "description": spec.description,
                "params": _jsonable(spec.params),
            }
            try:
                ranking_depth = max(eval_depth, int(spec.params.get("rank_depth", DEFAULT_EVAL_DEPTH)))
                rankings = rank_queries(
                    spec,
                    all_specs,
                    data.documents,
                    queries,
                    top_k=ranking_depth,
                    cache=cache,
                )
                for eval_part, frame in eval_frames:
                    mask = combined_eval["_eval_part"].eq(eval_part).to_numpy()
                    part_rankings = [ranking for ranking, keep in zip(rankings, mask, strict=True) if keep]
                    predictions = [[item.doc_id for item in ranking[:ranking_depth]] for ranking in part_rankings]
                    gold = frame["gold_doc_id"].astype(str).tolist()
                    qids = frame["qid"].astype(str).tolist()
                    record = {
                        **base_record,
                        "eval_part": eval_part,
                        "n_eval": len(frame),
                        "status": "ok",
                    }
                    record.update(evaluate_predictions(gold, predictions, ks=eval_ks))
                    record["duration_sec"] = (datetime.now(timezone.utc) - started).total_seconds()
                    records.append(record)
                    query_records.extend(
                        _query_hit_records(
                            base_record,
                            eval_part=eval_part,
                            qids=qids,
                            gold=gold,
                            predictions=predictions,
                            eval_ks=eval_ks,
                        )
                    )
                    with metrics_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")
            except Exception as exc:  # noqa: BLE001 - experiment failures should be logged, not hide previous metrics.
                for eval_part, frame in eval_frames:
                    record = {
                        **base_record,
                        "eval_part": eval_part,
                        "n_eval": len(frame),
                        "status": "failed",
                        "error": repr(exc),
                        "duration_sec": (datetime.now(timezone.utc) - started).total_seconds(),
                    }
                    records.append(record)
                    with metrics_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")

    raw = pd.DataFrame(records)
    query_hits = pd.DataFrame(query_records)
    summary = aggregate_validation_records(
        raw,
        query_hits=query_hits,
        comparison_baseline=comparison_baseline,
        metric_columns=[f"recall@{k}" for k in eval_ks],
    )
    raw_path = output / f"folds_{run_id}.csv"
    query_hits_path = output / f"query_hits_{run_id}.csv"
    summary_path = output / f"summary_{run_id}.csv"
    latest_path = output / "summary_latest.csv"
    raw.to_csv(raw_path, index=False)
    query_hits.to_csv(query_hits_path, index=False)
    summary.to_csv(summary_path, index=False)
    summary.to_csv(latest_path, index=False)
    return summary


def aggregate_validation_records(
    raw: pd.DataFrame,
    *,
    query_hits: pd.DataFrame | None = None,
    comparison_baseline: str = DEFAULT_COMPARISON_BASELINE,
    metric_columns: list[str] | None = None,
) -> pd.DataFrame:
    if raw.empty:
        return raw
    metric_columns = metric_columns or [metric for metric in METRIC_COLUMNS if metric in raw.columns]

    rows: list[dict[str, Any]] = []
    for experiment, group in raw.groupby("experiment", sort=False):
        first = group.iloc[0]
        ok_group = group[group["status"].eq("ok")]
        row: dict[str, Any] = {
            "run_id": first.get("run_id"),
            "mode": first.get("mode"),
            "experiment": experiment,
            "priority": first.get("priority"),
            "description": first.get("description"),
            "status": "failed" if group["status"].ne("ok").any() else "ok",
            "n_splits": group["split"].nunique(),
            "params": first.get("params"),
            "duration_sec": group["duration_sec"].max() if "duration_sec" in group else None,
        }
        for eval_part in EVAL_PART_ORDER:
            part = ok_group[ok_group["eval_part"].eq(eval_part)]
            if part.empty:
                continue
            row[f"{eval_part}_n_eval_mean"] = part["n_eval"].mean()
            row[f"{eval_part}_n_eval_total"] = part["n_eval"].sum()
            for metric in metric_columns:
                if metric not in part.columns:
                    continue
                row[f"{eval_part}_{metric}_mean"] = part[metric].mean()
                metric_std = part[metric].std(ddof=1) if len(part) > 1 else pd.NA
                row[f"{eval_part}_{metric}_std"] = metric_std
                row[f"{eval_part}_{metric}_se"] = metric_std / math.sqrt(len(part)) if len(part) > 1 else pd.NA
                row[f"{eval_part}_{metric}_micro"] = _weighted_metric(part, metric)
            if query_hits is not None and not query_hits.empty:
                row.update(
                    _paired_comparison_summary(
                        query_hits,
                        experiment=experiment,
                        eval_part=eval_part,
                        baseline=comparison_baseline,
                    )
                )
        rows.append(row)
    return pd.DataFrame(rows)


def _weighted_metric(part: pd.DataFrame, metric: str) -> float | Any:
    total = part["n_eval"].sum()
    if total == 0:
        return pd.NA
    return float((part[metric] * part["n_eval"]).sum() / total)


def _query_hit_records(
    base_record: dict[str, Any],
    *,
    eval_part: str,
    qids: list[str],
    gold: list[str],
    predictions: list[list[str]],
    eval_ks: tuple[int, ...] = EVAL_KS,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for qid, expected, predicted in zip(qids, gold, predictions, strict=True):
        record = {
            "run_id": base_record["run_id"],
            "split": base_record["split"],
            "mode": base_record["mode"],
            "experiment": base_record["experiment"],
            "eval_part": eval_part,
            "qid": qid,
            "gold_doc_id": expected,
        }
        for k in eval_ks:
            record[f"hit@{k}"] = int(str(expected) in dedupe_topk(predicted, k))
        records.append(record)
    return records


def _paired_comparison_summary(
    query_hits: pd.DataFrame,
    *,
    experiment: str,
    eval_part: str,
    baseline: str,
) -> dict[str, Any]:
    metric = "hit@5"
    part = query_hits[query_hits["eval_part"].eq(eval_part)]
    current = part[part["experiment"].eq(experiment)]
    base = part[part["experiment"].eq(baseline)]
    if current.empty or base.empty:
        return {}

    keys = ["split", "eval_part", "qid"]
    compared = current[keys + [metric]].merge(
        base[keys + [metric]],
        on=keys,
        suffixes=("", "_baseline"),
        how="inner",
    )
    if compared.empty:
        return {}

    hit = compared[metric].astype(int)
    base_hit = compared[f"{metric}_baseline"].astype(int)
    wins = int(((hit == 1) & (base_hit == 0)).sum())
    losses = int(((hit == 0) & (base_hit == 1)).sum())
    ties = int((hit == base_hit).sum())
    n_compared = int(len(compared))
    prefix = f"{eval_part}_recall@5"
    return {
        f"{prefix}_comparison_baseline": baseline,
        f"{prefix}_delta_vs_baseline": float((hit - base_hit).mean()),
        f"{prefix}_wins_vs_baseline": wins,
        f"{prefix}_losses_vs_baseline": losses,
        f"{prefix}_ties_vs_baseline": ties,
        f"{prefix}_n_compared_vs_baseline": n_compared,
    }


def rank_queries(
    spec: ExperimentSpec,
    all_specs: list[ExperimentSpec],
    documents: pd.DataFrame,
    queries: list[str],
    *,
    top_k: int,
    cache: dict[str, list[list[SearchResult]]] | None = None,
) -> list[list[SearchResult]]:
    cache = cache if cache is not None else {}
    if spec.name in cache:
        return cache[spec.name]

    if spec.kind == "rrf":
        by_name = {item.name: item for item in all_specs}
        member_names = spec.params["members"]
        member_rankings = [
            rank_queries(by_name[name], all_specs, documents, queries, top_k=top_k, cache=cache)
            for name in member_names
        ]
        rankings = []
        for query_rankings in zip(*member_rankings, strict=True):
            rankings.append(
                rrf_fusion(
                    [list(ranking) for ranking in query_rankings],
                    k=int(spec.params.get("rrf_k", 60)),
                    top_k=top_k,
                    source=spec.name,
                )
            )
        cache[spec.name] = rankings
        return rankings

    if spec.kind == "quota_rrf":
        by_name = {item.name: item for item in all_specs}
        member_names = spec.params["members"]
        member_rank_depth = max(top_k, int(spec.params.get("member_rank_depth", spec.params.get("rank_depth", top_k))))
        member_rankings = [
            rank_queries(by_name[name], all_specs, documents, queries, top_k=member_rank_depth, cache=cache)
            for name in member_names
        ]
        per_member_quota = spec.params.get("per_member_quota")
        rankings = []
        for query_rankings in zip(*member_rankings, strict=True):
            rankings.append(
                quota_union_fusion(
                    [list(ranking) for ranking in query_rankings],
                    quota=int(spec.params.get("quota", 8)),
                    per_ranking_quota=per_member_quota,
                    k=int(spec.params.get("rrf_k", 60)),
                    top_k=top_k,
                    source=spec.name,
                )
            )
        cache[spec.name] = rankings
        return rankings

    if spec.kind == "tfidf_doc":
        units = document_units(documents)
        retriever = TfidfRetriever(spec.name, TfidfConfig(**spec.params.get("config", {}))).fit(units)
        rankings = [dedupe_ranked_docs(items, top_k=top_k) for items in retriever.search(queries, top_k=top_k)]
    elif spec.kind == "bm25_doc":
        units = document_units(documents)
        retriever = BM25Retriever(spec.name, BM25Config(**spec.params.get("config", {}))).fit(units)
        rankings = [dedupe_ranked_docs(items, top_k=top_k) for items in retriever.search(queries, top_k=top_k)]
    elif spec.kind == "bm25_field_doc":
        units = document_units(add_field_aware_text(documents))
        retriever = BM25Retriever(spec.name, BM25Config(**spec.params.get("config", {}))).fit(units)
        rankings = [dedupe_ranked_docs(items, top_k=top_k) for items in retriever.search(queries, top_k=top_k)]
    elif spec.kind == "bm25_chunk":
        units = chunk_units(documents, ChunkConfig(**spec.params["chunk"]))
        retriever = BM25Retriever(spec.name, BM25Config(**spec.params.get("config", {}))).fit(units)
        raw_rankings = retriever.search(queries, top_k=int(spec.params.get("rank_depth", top_k)))
        rankings = [
            aggregate_chunk_results(
                items,
                method=str(spec.params.get("aggregation", "max")),
                top_k=top_k,
                source=spec.name,
            )
            for items in raw_rankings
        ]
    elif spec.kind == "dense_chunk":
        units = chunk_units(documents, ChunkConfig(**spec.params["chunk"]))
        retriever = DenseRetriever(spec.name, DenseConfig(**spec.params.get("config", {}))).fit(units)
        raw_rankings = retriever.search(queries, top_k=int(spec.params.get("rank_depth", top_k)))
        rankings = [
            aggregate_chunk_results(
                items,
                method=str(spec.params.get("aggregation", "max")),
                top_k=top_k,
                source=spec.name,
            )
            for items in raw_rankings
        ]
    else:
        raise ValueError(f"Unsupported experiment kind: {spec.kind}")

    cache[spec.name] = rankings
    return rankings


def run_recall_oracle_diagnostics(
    *,
    data_dir: str | Path,
    branch_names: list[str] | None = None,
    baseline_fusion_name: str = "rrf_sparse_legal_lemma_char",
    mode: str = "cv",
    include_optional: bool = False,
    extra_experiments: list[ExperimentSpec] | None = None,
    seed: int = 42,
    n_splits: int = 5,
    depth: int = 50,
    eval_ks: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Diagnose whether Recall@20/50 is limited by branches or fusion."""

    eval_ks = tuple(eval_ks or EVAL_KS)
    depth = max(depth, max(eval_ks))
    branch_names = branch_names or [
        "bm25_legal_lemma_doc",
        "bm25_legal_lemma_chunk_line_10_5_max",
        "tfidf_char_doc_3_5",
    ]
    paths = PathConfig.from_root(data_dir)
    data = load_data(paths.root)
    all_specs = _merge_specs(default_experiments(include_optional=include_optional), extra_experiments)
    by_name = {spec.name: spec for spec in all_specs}
    required = set(branch_names) | {baseline_fusion_name}
    missing = sorted(required.difference(by_name))
    if missing:
        raise ValueError(f"Unknown diagnostic experiments: {missing}")

    rows: list[dict[str, Any]] = []
    for split, eval_frames in _make_eval_folds(data.train, mode=mode, seed=seed, n_splits=n_splits):
        combined_eval = pd.concat(
            [
                frame.assign(_eval_part=eval_part, _eval_order=range(len(frame)))
                for eval_part, frame in eval_frames
            ],
            ignore_index=True,
        )
        queries = combined_eval["question"].astype(str).tolist()
        cache: dict[str, list[list[SearchResult]]] = {}
        ranking_names = [*branch_names, baseline_fusion_name]
        rankings_by_name = {
            name: rank_queries(by_name[name], all_specs, data.documents, queries, top_k=depth, cache=cache)
            for name in ranking_names
        }

        for eval_part, frame in eval_frames:
            mask = combined_eval["_eval_part"].eq(eval_part).to_numpy()
            idxs = [idx for idx, keep in enumerate(mask) if keep]
            gold = frame["gold_doc_id"].astype(str).tolist()

            for name in ranking_names:
                predictions = [[item.doc_id for item in rankings_by_name[name][idx][:depth]] for idx in idxs]
                record: dict[str, Any] = {
                    "split": split.name,
                    "eval_part": eval_part,
                    "kind": "branch",
                    "candidate": name,
                    "n_eval": len(gold),
                }
                record.update(evaluate_predictions(gold, predictions, ks=eval_ks))
                rows.append(record)

            for k_eval in [k for k in eval_ks if k >= 20]:
                union_hits = 0
                baseline_misses_union_hits = 0
                for expected, idx in zip(gold, idxs, strict=True):
                    branch_hit = any(
                        str(expected) in {item.doc_id for item in rankings_by_name[name][idx][:k_eval]}
                        for name in branch_names
                    )
                    baseline_hit = str(expected) in [
                        item.doc_id for item in rankings_by_name[baseline_fusion_name][idx][:k_eval]
                    ]
                    union_hits += int(branch_hit)
                    baseline_misses_union_hits += int(branch_hit and not baseline_hit)
                rows.append(
                    {
                        "split": split.name,
                        "eval_part": eval_part,
                        "kind": "oracle_union",
                        "candidate": f"oracle_union_top{k_eval}",
                        "n_eval": len(gold),
                        f"recall@{k_eval}": union_hits / max(1, len(gold)),
                        f"lost_by_{baseline_fusion_name}@{k_eval}": baseline_misses_union_hits,
                    }
                )

    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw
    rows_out: list[dict[str, Any]] = []
    metric_cols = [
        col
        for col in raw.columns
        if col.startswith("recall@") or col.startswith("lost_by_")
    ]
    for (kind, candidate), group in raw.groupby(["kind", "candidate"], sort=False):
        row: dict[str, Any] = {
            "kind": kind,
            "candidate": candidate,
            "n_splits": group["split"].nunique(),
            "n_eval_total": group["n_eval"].sum(),
        }
        for col in metric_cols:
            values = group[col].dropna()
            if values.empty:
                continue
            if col.startswith("lost_by_"):
                row[f"{col}_total"] = int(values.sum())
            else:
                row[f"{col}_mean"] = values.mean()
                row[f"{col}_micro"] = float((group.loc[values.index, col] * group.loc[values.index, "n_eval"]).sum() / group.loc[values.index, "n_eval"].sum())
        rows_out.append(row)
    return pd.DataFrame(rows_out)


def create_submission(
    *,
    data_dir: str | Path,
    experiment_name: str,
    output_path: str | Path | None = None,
    include_optional: bool = False,
    extra_experiments: list[ExperimentSpec] | None = None,
    top_k: int = 5,
) -> Path:
    paths = PathConfig.from_root(data_dir)
    paths.ensure_dirs()
    data = load_data(paths.root)
    specs = _merge_specs(default_experiments(include_optional=include_optional), extra_experiments)
    by_name = {spec.name: spec for spec in specs}
    if experiment_name not in by_name:
        raise ValueError(f"Unknown experiment name: {experiment_name}")
    rankings = rank_queries(
        by_name[experiment_name],
        specs,
        data.documents,
        data.test["question"].astype(str).tolist(),
        top_k=max(top_k, int(by_name[experiment_name].params.get("rank_depth", 100))),
    )
    predictions = [[item.doc_id for item in ranking[:top_k]] for ranking in rankings]
    output_path = output_path or paths.submissions_dir / f"submission_{experiment_name}.csv"
    return write_submission(
        data.test["qid"].astype(str).tolist(),
        predictions,
        output_path,
        test=data.test,
        documents=data.documents,
        top_k=top_k,
    )


def select_best_experiment(summary: pd.DataFrame, *, metric: str | None = None) -> str:
    ok = summary[summary["status"].eq("ok")].copy()
    if ok.empty:
        raise ValueError("No successful experiments in summary")
    metric = metric or _default_selection_metric(ok)
    if metric not in ok.columns:
        raise ValueError(f"Metric column not found in summary: {metric}")
    return str(ok.sort_values(metric, ascending=False).iloc[0]["experiment"])


def _default_selection_metric(summary: pd.DataFrame) -> str:
    for metric in ("holdout_recall@5_mean", "valid_recall@5_mean", "train_recall@5_mean"):
        if metric in summary.columns:
            return metric
    raise ValueError("No recall@5 metric found in summary")


def document_units(documents: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_id": documents["doc_id"].astype(str),
            "doc_id": documents["doc_id"].astype(str),
            "text": documents["text"].fillna("").astype(str),
        }
    )


def chunk_units(documents: pd.DataFrame, config: ChunkConfig) -> pd.DataFrame:
    chunks = build_chunk_index(documents, config)
    return chunks.rename(columns={"chunk_id": "unit_id"})[["unit_id", "doc_id", "text"]]


def _make_eval_folds(
    train: pd.DataFrame,
    *,
    mode: str,
    seed: int,
    n_splits: int,
) -> list[tuple[Split, list[tuple[str, pd.DataFrame]]]]:
    if mode == "holdout":
        split = make_group_holdout(train, seed=seed)
        _, holdout_df = _materialize_split(train, split)
        return [(split, [("holdout", holdout_df)])]
    if mode == "cv":
        outer_split = make_group_holdout(train, seed=seed)
        cv_pool = train.iloc[outer_split.train_idx].reset_index(drop=True)
        folds: list[tuple[Split, list[tuple[str, pd.DataFrame]]]] = []
        for split in make_group_kfold(cv_pool, n_splits=n_splits):
            _, valid_df = _materialize_split(cv_pool, split)
            folds.append((split, [("valid", valid_df)]))
        return folds
    if mode == "train":
        idx = list(range(len(train)))
        return [(Split("train_all", idx, idx), [("train", train.reset_index(drop=True))])]
    raise ValueError("mode must be one of: holdout, cv, train")


def _materialize_split(train: pd.DataFrame, split: Split) -> tuple[pd.DataFrame, pd.DataFrame]:
    return train.iloc[split.train_idx].reset_index(drop=True), train.iloc[split.valid_idx].reset_index(drop=True)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    return value
