from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from legal_hse.chunking import ChunkConfig, build_chunk_index
from legal_hse.config import PathConfig
from legal_hse.data import load_data
from legal_hse.fusion import aggregate_chunk_results, dedupe_ranked_docs, rrf_fusion
from legal_hse.features import add_field_aware_text
from legal_hse.metrics import evaluate_predictions
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


def run_suite(
    *,
    data_dir: str | Path,
    output_dir: str | Path | None = None,
    experiment_names: list[str] | None = None,
    mode: str = "holdout",
    include_optional: bool = False,
    seed: int = 42,
    n_splits: int = 5,
    eval_depth: int = DEFAULT_EVAL_DEPTH,
    run_id: str | None = None,
) -> pd.DataFrame:
    paths = PathConfig.from_root(data_dir)
    paths.ensure_dirs()
    data = load_data(paths.root)
    output = Path(output_dir) if output_dir is not None else paths.reports_dir
    output.mkdir(parents=True, exist_ok=True)
    metrics_dir = output / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    specs = default_experiments(include_optional=include_optional)
    if experiment_names:
        selected = set(experiment_names)
        specs = [spec for spec in specs if spec.name in selected]
        missing = selected.difference({spec.name for spec in specs})
        if missing:
            raise ValueError(f"Unknown experiment names: {sorted(missing)}")

    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    records: list[dict[str, Any]] = []
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
                    specs,
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
                    record = {
                        **base_record,
                        "eval_part": eval_part,
                        "n_eval": len(frame),
                        "status": "ok",
                    }
                    record.update(evaluate_predictions(gold, predictions, ks=EVAL_KS))
                    record["duration_sec"] = (datetime.now(timezone.utc) - started).total_seconds()
                    records.append(record)
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
    summary = aggregate_validation_records(raw)
    raw_path = output / f"folds_{run_id}.csv"
    summary_path = output / f"summary_{run_id}.csv"
    latest_path = output / "summary_latest.csv"
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    summary.to_csv(latest_path, index=False)
    return summary


def aggregate_validation_records(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw

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
            for metric in METRIC_COLUMNS:
                row[f"{eval_part}_{metric}_mean"] = part[metric].mean()
                row[f"{eval_part}_{metric}_std"] = part[metric].std(ddof=1) if len(part) > 1 else pd.NA
        rows.append(row)
    return pd.DataFrame(rows)


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


def create_submission(
    *,
    data_dir: str | Path,
    experiment_name: str,
    output_path: str | Path | None = None,
    include_optional: bool = False,
    top_k: int = 5,
) -> Path:
    paths = PathConfig.from_root(data_dir)
    paths.ensure_dirs()
    data = load_data(paths.root)
    specs = default_experiments(include_optional=include_optional)
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
