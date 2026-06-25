from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from legal_hse.chunking import ChunkConfig
from legal_hse.data import load_data
from legal_hse.experiments import (
    aggregate_validation_records,
    chunk_units,
    default_experiments,
    rank_queries,
    recall_candidate_experiments,
    select_best_experiment,
)
from legal_hse.metrics import dedupe_topk, evaluate_predictions
from legal_hse.rerankers.cross_encoder import CrossEncoderConfig, CrossEncoderReranker
from legal_hse.rerankers.flag import FlagEmbeddingReranker, FlagRerankerConfig
from legal_hse.retrievers.base import SearchResult
from legal_hse.retrievers.bm25 import BM25Config, BM25Retriever
from legal_hse.splits import make_group_holdout, make_group_kfold
from legal_hse.submission import write_submission


DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


@dataclass(frozen=True)
class RerankSuiteConfig:
    mode: str = "cv"
    seed: int = 42
    n_splits: int = 5
    eval_ks: tuple[int, ...] = (5, 10, 20)
    candidate_depth: int = 100
    depths: tuple[int, ...] = (20, 50, 100)
    chunks_per_doc: tuple[int, ...] = (1, 2)
    chunk_aggs: tuple[str, ...] = ("max", "top2_mean")
    score_modes: tuple[str, ...] = ("ce", "ce_plus_candidate")
    model_names: tuple[str, ...] = (DEFAULT_RERANK_MODEL,)
    batch_size: int = 16
    max_length: int = 512
    device: str | None = None
    chunk_search_depth: int = 2500
    pair_char_limit: int = 3500
    fallback_doc_chars: int = 2500
    create_submission: bool = True
    enable_e5_candidates: bool = False
    enable_bge_m3: bool = False
    candidate_experiments: tuple[str, ...] | None = None
    output_dir: str | Path | None = None
    run_id: str | None = None


@dataclass(frozen=True)
class RerankConfig:
    name: str
    candidate_experiment: str
    model_name: str
    depth: int
    chunks_per_doc: int
    chunk_agg: str
    score_mode: str


@dataclass(frozen=True)
class RerankSuiteResult:
    summary: pd.DataFrame
    raw: pd.DataFrame
    query_hits: pd.DataFrame
    best_rerank_experiment: str
    best_rerank_or_candidate: str
    summary_path: Path
    raw_path: Path
    query_hits_path: Path
    latest_path: Path
    submission_path: Path | None = None


def config_from_globals(namespace: Mapping[str, Any]) -> RerankSuiteConfig:
    """Build a suite config from Colab/global variables.

    This keeps notebooks as launch surfaces: users can override any
    `RERANK_*` variable before running the cell without keeping the experiment
    implementation in the notebook itself.
    """

    return RerankSuiteConfig(
        mode=str(namespace.get("RERANK_MODE", "cv")),
        seed=int(namespace.get("RERANK_SEED", 42)),
        n_splits=int(namespace.get("RERANK_N_SPLITS", 5)),
        eval_ks=tuple(namespace.get("RERANK_EVAL_KS", (5, 10, 20))),
        candidate_depth=int(namespace.get("RERANK_CANDIDATE_DEPTH", 100)),
        depths=tuple(namespace.get("RERANK_DEPTHS", (20, 50, 100))),
        chunks_per_doc=tuple(namespace.get("RERANK_CHUNKS_PER_DOC", (1, 2))),
        chunk_aggs=tuple(namespace.get("RERANK_CHUNK_AGGS", ("max", "top2_mean"))),
        score_modes=tuple(namespace.get("RERANK_SCORE_MODES", ("ce", "ce_plus_candidate"))),
        model_names=tuple(_as_list(namespace.get("RERANK_MODEL_NAMES", (DEFAULT_RERANK_MODEL,)))),
        batch_size=int(namespace.get("RERANK_BATCH_SIZE", 16)),
        max_length=int(namespace.get("RERANK_MAX_LENGTH", 512)),
        device=namespace.get("RERANK_DEVICE", None),
        chunk_search_depth=int(namespace.get("RERANK_CHUNK_SEARCH_DEPTH", 2500)),
        pair_char_limit=int(namespace.get("RERANK_PAIR_CHAR_LIMIT", 3500)),
        fallback_doc_chars=int(namespace.get("RERANK_FALLBACK_DOC_CHARS", 2500)),
        create_submission=bool(namespace.get("RERANK_CREATE_SUBMISSION", True)),
        enable_e5_candidates=bool(namespace.get("RERANK_ENABLE_E5_CANDIDATES", False)),
        enable_bge_m3=bool(namespace.get("RERANK_ENABLE_BGE_M3", False)),
        candidate_experiments=(
            tuple(_as_list(namespace["RERANK_CANDIDATE_EXPERIMENTS"]))
            if "RERANK_CANDIDATE_EXPERIMENTS" in namespace
            else None
        ),
        output_dir=namespace.get("RERANK_OUTPUT_DIR", None),
        run_id=namespace.get("RERANK_RUN_ID", None),
    )


def default_rerank_candidate_experiments(config: RerankSuiteConfig) -> tuple[str, ...]:
    candidates = [
        "rrf_sparse_deep_legal_lemma_char",
        "quota_sparse_legal_lemma_char_q10",
    ]
    if config.enable_e5_candidates:
        candidates += ["rrf_sparse_e5_line", "quota_sparse_e5_line_q8"]
    if config.enable_bge_m3:
        candidates += ["rrf_sparse_bge_m3_native_line", "quota_sparse_bge_m3_native_line_q8"]
    return tuple(candidates)


def run_rerank_suite(data_dir: str | Path, config: RerankSuiteConfig | None = None) -> RerankSuiteResult:
    config = config or RerankSuiteConfig()
    config = _normalized_config(config)

    data_dir = Path(data_dir).expanduser().resolve()
    data = load_data(data_dir)
    output_dir = Path(config.output_dir) if config.output_dir is not None else data_dir / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    extra_specs = recall_candidate_experiments(
        include_optional=config.enable_e5_candidates,
        include_bge_m3=config.enable_bge_m3,
    )
    all_specs = _merge_specs(default_experiments(include_optional=config.enable_e5_candidates), extra_specs)
    specs_by_name = {spec.name: spec for spec in all_specs}
    candidate_experiments = config.candidate_experiments or default_rerank_candidate_experiments(config)
    missing = sorted(set(candidate_experiments).difference(specs_by_name))
    if missing:
        raise ValueError(f"Unknown rerank candidate experiments: {missing}. Enable E5/BGE flags if needed.")

    chunk_selector_units = chunk_units(data.documents, ChunkConfig(unit="line", size=10, overlap=5))
    chunk_selector = BM25Retriever(
        "rerank_bm25_legal_lemma_chunk_line_10_5_raw",
        BM25Config(**_legal_bm25_config()),
    ).fit(chunk_selector_units)
    chunk_search_depth = min(config.chunk_search_depth, len(chunk_selector_units))
    doc_text_by_id = data.documents.set_index("doc_id")["text"].fillna("").astype(str).to_dict()

    run_id = config.run_id or datetime.now(timezone.utc).strftime("rerank_%Y%m%dT%H%M%SZ")
    records: list[dict[str, Any]] = []
    query_hit_rows: list[dict[str, Any]] = []
    rerankers: dict[str, Any] = {}
    configs_by_name: dict[str, RerankConfig] = {}

    print("Rerank suite")
    print("mode:", config.mode)
    print("candidate experiments:", list(candidate_experiments))
    print("models:", list(config.model_names))
    print("depths:", config.depths, "chunks_per_doc:", config.chunks_per_doc, "score_modes:", config.score_modes)
    print("chunk selector units:", len(chunk_selector_units), "search_depth:", chunk_search_depth)

    for split_name, eval_part, eval_frame in _make_rerank_eval_folds(
        data.train,
        mode=config.mode,
        seed=config.seed,
        n_splits=config.n_splits,
    ):
        queries = eval_frame["question"].astype(str).tolist()
        qids = eval_frame["qid"].astype(str).tolist()
        print(f"\nFold {split_name}: {len(queries)} queries")

        chunk_rankings = chunk_selector.search(queries, top_k=chunk_search_depth)
        rank_cache: dict[str, list[list[SearchResult]]] = {}

        for candidate_name in candidate_experiments:
            candidate_started = time.time()
            candidate_rankings = rank_queries(
                specs_by_name[candidate_name],
                all_specs,
                data.documents,
                queries,
                top_k=config.candidate_depth,
                cache=rank_cache,
            )
            baseline_predictions = [
                [item.doc_id for item in ranking[: config.candidate_depth]]
                for ranking in candidate_rankings
            ]
            baseline_name = f"candidate_{_slug(candidate_name, max_len=70)}"
            base_record, base_hits = _metric_record(
                run_id,
                split_name,
                eval_part,
                baseline_name,
                f"No-rerank candidate baseline from {candidate_name}.",
                eval_frame,
                baseline_predictions,
                time.time() - candidate_started,
                {"candidate_experiment": candidate_name, "depth": config.candidate_depth},
                config=config,
            )
            records.append(base_record)
            query_hit_rows.extend(base_hits)

            for model_name in config.model_names:
                print(f"Scoring {candidate_name} with {model_name}")
                pair_started = time.time()
                pair_frame = _score_candidate_pairs(
                    reranker=_get_reranker(model_name, config=config, rerankers=rerankers),
                    queries=queries,
                    qids=qids,
                    candidate_rankings=candidate_rankings,
                    chunk_rankings=chunk_rankings,
                    max_depth=max(config.depths),
                    max_chunks=max(config.chunks_per_doc),
                    doc_text_by_id=doc_text_by_id,
                    config=config,
                )
                scoring_sec = time.time() - pair_started

                for rerank_config in _make_rerank_configs(candidate_name, model_name, config):
                    configs_by_name[rerank_config.name] = rerank_config
                    predictions = _prediction_lists_from_pairs(pair_frame, qids, rerank_config)
                    record, hits = _metric_record(
                        run_id,
                        split_name,
                        eval_part,
                        rerank_config.name,
                        (
                            f"Cross-encoder rerank of {candidate_name}: depth={rerank_config.depth}, "
                            f"chunks={rerank_config.chunks_per_doc}, agg={rerank_config.chunk_agg}, "
                            f"score={rerank_config.score_mode}."
                        ),
                        eval_frame,
                        predictions,
                        scoring_sec,
                        asdict(rerank_config),
                        config=config,
                    )
                    records.append(record)
                    query_hit_rows.extend(hits)

    raw = pd.DataFrame(records)
    query_hits = pd.DataFrame(query_hit_rows)
    comparison_baseline = f"candidate_{_slug(candidate_experiments[0], max_len=70)}"
    summary = aggregate_validation_records(
        raw,
        query_hits=query_hits,
        comparison_baseline=comparison_baseline,
        metric_columns=[f"recall@{k}" for k in config.eval_ks],
    )

    raw_path = output_dir / f"folds_{run_id}.csv"
    query_hits_path = output_dir / f"query_hits_{run_id}.csv"
    summary_path = output_dir / f"summary_{run_id}.csv"
    latest_path = output_dir / "rerank_summary_latest.csv"
    raw.to_csv(raw_path, index=False)
    query_hits.to_csv(query_hits_path, index=False)
    summary.to_csv(summary_path, index=False)
    summary.to_csv(latest_path, index=False)

    sort_col = default_selection_metric(summary)
    rerank_only = summary[summary["experiment"].astype(str).str.startswith("rerank_")].copy()
    best_rerank_experiment = select_best_experiment(rerank_only, metric=sort_col)
    best_rerank_or_candidate = select_best_experiment(summary, metric=sort_col)

    submission_path = None
    if config.create_submission:
        submission_path = _make_rerank_submission(
            data=data,
            data_dir=data_dir,
            all_specs=all_specs,
            specs_by_name=specs_by_name,
            chunk_selector=chunk_selector,
            chunk_search_depth=chunk_search_depth,
            doc_text_by_id=doc_text_by_id,
            rerank_config=configs_by_name[best_rerank_experiment],
            suite_config=config,
            rerankers=rerankers,
        )

    return RerankSuiteResult(
        summary=summary,
        raw=raw,
        query_hits=query_hits,
        best_rerank_experiment=best_rerank_experiment,
        best_rerank_or_candidate=best_rerank_or_candidate,
        summary_path=summary_path,
        raw_path=raw_path,
        query_hits_path=query_hits_path,
        latest_path=latest_path,
        submission_path=submission_path,
    )


def default_selection_metric(summary: pd.DataFrame) -> str:
    for metric in ("holdout_recall@5_mean", "valid_recall@5_mean", "train_recall@5_mean"):
        if metric in summary.columns:
            return metric
    raise ValueError("No recall@5 metric found in rerank summary")


def ranked_summary(summary: pd.DataFrame) -> pd.DataFrame:
    sort_col = default_selection_metric(summary)
    visible_cols = [
        "experiment",
        "status",
        "n_splits",
        "valid_recall@5_mean",
        "valid_recall@5_delta_vs_baseline",
        "valid_recall@5_wins_vs_baseline",
        "valid_recall@5_losses_vs_baseline",
        "holdout_recall@5_mean",
        "holdout_recall@5_delta_vs_baseline",
        "holdout_recall@5_wins_vs_baseline",
        "holdout_recall@5_losses_vs_baseline",
        "duration_sec",
    ]
    visible_cols = [col for col in visible_cols if col in summary.columns]
    return summary[visible_cols].sort_values(["status", sort_col], ascending=[True, False])


def _normalized_config(config: RerankSuiteConfig) -> RerankSuiteConfig:
    candidate_depth = max(config.candidate_depth, max(config.depths), max(config.eval_ks))
    return RerankSuiteConfig(
        **{
            **asdict(config),
            "candidate_depth": int(candidate_depth),
            "eval_ks": tuple(int(k) for k in config.eval_ks),
            "depths": tuple(int(k) for k in config.depths),
            "chunks_per_doc": tuple(int(k) for k in config.chunks_per_doc),
            "chunk_aggs": tuple(str(item) for item in config.chunk_aggs),
            "score_modes": tuple(str(item) for item in config.score_modes),
            "model_names": tuple(str(item) for item in config.model_names),
            "candidate_experiments": (
                tuple(str(item) for item in config.candidate_experiments)
                if config.candidate_experiments is not None
                else None
            ),
        }
    )


def _make_rerank_eval_folds(train: pd.DataFrame, *, mode: str, seed: int, n_splits: int):
    if mode == "holdout":
        split = make_group_holdout(train, seed=seed)
        holdout_df = train.iloc[split.valid_idx].reset_index(drop=True)
        return [(split.name, "holdout", holdout_df)]
    if mode == "cv":
        outer_split = make_group_holdout(train, seed=seed)
        cv_pool = train.iloc[outer_split.train_idx].reset_index(drop=True)
        folds = []
        for split in make_group_kfold(cv_pool, n_splits=n_splits):
            valid_df = cv_pool.iloc[split.valid_idx].reset_index(drop=True)
            folds.append((split.name, "valid", valid_df))
        return folds
    if mode == "train":
        return [("train_all", "train", train.reset_index(drop=True))]
    raise ValueError("mode must be one of: cv, holdout, train")


def _score_candidate_pairs(
    *,
    reranker: Any,
    queries: list[str],
    qids: list[str],
    candidate_rankings: list[list[SearchResult]],
    chunk_rankings: list[list[SearchResult]],
    max_depth: int,
    max_chunks: int,
    doc_text_by_id: dict[str, str],
    config: RerankSuiteConfig,
) -> pd.DataFrame:
    meta_rows: list[dict[str, Any]] = []
    pairs: list[tuple[str, str]] = []
    for q_idx, (qid, query, candidates, chunks) in enumerate(
        zip(qids, queries, candidate_rankings, chunk_rankings, strict=True)
    ):
        grouped_chunks = _chunks_by_doc(chunks)
        seen_docs: set[str] = set()
        for candidate_rank, candidate in enumerate(candidates[:max_depth], start=1):
            doc_id = str(candidate.doc_id)
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            selected_chunks = grouped_chunks.get(doc_id, [])[:max_chunks]
            if not selected_chunks:
                fallback = candidate.text or doc_text_by_id.get(doc_id, "")[: config.fallback_doc_chars]
                selected_chunks = [
                    SearchResult(
                        doc_id=doc_id,
                        unit_id=f"{doc_id}::fallback",
                        score=float(candidate.score),
                        source="fallback_candidate_or_doc_intro",
                        text=fallback,
                    )
                ]
            for chunk_rank, chunk in enumerate(selected_chunks[:max_chunks], start=1):
                text = _clean_pair_text(chunk.text or "", char_limit=config.pair_char_limit)
                meta_rows.append(
                    {
                        "qid": qid,
                        "q_idx": q_idx,
                        "doc_id": doc_id,
                        "candidate_rank": candidate_rank,
                        "candidate_score": float(candidate.score),
                        "chunk_rank": chunk_rank,
                        "chunk_score": float(chunk.score),
                        "chunk_id": chunk.unit_id,
                        "text_hash": hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest(),
                    }
                )
                pairs.append((query, text))

    if not pairs:
        return pd.DataFrame(meta_rows)

    scores = _predict_scores(reranker, pairs, config=config)
    pair_frame = pd.DataFrame(meta_rows)
    pair_frame["ce_score"] = np.asarray(scores, dtype=np.float32).reshape(-1)
    return pair_frame


def _prediction_lists_from_pairs(pair_frame: pd.DataFrame, qids: list[str], config: RerankConfig) -> list[list[str]]:
    subset = pair_frame[
        (pair_frame["candidate_rank"] <= config.depth)
        & (pair_frame["chunk_rank"] <= config.chunks_per_doc)
    ].copy()
    if subset.empty:
        return [[] for _ in qids]

    doc_rows = []
    for (qid, doc_id), group in subset.groupby(["qid", "doc_id"], sort=False):
        doc_rows.append(
            {
                "qid": qid,
                "doc_id": doc_id,
                "candidate_rank": int(group["candidate_rank"].min()),
                "candidate_score": float(group["candidate_score"].iloc[0]),
                "ce_score": _aggregate_ce(group["ce_score"].tolist(), config.chunk_agg),
            }
        )
    docs = pd.DataFrame(doc_rows)

    predictions: list[list[str]] = []
    for qid in qids:
        part = docs[docs["qid"].eq(qid)].copy()
        if part.empty:
            predictions.append([])
            continue
        if config.score_mode == "ce":
            part["final_score"] = part["ce_score"]
        elif config.score_mode == "ce_plus_candidate":
            part["final_score"] = 0.85 * _zscore(part["ce_score"]) + 0.15 * _zscore(part["candidate_score"])
        else:
            raise ValueError(f"Unknown score_mode: {config.score_mode}")
        part = part.sort_values(["final_score", "candidate_rank"], ascending=[False, True])
        predictions.append(part["doc_id"].astype(str).tolist())
    return predictions


def _make_rerank_submission(
    *,
    data,
    data_dir: Path,
    all_specs,
    specs_by_name,
    chunk_selector: BM25Retriever,
    chunk_search_depth: int,
    doc_text_by_id: dict[str, str],
    rerank_config: RerankConfig,
    suite_config: RerankSuiteConfig,
    rerankers: dict[str, Any],
) -> Path:
    test_queries = data.test["question"].astype(str).tolist()
    test_qids = data.test["qid"].astype(str).tolist()
    test_chunk_rankings = chunk_selector.search(test_queries, top_k=chunk_search_depth)
    test_rank_cache: dict[str, list[list[SearchResult]]] = {}
    test_candidate_rankings = rank_queries(
        specs_by_name[rerank_config.candidate_experiment],
        all_specs,
        data.documents,
        test_queries,
        top_k=max(suite_config.candidate_depth, rerank_config.depth),
        cache=test_rank_cache,
    )
    test_pair_frame = _score_candidate_pairs(
        reranker=_get_reranker(rerank_config.model_name, config=suite_config, rerankers=rerankers),
        queries=test_queries,
        qids=test_qids,
        candidate_rankings=test_candidate_rankings,
        chunk_rankings=test_chunk_rankings,
        max_depth=rerank_config.depth,
        max_chunks=rerank_config.chunks_per_doc,
        doc_text_by_id=doc_text_by_id,
        config=suite_config,
    )
    test_predictions = _prediction_lists_from_pairs(test_pair_frame, test_qids, rerank_config)
    output_path = data_dir / "submissions" / f"submission_{rerank_config.name}.csv"
    return write_submission(
        test_qids,
        test_predictions,
        output_path,
        test=data.test,
        documents=data.documents,
        top_k=5,
    )


def _metric_record(
    run_id: str,
    split_name: str,
    eval_part: str,
    experiment: str,
    description: str,
    frame: pd.DataFrame,
    predictions: list[list[str]],
    duration_sec: float,
    params: dict[str, Any],
    *,
    config: RerankSuiteConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gold = frame["gold_doc_id"].astype(str).tolist()
    record: dict[str, Any] = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "split": split_name,
        "mode": config.mode,
        "experiment": experiment,
        "priority": "P0",
        "description": description,
        "params": json.dumps(params, ensure_ascii=False),
        "eval_part": eval_part,
        "n_eval": len(frame),
        "status": "ok",
        "duration_sec": float(duration_sec),
    }
    record.update(evaluate_predictions(gold, predictions, ks=config.eval_ks))
    return record, _query_hit_records(record, eval_part, frame, predictions, config=config)


def _query_hit_records(
    base_record: dict[str, Any],
    eval_part: str,
    frame: pd.DataFrame,
    predictions: list[list[str]],
    *,
    config: RerankSuiteConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for qid, expected, predicted in zip(
        frame["qid"].astype(str).tolist(),
        frame["gold_doc_id"].astype(str).tolist(),
        predictions,
        strict=True,
    ):
        row: dict[str, Any] = {
            "run_id": base_record["run_id"],
            "split": base_record["split"],
            "mode": base_record["mode"],
            "experiment": base_record["experiment"],
            "eval_part": eval_part,
            "qid": qid,
            "gold_doc_id": expected,
        }
        for k in config.eval_ks:
            row[f"hit@{k}"] = int(expected in dedupe_topk(predicted, k))
        rows.append(row)
    return rows


def _make_rerank_configs(candidate_name: str, model_name: str, config: RerankSuiteConfig) -> list[RerankConfig]:
    configs = []
    model_slug = _slug(model_name, max_len=42)
    candidate_slug = _slug(candidate_name, max_len=48)
    for depth in config.depths:
        for chunks_per_doc in config.chunks_per_doc:
            for chunk_agg in config.chunk_aggs:
                if chunks_per_doc == 1 and chunk_agg != "max":
                    continue
                for score_mode in config.score_modes:
                    name = f"rerank_{candidate_slug}_{model_slug}_d{depth}_c{chunks_per_doc}_{chunk_agg}_{score_mode}"
                    configs.append(
                        RerankConfig(
                            name=name,
                            candidate_experiment=candidate_name,
                            model_name=model_name,
                            depth=int(depth),
                            chunks_per_doc=int(chunks_per_doc),
                            chunk_agg=str(chunk_agg),
                            score_mode=str(score_mode),
                        )
                    )
    return configs


def _get_reranker(
    model_name: str,
    *,
    config: RerankSuiteConfig,
    rerankers: dict[str, Any],
) -> Any:
    if model_name not in rerankers:
        if _use_flag_embedding_backend(model_name):
            rerankers[model_name] = FlagEmbeddingReranker(
                FlagRerankerConfig(
                    model_name=model_name,
                    batch_size=config.batch_size,
                    use_fp16=config.device != "cpu",
                    normalize=False,
                )
            ).load()
        else:
            rerankers[model_name] = CrossEncoderReranker(
                CrossEncoderConfig(
                    model_name=model_name,
                    batch_size=config.batch_size,
                    max_length=config.max_length,
                    device=config.device,
                )
            ).load()
    return rerankers[model_name]


def _predict_scores(reranker: Any, pairs: list[tuple[str, str]], *, config: RerankSuiteConfig) -> list[float]:
    if isinstance(reranker, FlagEmbeddingReranker):
        return reranker.predict(pairs)
    if reranker.model is None:
        reranker.load()
    assert reranker.model is not None
    scores = reranker.model.predict(
        pairs,
        batch_size=config.batch_size,
        show_progress_bar=True,
    )
    return [float(score) for score in np.asarray(scores, dtype=np.float32).reshape(-1)]


def _use_flag_embedding_backend(model_name: str) -> bool:
    return str(model_name).startswith("BAAI/bge-reranker")


def _merge_specs(base, extra):
    by_name = {}
    for spec in [*base, *extra]:
        by_name.setdefault(spec.name, spec)
    return list(by_name.values())


def _legal_bm25_config() -> dict[str, Any]:
    return {
        "k1": 1.5,
        "b": 0.75,
        "lemmatize": True,
        "preserve_legal_refs": True,
        "legal_stop_words": True,
        "min_len": 2,
    }


def _chunks_by_doc(chunk_ranking: list[SearchResult]) -> dict[str, list[SearchResult]]:
    grouped: dict[str, list[SearchResult]] = {}
    for item in chunk_ranking:
        grouped.setdefault(item.doc_id, []).append(item)
    return grouped


def _aggregate_ce(scores: list[float], method: str) -> float:
    ordered = sorted([float(score) for score in scores], reverse=True)
    if not ordered:
        return float("-inf")
    if method == "max":
        return ordered[0]
    if method == "top2_mean":
        return float(np.mean(ordered[:2]))
    if method == "max_plus_second":
        return ordered[0] + (0.2 * ordered[1] if len(ordered) > 1 else 0.0)
    raise ValueError(f"Unknown chunk_agg: {method}")


def _zscore(values: pd.Series) -> pd.Series:
    arr = values.astype(float)
    std = arr.std(ddof=0)
    if not math.isfinite(float(std)) or std < 1e-9:
        return pd.Series(np.zeros(len(arr)), index=arr.index)
    return (arr - arr.mean()) / std


def _clean_pair_text(text: str, *, char_limit: int) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text[:char_limit]


def _slug(value: str, max_len: int = 80) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()
    return value[:max_len].strip("_") or "x"


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, str):
        return [value]
    return list(value)
