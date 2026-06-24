import pandas as pd

from legal_hse.experiments import _make_eval_folds, aggregate_validation_records
from legal_hse.splits import make_group_holdout


def test_cv_eval_folds_use_valid_part_and_exclude_outer_holdout():
    train = pd.DataFrame(
        {
            "qid": [f"q{i}" for i in range(20)],
            "question": [f"question {i}" for i in range(20)],
            "gold_doc_id": [f"doc{i}" for i in range(20)],
        }
    )
    outer = make_group_holdout(train, seed=42)
    outer_holdout_qids = set(train.iloc[outer.valid_idx]["qid"])

    folds = _make_eval_folds(train, mode="cv", seed=42, n_splits=4)

    assert {eval_part for _, frames in folds for eval_part, _ in frames} == {"valid"}
    cv_valid_qids = {qid for _, frames in folds for _, frame in frames for qid in frame["qid"]}
    assert cv_valid_qids.isdisjoint(outer_holdout_qids)
    assert len(cv_valid_qids) == len(train) - len(outer_holdout_qids)


def test_single_holdout_summary_std_is_missing():
    raw = pd.DataFrame(
        [
            {
                "run_id": "run",
                "split": "holdout",
                "mode": "holdout",
                "experiment": "bm25_doc",
                "priority": "P0",
                "description": "",
                "params": {},
                "eval_part": "holdout",
                "n_eval": 10,
                "status": "ok",
                "recall@5": 0.4,
                "recall@10": 0.5,
                "recall@20": 0.6,
                "recall@50": 0.7,
                "duration_sec": 1.0,
            }
        ]
    )

    summary = aggregate_validation_records(raw)

    assert summary.loc[0, "holdout_recall@5_mean"] == 0.4
    assert pd.isna(summary.loc[0, "holdout_recall@5_std"])
    assert pd.isna(summary.loc[0, "holdout_recall@5_se"])
    assert summary.loc[0, "holdout_recall@5_micro"] == 0.4


def test_summary_includes_micro_and_paired_baseline_comparison():
    raw = pd.DataFrame(
        [
            {
                "run_id": "run",
                "split": "fold_0",
                "mode": "cv",
                "experiment": "bm25_doc",
                "priority": "P0",
                "description": "",
                "params": {},
                "eval_part": "valid",
                "n_eval": 2,
                "status": "ok",
                "recall@5": 0.5,
                "recall@10": 0.5,
                "recall@20": 0.5,
                "recall@50": 0.5,
                "duration_sec": 1.0,
            },
            {
                "run_id": "run",
                "split": "fold_1",
                "mode": "cv",
                "experiment": "bm25_doc",
                "priority": "P0",
                "description": "",
                "params": {},
                "eval_part": "valid",
                "n_eval": 4,
                "status": "ok",
                "recall@5": 0.25,
                "recall@10": 0.25,
                "recall@20": 0.25,
                "recall@50": 0.25,
                "duration_sec": 1.0,
            },
            {
                "run_id": "run",
                "split": "fold_0",
                "mode": "cv",
                "experiment": "candidate",
                "priority": "P0",
                "description": "",
                "params": {},
                "eval_part": "valid",
                "n_eval": 2,
                "status": "ok",
                "recall@5": 1.0,
                "recall@10": 1.0,
                "recall@20": 1.0,
                "recall@50": 1.0,
                "duration_sec": 1.0,
            },
            {
                "run_id": "run",
                "split": "fold_1",
                "mode": "cv",
                "experiment": "candidate",
                "priority": "P0",
                "description": "",
                "params": {},
                "eval_part": "valid",
                "n_eval": 4,
                "status": "ok",
                "recall@5": 0.5,
                "recall@10": 0.5,
                "recall@20": 0.5,
                "recall@50": 0.5,
                "duration_sec": 1.0,
            },
        ]
    )
    query_hits = pd.DataFrame(
        [
            {"split": "fold_0", "eval_part": "valid", "qid": "q1", "experiment": "bm25_doc", "hit@5": 1},
            {"split": "fold_0", "eval_part": "valid", "qid": "q2", "experiment": "bm25_doc", "hit@5": 0},
            {"split": "fold_1", "eval_part": "valid", "qid": "q3", "experiment": "bm25_doc", "hit@5": 0},
            {"split": "fold_1", "eval_part": "valid", "qid": "q4", "experiment": "bm25_doc", "hit@5": 1},
            {"split": "fold_0", "eval_part": "valid", "qid": "q1", "experiment": "candidate", "hit@5": 1},
            {"split": "fold_0", "eval_part": "valid", "qid": "q2", "experiment": "candidate", "hit@5": 1},
            {"split": "fold_1", "eval_part": "valid", "qid": "q3", "experiment": "candidate", "hit@5": 1},
            {"split": "fold_1", "eval_part": "valid", "qid": "q4", "experiment": "candidate", "hit@5": 0},
        ]
    )

    summary = aggregate_validation_records(raw, query_hits=query_hits)
    candidate = summary[summary["experiment"].eq("candidate")].iloc[0]

    assert candidate["valid_recall@5_mean"] == 0.75
    assert candidate["valid_recall@5_micro"] == 4 / 6
    assert candidate["valid_recall@5_wins_vs_baseline"] == 2
    assert candidate["valid_recall@5_losses_vs_baseline"] == 1
    assert candidate["valid_recall@5_ties_vs_baseline"] == 1
    assert candidate["valid_recall@5_delta_vs_baseline"] == 0.25
