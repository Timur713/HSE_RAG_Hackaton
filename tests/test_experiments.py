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
