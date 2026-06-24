# Reports

Experiment runs write:

- `reports/metrics/<run_id>.jsonl` - append-only machine-readable metrics.
- `reports/folds_<run_id>.csv` - raw rows for each split, experiment, and eval part.
- `reports/summary_<run_id>.csv` - one aggregated row per experiment with train/holdout mean and std.
- `reports/summary_latest.csv` - latest run for quick inspection.

The intended workflow is: experiment -> metrics -> decision. Keep decisions in GitHub issues or PR comments, and commit the metrics that justify the decision.

Recorded decisions:

- `decision_tokenization_lemmatization_20260624.md` - choose `rrf_sparse_legal_lemma_char` after the tokenization/lemmatization sweep in `summary_20260624T175149Z.csv`.
