# Reports

Experiment runs write:

- `reports/metrics/<run_id>.jsonl` - append-only machine-readable metrics.
- `reports/summary_<run_id>.csv` - one row per experiment and split.
- `reports/summary_latest.csv` - latest run for quick inspection.

The intended workflow is: experiment -> metrics -> decision. Keep decisions in GitHub issues or PR comments, and commit the metrics that justify the decision.
