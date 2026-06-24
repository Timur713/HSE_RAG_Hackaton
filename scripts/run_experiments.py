from __future__ import annotations

import argparse

from legal_hse.experiments import run_suite, select_best_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval experiments and save metrics.")
    parser.add_argument("--data-dir", default=".", help="Directory with train/test/documents CSV files.")
    parser.add_argument("--output-dir", default=None, help="Directory for reports and metrics.")
    parser.add_argument("--mode", choices=["holdout", "cv", "train"], default="holdout")
    parser.add_argument("--experiment", action="append", dest="experiments", help="Experiment name. Repeatable.")
    parser.add_argument("--include-optional", action="store_true", help="Include dense optional experiments.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--eval-depth",
        "--top-k",
        dest="eval_depth",
        type=int,
        default=50,
        help="How many candidates to keep for recall metrics. Must be at least 50 for recall@50.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_suite(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        experiment_names=args.experiments,
        mode=args.mode,
        include_optional=args.include_optional,
        seed=args.seed,
        n_splits=args.n_splits,
        eval_depth=args.eval_depth,
    )
    cols = [
        "experiment",
        "status",
        "n_splits",
        "train_recall@5_mean",
        "train_recall@5_std",
        "valid_recall@5_mean",
        "valid_recall@5_std",
        "valid_recall@10_mean",
        "valid_recall@20_mean",
        "valid_recall@50_mean",
        "holdout_recall@5_mean",
        "holdout_recall@5_std",
        "holdout_recall@10_mean",
        "holdout_recall@20_mean",
        "holdout_recall@50_mean",
    ]
    cols = [col for col in cols if col in summary.columns]
    print(summary[cols].to_string(index=False))
    print(f"\nBest by mean recall@5: {select_best_experiment(summary)}")


if __name__ == "__main__":
    main()
