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
    parser.add_argument("--top-k", type=int, default=5)
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
        top_k=args.top_k,
    )
    print(summary[["split", "experiment", "status", "recall@1", "recall@5", "mrr@10"]].to_string(index=False))
    print(f"\nBest by mean recall@5: {select_best_experiment(summary)}")


if __name__ == "__main__":
    main()
