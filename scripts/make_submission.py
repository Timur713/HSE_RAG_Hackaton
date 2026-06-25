from __future__ import annotations

import argparse

from legal_hse.experiments import create_submission, recall_candidate_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create submission CSV from an experiment.")
    parser.add_argument("--data-dir", default=".", help="Directory with train/test/documents CSV files.")
    parser.add_argument("--experiment", required=True, help="Experiment name from legal_hse.experiments.")
    parser.add_argument("--output", default=None, help="Output CSV path.")
    parser.add_argument("--include-optional", action="store_true", help="Allow optional dense experiments.")
    parser.add_argument(
        "--recall-candidates",
        action="store_true",
        help="Allow experiments from the extended Recall@20/50 candidate-generation suite.",
    )
    parser.add_argument(
        "--include-bge-m3",
        action="store_true",
        help="Allow BGE-M3 candidate-generation experiments from the extended recall suite.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extra_experiments = None
    if args.recall_candidates or args.include_bge_m3:
        extra_experiments = recall_candidate_experiments(
            include_optional=args.include_optional,
            include_bge_m3=args.include_bge_m3,
        )
    path = create_submission(
        data_dir=args.data_dir,
        experiment_name=args.experiment,
        output_path=args.output,
        include_optional=args.include_optional,
        extra_experiments=extra_experiments,
        top_k=args.top_k,
    )
    print(path)


if __name__ == "__main__":
    main()
