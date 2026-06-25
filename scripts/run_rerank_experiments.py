from __future__ import annotations

import argparse
from pathlib import Path

from legal_hse.rerank_experiments import RerankSuiteConfig, ranked_summary, run_rerank_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cross-encoder rerank experiments.")
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=["cv", "holdout", "train"], default="cv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--candidate-experiment", action="append", default=None)
    parser.add_argument("--model-name", action="append", default=None)
    parser.add_argument("--enable-e5-candidates", action="store_true")
    parser.add_argument("--enable-bge-m3", action="store_true")
    parser.add_argument("--no-create-submission", action="store_true")
    parser.add_argument("--depths", default="20,50,100")
    parser.add_argument("--chunks-per-doc", default="1,2")
    parser.add_argument("--chunk-aggs", default="max,top2_mean")
    parser.add_argument("--score-modes", default="ce,ce_plus_candidate")
    parser.add_argument("--eval-ks", default="5,10,20")
    parser.add_argument("--candidate-depth", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = RerankSuiteConfig(
        mode=args.mode,
        seed=args.seed,
        n_splits=args.n_splits,
        eval_ks=_parse_int_tuple(args.eval_ks),
        candidate_depth=args.candidate_depth,
        depths=_parse_int_tuple(args.depths),
        chunks_per_doc=_parse_int_tuple(args.chunks_per_doc),
        chunk_aggs=_parse_str_tuple(args.chunk_aggs),
        score_modes=_parse_str_tuple(args.score_modes),
        model_names=tuple(args.model_name) if args.model_name else RerankSuiteConfig().model_names,
        create_submission=not args.no_create_submission,
        enable_e5_candidates=args.enable_e5_candidates,
        enable_bge_m3=args.enable_bge_m3,
        candidate_experiments=tuple(args.candidate_experiment) if args.candidate_experiment else None,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )
    result = run_rerank_suite(args.data_dir, config)
    print(ranked_summary(result.summary).head(30).to_string(index=False))
    print("Best rerank experiment:", result.best_rerank_experiment)
    print("Best rerank or candidate:", result.best_rerank_or_candidate)
    print("Summary:", result.summary_path)
    if result.submission_path is not None:
        print("Submission:", result.submission_path)


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_str_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


if __name__ == "__main__":
    main()
