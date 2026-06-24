from __future__ import annotations

import argparse
from pathlib import Path

from legal_hse.config import PathConfig
from legal_hse.data import load_data
from legal_hse.supervision import build_pairwise_training_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export query->evidence-window positives for fine-tuning.")
    parser.add_argument("--data-dir", default=".", help="Directory with train/test/documents CSV files.")
    parser.add_argument("--output", default=None, help="Output CSV path.")
    parser.add_argument("--window-chars", type=int, default=1200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = PathConfig.from_root(args.data_dir)
    paths.ensure_dirs()
    data = load_data(paths.root)
    pairs = build_pairwise_training_frame(data.train, data.documents, window_chars=args.window_chars)
    output = Path(args.output) if args.output else paths.artifacts_dir / "evidence_pairs.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(output, index=False)
    print(output)


if __name__ == "__main__":
    main()
