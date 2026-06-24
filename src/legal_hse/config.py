from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathConfig:
    """Centralized filesystem layout for local and Colab runs."""

    root: Path
    train: Path
    test: Path
    documents: Path
    sample_submission: Path
    artifacts_dir: Path
    reports_dir: Path
    metrics_dir: Path
    submissions_dir: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "PathConfig":
        root = Path(root).expanduser().resolve()
        reports_dir = root / "reports"
        return cls(
            root=root,
            train=root / "train.csv",
            test=root / "test.csv",
            documents=root / "documents.csv",
            sample_submission=root / "sample_submission.csv",
            artifacts_dir=root / "artifacts",
            reports_dir=reports_dir,
            metrics_dir=reports_dir / "metrics",
            submissions_dir=root / "submissions",
        )

    def ensure_dirs(self) -> None:
        for path in [
            self.artifacts_dir,
            self.reports_dir,
            self.metrics_dir,
            self.submissions_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def require_input_files(self) -> None:
        missing = [
            path
            for path in [self.train, self.test, self.documents, self.sample_submission]
            if not path.exists()
        ]
        if missing:
            names = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"Missing required input files: {names}")
