from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


@dataclass(frozen=True)
class Split:
    name: str
    train_idx: list[int]
    valid_idx: list[int]


def make_group_holdout(
    train: pd.DataFrame,
    *,
    group_col: str = "gold_doc_id",
    test_size: float = 0.2,
    seed: int = 42,
) -> Split:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    groups = train[group_col]
    train_idx, valid_idx = next(splitter.split(train, groups=groups))
    return Split("holdout", train_idx.tolist(), valid_idx.tolist())


def make_group_kfold(
    train: pd.DataFrame,
    *,
    group_col: str = "gold_doc_id",
    n_splits: int = 5,
) -> list[Split]:
    splitter = GroupKFold(n_splits=n_splits)
    groups = train[group_col]
    return [
        Split(f"fold_{fold}", train_idx.tolist(), valid_idx.tolist())
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(train, groups=groups))
    ]


def split_frame(train: pd.DataFrame, split: Split) -> tuple[pd.DataFrame, pd.DataFrame]:
    return train.iloc[split.train_idx].reset_index(drop=True), train.iloc[split.valid_idx].reset_index(drop=True)
