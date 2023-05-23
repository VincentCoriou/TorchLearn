"""TODO
"""
import os
from pathlib import Path
from typing import Mapping, Union, Dict

import pandas as pd

from torchlearn.metric.metric_value import MetricValue

PathLike = Union[str, os.PathLike]


class EpochResult:
    """TODO"""
    metrics: Mapping[str, MetricValue]

    def __init__(self, metrics: Mapping[str, MetricValue]) -> None:
        self.metrics = metrics

    def values(self, prefix: str = "") -> Mapping[str, float]:
        return {prefix + n: m.value() for n, m in self.metrics.items()}

    def to_pandas(self) -> pd.Series:
        return pd.Series({n: m.value() for n, m in self.metrics.items()})

    def save(self, *segments: PathLike) -> None:
        path = Path(*segments)
        self.to_pandas().to_csv(path)


class Result:
    """TODO"""
    results: Dict[int, Mapping[str, float]]

    def __init__(self) -> None:
        self.results = {}

    def add(self, epoch: int, result: EpochResult) -> None:
        self.results[epoch] = result.values()

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self.results, orient="index")

    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> "Result":
        result = cls()
        result.results = df.to_dict(orient="index")
        return result

    def save(self, *segments: PathLike) -> None:
        path = Path(*segments)
        self.to_pandas().to_csv(path)

    @classmethod
    def load(cls, *segments: PathLike) -> "Result":
        path = Path(*segments)
        df = pd.read_csv(path, index_col=0)
        return cls.from_pandas(df)

    def __getitem__(self, epoch: int) -> Mapping[str, float]:
        return self.results[epoch]

    def __call__(self, metric: str) -> Mapping[int, float]:
        return {e: r[metric] for e, r in self.results.items()}
