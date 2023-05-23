from pathlib import Path
from typing import Sequence

import pandas as pd


class EpochResult:
    def __init__(self, metrics: Sequence):
        self.metrics = metrics

    def values(self, prefix=""):
        return {prefix + n: m.value() for n, m in self.metrics.items()}

    def to_pandas(self):
        return pd.Series({n: m.value() for n, m in self.metrics.items()})

    def save(self, *path):
        path = Path(*path)
        self.to_pandas().to_csv(path)


class Result:
    def __init__(self):
        self.results = {}

    def add(self, epoch, result):
        self.results[epoch] = result.values()

    def to_pandas(self):
        return pd.DataFrame.from_dict(self.results, orient="index")

    @classmethod
    def from_pandas(cls, df: pd.DataFrame):
        result = cls()
        result.results = df.to_dict(orient="index")
        return result

    def save(self, *path):
        path = Path(*path)
        self.to_pandas().to_csv(path)

    @classmethod
    def load(cls, *path):
        path = Path(*path)
        df = pd.read_csv(path, index_col=0)
        return cls.from_pandas(df)

    def __getitem__(self, epoch):
        return self.results[epoch]

    def __call__(self, metric):
        return {e: r[metric] for e, r in self.results.items()}
