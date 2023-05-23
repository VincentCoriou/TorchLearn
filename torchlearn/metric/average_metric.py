"""TODO average_metric docstring
"""
from abc import ABC, abstractmethod
from typing import Any, Sequence

from torch import Tensor

from .metric import Metric


class AverageMetric(Metric, ABC):
    """TODO AverageMetric docstring"""

    _value: float
    _counter: int

    def __init__(self, parameters: Sequence[str]) -> None:
        super().__init__(parameters)
        self._value = 0
        self._counter = 0

    def value(self) -> float:
        if self._counter == 0:
            return 0
        return self._value / self._counter

    def reset(self) -> None:
        self._value = 0
        self._counter = 0

    @staticmethod
    @abstractmethod
    def compute(*args: Any) -> Tensor:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def compute_size(*args: Any) -> int:
        raise NotImplementedError

    def update(self, *args: Any) -> Tensor:  # type: ignore
        value = self.compute(*args)
        size = self.compute_size(*args)
        self._value += value.mean().item() * size
        self._counter += size
        return value
