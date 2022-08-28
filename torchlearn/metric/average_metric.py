"""TODO average_metric docstring
"""
from abc import ABC, abstractmethod
from typing import Any, Callable, Sequence

from .metric import Metric


def _compute_size_unimplemented(self: Any, *args: Any, **kwargs: Any) -> int:
    """TODO _compute_size_unimplemented docstring

    :param self:
    :param args:
    :param kwargs:
    :return:
    """
    raise NotImplementedError(f'AverageMetric [{type(self).__name__}] is missing the required "compute_size" function')


class AverageMetric(Metric, ABC):
    """TODO AverageMetric docstring"""

    _value: float
    _counter: int

    def __init__(self, name: str, keys: Sequence[str]) -> None:
        super().__init__(name, keys)
        self._value = 0
        self._counter = 0

    def value(self) -> float:
        if self._counter == 0:
            return 0
        return self._value / self._counter

    def reset(self) -> None:
        self._value = 0
        self._counter = 0

    compute_size: Callable[..., int] = staticmethod(abstractmethod(_compute_size_unimplemented))

    def update(self, *args: Any) -> Any:
        value = self.compute(*args)
        size = self.compute_size(*args)
        self._value += value.mean().item() * size
        self._counter += size
        return value
