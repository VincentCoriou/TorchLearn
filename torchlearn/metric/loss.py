"""TODO loss docstring
"""
from typing import Any

from torch import Tensor

from .average_metric import AverageMetric


class Loss(AverageMetric):
    """TODO Loss docstring"""

    def __init__(self, name: str = "loss", size_name: str = "batch_size") -> None:
        super().__init__((name, size_name))

    @staticmethod
    def compute(loss: Tensor, batch_size: int) -> Any: # type: ignore
        del batch_size
        return loss

    @staticmethod
    def compute_size(loss: Tensor, size: int) -> int: # type: ignore
        del loss
        return size
