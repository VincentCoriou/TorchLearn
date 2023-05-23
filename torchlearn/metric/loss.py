"""TODO loss docstring
"""
from typing import Any, Tuple, Optional

from torch import Tensor

from .average_metric import AverageMetric


class Loss(AverageMetric):
    """TODO Loss docstring"""

    def __init__(self, name: str = "loss", keys: Tuple[Optional[str], str] = (None, "batch_size")) -> None:
        keys = (keys[0] if keys[0] is not None else name, keys[1])
        super().__init__(name, keys)

    @staticmethod
    def compute(loss: Tensor, batch_size: int) -> Any:
        del batch_size
        return loss

    @staticmethod
    def compute_size(loss: Tensor, size: int) -> int:
        del loss
        return size
