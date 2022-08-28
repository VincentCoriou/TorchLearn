"""TODO loss docstring
"""
from typing import Any, Tuple, Union

from torch import Tensor

from .average_metric import AverageMetric


class Loss(AverageMetric):
    """TODO Loss docstring"""

    def __init__(
        self, name: str = "loss", keys: Union[Tuple[str, str], Tuple[None, str]] = (None, "batch_size")
    ) -> None:
        if keys[0] is None:
            keys = (name, *keys[1:])
        super().__init__(name, keys)

    @staticmethod
    def compute(loss: Tensor, batch_size: int) -> Any:
        del batch_size
        return loss

    @staticmethod
    def compute_size(loss: Tensor, size: int) -> int:
        del loss
        return size
