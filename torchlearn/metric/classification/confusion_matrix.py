"""TODO confusion_matrix docstring
"""
from typing import Tuple

import torch
from ..metric_state import MetricState
from torch import Tensor


class ConfusionMatrix(MetricState):
    """TODO ConfusionMatrix docstring"""

    _num_classes: int
    _confusion_matrix: Tensor

    def __init__(self, num_classes: int, keys: Tuple[str, str] = ("predicted", "target")) -> None:
        super().__init__(keys)
        self._num_classes = num_classes
        self._confusion_matrix = torch.empty(num_classes, num_classes).long()

    def reset(self) -> None:
        self._confusion_matrix.zero_()

    def update(self, predicted: Tensor, target: Tensor) -> Tensor:
        confusion_matrix = self.compute(predicted, target, self._num_classes)
        self._confusion_matrix = self._confusion_matrix + confusion_matrix
        return confusion_matrix

    @staticmethod
    def compute(predicted: Tensor, target: Tensor, num_classes: int) -> Tensor:
        confusion_matrix = torch.zeros(num_classes, num_classes).long()
        values = torch.ones(1).long()
        confusion_matrix.index_put_((target, predicted), values, accumulate=True)
        return confusion_matrix

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def confusion_matrix(self) -> Tensor:
        return self._confusion_matrix.clone()
