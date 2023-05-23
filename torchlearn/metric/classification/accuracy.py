"""TODO accuracy docstring
"""
from typing import Optional, Tuple

from torch import Tensor

from .classification_metric import ClassificationMetric
from .confusion_matrix import ConfusionMatrix


class Accuracy(ClassificationMetric):
    """TODO Accuracy docstring"""

    def __init__(
        self,
        num_classes: int,
        name: Optional[str] = None,
        confusion_matrix: Optional[ConfusionMatrix] = None,
        keys: Tuple[str, str] = ("predicted", "target"),
    ) -> None:
        if name is None:
            name = "Accuracy"
        super().__init__(name, num_classes, confusion_matrix, keys)

    def value(self) -> float:
        return self.compute(self._confusion_matrix.confusion_matrix)

    @staticmethod
    def compute(confusion_matrix: Tensor) -> float:
        total = confusion_matrix.sum()
        accuracy = confusion_matrix.diag().sum() / total
        if total == 0:
            return 0
        return accuracy.item()
