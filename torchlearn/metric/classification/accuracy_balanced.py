"""TODO accuracy_balanced docstring
"""
from typing import Optional, Tuple

from torch import Tensor

from .classification_metric import ClassificationMetric
from .confusion_matrix import ConfusionMatrix
from .recall import Recall


class AccuracyBalanced(ClassificationMetric):
    """TODO AccuracyBalanced docstring"""

    def __init__(
        self,
        num_classes: int,
        name: Optional[str] = None,
        confusion_matrix: Optional[ConfusionMatrix] = None,
        keys: Tuple[str, str] = ("predicted", "target"),
    ) -> None:
        if name is None:
            name = "AccuracyBalanced"
        super().__init__(name, num_classes, confusion_matrix, keys)

    def value(self) -> float:
        return AccuracyBalanced.compute(self._confusion_matrix.confusion_matrix)

    @staticmethod
    def compute(confusion_matrix: Tensor) -> float:
        recalls = Recall.compute(confusion_matrix)
        return recalls.mean().item()
