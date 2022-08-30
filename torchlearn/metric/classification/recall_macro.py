"""TODO recall_macro docstring
"""
from typing import Optional, Tuple

from torch import Tensor

from .classification_metric import (
    ClassificationMetric,
)
from .confusion_matrix import ConfusionMatrix
from .recall import Recall
from .support import Support


class RecallMacro(ClassificationMetric):
    """TODO RecallMacro docstring"""

    _weighted: bool

    def __init__(
        self,
        name: Optional[str] = None,
        num_classes: Optional[int] = None,
        weighted: bool = False,
        confusion_matrix: Optional[ConfusionMatrix] = None,
        keys: Tuple[str, str] = ("predicted", "target"),
    ) -> None:
        if name is None:
            name = "RecallMacro"
        super().__init__(name, num_classes, confusion_matrix, keys)
        self._weighted = weighted

    def value(self) -> float:
        return self.compute(self._confusion_matrix.confusion_matrix, self._weighted)

    @staticmethod
    def compute(confusion_matrix: Tensor, weighted: bool = False) -> float:
        recall = Recall.compute(confusion_matrix)
        if weighted:
            support = Support.compute(confusion_matrix)
            return ((support * recall) / support.sum(-1, keepdim=True)).sum().item()
        return recall.mean().item()
