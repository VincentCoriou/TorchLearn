"""TODO precision_macro docstring
"""
from typing import Optional, Tuple

from torch import Tensor

from .classification_metric import (
    ClassificationMetric,
)
from .confusion_matrix import ConfusionMatrix
from .precision import Precision
from .support import Support


class PrecisionMacro(ClassificationMetric):
    """TODO PrecisionMacro docstring"""

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
            name = "PrecisionMacro"
        super().__init__(name, num_classes, confusion_matrix, keys)
        self._weighted = weighted

    def value(self) -> float:
        return PrecisionMacro.compute(self._confusion_matrix.confusion_matrix, self._weighted)

    @staticmethod
    def compute(confusion_matrix: Tensor, weighted: bool = False) -> float:
        precision = Precision.compute(confusion_matrix)
        if weighted:
            support = Support.compute(confusion_matrix)
            return ((support * precision) / support.sum(-1, keepdim=True)).sum().item()
        return precision.mean().item()
