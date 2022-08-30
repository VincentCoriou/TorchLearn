"""TODO fscore_macro docstring
"""
from typing import Optional, Tuple

from torch import Tensor

from .classification_metric import (
    ClassificationMetric,
)
from .confusion_matrix import ConfusionMatrix
from .fscore import FScore
from .support import Support
from ...utils.numbers import check_positive


class FScoreMacro(ClassificationMetric):
    """TODO FScoreMacro docstring"""

    _beta: float
    _weighted: bool

    def __init__(
        self,
        beta: float,
        name: Optional[str] = None,
        num_classes: Optional[int] = None,
        weighted: bool = False,
        confusion_matrix: Optional[ConfusionMatrix] = None,
        keys: Tuple[str, str] = ("predicted", "target"),
    ) -> None:
        check_positive("beta", beta)
        if name is None:
            name = f"F-{beta}ScoreMacro"
        super().__init__(name, num_classes, confusion_matrix, keys)
        self._beta = beta
        self._weighted = weighted

    def value(self) -> float:
        return FScoreMacro.compute(self._confusion_matrix.confusion_matrix, self._beta, self._weighted)

    @staticmethod
    def compute(confusion_matrix: Tensor, beta: float, weighted: bool = False) -> float:
        fscore = FScore.compute(confusion_matrix, beta)
        if weighted:
            support = Support.compute(confusion_matrix)
            return ((support * fscore) / support.sum(-1, keepdim=True)).sum().item()
        return fscore.mean().item()
