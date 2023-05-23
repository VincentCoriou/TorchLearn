"""TODO fscore docstring
"""
from typing import Optional, Union, Tuple, overload

from torch import Tensor

from .classification_metric import (
    ClassificationMetric,
)
from .confusion_matrix import ConfusionMatrix
from .precision import Precision
from .recall import Recall
from ...utils.numbers import check_positive


class FScore(ClassificationMetric):
    """TODO FScore docstring"""

    _class: int
    _beta: float

    def __init__(
        self,
        beta: float,
        class_: int,
        name: Optional[str] = None,
        num_classes: Optional[int] = None,
        confusion_matrix: Optional[ConfusionMatrix] = None,
        keys: Tuple[str, str] = ("predicted", "target"),
    ) -> None:
        check_positive("beta", beta)
        if name is None:
            name = f"F-{beta} score_{class_}"
        super().__init__(name, num_classes, confusion_matrix, keys)
        self._class = class_
        self._beta = beta

    def value(self) -> float:
        return FScore.compute(self._confusion_matrix.confusion_matrix, self._beta, self._class)

    @staticmethod
    @overload
    def compute(confusion_matrix: Tensor, beta: float, class_: None = None) -> Tensor:
        ...

    @staticmethod
    @overload
    def compute(confusion_matrix: Tensor, beta: float, class_: int) -> float:
        ...

    @staticmethod
    def compute(confusion_matrix: Tensor, beta: float, class_: Optional[int] = None) -> Union[Tensor, float]:
        precision = Precision.compute(confusion_matrix)
        recall = Recall.compute(confusion_matrix)
        fscore = FScore.from_precision_recall(beta, precision, recall)
        mask = (precision == 0) & (recall == 0)
        fscore[mask] = 0
        if class_ is not None:
            return fscore[class_].item()
        return fscore

    @staticmethod
    @overload
    def from_precision_recall(beta: float, precision: Tensor, recall: Tensor) -> Tensor:
        ...

    @staticmethod
    @overload
    def from_precision_recall(beta: float, precision: float, recall: float) -> float:
        ...

    @staticmethod
    def from_precision_recall(
        beta: float,
        precision: Union[Tensor, float],
        recall: Union[Tensor, float],
    ) -> Union[Tensor, float]:
        check_positive("beta", beta)
        return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
