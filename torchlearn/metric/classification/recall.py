"""TODO recall docstring
"""
from typing import Optional, Union, overload, Tuple

from torch import Tensor

from .classification_metric import (
    ClassificationMetric,
)
from .confusion_matrix import ConfusionMatrix


class Recall(ClassificationMetric):
    """TODO Recall docstring"""

    _class: int

    def __init__(
        self,
        class_: int,
        name: Optional[str] = None,
        num_classes: Optional[int] = None,
        confusion_matrix: Optional[ConfusionMatrix] = None,
        keys: Tuple[str, str] = ("predicted", "target"),
    ) -> None:
        if name is None:
            name = f"Recall_{class_}"
        super().__init__(name, num_classes, confusion_matrix, keys)
        self._class = class_

    def value(self) -> float:
        return Recall.compute(self._confusion_matrix.confusion_matrix, self._class)

    @staticmethod
    @overload
    def compute(confusion_matrix: Tensor, class_: None = None) -> Tensor:
        ...

    @staticmethod
    @overload
    def compute(confusion_matrix: Tensor, class_: int) -> float:
        ...

    @staticmethod
    def compute(confusion_matrix: Tensor, class_: Optional[int] = None) -> Union[Tensor, float]:
        total = confusion_matrix.sum(1)
        recall = confusion_matrix.diag() / total
        recall[total == 0] = 0
        if class_ is not None:
            return recall[class_].item()
        return recall
