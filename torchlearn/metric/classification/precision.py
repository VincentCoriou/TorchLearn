"""TODO precision docstring
"""
from typing import Optional, Union, Tuple, overload

from torch import Tensor

from .classification_metric import ClassificationMetric
from .confusion_matrix import ConfusionMatrix


class Precision(ClassificationMetric):
    """TODO Precision docstring"""

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
            name = f"Precision_{class_}"
        super().__init__(name, num_classes, confusion_matrix, keys)
        self._class = class_

    def value(self) -> float:
        return Precision.compute(self._confusion_matrix.confusion_matrix, self._class)

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
        total = confusion_matrix.sum(0)
        precision = confusion_matrix.diag() / total
        precision[total == 0] = 0
        if class_ is not None:
            return precision[class_].item()
        return precision
