"""TODO sensitivity docstring
"""
from typing import Optional, Union, overload, Tuple

from torch import Tensor

from .classification_metric import (
    ClassificationMetric,
)
from .confusion_matrix import ConfusionMatrix


class Sensitivity(ClassificationMetric):
    """TODO Sensitivity docstring"""

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
            name = f"Sensitivity_{class_}"
        super().__init__(name, num_classes, confusion_matrix, keys)
        self._class = class_

    def value(self) -> float:
        return Sensitivity.compute(self._confusion_matrix.confusion_matrix, self._class)

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
        sensitivity = confusion_matrix.diag() / total
        sensitivity[total == 0] = 0
        if class_ is not None:
            return sensitivity[class_].item()
        return sensitivity
