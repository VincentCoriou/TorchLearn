"""TODO support docstring
"""
from typing import Optional, Union, Tuple, overload

from torch.fft import Tensor

from .classification_metric import (
    ClassificationMetric,
)
from .confusion_matrix import ConfusionMatrix


class Support(ClassificationMetric):
    """TODO Support docstring"""

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
            name = f"Support_{class_}"
        super().__init__(name, num_classes, confusion_matrix, keys)
        self._class = class_

    def value(self) -> float:
        return Support.compute(self._confusion_matrix.confusion_matrix, self._class)

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
        support = confusion_matrix.sum(1)
        if class_ is not None:
            return support[class_].item()
        return support
