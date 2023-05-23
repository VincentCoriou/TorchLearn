"""TODO classification_metric docstring
"""
from abc import ABC
from typing import Optional, Tuple

from .confusion_matrix import ConfusionMatrix
from ..metric_value import MetricValue
from ...utils.arguments import check_one_mandatory


class ClassificationMetric(MetricValue, ABC):
    """TODO ClassificationMetric docstring"""

    _confusion_matrix: ConfusionMatrix

    def __init__(
        self,
        name: str,
        num_classes: Optional[int] = None,
        confusion_matrix: Optional[ConfusionMatrix] = None,
        keys: Tuple[str, str] = ("predicted", "target"),
    ) -> None:
        super().__init__(name)
        check_one_mandatory(num_classes=num_classes, confusion_matrix=confusion_matrix)
        if confusion_matrix is None:
            assert num_classes is not None
            confusion_matrix = ConfusionMatrix(num_classes, keys)
        if num_classes is None:
            num_classes = confusion_matrix.num_classes
        if num_classes != confusion_matrix.num_classes:
            raise ValueError(
                f"Expected {num_classes} classes for confusion matrix," f"but got {confusion_matrix.num_classes}"
            )
        self._confusion_matrix = confusion_matrix
