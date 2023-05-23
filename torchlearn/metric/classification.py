"""TODO confusion_matrix docstring
"""
from abc import ABC
from functools import partial
from typing import Optional, Tuple, Type, Dict, Callable, Any, KeysView

import torch
from torch import Tensor, LongTensor

from torchlearn.metric.metric_value import MetricValue
from torchlearn.metric.state import State


class ConfusionMatrix(State):
    """TODO ConfusionMatrix docstring"""

    _num_classes: int
    _confusion_matrix: LongTensor

    metrics: Dict[str, Type["ClassificationMetric"]] = {}

    @classmethod
    def register_metric(cls, metric: Type["ClassificationMetric"], name: Optional[str] = None) -> Type[
        "ClassificationMetric"]:
        if name is None:
            name = metric.__name__
        if name in cls.metrics:
            raise NameError(f"Metric {name} is already registered.")
        cls.metrics[name] = metric
        return metric

    @classmethod
    def register(cls, name: Optional[str] = None) -> Callable[
        [Type["ClassificationMetric"]], Type["ClassificationMetric"]]:
        return partial(cls.register_metric, name=name)

    def get_metric(self, name: str, *args: Any, **kwargs: Any) -> "ClassificationMetric":
        return self.metrics[name](self, *args, **kwargs)

    def get_metrics(self) -> KeysView[str]:
        return self.metrics.keys()

    def __init__(self, num_classes: int, parameters: Optional[Tuple[str, str]] = None) -> None:
        super().__init__(parameters)
        self._num_classes = num_classes
        self._confusion_matrix = torch.empty(num_classes, num_classes).long()  # type: ignore

    def reset(self) -> None:
        self._confusion_matrix.zero_()

    def update(self, predicted: LongTensor, target: LongTensor) -> Tensor:  # type: ignore
        confusion_matrix = self.compute(predicted, target, self._num_classes)
        self._confusion_matrix += confusion_matrix # type: ignore
        return confusion_matrix

    @staticmethod
    def compute(predicted: LongTensor, target: LongTensor, num_classes: int) -> LongTensor:  # type: ignore
        confusion_matrix = torch.zeros(num_classes, num_classes).long()
        values = torch.ones(()).long()
        confusion_matrix.index_put_((target, predicted), values, accumulate=True)
        return confusion_matrix  # type: ignore

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def confusion_matrix(self) -> Tensor:
        return self._confusion_matrix.clone()

    def accuracy(self) -> Tensor:
        return Accuracy.compute(self._confusion_matrix)

    def balanced_accuracy(self) -> Tensor:
        return BalancedAccuracy.compute(self._confusion_matrix)

    def true_positive(self, class_: Optional[int] = None) -> Tensor:
        return TruePositive.compute(self._confusion_matrix, class_)

    def true_negative(self, class_: Optional[int] = None) -> Tensor:
        return TrueNegative.compute(self._confusion_matrix, class_)

    def false_positive(self, class_: Optional[int] = None) -> Tensor:
        return FalsePositive.compute(self._confusion_matrix, class_)

    def false_negative(self, class_: Optional[int] = None) -> Tensor:
        return FalseNegative.compute(self._confusion_matrix, class_)

    def retrieved(self, class_: Optional[int] = None) -> Tensor:
        return Retrieved.compute(self._confusion_matrix, class_)

    def relevant(self, class_: Optional[int] = None) -> Tensor:
        return Relevant.compute(self._confusion_matrix, class_)

    def sensitivity(self, class_: Optional[int] = None) -> Tensor:
        return Sensitivity.compute(self._confusion_matrix, class_)

    def specificity(self, class_: Optional[int] = None) -> Tensor:
        return Specificity.compute(self._confusion_matrix, class_)

    def support(self, class_: Optional[int] = None) -> Tensor:
        return Support.compute(self._confusion_matrix, class_)

    def precision(self, class_: Optional[int] = None) -> Tensor:
        return Precision.compute(self._confusion_matrix, class_)

    def recall(self, class_: Optional[int] = None) -> Tensor:
        return Recall.compute(self._confusion_matrix, class_)

    def fscore(self, beta: float, class_: Optional[int] = None) -> Tensor:
        return FScore.compute(self._confusion_matrix, beta, class_)

    def weighted_precision(self) -> Tensor:
        return WeightedPrecision.compute(self._confusion_matrix)

    def weighted_recall(self) -> Tensor:
        return WeightedRecall.compute(self._confusion_matrix)

    def weighted_fscore(self, beta: float) -> Tensor:
        return WeightedFScore.compute(self._confusion_matrix, beta)

    def macro_precision(self) -> Tensor:
        return MacroPrecision.compute(self._confusion_matrix)

    def macro_recall(self) -> Tensor:
        return MacroRecall.compute(self._confusion_matrix)

    def macro_fscore(self, beta: float) -> Tensor:
        return MacroFScore.compute(self._confusion_matrix, beta)


class ClassificationMetric(MetricValue, ABC):
    """TODO ClassificationMetric docstring"""

    _confusion_matrix: ConfusionMatrix

    def __init__(
            self,
            confusion_matrix: ConfusionMatrix,
    ) -> None:
        super().__init__()
        self._confusion_matrix = confusion_matrix


class ClassMetric(ClassificationMetric, ABC):
    """TODO ClassMetric docstring"""

    _class: int

    def __init__(self, confusion_matrix: ConfusionMatrix, class_: int) -> None:
        super().__init__(confusion_matrix)
        self._class = class_


@ConfusionMatrix.register()
class Accuracy(ClassificationMetric):
    """TODO Accuracy docstring"""

    def value(self) -> float:
        return self._confusion_matrix.accuracy().item()

    @staticmethod
    def compute(confusion_matrix: LongTensor) -> Tensor:
        total = Support.compute(confusion_matrix).sum()
        correct = TruePositive.compute(confusion_matrix).sum()
        if total == 0:
            return torch.ones(()).float()
        return correct / total


@ConfusionMatrix.register()
class BalancedAccuracy(ClassificationMetric):
    """TODO BalancedAccuracy docstring"""

    def value(self) -> float:
        return self._confusion_matrix.balanced_accuracy().item()

    @staticmethod
    def compute(confusion_matrix: LongTensor) -> Tensor:
        recalls = Recall.compute(confusion_matrix)
        return recalls.mean()


@ConfusionMatrix.register()
class TruePositive(ClassMetric):
    """TODO TruePositive docstring"""

    def value(self) -> float:
        return self._confusion_matrix.true_positive(self._class).item()

    @staticmethod
    def compute(confusion_matrix: LongTensor, class_: Optional[int] = None) -> LongTensor:
        tp = confusion_matrix.diag()
        if class_ is not None:
            return tp[class_]  # type: ignore
        return tp  # type: ignore


@ConfusionMatrix.register()
class TrueNegative(ClassMetric):
    """TODO TrueNegative docstring"""

    def value(self) -> float:
        return self._confusion_matrix.true_negative(self._class).item()

    @staticmethod
    def compute(confusion_matrix: LongTensor, class_: Optional[int] = None) -> LongTensor:
        total = confusion_matrix.sum()
        tp = TruePositive.compute(confusion_matrix, class_)
        ret = Retrieved.compute(confusion_matrix, class_)
        rel = Relevant.compute(confusion_matrix, class_)
        return total - ret - rel + tp  # type: ignore


@ConfusionMatrix.register()
class FalsePositive(ClassMetric):
    """TODO FalsePositive docstring"""

    def value(self) -> float:
        return self._confusion_matrix.false_positive(self._class).item()

    @staticmethod
    def compute(confusion_matrix: LongTensor, class_: Optional[int] = None) -> LongTensor:
        ret = Retrieved.compute(confusion_matrix, class_)
        tp = TruePositive.compute(confusion_matrix, class_)
        return ret - tp  # type: ignore


@ConfusionMatrix.register()
class FalseNegative(ClassMetric):
    """TODO FalseNegative docstring"""

    def value(self) -> float:
        return self._confusion_matrix.false_negative(self._class).item()

    @staticmethod
    def compute(confusion_matrix: LongTensor, class_: Optional[int] = None) -> LongTensor:
        rel = Relevant.compute(confusion_matrix, class_)
        tp = TruePositive.compute(confusion_matrix, class_)
        return rel - tp  # type: ignore


@ConfusionMatrix.register()
class Retrieved(ClassMetric):
    """TODO Retrieved docstring"""

    def value(self) -> float:
        return self._confusion_matrix.retrieved(self._class).item()

    @staticmethod
    def compute(confusion_matrix: LongTensor, class_: Optional[int] = None) -> LongTensor:
        ret = confusion_matrix.sum(0)
        if class_ is not None:
            return ret[class_]  # type: ignore
        return ret  # type: ignore


@ConfusionMatrix.register()
class Relevant(ClassMetric):
    """TODO Relevant docstring"""

    def value(self) -> float:
        return self._confusion_matrix.relevant(self._class).item()

    @staticmethod
    def compute(confusion_matrix: LongTensor, class_: Optional[int] = None) -> LongTensor:
        rel = confusion_matrix.sum(1)
        if class_ is not None:
            return rel[class_]  # type: ignore
        return rel  # type: ignore


@ConfusionMatrix.register()
class Sensitivity(ClassMetric):
    """TODO Sensitivity docstring"""

    def value(self) -> float:
        return self._confusion_matrix.sensitivity(self._class).item()

    @staticmethod
    def compute(confusion_matrix: LongTensor, class_: Optional[int] = None) -> Tensor:
        rel = Relevant.compute(confusion_matrix, class_)
        tp = TruePositive.compute(confusion_matrix, class_)
        sen = tp / rel
        sen[rel == 0] = 0.
        return sen


@ConfusionMatrix.register()
class Specificity(ClassMetric):
    """TODO Specificity docstring"""

    def value(self) -> float:
        return self._confusion_matrix.specificity(self._class).item()

    @staticmethod
    def compute(confusion_matrix: LongTensor, class_: Optional[int] = None) -> Tensor:
        tn = TrueNegative.compute(confusion_matrix, class_)
        fp = FalsePositive.compute(confusion_matrix, class_)
        spe = tn / (fp + tn)
        spe[(fp == 0) & (tn == 0)] = 0
        return spe


@ConfusionMatrix.register()
class Support(ClassMetric):
    """TODO Support docstring"""

    def value(self) -> float:
        return self._confusion_matrix.support(self._class).item()

    @staticmethod
    def compute(confusion_matrix: LongTensor, class_: Optional[int] = None) -> LongTensor:
        return Relevant.compute(confusion_matrix, class_)


@ConfusionMatrix.register()
class Precision(ClassMetric):
    """TODO Precision docstring"""

    def value(self) -> float:
        return self._confusion_matrix.precision(self._class).item()

    @staticmethod
    def compute(confusion_matrix: LongTensor, class_: Optional[int] = None) -> Tensor:
        ret = Retrieved.compute(confusion_matrix, class_)
        tp = TruePositive.compute(confusion_matrix, class_)
        pre = tp / ret
        pre[ret == 0] = 0.
        return pre


@ConfusionMatrix.register()
class Recall(ClassMetric):
    """TODO Recall docstring"""

    def value(self) -> float:
        return self._confusion_matrix.recall(self._class).item()

    @staticmethod
    def compute(confusion_matrix: LongTensor, class_: Optional[int] = None) -> Tensor:
        rel = Relevant.compute(confusion_matrix, class_)
        tp = TruePositive.compute(confusion_matrix, class_)
        rec = tp / rel
        rec[rel == 0] = 0.
        return rec


@ConfusionMatrix.register()
class FScore(ClassMetric):
    """TODO FScore docstring"""

    def __init__(self, confusion_matrix: ConfusionMatrix, class_: int, beta: float = 1.) -> None:
        super().__init__(confusion_matrix, class_)
        self._beta = beta

    def value(self) -> float:
        return self._confusion_matrix.fscore(self._beta, self._class).item()

    @staticmethod
    def compute(confusion_matrix: LongTensor, beta: float, class_: Optional[int] = None) -> Tensor:
        tp = TruePositive.compute(confusion_matrix, class_)
        fn = FalseNegative.compute(confusion_matrix, class_)
        fp = FalsePositive.compute(confusion_matrix, class_)
        fs = (1 + beta ** 2) * tp / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp)
        fs[(tp == 0) & (fn == 0) & (fp == 0)] = 0
        return fs


@ConfusionMatrix.register()
class WeightedPrecision(ClassificationMetric):
    """TODO WeightedPrecision docstring"""

    def value(self) -> float:
        return self._confusion_matrix.weighted_precision().item()

    @staticmethod
    def compute(confusion_matrix: LongTensor) -> Tensor:
        prec = Precision.compute(confusion_matrix)
        return _weighted_avg(confusion_matrix, prec)


@ConfusionMatrix.register()
class WeightedRecall(ClassificationMetric):
    """TODO WeightedRecall docstring"""

    def value(self) -> float:
        return self._confusion_matrix.weighted_recall().item()

    @staticmethod
    def compute(confusion_matrix: LongTensor) -> Tensor:
        rec = Recall.compute(confusion_matrix)
        return _weighted_avg(confusion_matrix, rec)


@ConfusionMatrix.register()
class WeightedFScore(ClassificationMetric):
    """TODO WeightedFScore docstring"""

    def __init__(self, confusion_matrix: ConfusionMatrix, beta: float = 1.) -> None:
        super().__init__(confusion_matrix)
        self._beta = beta

    def value(self) -> float:
        return self._confusion_matrix.weighted_fscore(self._beta).item()

    @staticmethod
    def compute(confusion_matrix: LongTensor, beta: float) -> Tensor:
        fs = FScore.compute(confusion_matrix, beta)
        return _weighted_avg(confusion_matrix, fs)


@ConfusionMatrix.register()
class MacroPrecision(ClassificationMetric):
    """TODO MacroPrecision docstring"""

    def value(self) -> float:
        return self._confusion_matrix.macro_precision().item()

    @staticmethod
    def compute(confusion_matrix: LongTensor) -> Tensor:
        prec = Precision.compute(confusion_matrix)
        return _macro_avg(confusion_matrix, prec)


@ConfusionMatrix.register()
class MacroRecall(ClassificationMetric):
    """TODO MacroRecall docstring"""

    def value(self) -> float:
        return self._confusion_matrix.macro_recall().item()

    @staticmethod
    def compute(confusion_matrix: LongTensor) -> Tensor:
        rec = Recall.compute(confusion_matrix)
        return _macro_avg(confusion_matrix, rec)


@ConfusionMatrix.register()
class MacroFScore(ClassificationMetric):
    """TODO MacroFScore docstring"""

    def __init__(self, confusion_matrix: ConfusionMatrix, beta: float = 1.) -> None:
        super().__init__(confusion_matrix)
        self._beta = beta

    def value(self) -> float:
        return self._confusion_matrix.macro_fscore(self._beta).item()

    @staticmethod
    def compute(confusion_matrix: LongTensor, beta: float) -> Tensor:
        fs = FScore.compute(confusion_matrix, beta)
        return _macro_avg(confusion_matrix, fs)


def _weighted_avg(confusion_matrix: LongTensor, values: Tensor) -> Tensor:
    sup = Support.compute(confusion_matrix)
    values = sup * values / sup.sum(-1, keepdim=True)
    return values.sum()


def _macro_avg(confusion_matrix: LongTensor, values: Tensor) -> Tensor:
    del confusion_matrix
    return values.mean()
