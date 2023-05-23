"""TODO metric_dict module docstring
"""
from functools import reduce
from operator import or_
from typing import FrozenSet

from torchlearn.metric.metric_value import MetricValue
from torchlearn.metric.state import State


class MetricDict(dict):
    """TODO MetricDict docstring"""
    _states: FrozenSet[State]
    _parameters: FrozenSet[str]

    def __init__(self, **metrics: MetricValue) -> None:
        super().__init__(**metrics)
        self._states = reduce(or_, (set(v.states()) for v in metrics.values()), frozenset())
        self._parameters = reduce(or_, (set(v.parameters) for v in self._states), frozenset())

    @property
    def states(self) -> FrozenSet[State]:
        return self._states

    @property
    def parameters(self) -> FrozenSet[str]:
        return self._parameters

    def reset(self) -> None:
        for s in self._states:
            s.reset()
