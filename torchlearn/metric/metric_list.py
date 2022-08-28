"""TODO metric_list docstring
"""
from typing import Any, Mapping, Sequence, Set

from torchlearn.metric.metric_state import MetricState
from torchlearn.metric.metric_value import MetricValue


class MetricList:
    """TODO MetricList docstring"""

    _values: Sequence[MetricValue]
    _states: Set[MetricState]

    @classmethod
    def _check_values(cls, values: Sequence[MetricValue]) -> None:
        unique = set()
        duplicates = []

        for x in map(lambda value: value.name, values):
            if x in unique:
                duplicates.append(x)
            else:
                unique.add(x)
        if len(duplicates) > 0:
            raise ValueError(f"Duplicated metric names: {tuple(duplicates)}.")

    def __init__(self, *args: MetricValue) -> None:
        self._check_values(args)
        self._values = args
        self._states = set(o for a in args for o in a.states())

    def arguments(self) -> Set[str]:
        return set(k for o in self._states for k in o.keys)

    def update(self, results: Mapping[str, Any]) -> None:
        for o in self._states:
            o.process(results)

    def values(self, prefix: str = "") -> Mapping[str, float]:
        return {prefix + v.name: v.value() for v in self._values}

    def keys(self, prefix: str = "") -> Set[str]:
        return {prefix + v.name for v in self._values}

    def reset(self) -> None:
        for o in self._states:
            o.reset()
