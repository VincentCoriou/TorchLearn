"""TODO metric docstring
"""
from abc import ABC
from typing import Sequence

from torchlearn.metric.metric_value import MetricValue
from torchlearn.metric.state import State


class Metric(MetricValue, State, ABC):
    """TODO Metric docstring"""

    def __init__(self, parameters: Sequence[str]) -> None:
        MetricValue.__init__(self)
        State.__init__(self, parameters)
