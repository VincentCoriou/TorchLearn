"""TODO metric docstring
"""
from abc import ABC
from typing import Sequence

from .metric_state import MetricState
from .metric_value import MetricValue


class Metric(MetricValue, MetricState, ABC):
    """TODO Metric docstring"""

    def __init__(self, name: str, keys: Sequence[str]) -> None:
        MetricValue.__init__(self, name)
        MetricState.__init__(self, keys)
