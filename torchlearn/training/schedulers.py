"""TODO
"""
from typing import Any, Protocol, runtime_checkable

from torch import nn


@runtime_checkable
class Scheduler(Protocol):
    def step(self, model: nn.Module, epoch: int) -> Any:
        """Steps the scheduler"""
