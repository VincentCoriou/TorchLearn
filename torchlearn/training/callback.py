"""TODO
"""

from typing import runtime_checkable, Protocol, Sequence, TypeVar

from torch import nn

from torchlearn.training.results import Result

T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class Callback(Protocol[T_co]):
    def __call__(self, module: nn.Module, epoch: int, results: Sequence[Result]) -> T_co:
        raise NotImplementedError
