from typing import runtime_checkable, Protocol, Sequence, TypeVar

from torch import nn

from torchlearn.training.results import Result

T = TypeVar("T")


@runtime_checkable
class Callback(Protocol[T]):
    def __call__(self, module: nn.Module, epoch: int, results: Sequence[Result]) -> T:
        raise NotImplementedError
