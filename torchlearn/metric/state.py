"""TODO metric_state docstring
"""
import inspect
from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence, Union, Optional

import torch


class State(ABC):
    """TODO MetricState docstring"""

    _parameters: Sequence[str]

    def __init__(self, parameters: Optional[Sequence[str]] = None) -> None:
        signature = inspect.signature(self.update)
        if parameters is None:
            parameters = list(signature.parameters)

        if len(parameters) != len(signature.parameters):
            raise ValueError(
                f"State expects {len(signature.parameters)} parameters ({signature.parameters}),"
                f"but only {len(parameters)} parameters were given ({parameters}).")

        self._parameters = tuple(parameters)

    @torch.no_grad()
    def __call__(self, kwargs: Mapping[str, Any]) -> Any:
        kwargs = {name: kwargs[name] for name in self.parameters if name in kwargs}
        return self.update(**kwargs)

    @property
    def parameters(self) -> Sequence[str]:
        return self._parameters

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(self, **kwargs: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def compute(*args: Any) -> Any:
        pass

    def process(self, outputs: Union[Sequence[Any], Mapping[str, Any]]) -> Any:
        if isinstance(outputs, Mapping):
            try:
                outputs = tuple(outputs[a] for a in self._parameters)
            except KeyError as exception:
                key = exception.args[0]
                raise ValueError(
                    f"Missing key from output: expected '{key}' to be present, but only found {tuple(outputs)}."
                ) from exception
        if len(outputs) != len(self._parameters):
            raise ValueError(
                f"Argument mismatch: expected {len(self._parameters)} " f"arguments, but got {len(outputs)} arguments."
            )
        return self.update(*outputs)
