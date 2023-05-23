"""TODO metric_state docstring
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping, Sequence, Union


def _update_unimplemented(self: Any, *outputs: Any) -> None:
    raise NotImplementedError(f'MetricState [{type(self).__name__}] is missing the required "update" function')


def _compute_unimplemented(self: Any, *outputs: Any) -> None:
    raise NotImplementedError(f'MetricState [{type(self).__name__}] is missing the required "compute" function')


class MetricState(ABC):
    """TODO MetricState docstring"""

    _keys: Sequence[str]

    def __init__(self, keys: Sequence[str]) -> None:
        self._keys = tuple(keys)

    @property
    def keys(self) -> Sequence[str]:
        return self._keys

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    update: Callable[..., Any] = abstractmethod(_update_unimplemented)

    compute: Callable[..., Any] = staticmethod(abstractmethod(_compute_unimplemented))

    def process(self, outputs: Union[Sequence[Any], Mapping[str, Any]]) -> Any:
        if isinstance(outputs, Mapping):
            try:
                outputs = tuple(outputs[a] for a in self._keys)
            except KeyError as exception:
                key = exception.args[0]
                raise ValueError(
                    f"Missing key from output: expected '{key}' " f"but only got {tuple(outputs)}."
                ) from exception
        if len(outputs) != len(self._keys):
            raise ValueError(
                f"Argument mismatch: expected {len(self._keys)} " f"arguments, but got {len(outputs)} arguments."
            )
        return self.update(*outputs)
