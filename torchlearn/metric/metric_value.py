"""TODO metric_value docstring
"""
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterator, Optional

from .metric_state import MetricState


def _compute_unimplemented(self: Any, *args: Any, **kwargs: Any) -> Any:
    """TODO _compute_unimplemented docstring

    :param self:
    :param args:
    :param kwargs:
    :return:
    """
    raise NotImplementedError(f'MetricValue [{type(self).__name__}] is missing the required "compute" function')


class MetricValue(ABC):
    """TODO MetricValue docstring"""

    _name: str
    _states: Dict[str, MetricState]

    def __init__(self, name: str) -> None:
        self._name = name
        self._states = {}

    def __setattr__(self, name: str, value: Optional[Any]) -> None:
        metric_objects = self.__dict__.get("_states")
        if isinstance(value, MetricState):
            if metric_objects is None:
                raise AttributeError(f"Cannot assign metric objects before " f"{MetricValue.__name__}.__init__ call")
            self.add_state(name, value)
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> Any:
        if "_states" in self.__dict__:
            metric_objects = self.__dict__["_states"]
            if name in metric_objects:
                return metric_objects[name]
        raise AttributeError(f'"{type(self).__name__}" as no attribute "{name}"')

    def __delattr__(self, name: str) -> None:
        if name in self._states:
            del self._states[name]
        else:
            object.__delattr__(self, name)

    def add_state(self, name: str, metric_state: MetricState) -> None:
        if "_states" not in self.__dict__:
            raise AttributeError(f"Cannot assign metric objects before " f"{MetricValue.__name__}.__init__ call")
        elif "." in name:
            raise KeyError('name cannot contain "."')
        elif name == "":
            raise KeyError("name cannot be empty")
        elif hasattr(self, name) and name not in self._states:
            raise KeyError(f"attribute {name} already exists")

        if metric_state is None:
            raise TypeError("object cannot be None")
        else:
            self._states[name] = metric_state

    @abstractmethod
    def value(self) -> float:
        raise NotImplementedError

    compute: Callable[..., Any] = staticmethod(abstractmethod(_compute_unimplemented))

    @property
    def name(self) -> str:
        return self._name

    def states(self) -> Iterator[MetricState]:
        if isinstance(self, MetricState):
            yield self
        if "_states" not in self.__dict__:
            raise AttributeError(f"Cannot access metric objects before " f"{MetricValue.__name__}.__init__ call")
        yield from self._states.values()
