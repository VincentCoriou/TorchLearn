"""TODO
"""
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor

_BetterFN = Callable[[Tensor, Tensor], Tensor]
_SameFN = Callable[[Tensor, Tensor], Tensor]
_ThresholdFN = Callable[[Tensor, Tensor, float], Tensor]


def parse_objectives(objectives_configuration: Mapping[str, str]) -> Sequence["Objective"]:
    return tuple(Objective.from_config(name, config) for name, config in objectives_configuration.items())


def _gt_threshold(x: Tensor, y: Tensor, t: float) -> Tensor:
    return torch.gt(x - t, y)


def _lt_threshold(x: Tensor, y: Tensor, t: float) -> Tensor:
    return torch.lt(x + t, y)


class Objective:
    """TODO"""

    def __init__(self, name: str, better_fn: _BetterFN, same_fn: _SameFN, threshold_fn: _ThresholdFN) -> None:
        self.name = name
        self.better_fn = better_fn
        self.same_fn = same_fn
        self.threshold_fn = threshold_fn

    @classmethod
    def minimize(cls, name: str) -> "Objective":
        return cls(name, torch.lt, torch.le, _lt_threshold)

    @classmethod
    def maximize(cls, name: str) -> "Objective":
        return cls(name, torch.gt, torch.ge, _gt_threshold)

    @classmethod
    def from_config(cls, name: str, config: Any) -> "Objective":
        if config in {"maximize", "max"}:
            return cls.maximize(name)
        elif config in {"minimize", "min"}:
            return cls.minimize(name)
        raise ValueError(f"Unrecognized configuration for objective '{name}': {config}.")
