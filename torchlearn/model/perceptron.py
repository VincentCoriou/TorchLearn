"""TODO
"""
from typing import Any, Optional, Type, List

import torch
from torch import nn


class Perceptron(nn.Sequential):
    """TODO"""

    def __init__(
            self,
            in_size: int,
            out_size: int,
            activation: Optional[nn.Module | Type[nn.Module]] = None,
            bias: bool = True,
            device: Optional[torch.device] = None,
            dropout: float = 0.,
            **kwargs: Any
    ) -> None:
        modules: List[nn.Module] = [nn.Linear(in_size, out_size, bias=bias, device=device)]
        if activation is not None:
            if isinstance(activation, type):
                activation = activation(**kwargs)
            modules.append(activation)
        if dropout > 0:
            modules.append(nn.Dropout(dropout))
        super().__init__(*modules)
