"""TODO
"""
from typing import Optional, Type, Sequence

import torch
from torch import nn

from torchlearn.model.perceptron import Perceptron
from torchlearn.utils.arguments import expand_list


class MultiLayerPerceptron(nn.Sequential):
    """TODO"""

    def __init__(
            self,
            input_size: int,
            output_size: int,
            num_layers: int,
            hidden_units: int | Sequence[int],
            activation: nn.Module | Type[nn.Module] = nn.ReLU,
            dropout: float | Sequence[float] = 0.0,
            bias: bool = True,
            device: Optional[torch.device] = None,
    ) -> None:
        hidden_units = expand_list(num_layers, hidden_units)
        sizes = (input_size, *hidden_units)
        activations = expand_list(num_layers, activation)
        dropouts = expand_list(num_layers, dropout)
        input_sizes = sizes[:-1]
        output_sizes = sizes[1:]
        layers = [
            Perceptron(i, o, a, bias, device, d) for i, o, d, a in zip(input_sizes, output_sizes, dropouts, activations)
        ]
        last_layer = Perceptron(sizes[-1], output_size, bias=bias)
        super().__init__(*layers, last_layer)
