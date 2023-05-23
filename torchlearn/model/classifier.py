"""TODO
"""
from typing import Any

import torch
from torch import nn, Tensor, LongTensor


class Classifier(nn.Module):
    """TODO"""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, *args: Any) -> Tensor:
        return self.model(*args)  # type: ignore

    def predict(self, *args: Any) -> LongTensor:
        return self.classify(self(*args))

    @classmethod
    def classify(cls, outputs: Tensor) -> LongTensor:
        return torch.argmax(outputs, -1) # type: ignore
