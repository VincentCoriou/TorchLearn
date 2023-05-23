"""TODO
"""
from abc import abstractmethod
from typing import Mapping
from typing import Optional

import torch

from torchlearn.objective.objective import Objective
from torchlearn.training.results import Result


class EarlyStopping:
    @abstractmethod
    def step(self, epoch: int, train_result: Result, validation_result: Optional[Result] = None) -> bool:
        pass


class Patience(EarlyStopping):
    """TODO"""

    def __init__(self, patience: int, objective: Objective | Mapping[str, str], validation: bool = False,
                 threshold: float = 0.) -> None:
        if isinstance(objective, Mapping):
            if not len(objective) == 1:
                raise ValueError(
                    f"Early stopping expects only 1 objective, but {len(objective)} were given: '{objective}'.")
            name, config = next(iter(objective.items()))
            obj = Objective.from_config(name, config)
        else:
            obj = objective
        self.patience = patience
        self.objective = obj
        self.validation = validation
        self.threshold = threshold

    def step(self, epoch: int, train_result: Result, validation_result: Optional[Result] = None) -> bool:
        if self.validation and validation_result is None:
            raise ValueError("'validation_result' should not be None when 'validation' is 'True'.")
        result = validation_result if self.validation else train_result
        assert result is not None

        if epoch <= self.patience:
            return False

        results = torch.tensor([result[e][self.objective.name] for e in range(epoch - self.patience, epoch + 1)])
        return self.objective.threshold_fn(results[0], results[1:], self.threshold).all().item()  # type: ignore
