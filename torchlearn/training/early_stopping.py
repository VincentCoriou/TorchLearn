from abc import abstractmethod
from typing import Mapping
from typing import Optional

import torch

from fade.objective.pareto import Objective
from fade.training.result import Result
from fade.utils import Registry, parse_config


class EarlyStoppingRegistry(Registry):
    pass


EARLY_STOPPINGS = EarlyStoppingRegistry()


def parse_early_stopping(early_stopping):
    name, args, kwargs = parse_config(early_stopping, "class")
    return EARLY_STOPPINGS[name](*args, **kwargs)


class EarlyStopping:
    @abstractmethod
    def step(self, epoch, train_result, validation_result=None) -> bool:
        pass


@EARLY_STOPPINGS.register("Patience")
class Patience(EarlyStopping):
    def __init__(self, patience, objective, validation=False, threshold=0.):
        if isinstance(objective, Mapping):
            if not len(objective) == 1:
                raise ValueError(
                    f"Early stopping expects only 1 objective, but {len(objective)} were given: '{objective}'.")
            name, config = next(iter(objective.items()))
            objective = Objective.from_config(name, config)
        self.patience = patience
        self.objective = objective
        self.validation = validation
        self.threshold = threshold

    def step(self, epoch, train_result: Result, validation_result: Optional[Result] = None):
        if self.validation and validation_result is None:
            raise ValueError("'validation_result' should not be None when 'validation' is 'True'.")
        result = validation_result if self.validation else train_result

        if epoch <= self.patience:
            return False

        results = torch.tensor([result[e][self.objective.name] for e in range(epoch - self.patience, epoch + 1)])
        return self.objective.threshold_fn(results[0], results[1:], self.threshold).all()
