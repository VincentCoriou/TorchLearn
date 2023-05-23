"""TODO
"""
import os
from pathlib import Path
from typing import Sequence, Callable, Union

import torch
from torch import Tensor
from torch import nn

from torchlearn.objective.objective import Objective
from torchlearn.objective.pareto import ParetoSet
from torchlearn.training.results import Result

PathLike = Union[str, os.PathLike[str]]


class ParetoManager:
    """TODO"""

    def __init__(self, *path: PathLike, objectives: Sequence[Objective], validation: bool = False) -> None:
        self.path = Path(*path)
        self.objectives = objectives
        self.validation = validation
        self.pareto_set = ParetoSet(objectives)

        self.path.mkdir(parents=True, exist_ok=False)

    def __call__(self, model: nn.Module, epoch: int, results: Sequence[Result]) -> None:
        epoch_results = results[1] if self.validation and len(results) == 2 else results[0]
        solution = torch.tensor([epoch_results[epoch][o.name] for o in self.objectives])
        s_id, pareto_optimal, deleted = self.pareto_set.add(solution)
        if pareto_optimal:
            self._save_solution(s_id, model)
        for id_ in deleted:
            self._delete_solution(id_)
        self.pareto_set.save(self.path, "solutions.csv")

    def _save_solution(self, s_id: int, model: nn.Module) -> None:
        path = self.path / f"{s_id}.pth"
        torch.save(model.state_dict(), path)

    def _delete_solution(self, s_id: int) -> None:
        path = self.path / f"{s_id}.pth"
        path.unlink()

    def _load_solution(self, s_id: int, model: nn.Module) -> None:
        path = self.path / f"{s_id}.pth"
        model.load_state_dict(torch.load(path))

    def load_best_model(self, model: nn.Module, function: Callable[[Tensor], float]) -> nn.Module:
        s_id = self.pareto_set.get_best(function)
        self._load_solution(s_id, model)
        return model
