"""TODO
"""
import os
from pathlib import Path
from typing import Sequence, Callable, List, Set, Tuple, Any, Union

import pandas as pd
import torch
from matplotlib import Axes
from torch import Tensor

from torchlearn.objective.objective import Objective

PathLike = Union[str, os.PathLike[str]]


def dominates(s1: Tensor, s2: Tensor, objectives: Sequence[Objective]) -> Tensor:
    size = torch.broadcast_shapes(s1.size(), s2.size())
    s1 = s1.expand(size)
    s2 = s2.expand(size)
    better = torch.empty(size)
    same = torch.empty(size)
    for o, b, s, s1i, s2i in zip(objectives, better.split(1, -1), same.split(1, -1), s1.split(1, -1), s2.split(1, -1)):
        b[..., :] = o.better_fn(s1i, s2i)
        s[..., :] = o.same_fn(s1i, s2i)
    return better.any(-1) & same.all(-1)


class ParetoSet:
    """TODO"""
    all_solutions: List[Tensor]
    pareto_set: Set[int]
    objectives: Sequence[Objective]

    def __init__(self, objectives: Sequence[Objective]):
        self.all_solutions = []
        self.pareto_set = set()
        self.objectives = tuple(objectives)

    def add(self, solution: Tensor) -> Tuple[int, bool, Set[int]]:
        s_id = len(self.all_solutions)
        self.all_solutions.append(solution)
        front_ids = torch.empty(len(self.pareto_set)).long()
        front = torch.empty(len(self.pareto_set), len(self.objectives))
        for i, id_ in enumerate(self.pareto_set):
            front_ids[i] = id_
            front[i] = self.all_solutions[id_]
        dom = dominates(solution, front, self.objectives)
        is_dom = dominates(front, solution, self.objectives)
        deleted = set()
        if not is_dom.any():
            self.pareto_set.add(s_id)
            for s in front_ids[dom]:
                id_ = s.item()
                self.pareto_set.remove(id_)
                deleted.add(id_)
            return s_id, True, deleted
        return s_id, False, deleted

    def get_best(self, function: Callable[[Tensor], float]) -> int:
        return max(self.pareto_set, key=lambda x: function(self.all_solutions[x]))

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(torch.stack(self.all_solutions, 0).numpy(), columns=[o.name for o in self.objectives])

    def clear(self) -> None:
        self.pareto_set.clear()
        self.all_solutions.clear()

    def from_pandas(self, df: pd.DataFrame) -> "ParetoSet":
        if self.all_solutions or self.pareto_set:
            raise ValueError("Cannot load in non-empty ParetoSet. Please clear the pareto set beforehand.")
        df = df[[o.name for o in self.objectives]].sort_index()
        for solution in torch.from_numpy(df.values):
            self.add(solution)
        return self

    def save(self, *segments: PathLike) -> None:
        path = Path(*segments)
        df = self.to_pandas()
        df.to_csv(path)

    def load(self, *segments: PathLike) -> "ParetoSet":
        path = Path(*segments)
        df = pd.read_csv(path, index_col=0)
        return self.from_pandas(df)

    def plot(self, ax: Axes, **kwargs: Any) -> Axes:
        if len(self.objectives) != 2:
            raise ValueError("Cannot plot pareto front with more than 2 objectives")

        solutions = torch.stack(self.all_solutions, -1)

        ax.scatter(*solutions, c="grey", alpha=0.2, **kwargs)
        ax.scatter(*solutions[..., list(self.pareto_set)], c="blue", **kwargs)

        for i, (x, y) in enumerate(self.all_solutions):
            ax.annotate(i, (x, y))

        ax.set_xlabel(self.objectives[0])
        ax.set_ylabel(self.objectives[1])
        return ax
