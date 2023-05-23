"""TODO
"""
import itertools
from typing import Union, Sequence, Tuple, Generic, TypeVar, Hashable, Iterator

import torch
from torch import Tensor

T = TypeVar("T", bound=Hashable)
Edge = Union[Tuple[T, T], Tuple[T, T, float]]


class CyclicGraphException(BaseException):
    pass


class Graph(Generic[T]):
    """TODO"""

    nodes: Sequence[T]
    n: int
    adjacency: torch.Tensor

    def __init__(self, nodes: Sequence[T], adjacency: torch.Tensor) -> None:
        for x, y in itertools.combinations(nodes, 2):
            if x == y:
                raise ValueError(f"Duplicated node found: '{x}'")
        n = len(nodes)
        if adjacency.size() != (n, n):
            raise ValueError(
                f"Expected square adjacency matrix of size ({n}, {n}) but found {tuple(adjacency.size())}."
            )
        self.nodes = nodes
        self.n = len(nodes)
        self.adjacency = adjacency

    @classmethod
    def from_edge_list(cls, nodes: Sequence[T], edges: Sequence[Edge]) -> "Graph":
        n = len(nodes)
        adjacency = torch.zeros(n, n)
        id2node = dict(enumerate(nodes))
        node2id = {k: i for i, k in id2node.items()}
        for edge in edges:
            src, dst, weight = (*edge, 1) if len(edge) == 2 else edge
            for node in (src, dst):
                if node not in node2id:
                    raise ValueError(f"Found invalid node '{node}', should be one of '{tuple(nodes)}'.")
            src_id = node2id[src]
            dst_id = node2id[dst]
            adjacency[src_id, dst_id] += weight
        return cls(nodes, adjacency)

    def _dfs_traversal(self, node: int, visited: Tensor, path: Tensor, detect_cycle: bool = False) -> Iterator[int]:
        if path[node] and detect_cycle:
            raise CyclicGraphException()
        if not visited[node]:
            yield node
            visited[node] = True
            path[node] = True
            (neighbors,) = self.adjacency[node].nonzero(as_tuple=True)
            for neighbor in neighbors:
                yield from self._dfs_traversal(neighbor.item(), visited, path, detect_cycle)
            path[node] = False

    def dfs(self, detect_cycle: bool = False) -> Iterator[int]:
        visited = torch.zeros(self.n).bool()
        path = torch.zeros(self.n).bool()
        for node in range(self.n):
            yield from self._dfs_traversal(node, visited, path, detect_cycle)

    def is_cyclic(self) -> bool:
        try:
            for _ in self.dfs(True):
                pass
            return False
        except CyclicGraphException:
            return True
