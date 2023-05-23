import pytest

from torchlearn.utils.graph import Graph

nodes = [0, 1, 2, 3, 4]
cyclic_edges = [(0, 1), (1, 2), (2, 0)]
not_cyclic_edges = [(0, 1), (1, 2), (2, 4), (4, 3)]
cyclic_graph = (nodes, cyclic_edges)
not_cyclic_graph = (nodes, not_cyclic_edges)

cyclic_dfs = [0, 1, 2, 3, 4]
not_cyclic_dfs = [0, 1, 2, 4, 3]


@pytest.fixture()
def graph(request):
    return Graph.from_edge_list(*request.param)


parametrized_graphs = [(cyclic_graph, True, cyclic_dfs), (not_cyclic_graph, False, not_cyclic_dfs)]


@pytest.mark.parametrize("graph, cyclic, dfs", parametrized_graphs, indirect=["graph"])
class TestGraph:
    def test_cyclic(self, graph, cyclic, dfs):
        assert graph.is_cyclic() == cyclic

    def test_dfs(self, graph, cyclic, dfs):
        assert list(graph.dfs()) == dfs
