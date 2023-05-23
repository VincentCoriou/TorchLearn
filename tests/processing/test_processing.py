from typing import Any, Mapping, Sequence, Union

import pytest

from torchlearn.processing.graph import ProcessingGraph


class DummyModel:
    """Class implementing a dummy ML model."""

    def f3(self, f0: Any, f1: Any) -> Any:
        return f0 + f1

    def f4(self, f0: Any, f2: Any) -> Any:
        return f0 * f2

    def f5_f6(self, f0: Any, f1: Any, f2: Any, f3: Any) -> Any:
        return f0 / f2, f1 / f3

    def f7(self, f3: Any, f6: Any) -> Any:
        return f3 % f6

    def f8(self, f1: Any) -> Any:
        return f1


@pytest.fixture(name="model")
def fixture_model() -> DummyModel:
    return DummyModel()


@pytest.fixture(name="processing_graph")
def fixture_processing_graph(model: DummyModel) -> ProcessingGraph:
    inputs = ("f0", "f1", "f2")
    functions = (
        ("f3", model.f3, ("f0", "f1")),
        ("f4", model.f4, ("f0", "f2")),
        (("f5", "f6"), model.f5_f6, ("f0", "f1", "f2", "f3")),
        ("f7", model.f7, ("f3", "f6")),
        ("f8", model.f8, "f1"),
    )

    return ProcessingGraph(inputs=inputs, functions=functions)


class TestProcessingGraph:
    """Class encapsulating Processing Graph Tests."""

    @pytest.mark.parametrize(
        "inputs, outputs",
        [
            ({"f0": 1, "f1": 2, "f2": 3}, ("f5",)),
            ((1, 2, 3), ("f5",)),
            ((1, 2, 3), ()),
        ],
    )
    def test_call(
        self,
        processing_graph: ProcessingGraph,
        inputs: Union[Sequence[Any], Mapping[str, Any]],
        outputs: Sequence[str],
    ) -> None:
        processing_graph(*outputs, inputs=inputs)

    @pytest.mark.parametrize("elements", [("f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8")])
    def test_contains(self, processing_graph: ProcessingGraph, elements: Sequence[str]) -> None:
        for e in elements:
            assert e in processing_graph
