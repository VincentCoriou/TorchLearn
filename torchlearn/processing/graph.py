"""TODO
"""

from typing import Dict, Any, Union, Sequence, Callable, Mapping, Optional, Tuple, Protocol, TypeVar

from torch import nn

from torchlearn.utils.graph import Graph


class ProcessingValues:
    """TODO"""

    graph: "ProcessingGraph"
    values: Dict[str, Any]

    def __init__(self, graph: "ProcessingGraph") -> None:
        self.graph = graph
        self.values = {}

    def input_values(self, **inputs: Any) -> None:
        for k, v in inputs.items():
            self.values[k] = v

    def reset(self) -> None:
        self.values.clear()

    def __getitem__(self, key: str) -> Any:
        if key not in self.values:
            return self.graph.compute(key)
        return self.values[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.values[key] = value


_Keys = Union[str, Sequence[str]]
_KeysMapping = Mapping[str, str]
_Function = Callable[..., Any]
_Inputs = _Keys
_Outputs = _Keys
_InputsKW = _KeysMapping


class ProcessingNode:
    """TODO"""

    _inputs: Sequence[str]
    _outputs: Sequence[str]
    _function: _Function
    _inputs_kw: Dict[str, str]

    def __init__(self, outputs: _Outputs, function: _Function, inputs: _Inputs, inputs_kw: Optional[_InputsKW] = None):
        if isinstance(inputs, str):
            inputs = (inputs,)
        if isinstance(outputs, str):
            outputs = (outputs,)
        if inputs_kw is None:
            inputs_kw = {}
        self._inputs = tuple(inputs)
        self._outputs = tuple(outputs)
        self._function = function
        self._inputs_kw = dict(inputs_kw)

    def compute(self, *inputs: Any, **input_kwargs: Any) -> Sequence[Any]:
        outputs = self._function(*inputs, **input_kwargs)
        return outputs if isinstance(outputs, tuple) else (outputs,)

    @property
    def inputs(self) -> Sequence[str]:
        return self._inputs

    @property
    def outputs(self) -> Sequence[str]:
        return self._outputs

    @property
    def inputs_kw(self) -> _KeysMapping:
        return dict(self._inputs_kw)


_GraphFunction = Union[
    ProcessingNode, Tuple[_Outputs, _Function, _Inputs], Tuple[_Outputs, _Function, _Inputs, _InputsKW]
]
_Constants = Mapping[str, Any]


class ProcessingGraph:
    """TODO"""

    graph: Graph
    nodes: Mapping[str, ProcessingNode]

    def __init__(
            self, *, inputs: Sequence[str], functions: Sequence[_GraphFunction], constants: Optional[_Constants] = None
    ):
        inputs = tuple(inputs)
        processing_nodes = tuple(
            ProcessingNode(*data) if not isinstance(data, ProcessingNode) else data for data in functions
        )
        nodes = {o: n for n in processing_nodes for o in n.outputs}
        if constants is None:
            constants = {}

        overlapping_keys = nodes.keys() & constants.keys()
        if overlapping_keys:
            raise ValueError(f"Found duplicated keys: '{overlapping_keys}'.")

        keys = (*inputs, *constants, *nodes)
        edges = tuple((i, o) for o, n in nodes.items() for i in n.inputs)
        graph = Graph.from_edge_list(keys, edges)
        if graph.is_cyclic():
            raise ValueError("Processing graph needs to be acyclic.")

        self.graph = graph
        self.nodes = nodes
        self.inputs = inputs
        self.constants = constants
        self.values = ProcessingValues(self)

    def __call__(self, *outputs: str, inputs: Union[Sequence[Any], Mapping[str, Any]]) -> Mapping[str, Any]:
        self.values.reset()
        if isinstance(inputs, Sequence):
            inputs = dict(zip(self.inputs, inputs))
        assert self.check_inputs(inputs)
        self.values.input_values(**inputs, **self.constants)
        return self.get_outputs(*outputs)

    def check_inputs(self, inputs: Mapping[str, Any]) -> bool:
        return all(i in inputs for i in self.inputs)

    def __contains__(self, item: str) -> bool:
        return item in self.inputs or item in self.nodes or item in self.constants

    def compute(self, name: str) -> Any:
        node = self.nodes[name]
        inputs = tuple(self.values[a] for a in node.inputs)
        inputs_kw = {k: self.values[v] for k, v in node.inputs_kw.items()}
        outputs = node.compute(*inputs, **inputs_kw)
        if len(node.outputs) != len(outputs):
            raise ValueError(
                f"Invalid output size for outputs {node.outputs}. "
                f"Expected {len(node.outputs)}, but got {len(outputs)}"
            )
        for output, value in zip(node.outputs, outputs):
            self.values[output] = value
        return self.values[name]

    def get_outputs(self, *outputs: str) -> Mapping[str, Any]:
        if len(outputs) == 0:
            outputs = tuple(self.nodes)
        return {a: self.values[a] for a in outputs}


T_contra = TypeVar("T_contra", bound=nn.Module, contravariant=True)


class ProcessingFunction(Protocol[T_contra]):
    def __call__(self, model: T_contra, *args: Any, **kwargs: Any) -> ProcessingGraph:
        pass
