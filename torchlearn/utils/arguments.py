"""TODO arguments docstring
"""
from typing import Any, Sequence, TypeGuard


def check_one_mandatory(**kwargs: Any) -> Sequence[str]:
    arguments = tuple(a for a, v in kwargs.items() if v is not None)
    if len(arguments) == 0:
        raise ValueError(f"Need to provide at least one argument of {tuple(kwargs)}")
    return arguments


def is_sequence(values: Any) -> TypeGuard[Sequence[Any]]:
    return isinstance(values, Sequence)


def expand_list(size: int, values: Any) -> Sequence[Any]:
    if not isinstance(values, Sequence):
        values = (values,) * size
    assert is_sequence(values)
    if len(values) != size:
        raise ValueError(f"Expected sequence of length {size}, but got {len(values)}")
    return values
