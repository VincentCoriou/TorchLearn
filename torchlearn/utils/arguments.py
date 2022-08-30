"""TODO arguments docstring
"""
from typing import Any, Sequence


def check_one_mandatory(**kwargs: Any) -> Sequence[str]:
    arguments = tuple(a for a, v in kwargs.items() if v is not None)
    if len(arguments) == 0:
        raise ValueError(f"Need to provide at least one argument of {tuple(kwargs)}")
    return arguments
