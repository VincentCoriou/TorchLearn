"""TODO numbers docstring
"""
from typing import Union


def check_positive(name: str, value: Union[int, float]) -> None:
    if value < 0:
        raise ValueError(f"Expected value for argument '{name}' to be >= 0, " f"but got {value}.")
