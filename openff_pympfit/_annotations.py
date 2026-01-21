from collections.abc import Callable
from functools import partial
from typing import Annotated

import numpy as np
from openff.toolkit import Quantity
from pydantic import BeforeValidator


def _array_validator(
    value: np.ndarray | Quantity,
    unit: str,
) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, Quantity):
        return value.m_as(unit)
    raise ValueError(f"Invalid type {type(value)}")


def validator_factory(unit: str) -> Callable:
    """Return a function that converts the input array in given implicit units.

    This is meant to be used as the argument to pydantic.BeforeValidator
    in an Annotated type.
    """
    return partial(_array_validator, unit=unit)


Coordinates = Annotated[
    np.ndarray[float],
    BeforeValidator(validator_factory(unit="angstrom")),
]

MP = Annotated[
    np.ndarray[float],
    BeforeValidator(validator_factory(unit="AU")),
]
