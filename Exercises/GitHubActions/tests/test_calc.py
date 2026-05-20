"""Tests that are intentionally simple and green out of the box.

The point of this exercise is to watch a CI run go green when you push, not
to debug failing tests. If you want a richer test suite, see the
``PythonUnitTesting`` exercise.
"""

from __future__ import annotations

import pytest

from simple_mlops.calc import add, divide


def test_add_returns_sum() -> None:
    """add(2, 3) returns 5."""
    assert add(2, 3) == 5, "add should return the arithmetic sum"


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (0, 0, 0),
        (1, 1, 2),
        (-1, 1, 0),
        (1.5, 2.5, 4.0),
    ],
)
def test_add_parametrized(a: float, b: float, expected: float) -> None:
    """add behaves over a small grid of cases."""
    assert add(a, b) == expected


def test_divide_returns_quotient() -> None:
    """divide(10, 2) returns 5.0."""
    assert divide(10, 2) == 5.0


def test_divide_by_zero_raises() -> None:
    """Dividing by zero raises ZeroDivisionError."""
    with pytest.raises(ZeroDivisionError):
        divide(1, 0)
