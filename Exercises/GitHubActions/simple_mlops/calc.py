"""A deliberately tiny module so we can focus on CI plumbing, not the code.

The GitHubActions exercise is about wiring up workflows, not about teaching
new domain logic. Two trivial functions are enough to demonstrate "tests run
on every push and PR" without distracting students.
"""

from __future__ import annotations


def add(a: float, b: float) -> float:
    """Return ``a + b``.

    Floats only so we don't have to think about overflow. The whole point is
    that this is boring enough that *the test pipeline* is the interesting
    object of study.
    """
    return a + b


def divide(numerator: float, denominator: float) -> float:
    """Return ``numerator / denominator``.

    Raises ``ZeroDivisionError`` if ``denominator`` is zero, just like Python's
    built-in ``/`` operator. The test suite uses this to demonstrate
    ``pytest.raises``.
    """
    if denominator == 0:
        raise ZeroDivisionError("denominator must be non-zero")
    return numerator / denominator
