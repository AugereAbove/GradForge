import math

import pytest

from gradforge import Value


def finite_difference(fn, x: float, epsilon: float = 1e-6) -> float:
    return (fn(x + epsilon) - fn(x - epsilon)) / (2 * epsilon)


def test_power_backward() -> None:
    x = Value(2.0)
    (x**3).backward()
    assert x.grad == pytest.approx(12.0)


def test_branching_accumulates_reused_value() -> None:
    x = Value(3.0)
    y = x * x + x
    y.backward()
    assert y.data == 12.0
    assert x.grad == pytest.approx(7.0)


@pytest.mark.parametrize("name", ["exp", "log", "tanh", "relu"])
def test_unary_derivatives_match_finite_difference(name: str) -> None:
    x = Value(1.2 if name != "log" else 1.7)
    y = getattr(x, name)()
    y.backward()
    fn = getattr(math, name) if name != "relu" else lambda v: max(0.0, v)
    assert x.grad == pytest.approx(finite_difference(fn, x.data), rel=1e-5, abs=1e-7)


def test_chain_rule_expression() -> None:
    x = Value(2.0)
    y = ((x * 3.0 + 1.0).tanh())
    y.backward()
    expected = 3.0 * (1.0 - math.tanh(7.0) ** 2)
    assert x.grad == pytest.approx(expected)


def test_backward_accumulates_across_calls() -> None:
    x = Value(2.0)
    y = x * x
    y.backward()
    y.backward()
    assert x.grad == pytest.approx(8.0)
