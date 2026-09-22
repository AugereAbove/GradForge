import math

import pytest

from gradforge import Value


def finite_difference(fn, x: float, epsilon: float = 1e-6) -> float:
    return (fn(x + epsilon) - fn(x - epsilon)) / (2 * epsilon)


def test_power_backward() -> None:
    x = Value(2.0)
    (x**3).backward()
    assert x.grad == pytest.approx(12.0)


def test_variable_exponent_backward_matches_analytical_derivatives() -> None:
    base = Value(2.0)
    exponent = Value(3.0)

    (base**exponent).backward()

    assert base.grad == pytest.approx(12.0)
    assert exponent.grad == pytest.approx(8.0 * math.log(2.0))


def test_self_power_accumulates_base_and_exponent_partials() -> None:
    x = Value(2.0)

    (x**x).backward()

    assert x.grad == pytest.approx(4.0 * (1.0 + math.log(2.0)))


def test_variable_exponent_gradients_match_finite_differences() -> None:
    base_data = 1.7
    exponent_data = 2.3
    base = Value(base_data)
    exponent = Value(exponent_data)

    (base**exponent).backward()

    assert base.grad == pytest.approx(
        finite_difference(lambda candidate: candidate**exponent_data, base_data), rel=1e-5
    )
    assert exponent.grad == pytest.approx(
        finite_difference(lambda candidate: base_data**candidate, exponent_data), rel=1e-5
    )


def test_reverse_power_backpropagates_to_exponent() -> None:
    exponent = Value(2.0)
    output = 3.0**exponent

    output.backward()

    assert output.data == pytest.approx(9.0)
    assert exponent.grad == pytest.approx(9.0 * math.log(3.0))


def test_differentiable_exponent_rejects_non_positive_base() -> None:
    with pytest.raises(ValueError, match="positive base"):
        Value(0.0) ** Value(2.0)


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


def test_backward_accumulates_shared_leaf_gradients_across_outputs() -> None:
    x = Value(2.0)
    linear_output = x * 2.0
    cubic_output = x**3

    linear_output.backward()
    cubic_output.backward()

    assert x.grad == pytest.approx(14.0)
