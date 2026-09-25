"""Scalar reverse-mode automatic differentiation.

Each ``Value`` stores a scalar and a local backward rule. Calling ``backward``
topologically visits the computation graph in reverse and accumulates the
chain-rule contributions into each node's ``grad`` field.
"""

from __future__ import annotations

import math
from collections.abc import Callable


class Value:
    """A scalar value participating in a dynamic reverse-mode graph."""

    def __init__(
        self,
        data: float,
        _children: tuple[Value, ...] = (),
        _op: str = "",
        requires_grad: bool = True,
    ) -> None:
        self.data = float(data)
        self.grad = 0.0
        self.requires_grad = requires_grad
        self._prev = set(_children)
        self._op = _op
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

    def _binary(self, other: Value | float, op: str, fn: Callable[[float, float], float]) -> Value:
        rhs = other if isinstance(other, Value) else Value(other, requires_grad=False)
        out = Value(
            fn(self.data, rhs.data),
            (self, rhs),
            op,
            requires_grad=self.requires_grad or rhs.requires_grad,
        )

        def backward() -> None:
            if self.requires_grad:
                self.grad += out.grad * self._local_derivative(rhs.data, op, left=True)
            if rhs.requires_grad:
                rhs.grad += out.grad * self._local_derivative(self.data, op, left=False)

        out._backward = backward
        return out

    def _local_derivative(self, other: float, op: str, *, left: bool) -> float:
        if op == "+":
            return 1.0
        if op == "-":
            return 1.0 if left else -1.0
        if op == "*":
            return other
        if op == "/":
            return 1.0 / other if left else -self.data / (other**2)
        raise ValueError(f"Unknown operation: {op}")

    def __add__(self, other: Value | float) -> Value:
        return self._binary(other, "+", lambda a, b: a + b)

    def __radd__(self, other: float) -> Value:
        return self + other

    def __sub__(self, other: Value | float) -> Value:
        return self._binary(other, "-", lambda a, b: a - b)

    def __rsub__(self, other: float) -> Value:
        return Value(other, requires_grad=False) - self

    def __mul__(self, other: Value | float) -> Value:
        return self._binary(other, "*", lambda a, b: a * b)

    def __rmul__(self, other: float) -> Value:
        return self * other

    def __truediv__(self, other: Value | float) -> Value:
        return self._binary(other, "/", lambda a, b: a / b)

    def __rtruediv__(self, other: float) -> Value:
        return Value(other, requires_grad=False) / self

    def __neg__(self) -> Value:
        return self * -1.0

    def __pow__(self, exponent: Value | float) -> Value:
        """Raise this value to a scalar exponent.

        An exponent that requires gradients is differentiable with respect to
        both operands. Its derivative uses ``log(self.data)``, so that
        real-valued operation requires a strictly positive base.
        """
        rhs = exponent if isinstance(exponent, Value) else Value(exponent, requires_grad=False)
        if rhs.requires_grad and self.data <= 0.0:
            raise ValueError("A differentiable exponent requires a positive base")
        if self.data < 0.0 and not rhs.data.is_integer():
            raise ValueError("A negative base requires an integer exponent in real-valued mode")
        if self.data == 0.0:
            if rhs.data <= 0.0:
                raise ValueError("A zero base requires a positive exponent")
            if self.requires_grad and rhs.data < 1.0:
                raise ValueError("A zero base requires an exponent of at least 1 for a finite gradient")

        out = Value(
            self.data**rhs.data,
            (self, rhs),
            "**",
            requires_grad=self.requires_grad or rhs.requires_grad,
        )

        def backward() -> None:
            if self.requires_grad:
                base_derivative = 0.0 if rhs.data == 0.0 else rhs.data * (self.data ** (rhs.data - 1))
                self.grad += out.grad * base_derivative
            if rhs.requires_grad:
                rhs.grad += out.grad * out.data * math.log(self.data)

        out._backward = backward
        return out

    def __rpow__(self, other: float) -> Value:
        return Value(other, requires_grad=False) ** self

    def exp(self) -> Value:
        out = Value(math.exp(self.data), (self,), "exp", requires_grad=self.requires_grad)

        def backward() -> None:
            if self.requires_grad:
                self.grad += out.grad * out.data

        out._backward = backward
        return out

    def log(self) -> Value:
        out = Value(math.log(self.data), (self,), "log", requires_grad=self.requires_grad)

        def backward() -> None:
            if self.requires_grad:
                self.grad += out.grad / self.data

        out._backward = backward
        return out

    def tanh(self) -> Value:
        out = Value(math.tanh(self.data), (self,), "tanh", requires_grad=self.requires_grad)

        def backward() -> None:
            if self.requires_grad:
                self.grad += out.grad * (1.0 - out.data**2)

        out._backward = backward
        return out

    def relu(self) -> Value:
        out = Value(max(0.0, self.data), (self,), "relu", requires_grad=self.requires_grad)

        def backward() -> None:
            if self.requires_grad:
                self.grad += out.grad * float(self.data > 0.0)

        out._backward = backward
        return out

    def backward(self) -> None:
        """Accumulate gradients from this scalar output through its graph."""
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward on a Value that does not require gradients")

        topo: list[Value] = []
        visited: set[Value] = set()

        def build(node: Value) -> None:
            if node not in visited:
                visited.add(node)
                for parent in node._prev:
                    build(parent)
                topo.append(node)

        build(self)
        # Intermediate gradients belong to this traversal, so reset every node
        # that has parents (including this output when it is non-leaf). Leaf
        # gradients are intentionally preserved: backpropagating independent
        # outputs that share a leaf must accumulate their contributions.
        for node in topo:
            if node._prev:
                node.grad = 0.0
        self.grad += 1.0
        for node in reversed(topo):
            node._backward()
