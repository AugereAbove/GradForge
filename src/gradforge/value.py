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
        out = Value(fn(self.data, rhs.data), (self, rhs), op)

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

    def __pow__(self, exponent: float) -> Value:
        out = Value(self.data**exponent, (self,), "**")

        def backward() -> None:
            if self.requires_grad:
                self.grad += out.grad * exponent * (self.data ** (exponent - 1))

        out._backward = backward
        return out

    def exp(self) -> Value:
        out = Value(math.exp(self.data), (self,), "exp")

        def backward() -> None:
            if self.requires_grad:
                self.grad += out.grad * out.data

        out._backward = backward
        return out

    def log(self) -> Value:
        out = Value(math.log(self.data), (self,), "log")

        def backward() -> None:
            if self.requires_grad:
                self.grad += out.grad / self.data

        out._backward = backward
        return out

    def tanh(self) -> Value:
        out = Value(math.tanh(self.data), (self,), "tanh")

        def backward() -> None:
            if self.requires_grad:
                self.grad += out.grad * (1.0 - out.data**2)

        out._backward = backward
        return out

    def relu(self) -> Value:
        out = Value(max(0.0, self.data), (self,), "relu")

        def backward() -> None:
            if self.requires_grad:
                self.grad += out.grad * float(self.data > 0.0)

        out._backward = backward
        return out

    def backward(self) -> None:
        """Accumulate gradients from this scalar output through its graph."""
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
