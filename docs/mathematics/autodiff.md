# Scalar reverse-mode autodiff

GradForge represents each scalar as a node in a dynamic computation graph. A node stores its value, its parent nodes, and a local backward function describing how an output gradient contributes to each parent.

For an operation `z = f(x, y)`, reverse mode receives `∂L/∂z` and applies the chain rule:

`∂L/∂x += (∂L/∂z)(∂z/∂x)`

The `+=` is important: when a value is reused in multiple branches, each branch contributes to the same gradient. `Value.backward()` first builds a topological ordering of reachable nodes, clears gradients on graph intermediates, seeds the output with gradient 1, and visits that ordering in reverse. Leaf gradients are deliberately retained, so backward calls on independent outputs that share a leaf accumulate their contributions.

Exponentiation supports either a Python scalar exponent or another `Value`. For `z = x^y` with a differentiable exponent and `x > 0`, the local derivatives are:

`∂z/∂x = yx^(y - 1)`

`∂z/∂y = x^y ln(x)`

The positive-base restriction is necessary for the real-valued derivative with respect to `y`. Reverse exponentiation is also supported, so `3.0 ** y` differentiates as `3.0^y ln(3.0)`.

Values can be frozen with `requires_grad=False`. That state propagates through operations: a result requires gradients only when at least one input does. A frozen `Value` exponent does not need the logarithmic derivative, so negative bases remain valid whenever their forward power is real (for example, `(-2)^3`). Calling `backward()` on an entirely frozen graph raises an error rather than silently writing a gradient into it.

Power operations explicitly stay in the real-valued, finite-gradient domain. A negative base therefore requires an integer frozen exponent, and a zero base requires a positive exponent. When the base needs gradients, zero bases also reject exponents below one because their base derivative is not finite. Fully frozen expressions may still evaluate a real forward-only boundary case such as `0^0.5`.

The implementation lives in `src/gradforge/value.py`. Tests compare elementary derivatives with centered finite differences and explicitly cover reused values, repeated backward calls, shared leaves reached through distinct outputs, and differentiable exponentiation.
