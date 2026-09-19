# Scalar reverse-mode autodiff

GradForge represents each scalar as a node in a dynamic computation graph. A node stores its value, its parent nodes, and a local backward function describing how an output gradient contributes to each parent.

For an operation `z = f(x, y)`, reverse mode receives `∂L/∂z` and applies the chain rule:

`∂L/∂x += (∂L/∂z)(∂z/∂x)`

The `+=` is important: when a value is reused in multiple branches, each branch contributes to the same gradient. `Value.backward()` first builds a topological ordering of reachable nodes, seeds the output with gradient 1, and visits that ordering in reverse.

The implementation lives in `src/gradforge/value.py`. Tests compare elementary derivatives with centered finite differences and explicitly cover reused values and repeated backward calls.
