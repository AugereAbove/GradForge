# GradForge

GradForge is an educational deep-learning engine and automatic-differentiation framework implemented from first principles with NumPy.

The project is intentionally built in small, validated phases. The current implementation is Phase 1: a scalar reverse-mode autodiff engine with a dynamic computation graph, gradient accumulation, elementary functions, and finite-difference tests. Tensor arrays and neural-network layers are deliberately not part of this first cycle.

## Current example

```python
from gradforge import Value

x = Value(2.0)
y = x**3
y.backward()
print(x.grad)  # 12.0
```

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
mypy src
```

See [ROADMAP.md](ROADMAP.md) for the staged plan and [docs/mathematics/autodiff.md](docs/mathematics/autodiff.md) for the mathematical foundation.
