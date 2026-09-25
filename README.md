# GradForge

GradForge is an educational deep-learning engine and automatic-differentiation framework implemented from first principles with NumPy.

The project is intentionally built in small, validated phases. Phase 1 is complete and awaiting owner review: it provides a scalar reverse-mode autodiff engine with a dynamic computation graph, gradient accumulation, elementary functions, finite-difference tests, and explicit real-domain power boundaries. Tensor arrays and neural-network layers remain deliberately out of scope until that review approves advancement.

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
mypy src tests --strict
```

See [ROADMAP.md](ROADMAP.md) for the staged plan and [docs/mathematics/autodiff.md](docs/mathematics/autodiff.md) for the mathematical foundation.
