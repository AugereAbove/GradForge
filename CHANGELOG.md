# Changelog

## Unreleased

- Started Phase 1 with scalar reverse-mode autodiff and mathematical regression tests.
- Preserve shared leaf gradients when backward passes originate from distinct outputs.
- Add differentiable scalar exponents, including reverse powers and positive-domain validation.
- Keep scalar numerical regression tests clean under strict mypy checking.
