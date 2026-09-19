# Decisions

One line per dependency: what it is, what it replaced, why.

- ruff — lint + format. Replaces flake8 + isort + black; one tool, one config, far faster.
- mypy — static types. Chosen over pyright to keep the toolchain inside the Python package set.
- pytest — test runner. Replaces unittest; fixtures and parametrisation are worth the dependency.
