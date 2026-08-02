# Quant Risk Engine

A reproducible portfolio-analytics and market-risk engine for public
securities. You supply price series; it computes risk and performance
evidence — nothing here predicts prices or recommends trades.

This repository is being bootstrapped. The scaffolding, quality gates,
and CI exist; the engine itself lands next, in small reviewed pieces.

## Development

Python 3.12 or newer.

```bash
python -m pip install -e ".[dev]"
ruff format --check .
ruff check .
mypy
coverage run -m unittest discover -s tests
coverage report --fail-under=90
```
