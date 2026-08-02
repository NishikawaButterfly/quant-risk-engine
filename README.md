# Quant Risk Engine

A reproducible portfolio-analytics and market-risk engine for public
securities — the quantitative sibling of
[energy-investment-lab](https://github.com/NishikawaButterfly/energy-investment-lab).
You supply price series; it computes risk and performance evidence
with every convention documented and every worked example asserted by
tests.

This is not a prediction tool. Nothing in this repository forecasts
prices, generates trading signals, or recommends investments, and it
never will claim to. It measures what a supplied price history did,
under stated conventions — that is the whole product.

## What exists today

- Validated price series: strictly increasing ISO dates, finite
  positive prices, simple and log returns, and multi-series alignment
  by date intersection. Forward-filling is a deliberate non-feature:
  silent fills poison risk numbers, so gaps must be resolved upstream.
- Core metrics: annualized volatility, Sharpe and Sortino ratios,
  maximum drawdown with its peak and trough dates, historical VaR and
  CVaR at any confidence, and parametric (normal) VaR with a
  documented warning about its thin tails.
- A seeded synthetic fixture: one weekday year of daily closes for
  four fictional tickers, regenerated and byte-compared in the test
  suite. Every dataset in this repository is fictional.

Each convention — the 252-day annualization, the n − 1 volatility
denominator, the Sortino target, the interpolated percentile, the sign
of a VaR — is written down in [docs/methodology.md](docs/methodology.md)
next to a hand-worked example the tests assert digit for digit.

## What does not exist yet

Correlations and risk contribution, the efficient frontier,
look-ahead-safe backtesting with transaction costs and rebalancing,
benchmark comparisons, seeded Monte Carlo, stress scenarios, and
reproducible reports are planned but not built. The README will say so
when they land, and not before.

## Usage

```python
from quantrisk import PriceSeries, annualized_volatility, historical_var, max_drawdown

series = PriceSeries(
    name="AAA",
    dates=("2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"),
    prices=(100.0, 102.0, 99.0, 101.0),
)
returns = series.simple_returns()
volatility = annualized_volatility(returns)
var_95 = historical_var(returns, confidence=0.95)
drawdown = max_drawdown(series)
```

## Development

Python 3.12 or newer; NumPy and SciPy are the only runtime
dependencies (exact pins in `requirements.txt`).

```bash
python -m pip install -e ".[dev]"
ruff format --check .
ruff check .
mypy
coverage run -m unittest discover -s tests
coverage report --fail-under=90
```

`python tools/make_synthetic_prices.py --seed 2026` regenerates the
sample data; the test suite fails if the committed CSV and the
generator ever disagree.

## License

MIT — see [LICENSE](LICENSE).
