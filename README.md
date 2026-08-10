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
- Correlations and risk contribution: sample covariance and
  correlation matrices over aligned series, and a `Portfolio` of named
  weights — daily weighted return series, annualized volatility via
  `w'Σw × 252`, per-asset risk contributions that sum to one, and the
  diversification benefit against the weighted sum of individual
  volatilities. Short positions are allowed and flagged, never hidden.
- The efficient frontier: the unconstrained minimum-variance portfolio
  in closed form via the normal equations (solved, never inverted),
  SLSQP for long-only bounds and target-return sweeps with every solver
  answer validated, and unreachable targets rejected with the reachable
  range in the message. Each swept point is labeled `efficient` or
  dominated against the minimum-variance return under the same
  constraint set, so the dominated lower branch is never passed off as
  part of the efficient frontier. Expected returns must be supplied by
  the caller: historical means are available as a helper but are noisy
  estimators, and the engine never defaults to them silently.
- Look-ahead-safe backtesting: a walk-forward engine whose policies
  receive a `PolicyWindow` sliced strictly before the decision date —
  the interface exposes only observations strictly preceding the
  decision, so a policy that works from its window alone has no
  future data to read. Caller-supplied policies remain responsible
  for not reaching around the interface to future data through
  external state or external data sources; a deliberately cheating
  policy in the test suite does exactly that and proves peeking would
  change the answer. Rebalancing runs on an explicit trading-day interval with
  buy-and-hold drift between decisions, proportional transaction
  costs charged on turnover at every rebalance, and a cash warmup
  (earning zero) until the policy's declared minimum history.
- Benchmark comparison: beta, arithmetic CAPM alpha (daily and
  annualized), tracking error, information ratio, and up/down capture
  ratios over the portfolio's and the benchmark's returns on one
  verified shared date grid — series on mismatched grids are rejected,
  not paired — with undefined cases reported as `None` for stated
  reasons rather than as fabricated numbers.
- A seeded synthetic fixture: one weekday year of daily closes for
  four fictional tickers, regenerated and byte-compared in the test
  suite. Every dataset in this repository is fictional.
- Seeded Monte Carlo on portfolios: bootstrap resampling of the
  portfolio's historical daily returns and a parametric normal
  alternative — both requiring an explicit seed for exact
  reproducibility — compounded into a terminal-value distribution
  with interpolated percentiles, the probability of finishing below
  the initial value, and each mode's blind spot documented.
- Stress scenarios: named historical windows replayed through the
  portfolio's own return series (total return, volatility, and
  drawdown inside the window), and named hypothetical one-day shock
  vectors reported as the weighted sum — exact for one day of plain
  positions, and documented as linear and correlation-free, so the
  scenario author owns the co-movements.
- A CLI and reproducible reports: one JSON spec (prices CSV, weights,
  optional risk-free rate, benchmark, stress, and seeded Monte Carlo)
  drives `quantrisk run`, which writes `results.json` and a
  committee-style `report.md` with no timestamp anywhere — identical
  inputs evaluated under identical library versions produce
  byte-identical artifacts, a test asserts it, and each artifact
  records the provenance (input SHA-256 hashes plus the quantrisk,
  Python, NumPy, and SciPy versions) that makes the condition
  checkable. `quantrisk validate` checks a spec without computing.

Each convention — the 252-day annualization, the n − 1 volatility
denominator, the Sortino target, the interpolated percentile, the sign
of a VaR — is written down in [docs/methodology.md](docs/methodology.md)
next to a hand-worked example the tests assert digit for digit.

## What does not exist yet

Everything on the original roadmap — stress scenarios, the CLI, and
reproducible reports — has now landed. What remains open is tracked in
the issues: a block bootstrap that preserves short-range dependence,
and an interest convention for backtest cash are the known candidates.
Nothing else is claimed.

## Usage

Install the package and point the CLI at a spec file:

```bash
python -m pip install -e .
quantrisk validate --spec sample-data/portfolio-spec.json
quantrisk run --spec sample-data/portfolio-spec.json --output results
```

The run writes `results.json`, with every computed number plus the
Monte Carlo seed and run count, and `report.md`, a short report meant
to be read in a few minutes: the holdings and conventions, per-asset
and portfolio risk with contributions, the benchmark comparison, the
stress tables, the Monte Carlo percentiles, and the caveats. Both
artifacts record their provenance — the SHA-256 of each input file and
the versions of quantrisk, Python, NumPy, and SciPy that produced them
— and neither embeds a timestamp, so identical inputs under identical
versions produce byte-identical output, and the provenance says
whether that condition holds. The spec format and the verification
recipe are documented in [docs/cli.md](docs/cli.md).

The library remains directly usable:

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
