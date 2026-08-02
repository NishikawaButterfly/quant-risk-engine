# Methodology

Every metric in this engine follows a stated convention, and every
convention is exercised here on one tiny fixture a reviewer can
recompute by hand. The test suite asserts the same digits
(`tests/test_metrics.py`), so the document and the code cannot drift
apart silently.

Nothing in this document forecasts anything. Each number is a
description of the supplied price history, not a statement about the
future.

## Data model and units

Prices are daily closes, positive and finite, on strictly increasing
ISO dates. Returns are plain fractions: 0.01 means one percent.
Metrics over returns take periodic (daily) returns; the annualization
convention is 252 trading days per year. Mean returns scale linearly
with time and volatilities with its square root.

## Returns

Simple and log returns of a price series `p`:

```text
simple[t] = p[t] / p[t-1] - 1
log[t]    = ln(p[t] / p[t-1])
```

Log returns are additive: their sum telescopes to `ln(p[n] / p[0])`.
The tests assert this identity, which pins both definitions at once.

## Alignment policy

Aligning several series keeps only the dates every series shares —
intersection only, nothing else. Forward-filling missing dates is a
deliberate non-feature of this version: a filled price is a fabricated
observation with zero return, it deflates volatility and correlation
estimates, and the caller cannot see that it happened. Gaps must be
resolved upstream, where they are visible.

## The hand-worked fixture

Five daily returns:

```text
R = (0.01, 0.02, -0.03, 0.04, -0.02)
```

and one six-day price path for the drawdown:

| Date | Price |
| --- | ---: |
| 2026-01-05 | 100 |
| 2026-01-06 | 102 |
| 2026-01-07 | 99 |
| 2026-01-08 | 101 |
| 2026-01-09 | 98 |
| 2026-01-12 | 100 |

## Annualized volatility

The sample standard deviation of periodic returns (denominator n − 1),
times the square root of 252:

```text
mean            = (0.01 + 0.02 - 0.03 + 0.04 - 0.02) / 5 = 0.004
deviations      = 0.006, 0.016, -0.034, 0.036, -0.024
sum of squares  = 0.000036 + 0.000256 + 0.001156 + 0.001296 + 0.000576
                = 0.00332
sample variance = 0.00332 / 4 = 0.00083
sample stddev   = sqrt(0.00083) = 0.02880972...
volatility      = 0.02880972 * sqrt(252) = 0.457340
```

## Sharpe ratio

The caller supplies an annual risk-free rate; it converts to a daily
rate by plain division by 252. The ratio of mean daily excess return
to daily sample standard deviation is annualized with sqrt(252). An
annual rate of 2.52% is exactly 0.0001 per day:

```text
excess mean = 0.004 - 0.0001 = 0.0039
Sharpe      = 0.0039 / 0.02880972 * sqrt(252) = 2.148948
```

The ratio is undefined for constant returns and the engine raises
rather than dividing by zero.

## Sortino ratio

The target is a periodic (daily) return and defaults to zero. The
downside deviation is the root mean squared shortfall below the
target, averaged over all n observations — not only the losing days,
and not n − 1. Only −0.03 and −0.02 fall below the zero target:

```text
downside deviation = sqrt((0.0009 + 0.0004) / 5) = sqrt(0.00026)
                   = 0.01612451...
Sortino            = (0.004 - 0) / 0.01612451 * sqrt(252) = 3.937981
```

When no return falls below the target the ratio is undefined and the
engine raises.

## Maximum drawdown

The running peak is the highest price seen so far; each day's drawdown
is `1 - price / running peak`. The result carries the peak and trough
dates; ties keep the earliest trough.

| Date | Price | Running peak | Drawdown |
| --- | ---: | ---: | ---: |
| 2026-01-05 | 100 | 100 | 0 |
| 2026-01-06 | 102 | 102 | 0 |
| 2026-01-07 | 99 | 102 | 3/102 = 0.029412 |
| 2026-01-08 | 101 | 102 | 1/102 = 0.009804 |
| 2026-01-09 | 98 | 102 | 4/102 = 0.039216 |
| 2026-01-12 | 100 | 102 | 2/102 = 0.019608 |

Maximum drawdown 0.039216, peak 2026-01-06, trough 2026-01-09. A
series that never declines reports depth zero with both dates on the
first observation.

## Historical VaR and CVaR

Percentiles use linear interpolation between closest ranks: the sorted
values take ranks 0 through n − 1, the target rank for fraction `q` is
`q × (n − 1)`, and a fractional rank interpolates linearly between its
two neighbours. At confidence `c` the VaR is the negated `(1 − c)`
percentile of the returns, reported as a positive loss fraction.

Sorted returns: −0.03, −0.02, 0.01, 0.02, 0.04.

```text
95%: rank = 0.05 * 4 = 0.2   quantile = -0.03 + 0.2 * 0.01 = -0.028   VaR = 0.028
90%: rank = 0.10 * 4 = 0.4   quantile = -0.03 + 0.4 * 0.01 = -0.026   VaR = 0.026
75%: rank = 0.25 * 4 = 1.0   quantile = -0.02 exactly                 VaR = 0.020
```

VaR is monotone in confidence, and a negative VaR simply means even
that percentile of days was a gain.

CVaR (expected shortfall) is the negated mean of every observed return
at or below that quantile. Only −0.03 lies at or below −0.028, so the
95% CVaR is 0.03. The mean of a set bounded above by the quantile can
never exceed the quantile, so CVaR never falls below VaR at the same
confidence.

## Parametric (normal) VaR

Fit a normal distribution by the sample mean and sample standard
deviation and negate its `(1 − c)` quantile:

```text
z(0.05) = -1.6448536...
VaR     = -(0.004 + (-1.6448536) * 0.02880972) = 0.043388
```

On this tiny fixture the normal VaR exceeds the historical one because
five observations barely populate the tail. On real daily equity
returns the opposite failure matters more: returns have fatter tails
than a normal distribution, so at high confidences (99% and beyond)
parametric VaR systematically understates tail losses. Treat it as a
cross-check on the historical figures, not a replacement.

## Synthetic sample data

`sample-data/prices.csv` holds one weekday year (2025, 261 rows) of
daily closes for four fictional tickers — AAA, BBB, CCC, DDD — each
following a seeded geometric-brownian-ish walk with its own invented
drift and volatility. `tools/make_synthetic_prices.py --seed 2026`
reproduces the file byte for byte, and a test asserts that equality,
so the committed fixture can never drift from its generator. No real
market data appears anywhere in this repository.
