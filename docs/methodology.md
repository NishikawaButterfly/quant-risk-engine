# Methodology

Every metric in this engine follows a stated convention, and every
convention is exercised here on one tiny fixture a reviewer can
recompute by hand. The test suite asserts the same digits
(`tests/test_metrics.py`, `tests/test_portfolio.py`,
`tests/test_frontier.py`, `tests/test_backtest.py`), so the document
and the code cannot drift apart silently.

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

## Covariance and correlation

Cross-asset statistics run over the simple returns of series that are
already on one shared date grid (the alignment policy above); anything
else is rejected, never silently truncated. Covariances are sample
covariances — deviations from the sample mean, summed and divided by
n − 1 — matching the sample standard deviation used everywhere else.

The two-asset fixture: seven prices give six returns per asset. The
returns were chosen so the covariance matrix lands on round numbers.

| Date | A return | B return |
| --- | ---: | ---: |
| 2026-01-06 | 0.02 | 0.03 |
| 2026-01-07 | -0.01 | 0.03 |
| 2026-01-08 | -0.02 | -0.03 |
| 2026-01-09 | 0.02 | 0.035 |
| 2026-01-12 | 0.03 | -0.01 |
| 2026-01-13 | 0.02 | -0.025 |

Means: A `0.06 / 6 = 0.01`, B `0.03 / 6 = 0.005`. Deviations from the
mean and their products:

| Date | dev A | dev B | dev A² | dev B² | dev A · dev B |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026-01-06 | 0.01 | 0.025 | 0.0001 | 0.000625 | 0.00025 |
| 2026-01-07 | -0.02 | 0.025 | 0.0004 | 0.000625 | -0.0005 |
| 2026-01-08 | -0.03 | -0.035 | 0.0009 | 0.001225 | 0.00105 |
| 2026-01-09 | 0.01 | 0.03 | 0.0001 | 0.0009 | 0.0003 |
| 2026-01-12 | 0.02 | -0.015 | 0.0004 | 0.000225 | -0.0003 |
| 2026-01-13 | 0.01 | -0.03 | 0.0001 | 0.0009 | -0.0003 |
| sum | 0 | 0 | 0.002 | 0.0045 | 0.0005 |

Dividing each sum by n − 1 = 5:

```text
var(A) = 0.002  / 5 = 0.0004    stddev(A) = 0.02
var(B) = 0.0045 / 5 = 0.0009    stddev(B) = 0.03
cov    = 0.0005 / 5 = 0.0001
```

so the sample covariance matrix is exactly

```text
S = | 0.0004  0.0001 |
    | 0.0001  0.0009 |
```

Correlation divides the covariance by both standard deviations:

```text
corr = 0.0001 / (0.02 * 0.03) = 1/6 = 0.166667
```

The engine pins the correlation diagonal to exactly 1.0 and clips
off-diagonal entries into [-1, 1], removing float noise in the last
digit; a constant-return series has zero variance, so its correlation
is undefined and rejected.

## Portfolio volatility and risk contribution

A portfolio holds named weights that must sum to one (within 1e-9).
Its daily return is the weighted sum of that day's simple returns —
exact for a single day, and equivalent to rebalancing back to the
target weights every day. Buy-and-hold weights would drift with
prices; this engine does not model that drift.

Weights 0.6 on A and 0.4 on B over the fixture:

```text
r(p) = (0.024, 0.006, -0.024, 0.026, 0.014, 0.002)
```

(first day: 0.6 × 0.02 + 0.4 × 0.03 = 0.024, and so on). The daily
portfolio variance is the quadratic form w'Sw:

```text
w'Sw = 0.6² × 0.0004 + 2 × 0.6 × 0.4 × 0.0001 + 0.4² × 0.0009
     = 0.000144 + 0.000048 + 0.000144
     = 0.000336
```

The same number falls out of the return series directly: r(p) has mean
0.008, squared deviations 0.000256, 0.000004, 0.001024, 0.000324,
0.000036, 0.000036, sum 0.00168, divided by 5 = 0.000336. Annualizing
with the usual 252-day convention:

```text
vol(p) = sqrt(0.000336 × 252) = sqrt(0.084672) = 0.290985
```

Risk contribution attributes that variance to the assets. Asset i
contributes `w_i (Sw)_i / (w'Sw)` — its weight times its marginal
covariance with the whole portfolio, as a fraction of total variance:

```text
(Sw)_A = 0.0004 × 0.6 + 0.0001 × 0.4 = 0.00028
(Sw)_B = 0.0001 × 0.6 + 0.0009 × 0.4 = 0.00042

RC_A = 0.6 × 0.00028 / 0.000336 = 0.000168 / 0.000336 = 0.5
RC_B = 0.4 × 0.00042 / 0.000336 = 0.000168 / 0.000336 = 0.5
```

The numerators sum to the denominator by construction, so the
contributions always sum to one. Here they are exactly equal: B's
larger variance and its smaller weight cancel precisely. A short or
strongly diversifying position can carry a negative contribution;
shorts are allowed and flagged (`has_short_positions`), never hidden.

## Diversification

Individually, annualized: `vol(A) = 0.02 × sqrt(252) = 0.317490` and
`vol(B) = 0.03 × sqrt(252) = 0.476235`. Their weighted sum is

```text
0.6 × 0.317490 + 0.4 × 0.476235 = 0.380988
```

against the portfolio's 0.290985 — a diversification benefit of
0.090004 in annualized volatility, earned because the correlation is
1/6 rather than 1. For a long-only portfolio the benefit is never
negative, and it is zero exactly when every pair of held assets is
perfectly correlated; the tests exercise both sides. With short
positions the inequality has no guaranteed sign, which is one reason
the flag exists.

## Minimum variance and the efficient frontier

The frontier machinery takes two inputs of very different standing.
The daily covariance matrix comes from the engine itself — here the
two-asset fixture above. Expected *annual* returns must be supplied by
the caller. The obvious candidate, the historical mean
(`annualized_mean_returns`), is a noisy estimator: its standard error
is the volatility over the square root of the sample size, comparable
to the mean itself on realistic daily windows. On this fixture it
yields A `0.01 × 252 = 2.52` and B `0.005 × 252 = 1.26` — six days of
data annualized into a 252% "expected return", which is exactly why
the engine never defaults to it. The examples below use supplied round
values:

```text
μ = (0.05, 0.10)        S = | 0.0004  0.0001 |
                            | 0.0001  0.0009 |
```

### The minimum-variance portfolio

Minimizing `w'Sw` subject only to `1'w = 1` gives the Lagrangian
condition `Sw = λ1`: the weights solve one linear system, then rescale
to sum to one. The code hands that system to `numpy.linalg.solve` —
forming an explicit inverse would cost more and lose accuracy roughly
with the square of the matrix's condition number, and only `S⁻¹1` is
ever needed. By hand, with the 2×2 inverse formula:

```text
det  = 0.0004 × 0.0009 − 0.0001² = 3.6e-7 − 0.1e-7 = 3.5e-7

S⁻¹1 = (1/det) | 0.0009  −0.0001 | |1|  = (1/det) (0.0008, 0.0003)
               | −0.0001  0.0004 | |1|

w    = (0.0008, 0.0003) / (0.0008 + 0.0003)
     = (8/11, 3/11) = (0.727273, 0.272727)
```

The minimized daily variance is `1/(1'S⁻¹1) = 3.5e-7 / 0.0011 =
7/22000 = 0.000318182`, which the quadratic form confirms:

```text
w'Sw = (8² × 0.0004 + 2 × 8 × 3 × 0.0001 + 3² × 0.0009) / 11²
     = (0.0256 + 0.0048 + 0.0081) / 121 = 0.0385 / 121 = 7/22000
vol  = sqrt(7/22000 × 252) = sqrt(0.080182) = 0.283164
```

With the supplied μ its expected return is
`(8 × 0.05 + 3 × 0.10) / 11 = 0.7/11 = 0.063636`.

### One frontier point

A frontier point minimizes the same variance under one more equality:
`μ'w = target`. With two assets the two constraints pin the weights
completely — there is nothing left to minimize — so the point at
target 0.08 is pure algebra:

```text
w_A + w_B = 1        0.05 w_A + 0.10 w_B = 0.08
w_A  = (0.10 − 0.08) / (0.10 − 0.05) = 0.4        w_B = 0.6

w'Sw = 0.16 × 0.0004 + 2 × 0.24 × 0.0001 + 0.36 × 0.0009
     = 0.000064 + 0.000048 + 0.000324 = 0.000436
vol  = sqrt(0.000436 × 252) = sqrt(0.109872) = 0.331469
```

The tests assert every digit of both portfolios against the solver's
output. With three or more assets the constraints stop pinning the
weights and the quadratic program does real work; SLSQP solves it, and
its answer is validated — success flag, weight-sum and target-return
residuals within 1e-8, no long-only weight below zero — never trusted.

### Reachability and shape

Long-only, the achievable expected returns are exactly the interval
between the worst and the best single asset; targets outside it are
rejected up front with that interval in the message. Without bounds,
leverage reaches any target — here 0.12 needs `w = (−0.4, 1.4)` —
unless every asset carries the same expected return, in which case
only that value is achievable and anything else is rejected.

Volatility along the frontier falls as the target rises toward the
minimum-variance return (0.063636 above) and rises past it, and no
frontier point undercuts the closed-form minimum. The tests assert
that shape on a grid of targets, and separately that a deeply
negative-return asset receives exactly zero weight in long-only
frontiers at high targets.

## Look-ahead-safe backtesting

A backtest walks a weight policy forward through the aligned series:
at each scheduled rebalance the policy is asked for target weights,
trades execute at that day's prices, a proportional cost is charged,
and the portfolio then drifts untouched until the next rebalance.
Nothing here forecasts anything either — a backtest describes what a
stated policy would have done on the supplied history, under the
conventions below.

### The look-ahead guarantee

The design's core claim: a policy cannot peek, because there is
nothing to peek at. At every decision the engine hands the policy a
`PolicyWindow` built by *slicing* the aligned data at the decision
date — the object physically contains the dates and prices strictly
before that date and nothing else. Look-ahead is not a rule the policy
is trusted to follow; the data on or after the decision date is absent
from the only object the policy receives, so expressing a peek raises
an error instead of returning a number. The window's constructor
rejects any history that touches its own decision date, and the tests
assert, for every rebalance of a run, that the window ends exactly one
trading day before the decision date.

The tests also prove peeking would change the answer: a deliberately
cheating policy — handed the full series separately, outside the
interface — strictly beats an honest momentum policy on a
mean-reverting fixture where the trailing winner is always the forward
loser. That gap is exactly why the interface must make the cheat
inexpressible, and the same tests show the cheat's first move
(locating the decision date in the data) raises when attempted through
the window.

### Drift versus daily rebalancing

`Portfolio.return_series` weights every day's returns, which is
implicitly rebalancing back to the target weights every day. That
convention is exact for risk description but wrong for a backtest with
a stated schedule: a path that trades daily while charging costs only
at scheduled rebalances is one no real portfolio could achieve. The
backtest therefore holds positions between decisions — buy-and-hold,
weights drifting with prices — and trades only on rebalance days,
where the cost is charged. With a daily schedule and zero costs the
two conventions coincide, and a test asserts that equivalence.

The schedule is an explicit interval in trading days: the first
decision falls on the first day with the policy's declared
`min_history_days` observed days behind it, and further decisions come
every `schedule` trading days after that. Days before the first
decision are held in cash, and cash earns exactly zero in this
version — no interest convention is smuggled in. The value path stays
flat at the initial value until the first rebalance day.

### Transaction costs, worked by hand

At each rebalance the drifted (pre-trade) weights are compared with
the policy's targets. Turnover is the weight-space distance
`sum |target − drifted|` and the charge is
`cost_rate × turnover × pre-trade value`. The first rebalance out of
cash has drifted weights of zero, so a fully invested long-only target
carries turnover 1 and costs `cost_rate × value` — it is charged like
any other rebalance, not waived.

The fixture: two assets on five days, A at 100, 100, 150, 150, 150
and B flat at 100. A constant 50/50 policy with one day of warmup,
rebalancing every 2 trading days, cost rate 2%, initial value 1000:

| Date | A | B | Event | Value |
| --- | ---: | ---: | --- | ---: |
| 2026-01-05 | 100 | 100 | cash warmup | 1000 |
| 2026-01-06 | 100 | 100 | rebalance 1 | 980 |
| 2026-01-07 | 150 | 100 | drift | 1225 |
| 2026-01-08 | 150 | 100 | rebalance 2 | 1220.1 |
| 2026-01-09 | 150 | 100 | drift | 1220.1 |

Rebalance 1 (2026-01-06): out of cash, drifted (0, 0), target
(0.5, 0.5), turnover |0.5 − 0| + |0.5 − 0| = 1:

```text
cost  = 0.02 × 1 × 1000 = 20
value = 1000 − 20 = 980, invested 490 + 490
```

Drift (2026-01-07): A gains 50%, so the legs become 735 and 490 —
value 1225, weights (0.6, 0.4). Nobody traded; prices moved.

Rebalance 2 (2026-01-08): drifted (0.6, 0.4), target (0.5, 0.5),
turnover |0.5 − 0.6| + |0.5 − 0.4| = 0.2:

```text
cost  = 0.02 × 0.2 × 1225 = 4.9
value = 1225 − 4.9 = 1220.1, split 610.05 + 610.05
```

Nothing moves on the last day. Total costs 20 + 4.9 = 24.9; total
return 1220.1 / 1000 − 1 = 0.2201. The tests assert every one of
these digits, including the drifted (0.6, 0.4).

### Reported figures

`total_return` covers the whole value path, cash warmup included.
`annualized_return` compounds it geometrically over the path's return
days with the 252-day convention: `(1 + total)^(252 / (n − 1)) − 1`.
The path's volatility and maximum drawdown reuse
`annualized_volatility` and `max_drawdown` from the metrics module
unchanged — the value path is itself a `PriceSeries`, so every metric
in the engine applies to it directly.

## Synthetic sample data

`sample-data/prices.csv` holds one weekday year (2025, 261 rows) of
daily closes for four fictional tickers — AAA, BBB, CCC, DDD — each
following a seeded geometric-brownian-ish walk with its own invented
drift and volatility. `tools/make_synthetic_prices.py --seed 2026`
reproduces the file byte for byte, and a test asserts that equality,
so the committed fixture can never drift from its generator. No real
market data appears anywhere in this repository.
