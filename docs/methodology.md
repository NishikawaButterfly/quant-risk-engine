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

### Input contract for numbers

Every number crossing a public boundary — prices, weights, rates,
shocks, confidence levels, expected and target returns, spec fields —
passes one shared validation path (`quantrisk._validation`). The value
must be a real number in the `numbers.Real` sense, must not be a
`bool`, and must be finite; failures name the offending parameter (and
its index, inside a sequence). `bool` is rejected explicitly because
Python's `bool` subclasses `int`, so `True` would otherwise compute
silently as 1.0. The `numbers.Real` rule means NumPy scalars pass —
`np.float64` (a `float` subclass) and `np.int64` (not an `int`
subclass) alike — and are normalized to built-in floats on entry.
JSON specs are stricter still: `true`/`false` never count as numbers,
and the non-standard `NaN`/`Infinity`/`-Infinity` tokens are refused
when the document is parsed.

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

## Numerical conditioning

Every covariance matrix entering the engine — the portfolio methods
and the frontier solvers alike — passes through one validation path
before any arithmetic runs. The matrix must be square, finite,
symmetric within 1e-12, and positive semidefinite: one
`numpy.linalg.eigvalsh` decomposition checks the eigenvalues against
the relative floor

```text
min_eig >= -1e-10 × max(1, max_eig)
```

and an indefinite matrix is rejected with the offending eigenvalue
named. The floor exists because the *computed* sample covariance of
near-collinear return series can carry a tiny negative eigenvalue that
is pure floating-point rounding on a matrix that is PSD by
construction; rejecting it would reject legitimate data. A genuinely
indefinite matrix — which no return data can produce — lands far below
the floor and is refused.

The same decomposition yields the 2-norm condition number, the largest
eigenvalue magnitude over the smallest (infinite for an exactly
singular matrix). On the hand fixture the eigenvalues are
(13 ± √29)e-4 / 2, so

```text
cond2 = (13 + √29) / (13 − √29) = 2.414388
```

— thoroughly well conditioned. The number matters because the
minimum-variance weights come from solving `Σx = 1`, and a linear
solve amplifies relative input error by up to cond2: a covariance
known to 12 digits fed through a matrix at cond2 = 1e9 yields weights
trustworthy to about 3. Two thresholds apply, both deliberately
asymmetric about what they protect:

- **Above 1e8** (warn): results still compute, but they carry a
  `conditioning_warning` — on every `FrontierPoint`, in the
  portfolio's `covariance_diagnostics`, and in `results.json` — and
  the CLI report surfaces it under Caveats. The condition number
  itself is always exposed alongside.
- **Above 1e12** (refuse): the frontier functions refuse with a clear
  message instead of solving. At that conditioning fewer than four
  significant digits survive the solve, so the "weights" would be
  noise formatted as precision; an error the caller sees beats a
  confident number nobody can trust. The portfolio's own methods
  (volatility, risk contributions, diversification) never refuse on
  conditioning, because quadratic forms like `w'Sw` involve no
  inverse and do not amplify error this way — for them the condition
  number is disclosure, not a gate.

The previously existing late guards — the negative-variance check and
the singular-solve rejection — stay in the code as defense in depth,
but validation now happens at entry, so bad matrices are named for
what they are instead of surfacing as solver failures.

## Look-ahead-safe backtesting

A backtest walks a weight policy forward through the aligned series:
at each scheduled rebalance the policy is asked for target weights,
trades execute at that day's prices, a proportional cost is charged,
and the portfolio then drifts untouched until the next rebalance.
Nothing here forecasts anything either — a backtest describes what a
stated policy would have done on the supplied history, under the
conventions below.

### The look-ahead boundary

The design's core claim: the policy interface exposes only
observations strictly preceding the decision date. At every decision
the engine hands the policy a `PolicyWindow` built by *slicing* the
aligned data at the decision date — the object physically contains
the dates and prices strictly before that date and nothing else, so a
policy that works from its window alone has no future data to read,
and looking up the decision date or indexing past the window's end
raises an error instead of returning a number. The window's
constructor rejects any history that touches its own decision date,
and the tests assert, for every rebalance of a run, that the window
ends exactly one trading day before the decision date.

The boundary stops at the interface: a Python callable can still
reach future data through closures, external state, files, or APIs,
so caller-supplied policies remain responsible for not accessing
future data through anything outside their window. The tests prove
why that responsibility matters: a deliberately cheating policy —
handed the full series separately, reaching around the interface —
strictly beats an honest momentum policy on a mean-reverting fixture
where the trailing winner is always the forward loser. That gap is
exactly what the interface removes from the data a policy is handed,
and the same tests show the cheat's first move (locating the decision
date in the data) raises when attempted through the window alone.

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

## Benchmark comparison

Benchmark statistics run over two return sequences paired on the same
date grid — the caller aligns the price series first (the alignment
policy above) and supplies the resulting returns; the engine re-checks
lengths, finiteness, and a minimum of three observations, since a line
fits any two points exactly. All second moments are sample moments
(n − 1), matching the rest of the engine. The risk-free rate here is a
*daily* rate, zero by default, and enters only the alpha: a constant
drops out of every covariance, so beta and the active-return
statistics never see it.

The hand-worked fixture, six pairs of daily returns
(`tests/test_benchmark.py` asserts every digit):

| Day | Portfolio p | Benchmark b |
| --- | ---: | ---: |
| 1 | 0.04 | 0.02 |
| 2 | 0.00 | -0.01 |
| 3 | 0.06 | 0.03 |
| 4 | -0.03 | -0.02 |
| 5 | 0.03 | 0.02 |
| 6 | 0.02 | 0.02 |

Means: p `0.12 / 6 = 0.02`, b `0.06 / 6 = 0.01`. Deviations from the
means and their products:

| Day | dev p | dev b | dev b² | dev p · dev b |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.02 | 0.01 | 0.0001 | 0.0002 |
| 2 | -0.02 | -0.02 | 0.0004 | 0.0004 |
| 3 | 0.04 | 0.02 | 0.0004 | 0.0008 |
| 4 | -0.05 | -0.03 | 0.0009 | 0.0015 |
| 5 | 0.01 | 0.01 | 0.0001 | 0.0001 |
| 6 | 0.00 | 0.01 | 0.0001 | 0.0000 |
| sum | 0 | 0 | 0.002 | 0.003 |

### Beta

The slope of the sample regression of p on b: covariance over
benchmark variance, both with denominator n − 1 = 5:

```text
cov(p, b) = 0.003 / 5 = 0.0006
var(b)    = 0.002 / 5 = 0.0004
beta      = 0.0006 / 0.0004 = 3/2 = 1.5
```

A constant-return benchmark has zero variance — the denominator holds
no information — and is rejected rather than divided by.

### Alpha

The arithmetic CAPM residual over excess returns, per day, then times
252. With the default zero risk-free rate:

```text
alpha (daily)  = 0.02 - 1.5 × 0.01 = 0.005
alpha (annual) = 0.005 × 252 = 1.26
```

The annualization is arithmetic — the daily residual times 252 — not
geometric compounding. Alpha is a regression intercept, not a return
anyone can hold; compounding it as `(1 + a)^252 − 1` would smuggle a
growth model into a linear residual, while the arithmetic convention
keeps the exact identity that adding a constant `c` to every portfolio
return moves annual alpha by exactly `252 c` (a property test asserts
it). The 1.26 itself — a 126% annual alpha from six good days — is the
same annualize-a-tiny-sample absurdity flagged in the frontier
section, on display rather than hidden. With a nonzero daily rate
`rf`, alpha becomes `(mean p − rf) − beta × (mean b − rf)`, a shift of
exactly `(beta − 1) × rf` per day; beta is unchanged.

### Tracking error and information ratio

Active returns `a = p − b`, per day:

```text
a      = (0.02, 0.01, 0.03, -0.01, 0.01, 0.00)
mean   = 0.06 / 6 = 0.01
devs   = 0.01, 0.00, 0.02, -0.02, 0.00, -0.01
var    = (0.0001 + 0 + 0.0004 + 0.0004 + 0 + 0.0001) / 5 = 0.0002
TE     = sqrt(0.0002) × sqrt(252) = sqrt(0.0504) = 0.224499
IR     = (0.01 × 252) / sqrt(0.0504) = sqrt(126) = 11.224972
```

The information ratio is the annualized mean active return over the
tracking error. A zero tracking error means the active return is
constant — the portfolio is the benchmark plus a fixed daily offset —
so the ratio has a zero denominator and is reported as `None`, not a
number and not an exception: unlike a constant benchmark, the
comparison itself is legitimate and every other statistic stands.

### Capture ratios

Each side conditions on the benchmark's sign; a day the benchmark
returned exactly zero belongs to neither. The benchmark rose on days
1, 3, 5, 6 and fell on exactly two days, 2 and 4:

```text
up   = (0.15 / 4) / (0.09 / 4)   = 0.0375 / 0.0225  = 5/3 = 1.666667
down = (-0.03 / 2) / (-0.03 / 2) = -0.015 / -0.015  = 1.0
```

The portfolio captured five-thirds of the benchmark's rises and
exactly its falls — beta above one bought the upside without, on this
fixture, extra downside. A benchmark that never fell has no falling
days to measure, so that side's ratio is `None` rather than a
fabricated number; the tests exercise both empty sides.

## Stress scenarios

Two kinds of stress with different epistemic standing, kept apart in
the code and in this document (`tests/test_stress.py` asserts every
digit below).

### Historical windows

A historical stress names a date window and replays the portfolio's
own daily return series — the weighted, daily-rebalanced returns of
the portfolio section above — through it, reporting the window's total
return, its annualized volatility, and its deepest drawdown. Both
window dates must lie inside the aligned grid (a window reaching
outside the data would silently describe a shorter period than its
name claims), the bounds themselves need not be trading days, and the
window must cover at least five observations, so at least four returns
fall inside it. The return landing on the window's first date belongs
to the day before the window and is excluded.

Worked by hand on the two-asset fixture with the 60/40 portfolio: the
window 2026-01-06 through 2026-01-12 covers five dates, and the
portfolio returns inside it are the middle four entries of r(p):

```text
r(window) = (0.006, -0.024, 0.026, 0.014)
```

Total return compounds them:

```text
1.006 × 0.976 × 1.026 × 1.014 = 1.021487635584
total return                  = 0.021488
```

Volatility inside the window is the usual sample convention over the
four returns:

```text
mean            = 0.022 / 4 = 0.0055
deviations      = 0.0005, -0.0295, 0.0205, 0.0085
sum of squares  = 0.00000025 + 0.00087025 + 0.00042025 + 0.00007225
                = 0.001363
sample variance = 0.001363 / 3
volatility      = sqrt(0.001363 / 3 × 252) = sqrt(0.114492) = 0.338367
```

The drawdown runs over the compounded value path inside the window,
starting from 1.0:

| Date | Value | Running peak | Drawdown |
| --- | ---: | ---: | ---: |
| 2026-01-06 | 1.0 | 1.0 | 0 |
| 2026-01-07 | 1.006 | 1.006 | 0 |
| 2026-01-08 | 0.981856 | 1.006 | 1 − 0.976 = 0.024 |
| 2026-01-09 | 1.007384256 | 1.007384256 | 0 |
| 2026-01-12 | 1.021487635584 | 1.021487635584 | 0 |

Maximum drawdown exactly 0.024 (0.981856 / 1.006 = 0.976), peak
2026-01-07, trough 2026-01-08.

### Hypothetical shocks

A shock stress names a one-day shock vector: one simple-return shock
per held asset, each strictly above −1 because a positive price cannot
lose more than everything. A missing asset is rejected, not defaulted
to zero — an unshocked asset is itself a scenario assumption the
author must write down. The portfolio's one-day return under the
vector is the weight-weighted sum. With the 60/40 portfolio and shocks
of −20% on A and −5% on B:

```text
0.6 × (−0.20) + 0.4 × (−0.05) = −0.12 − 0.02 = −0.14
```

exactly, and the test asserts exact equality, not a tolerance. An
all-zero vector returns exactly zero.

For plain long or short positions in the assets themselves that
weighted sum is exact for a single day. It is still an approximation
of any real event, in three stated ways: the vector is *chosen*, so no
probability attaches to the result; the co-movements are frozen
exactly as written, so shocking one asset while holding the others at
zero asserts a correlation the history may flatly contradict; and
nothing propagates past the single day — no follow-on volatility, no
liquidity effect, no rebalancing. A shock table describes the
scenarios its author wrote down, never their likelihood.

## Synthetic sample data

`sample-data/prices.csv` holds one weekday year (2025, 261 rows) of
daily closes for four fictional tickers — AAA, BBB, CCC, DDD — each
following a seeded geometric-brownian-ish walk with its own invented
drift and volatility. `tools/make_synthetic_prices.py --seed 2026`
reproduces the file byte for byte, and a test asserts that equality,
so the committed fixture can never drift from its generator. No real
market data appears anywhere in this repository.

## Seeded Monte Carlo on portfolios

The Monte Carlo module simulates a portfolio's terminal value: each
run draws daily returns over a stated horizon in trading days,
compounds them from an initial value of 1.0, and the collected runs
form a terminal-value distribution — mean, sample standard deviation
(n − 1, as everywhere else), the 5/25/50/75/95 percentiles, the
probability of finishing strictly below the initial value, and the
full sorted array of terminal values retained on the result so every
summary figure can be recomputed and checked.

The seed is a required argument on purpose: an unseeded simulation
cannot be reproduced, so its numbers cannot be checked, and a number
that cannot be checked does not ship. The same seed reproduces the
same result number for number — the tests assert strict equality on
the full percentile vector and the full terminal array — and run
counts are bounded to [100, 20000]: fewer runs make tail percentiles
meaningless, more buy precision the input data cannot support.

### Memory shape: draws are processed in blocks

The draws are never materialized as one `runs × horizon` matrix.
Both modes draw and compound in blocks of at most `BLOCK_RUNS = 512`
runs: each block's `(rows × horizon)` matrix is compounded into that
block's terminal values and discarded before the next block is drawn,
so the peak footprint is O(`BLOCK_RUNS` × horizon) regardless of the
run count — one 512 × 2,520 float64 block at the horizon cap is about
10 MB — where a whole matrix at both caps (20,000 × 2,520) would put
over a gigabyte across the draw matrix and its compounding
temporaries. Blockwise drawing changes no result digit: NumPy's
generator fills arrays in row-major order and consumes the PCG64
stream value by value for both `integers` and `normal`, so drawing
the same rows across consecutive block calls yields the same numbers
in the same positions as one whole-matrix call. The tests assert this
digit for digit against an inline whole-matrix reference
implementation at run counts below, at, and across block boundaries,
and the pinned report values elsewhere in the test suite span a block
boundary themselves.

### The two modes, and when each misleads

**Bootstrap** resamples the portfolio's historical daily returns
i.i.d. with replacement, so each terminal value is a product of
factors the portfolio actually printed and lies inside
`[(1 + min r)^h, (1 + max r)^h]` by construction; if every historical
return is one value `r`, the whole distribution collapses to the
single point `(1 + r)^h`. The tests assert both properties, the
collapse exactly. The limitation is the i.i.d. assumption itself:
independent redraws destroy autocorrelation and volatility clustering,
so when turbulent days cluster in the history — as they do in real
markets — multi-day drawdown risk is understated. A block bootstrap,
which resamples contiguous runs of days to preserve short-range
dependence, is future work; it is not silently approximated.

**Parametric normal** draws each day from `N(mean, stddev)` fitted to
the same historical returns by the sample mean and sample standard
deviation. It fails exactly as the parametric VaR section above warns:
daily equity returns have fatter tails than a normal distribution, so
the simulated extremes are too mild and the tail percentiles
understate risk precisely where they matter. The normal's unbounded
support can also produce a daily draw below −100%, which no real
asset return can. Treat this mode as a smooth cross-check on the
bootstrap, never a replacement.

Neither mode forecasts anything: both assume the future resembles the
sampled history, and the outputs describe that assumption.

### Percentile method

Terminal-value percentiles come from `numpy.percentile` with its
default linear interpolation between closest ranks — the sorted
values take ranks 0 through n − 1 and the target rank for level `p`
is `p / 100 × (n − 1)` — which is the same convention as the
historical VaR percentile earlier in this document and matches the
interpolated percentile used by the sibling energy-investment-lab, so
percentiles from both engines are comparable digit for digit. The
tests recompute the interpolation independently and assert agreement.
