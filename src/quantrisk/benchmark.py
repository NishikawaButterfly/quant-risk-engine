"""Benchmark-relative performance: beta, alpha, tracking error, capture.

Everything here compares two daily return series on one shared date
grid — the portfolio's and the benchmark's, paired date by date. The
comparison takes the dated inputs themselves (the portfolio's weights
over its aligned asset series, and the benchmark's price series), so
the pairing is checked, not trusted: every series must carry the
identical date grid, and anything else is rejected with instructions to
run :func:`~quantrisk.series.align` first. Alignment stays a visible
act of the caller — the intersection policy is a policy over *price*
series, and dates are only ever dropped where the caller can see it —
while this module re-checks the grid, finite returns, and enough
observations for the statistics to mean anything.

Conventions, stated once:

* **Beta** is ``Cov(p, b) / Var(b)`` over the raw returns, both sample
  moments with denominator ``n - 1`` (``ddof=1``), matching every other
  second moment in the engine. A constant risk-free rate drops out of
  both covariances, so beta is the same over raw and excess returns.
* **Alpha** is the arithmetic CAPM residual over *excess* returns:
  ``mean(p - rf) - beta * mean(b - rf)`` per day, with
  ``risk_free_rate_daily`` a daily rate (zero by default). The annual
  figure multiplies the daily one by 252 — arithmetic scaling, not
  geometric compounding. Alpha is a regression intercept, not a
  tradable return; compounding it as ``(1 + a)^252 - 1`` would smuggle
  a growth model into a linear residual, while the arithmetic
  convention keeps alpha in the same linear units as the regression
  that produced it (and keeps exact the identity that adding a
  constant ``c`` to every portfolio return moves annual alpha by
  exactly ``252 c``).
* **Tracking error** is the sample standard deviation of the active
  returns ``p - b``, annualized with sqrt(252). The **information
  ratio** is the annualized mean active return (mean times 252) over
  that tracking error. An active return constant to within the
  engine's near-zero-variance tolerance
  (:data:`~quantrisk._validation.MIN_VOLATILITY`, compared on the
  daily stddev) means the portfolio is the benchmark plus a fixed
  daily offset — the ratio's denominator holds nothing but rounding
  noise, and the field is reported as ``None`` rather than a number
  or an exception: unlike a degenerate benchmark, this is a
  legitimate comparison whose other statistics remain meaningful.
  This is the tolerance's one *degrade* site; every other site
  rejects.
* **Capture ratios** are geometric and condition on the benchmark's
  sign. Each side's returns compound into one cumulative move —
  ``prod(1 + r) - 1`` — and up capture divides the portfolio's
  compounded move over the days the benchmark rose by the benchmark's
  own compounded move over those same days; down capture is the same
  over the days the benchmark fell. Days the benchmark returned
  exactly zero belong to neither side. Compounding, not an arithmetic
  mean of the days, is the convention because it is how the engine
  treats returns everywhere they aggregate across days (the
  backtest's annualized return, the stress windows' total return, the
  report's value path), and the two conventions disagree exactly over
  runs of same-signed days — when capture matters most. A side with
  no days has no ratio and is reported as ``None`` — a benchmark that
  never fell simply has no down capture to report. A nonempty side
  whose benchmark move compounds to exactly zero — reachable only
  when a sub-ulp return rounds away in ``1 + r``, never from
  validated prices — is rejected rather than divided by.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from quantrisk._validation import (
    MIN_VOLATILITY,
    require_finite_number,
    require_finite_numbers,
    require_meaningful_volatility,
)
from quantrisk.metrics import TRADING_DAYS_PER_YEAR, _mean, _sample_stddev
from quantrisk.portfolio import Portfolio, _validate_aligned
from quantrisk.series import PriceSeries

#: A comparison needs at least three paired observations: a straight
#: line fits any two points exactly, so a two-day beta is 1.0-R² by
#: construction and its alpha describes nothing but the fixture.
MIN_COMPARISON_RETURNS = 3


@dataclass(frozen=True, slots=True)
class BenchmarkComparison:
    """Benchmark-relative statistics of one portfolio return series.

    ``alpha_daily`` is the per-day arithmetic CAPM residual and
    ``alpha_annualized`` is that figure times 252 (arithmetic, not
    geometric — see the module docstring). ``tracking_error`` is
    annualized. ``information_ratio`` is ``None`` exactly when the
    tracking error is zero: the active return is then constant and the
    ratio's denominator vanishes. ``up_capture`` (``down_capture``) is
    the *geometric* capture ratio — the portfolio's compounded return
    over the benchmark's, on the days the benchmark rose (fell) — and
    is ``None`` exactly when the benchmark never rose (never fell), so
    that side of the market has no days to measure.
    """

    beta: float
    alpha_daily: float
    alpha_annualized: float
    tracking_error: float
    information_ratio: float | None
    up_capture: float | None
    down_capture: float | None


def _paired_returns(
    portfolio: Portfolio,
    asset_series: Sequence[PriceSeries],
    benchmark_series: PriceSeries,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """The portfolio's and the benchmark's returns on one verified grid.

    The combined grid check puts the benchmark on the assets' own dates
    (and keeps its name distinct from every holding); the portfolio then
    validates its side — names matched exactly, no extra series. Only
    after that are the two return series read off, equal in length by
    construction. Prices are validated finite and positive, but a ratio
    of finite prices can still overflow, so the derived returns pass
    through the engine's shared validation path as well.
    """

    _validate_aligned([*asset_series, benchmark_series])
    portfolio_returns = portfolio.return_series(asset_series)
    benchmark_returns = benchmark_series.simple_returns()
    if len(portfolio_returns) < MIN_COMPARISON_RETURNS:
        raise ValueError(
            f"need at least {MIN_COMPARISON_RETURNS} paired returns, got {len(portfolio_returns)}"
        )
    return (
        require_finite_numbers(portfolio_returns, "portfolio return"),
        require_finite_numbers(benchmark_returns, "benchmark return"),
    )


def _sample_covariance(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    mean_first = _mean(first)
    mean_second = _mean(second)
    paired = zip(first, second, strict=True)
    return sum((x - mean_first) * (y - mean_second) for x, y in paired) / (len(first) - 1)


def _capture_ratio(
    portfolio: tuple[float, ...], benchmark: tuple[float, ...], *, rising: bool
) -> float | None:
    """Compounded portfolio move over compounded benchmark move, one side.

    The geometric convention: each side's returns compound into one
    cumulative return, ``prod(1 + r) - 1``, and the ratio divides the
    portfolio's by the benchmark's. ``rising`` selects the days the
    benchmark rose (fell when false); zero-return benchmark days belong
    to neither side. A side with no days has nothing to measure and
    yields ``None``. Rising factors all exceed one and falling factors
    all sit strictly inside (0, 1), so a nonempty side's benchmark move
    is never truly zero; only a sub-ulp return — one that ``1.0 + move``
    rounds away entirely — can compound the denominator to exactly
    zero, and that is rejected rather than divided by.
    """

    pairs = [
        (gain, move)
        for gain, move in zip(portfolio, benchmark, strict=True)
        if (move > 0.0 if rising else move < 0.0)
    ]
    if not pairs:
        return None
    portfolio_move = math.prod(1.0 + gain for gain, _ in pairs) - 1.0
    benchmark_move = math.prod(1.0 + move for _, move in pairs) - 1.0
    if benchmark_move == 0.0:
        side = "rose" if rising else "fell"
        raise ValueError(
            f"benchmark returns compound to exactly zero over the days the benchmark {side}; "
            "the capture ratio divides by that compounded move and is undefined"
        )
    return portfolio_move / benchmark_move


def compare_to_benchmark(
    portfolio: Portfolio,
    asset_series: Sequence[PriceSeries],
    benchmark_series: PriceSeries,
    *,
    risk_free_rate_daily: float = 0.0,
) -> BenchmarkComparison:
    """Compare a portfolio's daily returns against a benchmark's.

    ``asset_series`` are the portfolio's holdings and
    ``benchmark_series`` the benchmark, all on one identical date grid —
    put them there with :func:`~quantrisk.series.align`; any series on a
    different grid is rejected rather than paired, since two return
    sequences of equal length prove nothing about their dates. The
    shared grid must yield at least :data:`MIN_COMPARISON_RETURNS`
    paired returns, the benchmark's name must not be a holding, and both
    return series must be finite. ``risk_free_rate_daily`` is a *daily*
    rate entering only the alpha (beta and the active-return statistics
    are invariant to a constant shift applied to both series). A
    benchmark whose returns are constant to within the engine's
    near-zero-variance tolerance
    (:data:`~quantrisk._validation.MIN_VOLATILITY`, squared for this
    variance denominator) leaves beta's denominator empty of
    information and is rejected rather than divided by. See the module
    docstring for each statistic's convention and for when a field is
    ``None``.
    """

    portfolio_returns, benchmark_returns = _paired_returns(
        portfolio, asset_series, benchmark_series
    )
    rate = require_finite_number(risk_free_rate_daily, "risk_free_rate_daily")

    benchmark_variance = _sample_covariance(benchmark_returns, benchmark_returns)
    # A variance site: the shared near-zero tolerance is stated as a
    # daily volatility, so the guard runs on sqrt(variance) — that is,
    # MIN_VOLATILITY squared in variance terms.
    require_meaningful_volatility(
        math.sqrt(benchmark_variance),
        "benchmark returns are effectively constant (near-zero variance); beta "
        "divides by the benchmark variance and is undefined",
    )
    beta = _sample_covariance(portfolio_returns, benchmark_returns) / benchmark_variance
    alpha_daily = (_mean(portfolio_returns) - rate) - beta * (_mean(benchmark_returns) - rate)

    active = tuple(
        gain - move for gain, move in zip(portfolio_returns, benchmark_returns, strict=True)
    )
    active_stddev = _sample_stddev(active)
    tracking_error = active_stddev * math.sqrt(TRADING_DAYS_PER_YEAR)
    # The engine's one *degrade* site for the shared tolerance: an
    # active return constant to within MIN_VOLATILITY (compared on the
    # daily stddev, before annualization) leaves the information ratio
    # without a denominator, but the comparison itself stays legitimate
    # and every other statistic stands — so the field is None rather
    # than an exception, exactly as for an exactly-zero tracking error.
    information_ratio = (
        None
        if active_stddev < MIN_VOLATILITY
        else _mean(active) * TRADING_DAYS_PER_YEAR / tracking_error
    )

    return BenchmarkComparison(
        beta=beta,
        alpha_daily=alpha_daily,
        alpha_annualized=alpha_daily * TRADING_DAYS_PER_YEAR,
        tracking_error=tracking_error,
        information_ratio=information_ratio,
        up_capture=_capture_ratio(portfolio_returns, benchmark_returns, rising=True),
        down_capture=_capture_ratio(portfolio_returns, benchmark_returns, rising=False),
    )
