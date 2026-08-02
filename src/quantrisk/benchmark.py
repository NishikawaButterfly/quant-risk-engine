"""Benchmark-relative performance: beta, alpha, tracking error, capture.

Everything here compares two return sequences on the same date grid —
the portfolio's and the benchmark's, paired element by element. The
engine does not align them: alignment is a policy over *price* series
(:func:`~quantrisk.series.align`), so the caller builds the aligned
intersection first and hands over the resulting returns. What this
module re-checks is what it can see: equal lengths, finite values, and
enough observations for the statistics to mean anything.

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
  that tracking error. A zero tracking error means the active return
  is constant — the portfolio is the benchmark plus a fixed daily
  offset — so the ratio has a zero denominator and is reported as
  ``None`` rather than a number or an exception: unlike a constant
  benchmark, this is a legitimate comparison whose other statistics
  remain meaningful.
* **Capture ratios** condition on the benchmark's sign. Up capture is
  the mean portfolio return over the days the benchmark rose, divided
  by the mean benchmark return over those same days; down capture is
  the same over the days the benchmark fell. Days the benchmark
  returned exactly zero belong to neither side. A side with no days
  has no ratio and is reported as ``None`` — a benchmark that never
  fell simply has no down capture to report.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from quantrisk.metrics import TRADING_DAYS_PER_YEAR, _mean, _sample_stddev

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
    ``None`` exactly when the benchmark never rose (never fell), so
    that side of the market has no days to measure.
    """

    beta: float
    alpha_daily: float
    alpha_annualized: float
    tracking_error: float
    information_ratio: float | None
    up_capture: float | None
    down_capture: float | None


def _validate_pair(
    portfolio_returns: Sequence[float], benchmark_returns: Sequence[float]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    portfolio = tuple(portfolio_returns)
    benchmark = tuple(benchmark_returns)
    if len(portfolio) != len(benchmark):
        raise ValueError(
            f"portfolio has {len(portfolio)} returns but the benchmark has "
            f"{len(benchmark)}; both must cover the same date grid"
        )
    if len(portfolio) < MIN_COMPARISON_RETURNS:
        raise ValueError(
            f"need at least {MIN_COMPARISON_RETURNS} paired returns, got {len(portfolio)}"
        )
    for label, values in (("portfolio", portfolio), ("benchmark", benchmark)):
        for value in values:
            if not math.isfinite(value):
                raise ValueError(f"{label} returns must be finite, got {value!r}")
    return portfolio, benchmark


def _sample_covariance(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    mean_first = _mean(first)
    mean_second = _mean(second)
    paired = zip(first, second, strict=True)
    return sum((x - mean_first) * (y - mean_second) for x, y in paired) / (len(first) - 1)


def _capture_ratio(
    portfolio: tuple[float, ...], benchmark: tuple[float, ...], *, rising: bool
) -> float | None:
    """Mean portfolio return over mean benchmark return on one side.

    ``rising`` selects the days the benchmark rose (fell when false);
    zero-return benchmark days belong to neither side. A side with no
    days has nothing to measure and yields ``None``.
    """

    pairs = [
        (gain, move)
        for gain, move in zip(portfolio, benchmark, strict=True)
        if (move > 0.0 if rising else move < 0.0)
    ]
    if not pairs:
        return None
    mean_portfolio = math.fsum(gain for gain, _ in pairs) / len(pairs)
    mean_benchmark = math.fsum(move for _, move in pairs) / len(pairs)
    return mean_portfolio / mean_benchmark


def compare_to_benchmark(
    portfolio_returns: Sequence[float],
    benchmark_returns: Sequence[float],
    *,
    risk_free_rate_daily: float = 0.0,
) -> BenchmarkComparison:
    """Compare paired daily returns against a benchmark's.

    Both sequences must cover the same date grid in the same order —
    build them from series aligned with
    :func:`~quantrisk.series.align` — with equal lengths, finite
    values, and at least :data:`MIN_COMPARISON_RETURNS` observations.
    ``risk_free_rate_daily`` is a *daily* rate entering only the alpha
    (beta and the active-return statistics are invariant to a constant
    shift applied to both series). A constant-return benchmark has zero
    variance, leaving beta's denominator empty of information, and is
    rejected rather than divided by. See the module docstring for each
    statistic's convention and for when a field is ``None``.
    """

    portfolio, benchmark = _validate_pair(portfolio_returns, benchmark_returns)
    if not isinstance(risk_free_rate_daily, int | float) or isinstance(risk_free_rate_daily, bool):
        raise ValueError("risk_free_rate_daily is not a number")
    if not math.isfinite(risk_free_rate_daily):
        raise ValueError(f"risk_free_rate_daily is {risk_free_rate_daily!r}; it must be finite")
    rate = float(risk_free_rate_daily)

    benchmark_variance = _sample_covariance(benchmark, benchmark)
    if benchmark_variance == 0.0:
        raise ValueError(
            "benchmark returns are constant (zero variance); beta divides by the "
            "benchmark variance and is undefined"
        )
    beta = _sample_covariance(portfolio, benchmark) / benchmark_variance
    alpha_daily = (_mean(portfolio) - rate) - beta * (_mean(benchmark) - rate)

    active = tuple(gain - move for gain, move in zip(portfolio, benchmark, strict=True))
    active_stddev = _sample_stddev(active)
    tracking_error = active_stddev * math.sqrt(TRADING_DAYS_PER_YEAR)
    information_ratio = (
        None if tracking_error == 0.0 else _mean(active) * TRADING_DAYS_PER_YEAR / tracking_error
    )

    return BenchmarkComparison(
        beta=beta,
        alpha_daily=alpha_daily,
        alpha_annualized=alpha_daily * TRADING_DAYS_PER_YEAR,
        tracking_error=tracking_error,
        information_ratio=information_ratio,
        up_capture=_capture_ratio(portfolio, benchmark, rising=True),
        down_capture=_capture_ratio(portfolio, benchmark, rising=False),
    )
