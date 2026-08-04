"""Core risk and performance metrics over daily return series.

Every metric is a small pure function over a sequence of periodic
(daily) returns, or over a :class:`~quantrisk.series.PriceSeries` when
the metric needs the price path itself. A "day" here is a trading
session in the sense of the session model (``docs/methodology.md``):
one return per pair of consecutive rows, annualized by session counts,
with the calendar spacing of the underlying dates playing no role. The
conventions — annualization factor, denominators, percentile method,
sign of a VaR — are stated in ``docs/methodology.md`` next to a
hand-worked example whose digits the test suite asserts.

Every number entering a metric — each return, the risk-free rate, the
Sortino target, the confidence level — passes the engine's shared
validation path (:mod:`quantrisk._validation`): booleans and
non-finite values are rejected with the offending parameter named.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from scipy.stats import norm

from quantrisk._validation import require_finite_number, require_finite_numbers
from quantrisk.series import PriceSeries

#: Annualization convention: 252 trading sessions per year. The factor
#: counts sessions — rows of a price series — never calendar days: see
#: the session model in ``docs/methodology.md``. Volatility scales with
#: the square root of time, mean returns scale linearly.
TRADING_DAYS_PER_YEAR = 252

#: A sample standard deviation needs at least two observations.
MIN_RETURNS = 2


def _validate_returns(returns: Sequence[float]) -> tuple[float, ...]:
    values = tuple(returns)
    if len(values) < MIN_RETURNS:
        raise ValueError(f"need at least {MIN_RETURNS} returns, got {len(values)}")
    return require_finite_numbers(values, "return")


def _validate_confidence(confidence: float) -> float:
    value = require_finite_number(confidence, "confidence")
    if not 0.0 < value < 1.0:
        raise ValueError(f"confidence must lie strictly between 0 and 1, got {value}")
    return value


def _mean(values: tuple[float, ...]) -> float:
    return sum(values) / len(values)


def _sample_stddev(values: tuple[float, ...]) -> float:
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def annualized_volatility(returns: Sequence[float]) -> float:
    """Sample standard deviation of periodic returns times sqrt(252).

    The denominator is ``n - 1`` (sample, not population). The 252
    annualizes session counts: every return is one session, however
    many calendar days its two observations spanned.
    """

    values = _validate_returns(returns)
    return _sample_stddev(values) * math.sqrt(TRADING_DAYS_PER_YEAR)


def sharpe_ratio(returns: Sequence[float], risk_free_rate_annual: float) -> float:
    """Annualized Sharpe ratio of periodic returns over a risk-free rate.

    The annual risk-free rate is converted to a periodic rate by plain
    division by 252. The ratio of periodic mean excess return to periodic
    sample standard deviation is annualized with sqrt(252).
    """

    values = _validate_returns(returns)
    rate = require_finite_number(risk_free_rate_annual, "risk_free_rate_annual")
    stddev = _sample_stddev(values)
    if stddev == 0:
        raise ValueError("Sharpe ratio is undefined for constant returns")
    risk_free_periodic = rate / TRADING_DAYS_PER_YEAR
    excess = _mean(values) - risk_free_periodic
    return excess / stddev * math.sqrt(TRADING_DAYS_PER_YEAR)


def sortino_ratio(returns: Sequence[float], target_return_periodic: float = 0.0) -> float:
    """Annualized Sortino ratio of periodic returns against a periodic target.

    The target is a periodic (daily) return and defaults to zero. The
    downside deviation is the root of the mean squared shortfall below
    the target, averaged over all ``n`` observations — not only the
    losing ones and not ``n - 1``. Annualization mirrors the Sharpe
    ratio: mean excess over downside deviation, times sqrt(252).
    """

    values = _validate_returns(returns)
    target = require_finite_number(target_return_periodic, "target_return_periodic")
    shortfalls = tuple(min(value - target, 0.0) for value in values)
    downside = math.sqrt(sum(shortfall**2 for shortfall in shortfalls) / len(values))
    if downside == 0:
        raise ValueError("Sortino ratio is undefined when no return falls below the target")
    excess = _mean(values) - target
    return excess / downside * math.sqrt(TRADING_DAYS_PER_YEAR)


@dataclass(frozen=True, slots=True)
class Drawdown:
    """Deepest peak-to-trough decline of a price path.

    ``depth`` is a nonnegative fraction: 0.04 means 4% below the peak.
    When the series never declines, ``depth`` is zero and both dates are
    the first date of the series.

    The dates are session labels under the session model
    (``docs/methodology.md``): the drawdown's duration is the number of
    sessions between them, and the calendar span they suggest can be
    longer, because weekends and holidays between the two dates are not
    sessions and no return accrued across them.
    """

    depth: float
    peak_date: str
    trough_date: str


def max_drawdown(series: PriceSeries) -> Drawdown:
    """Maximum drawdown of a price series with its peak and trough dates.

    The running peak is the highest price seen so far; each price's
    drawdown is ``1 - price / running peak``. Ties keep the earliest
    trough, and the peak is the date the running peak was first set.
    """

    peak_price = series.prices[0]
    peak_date = series.dates[0]
    worst = Drawdown(depth=0.0, peak_date=series.dates[0], trough_date=series.dates[0])
    for day, price in zip(series.dates, series.prices, strict=True):
        if price > peak_price:
            peak_price = price
            peak_date = day
        depth = 1.0 - price / peak_price
        if depth > worst.depth:
            worst = Drawdown(depth=depth, peak_date=peak_date, trough_date=day)
    return worst


def _interpolated_percentile(sorted_values: tuple[float, ...], level: float) -> float:
    """Percentile with linear interpolation between closest ranks.

    The sorted values take ranks 0 through n - 1; the target rank for
    fraction ``level`` is ``level * (n - 1)``, and a fractional rank
    interpolates linearly between its two neighbours. This is the same
    convention NumPy calls ``linear``.
    """

    rank = level * (len(sorted_values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    fraction = rank - lower
    return sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])


def historical_var(returns: Sequence[float], confidence: float) -> float:
    """Historical value-at-risk as a positive loss fraction.

    At confidence ``c`` the VaR is the negated ``(1 - c)`` interpolated
    percentile of the observed returns: the loss that the worst
    ``(1 - c)`` share of observed days reached or exceeded. A negative
    result means even that percentile was a gain.
    """

    values = _validate_returns(returns)
    level = _validate_confidence(confidence)
    return -_interpolated_percentile(tuple(sorted(values)), 1.0 - level)


def historical_cvar(returns: Sequence[float], confidence: float) -> float:
    """Historical conditional value-at-risk (expected shortfall).

    The negated mean of every observed return at or below the
    ``(1 - c)`` interpolated percentile. At least the worst observation
    always qualifies, and the result is never below the historical VaR
    at the same confidence.
    """

    values = _validate_returns(returns)
    level = _validate_confidence(confidence)
    threshold = _interpolated_percentile(tuple(sorted(values)), 1.0 - level)
    tail = [value for value in values if value <= threshold]
    return -sum(tail) / len(tail)


def parametric_var(returns: Sequence[float], confidence: float) -> float:
    """Normal (variance-covariance) value-at-risk as a positive loss fraction.

    Fits a normal distribution by the sample mean and sample standard
    deviation and negates its ``(1 - c)`` quantile:
    ``-(mean + z * stddev)`` with ``z = norm.ppf(1 - c)``. Daily equity
    returns have fatter tails than a normal, so at high confidences this
    understates tail losses; compare it with the historical figures.
    """

    values = _validate_returns(returns)
    level = _validate_confidence(confidence)
    z_score = float(norm.ppf(1.0 - level))
    return -(_mean(values) + z_score * _sample_stddev(values))
