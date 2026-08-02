"""Cross-asset covariance, correlation, and portfolio risk decomposition.

Everything here works on the simple returns of *already aligned* price
series: the caller runs :func:`~quantrisk.series.align` first, and every
function re-checks that the series share one identical date grid rather
than trusting the caller. Covariances are sample covariances
(denominator ``n - 1``, the ``ddof=1`` convention), matching the sample
standard deviation used throughout :mod:`quantrisk.metrics`.

A :class:`Portfolio` is a set of named weights over those series.
Weights must be finite and sum to one within a stated tolerance. Short
positions (negative weights) are allowed, not hidden: the
:attr:`Portfolio.has_short_positions` flag exists because several
long-only guarantees — most importantly that portfolio volatility never
exceeds the weighted sum of individual volatilities — do not survive
negative weights.

The aggregation convention is arithmetic: the portfolio's return each
day is the weight-weighted sum of that day's simple returns, which is
exact for a single period and equivalent to rebalancing back to the
target weights every day. Buy-and-hold weights would drift with prices
instead; over long horizons the two diverge, and this module makes no
attempt to model that drift.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from quantrisk.metrics import TRADING_DAYS_PER_YEAR
from quantrisk.series import MIN_ALIGN_SERIES, PriceSeries

#: Weights must sum to one within this absolute tolerance. It absorbs
#: float rounding (for example thirds) but still rejects a forgotten
#: leg or weights quoted in percent.
WEIGHT_SUM_TOLERANCE = 1e-9


def _validate_aligned(series: Sequence[PriceSeries]) -> tuple[PriceSeries, ...]:
    items = tuple(series)
    if len(items) < MIN_ALIGN_SERIES:
        raise ValueError("need at least two aligned series")
    names = [item.name for item in items]
    if len(set(names)) != len(names):
        raise ValueError(f"series names must be unique, got {names}")
    grid = items[0].dates
    for item in items[1:]:
        if item.dates != grid:
            raise ValueError(
                f"series {item.name!r} is not on the same date grid as {items[0].name!r}; "
                "run align() first"
            )
    return items


def _returns_matrix(items: tuple[PriceSeries, ...]) -> npt.NDArray[np.float64]:
    return np.array([item.simple_returns() for item in items], dtype=np.float64)


def covariance_matrix(series: Sequence[PriceSeries]) -> tuple[tuple[float, ...], ...]:
    """Sample covariance matrix of the series' simple returns.

    Entry ``[i][j]`` is the sample covariance (denominator ``n - 1``,
    ``ddof=1``) between the daily simple returns of series ``i`` and
    ``j``, in input order. The series must already share one date grid;
    anything else is rejected rather than silently truncated.
    """

    items = _validate_aligned(series)
    matrix = np.cov(_returns_matrix(items), ddof=1)
    return tuple(tuple(float(value) for value in row) for row in matrix)


def correlation_matrix(series: Sequence[PriceSeries]) -> tuple[tuple[float, ...], ...]:
    """Sample correlation matrix of the series' simple returns.

    Each covariance is divided by the product of the two sample standard
    deviations. The diagonal is set to exactly 1.0 and off-diagonal
    entries are clipped into [-1, 1], removing float noise in the last
    digit; a constant-return series has zero variance and no defined
    correlation, so it is rejected.
    """

    items = _validate_aligned(series)
    matrix = np.asarray(covariance_matrix(items))
    deviations = np.sqrt(np.diag(matrix))
    for item, deviation in zip(items, deviations, strict=True):
        if deviation == 0.0:
            raise ValueError(
                f"series {item.name!r} has constant returns; its correlation is undefined"
            )
    correlations = np.clip(matrix / np.outer(deviations, deviations), -1.0, 1.0)
    np.fill_diagonal(correlations, 1.0)
    return tuple(tuple(float(value) for value in row) for row in correlations)


@dataclass(frozen=True, slots=True)
class Portfolio:
    """Named weights over aligned price series.

    Weights are plain fractions of portfolio value and must be finite
    and sum to one within :data:`WEIGHT_SUM_TOLERANCE`. Negative weights
    (short positions) are legal and *flagged*, never hidden — see
    :attr:`has_short_positions`.
    """

    names: tuple[str, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.names) < MIN_ALIGN_SERIES:
            raise ValueError("a portfolio needs at least two named weights")
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"portfolio names must be unique, got {list(self.names)}")
        if len(self.names) != len(self.weights):
            raise ValueError(
                f"portfolio has {len(self.names)} names but {len(self.weights)} weights"
            )
        for name in self.names:
            if not name or not name.strip():
                raise ValueError("portfolio names must be nonempty strings")
        for name, weight in zip(self.names, self.weights, strict=True):
            if not isinstance(weight, int | float) or isinstance(weight, bool):
                raise ValueError(f"weight for {name!r} is not a number")
            if not math.isfinite(weight):
                raise ValueError(f"weight for {name!r} is {weight!r}; weights must be finite")
        total = math.fsum(self.weights)
        if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
            raise ValueError(
                f"portfolio weights sum to {total!r}; they must sum to 1 "
                f"within {WEIGHT_SUM_TOLERANCE}"
            )

    @property
    def has_short_positions(self) -> bool:
        """True when any weight is negative.

        Shorts are allowed, but the long-only diversification guarantee
        (portfolio volatility never above the weighted sum of individual
        volatilities) does not hold for them, so the flag is explicit.
        """

        return any(weight < 0.0 for weight in self.weights)

    def _ordered(self, series: Sequence[PriceSeries]) -> tuple[PriceSeries, ...]:
        """The aligned series in portfolio order, names matched exactly."""

        items = _validate_aligned(series)
        by_name = {item.name: item for item in items}
        missing = [name for name in self.names if name not in by_name]
        if missing:
            raise ValueError(f"portfolio names {missing} match no supplied series")
        extra = [item.name for item in items if item.name not in set(self.names)]
        if extra:
            raise ValueError(f"series {extra} have no weight in this portfolio")
        return tuple(by_name[name] for name in self.names)

    def _weights_and_covariance(
        self, series: Sequence[PriceSeries]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        ordered = self._ordered(series)
        weights = np.array(self.weights, dtype=np.float64)
        matrix = np.asarray(covariance_matrix(ordered))
        return weights, matrix

    def return_series(self, series: Sequence[PriceSeries]) -> tuple[float, ...]:
        """Daily portfolio simple returns: the weighted sum of asset returns.

        Weighting each day's simple returns is exact for that day and
        amounts to rebalancing to the target weights daily; buy-and-hold
        weights would drift instead. The result feeds directly into the
        return-based metrics in :mod:`quantrisk.metrics`.
        """

        ordered = self._ordered(series)
        weights = np.array(self.weights, dtype=np.float64)
        combined = weights @ _returns_matrix(ordered)
        return tuple(float(value) for value in combined)

    def annualized_volatility(self, series: Sequence[PriceSeries]) -> float:
        """Annualized portfolio volatility ``sqrt(w' S w * 252)``.

        ``S`` is the sample covariance matrix of daily simple returns in
        portfolio order, so the daily variance ``w' S w`` annualizes by
        the same 252-day convention as every other metric.
        """

        weights, matrix = self._weights_and_covariance(series)
        variance = float(weights @ matrix @ weights)
        return math.sqrt(variance * TRADING_DAYS_PER_YEAR)

    def risk_contributions(self, series: Sequence[PriceSeries]) -> dict[str, float]:
        """Each asset's fractional contribution to portfolio variance.

        Asset ``i`` contributes ``w_i * (S w)_i / (w' S w)``. The
        numerators sum to the denominator by construction, so the
        contributions sum to one; each one is a fraction of total
        portfolio variance, not an isolated volatility. A short or
        strongly diversifying position can contribute a negative share.
        A zero-variance portfolio has nothing to attribute and is
        rejected.
        """

        weights, matrix = self._weights_and_covariance(series)
        marginal = matrix @ weights
        variance = float(weights @ marginal)
        if variance == 0.0:
            raise ValueError("risk contributions are undefined for a zero-variance portfolio")
        return {
            name: float(weight * against / variance)
            for name, weight, against in zip(self.names, weights, marginal, strict=True)
        }

    def diversification_benefit(self, series: Sequence[PriceSeries]) -> float:
        """Weighted sum of individual annualized volatilities minus the portfolio's.

        For a long-only portfolio this is nonnegative, and it is zero
        exactly when every pair of held assets is perfectly correlated —
        the inequality ``vol(w) <= sum_i w_i vol_i`` needs ``w_i >= 0``,
        so no sign is guaranteed once :attr:`has_short_positions` is
        true.
        """

        weights, matrix = self._weights_and_covariance(series)
        individual = np.sqrt(np.diag(matrix) * TRADING_DAYS_PER_YEAR)
        weighted_sum = float(weights @ individual)
        return weighted_sum - self.annualized_volatility(series)
