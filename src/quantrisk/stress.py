"""Stress scenarios: historical window replays and hypothetical shocks.

Two kinds of stress, with very different epistemic standing, and the
code keeps them apart on purpose.

**Historical stress** replays the portfolio's own daily return series
through a named date window of the supplied history and reports what
the portfolio would have printed: the window's total return, the
annualized volatility inside it, and the deepest drawdown inside it.
Nothing is invented — every number is arithmetic over returns the
aligned series actually contain — but the replay inherits the daily
rebalancing convention of
:meth:`quantrisk.portfolio.Portfolio.return_series`, and a window is
only as informative as the history behind it.

**Shock stress** applies a named hypothetical one-day shock vector —
one simple-return shock per held asset — and reports the portfolio's
one-day return as the weight-weighted sum of the shocks. For a
portfolio of plain long or short positions in the assets themselves
that sum is exact for a single day. What makes it an *approximation*
is everything the vector leaves out: the shocks are chosen, not drawn
from any distribution, so no probability attaches to the result; the
co-movements are frozen exactly as written, so shocking one asset
while holding the others at zero asserts a correlation of zero that
the history may flatly contradict; and nothing propagates beyond the
single day — no follow-on volatility, no liquidity effect, no
rebalancing. A shock table is a statement about the scenarios its
author wrote down, never about their likelihood.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from quantrisk.metrics import Drawdown, annualized_volatility, max_drawdown
from quantrisk.portfolio import Portfolio
from quantrisk.series import PriceSeries, _parse_iso_date

#: A window must hold at least five price observations, giving four
#: returns inside it — enough for a sample standard deviation and a
#: drawdown that describe the window rather than a single day.
MIN_WINDOW_OBSERVATIONS = 5


@dataclass(frozen=True, slots=True)
class StressWindow:
    """One named historical window: two ISO dates, start before end.

    Construction validates the dates and their order; whether the
    window lies inside a particular aligned grid is checked by
    :func:`historical_stress`, which is the first place the grid is
    known.
    """

    name: str
    start: str
    end: str

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("stress window name must be a nonempty string")
        start = _parse_iso_date(self.start)
        end = _parse_iso_date(self.end)
        if end < start:
            raise ValueError(
                f"stress window {self.name!r} is reversed: start {self.start} "
                f"follows end {self.end}"
            )


@dataclass(frozen=True, slots=True)
class WindowStressResult:
    """What the portfolio printed inside one historical window.

    ``observations`` counts the grid dates inside the window; the
    ``returns`` inside it number one fewer, because the return landing
    on the window's first date belongs to the day before the window.
    ``total_return`` compounds those returns, ``annualized_volatility``
    is their sample standard deviation times sqrt(252), and
    ``max_drawdown`` is the deepest peak-to-trough decline of the
    compounded value path inside the window.
    """

    window: StressWindow
    observations: int
    total_return: float
    annualized_volatility: float
    max_drawdown: Drawdown


@dataclass(frozen=True, slots=True)
class ShockStressResult:
    """The portfolio's one-day return under one named shock vector.

    ``asset_shocks`` echoes the applied per-asset shocks in portfolio
    order, and ``portfolio_return`` is their weight-weighted sum — the
    linear, correlation-free approximation the module docstring
    qualifies.
    """

    name: str
    asset_shocks: tuple[float, ...]
    portfolio_return: float


def _validate_windows(windows: Sequence[StressWindow]) -> tuple[StressWindow, ...]:
    items = tuple(windows)
    if not items:
        raise ValueError("historical stress needs at least one window")
    names = [item.name for item in items]
    if len(set(names)) != len(names):
        raise ValueError(f"stress window names must be unique, got {names}")
    return items


def historical_stress(
    portfolio: Portfolio,
    aligned_series: Sequence[PriceSeries],
    windows: Sequence[StressWindow],
) -> tuple[WindowStressResult, ...]:
    """Replay the portfolio's return series through each named window.

    The aligned series and the portfolio are revalidated by
    :meth:`~quantrisk.portfolio.Portfolio.return_series`, which also
    fixes the convention: the replayed returns are the daily-rebalanced
    weighted returns, not a buy-and-hold path. Each window's start and
    end must lie inside the aligned date grid — a window reaching
    outside the data would silently describe a shorter period than its
    name claims, so it is rejected instead — and must cover at least
    :data:`MIN_WINDOW_OBSERVATIONS` grid dates. The dates only bound
    the window; they need not be trading days themselves.

    Results come back in the order the windows were given.
    """

    items = _validate_windows(windows)
    returns = portfolio.return_series(aligned_series)
    grid = next(iter(aligned_series)).dates
    results = []
    for window in items:
        if window.start < grid[0] or grid[-1] < window.end:
            raise ValueError(
                f"stress window {window.name!r} [{window.start}, {window.end}] lies "
                f"outside the aligned grid [{grid[0]}, {grid[-1]}]"
            )
        inside = [index for index, day in enumerate(grid) if window.start <= day <= window.end]
        if len(inside) < MIN_WINDOW_OBSERVATIONS:
            raise ValueError(
                f"stress window {window.name!r} covers {len(inside)} observations; "
                f"at least {MIN_WINDOW_OBSERVATIONS} are required"
            )
        first, last = inside[0], inside[-1]
        # returns[k] is the return landing on grid[k + 1], so the returns
        # inside the window are exactly returns[first:last].
        window_returns = returns[first:last]
        path = [1.0]
        for value in window_returns:
            path.append(path[-1] * (1.0 + value))
        if min(path) <= 0.0:
            raise ValueError(
                f"portfolio value inside stress window {window.name!r} reaches "
                f"{min(path)!r}; short positions drove it to zero or below"
            )
        values = PriceSeries(
            name=window.name,
            dates=grid[first : last + 1],
            prices=tuple(path),
        )
        results.append(
            WindowStressResult(
                window=window,
                observations=len(inside),
                total_return=path[-1] - 1.0,
                annualized_volatility=annualized_volatility(window_returns),
                max_drawdown=max_drawdown(values),
            )
        )
    return tuple(results)


def _validate_shock_vector(
    name: str, shocks: Mapping[str, float], portfolio: Portfolio
) -> tuple[float, ...]:
    """The shock vector in portfolio order, complete and each entry > -1."""

    if not name or not name.strip():
        raise ValueError("shock name must be a nonempty string")
    missing = [asset for asset in portfolio.names if asset not in shocks]
    if missing:
        raise ValueError(f"shock {name!r} is missing a value for {missing}")
    unknown = [asset for asset in shocks if asset not in set(portfolio.names)]
    if unknown:
        raise ValueError(f"shock {name!r} names {unknown}, which the portfolio does not hold")
    ordered = []
    for asset in portfolio.names:
        value = shocks[asset]
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise ValueError(f"shock {name!r} value for {asset!r} is not a number")
        if not math.isfinite(value):
            raise ValueError(f"shock {name!r} value for {asset!r} is {value!r}; not finite")
        if value <= -1.0:
            raise ValueError(
                f"shock {name!r} value for {asset!r} is {value!r}; a simple-return "
                "shock must exceed -1, because a positive price cannot lose more "
                "than everything"
            )
        ordered.append(float(value))
    return tuple(ordered)


def shock_stress(
    portfolio: Portfolio, shocks: Mapping[str, Mapping[str, float]]
) -> tuple[ShockStressResult, ...]:
    """The portfolio's one-day return under each named shock vector.

    Each vector must state one simple-return shock strictly above -1
    for *every* held asset — a missing asset is not defaulted to zero,
    because an unshocked asset is itself a scenario assumption the
    author must write down. The reported return is the weight-weighted
    sum of the shocks: exact for one day of plain positions in the
    assets, but linear and correlation-free as an approximation of any
    real event — see the module docstring for exactly when that
    misleads. Results come back in the order the vectors were given.
    """

    if not shocks:
        raise ValueError("shock stress needs at least one shock vector")
    results = []
    for name, vector in shocks.items():
        ordered = _validate_shock_vector(name, vector, portfolio)
        combined = math.fsum(
            weight * value for weight, value in zip(portfolio.weights, ordered, strict=True)
        )
        results.append(
            ShockStressResult(name=name, asset_shocks=ordered, portfolio_return=combined)
        )
    return tuple(results)
