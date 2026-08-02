"""Seeded Monte Carlo simulation of a portfolio's terminal value.

Both simulation modes draw daily returns over a fixed horizon, compound
them from an initial value of 1.0, and summarize the terminal-value
distribution. The historical evidence is the portfolio's own daily
return series — the weighted, daily-rebalanced returns of
:meth:`quantrisk.portfolio.Portfolio.return_series` — and nothing here
invents data beyond redrawing or refitting that history.

The seed is a required argument on purpose: an unseeded simulation
cannot be reproduced, so its numbers cannot be checked, and a number
that cannot be checked does not ship.

A simulated distribution is not a forecast. Both modes assume the
future resembles the sampled history; the docstrings below state when
each assumption misleads, and the numbers always describe the
assumption, not the world.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from quantrisk.portfolio import Portfolio
from quantrisk.series import PriceSeries

#: Fewer than 100 runs makes the tail percentiles meaningless; more
#: than 20,000 buys precision the input data cannot support.
MIN_RUNS = 100
MAX_RUNS = 20_000

#: Horizons are trading days, at least one and at most ten trading
#: years. The cap bounds the ``runs x horizon`` draw matrix and keeps a
#: typo (a calendar-year count, a milliseconds value) from silently
#: allocating gigabytes.
MIN_HORIZON_DAYS = 1
MAX_HORIZON_DAYS = 2_520

_PERCENTILE_LEVELS = (5.0, 25.0, 50.0, 75.0, 95.0)

SimulationMode = Literal["bootstrap", "parametric_normal"]


@dataclass(frozen=True, slots=True)
class Percentiles:
    """The 5/25/50/75/95 percentiles of the terminal-value distribution.

    Computed by ``numpy.percentile`` with its default linear
    interpolation between closest ranks: the sorted values take ranks 0
    through n - 1, the target rank for level ``p`` is
    ``p / 100 * (n - 1)``, and a fractional rank interpolates linearly
    between its two neighbours. This is the same convention as
    ``_interpolated_percentile`` in :mod:`quantrisk.metrics` and the
    interpolated percentile of the sibling energy-investment-lab, so
    every percentile in both engines is comparable digit for digit.
    """

    p5: float
    p25: float
    p50: float
    p75: float
    p95: float


@dataclass(frozen=True, slots=True)
class MonteCarloResult:
    """Every number a simulation produces, plus the seed that reproduces it.

    Terminal values are growth factors of an initial value of 1.0, kept
    complete and sorted ascending in :attr:`terminal_values` so any
    summary figure can be recomputed, checked, or fed to downstream
    analysis. ``terminal_stddev`` is the sample standard deviation
    (denominator ``n - 1``), matching the convention used throughout
    :mod:`quantrisk.metrics`; ``probability_below_initial`` is the
    fraction of runs that finish strictly below 1.0.
    """

    mode: SimulationMode
    seed: int
    runs: int
    horizon_days: int
    terminal_mean: float
    terminal_stddev: float
    terminal_percentiles: Percentiles
    probability_below_initial: float
    terminal_values: tuple[float, ...]


def _validate_arguments(horizon_days: int, runs: int, seed: int) -> None:
    if isinstance(horizon_days, bool) or not isinstance(horizon_days, int):
        raise ValueError("horizon_days must be an integer")
    if not MIN_HORIZON_DAYS <= horizon_days <= MAX_HORIZON_DAYS:
        raise ValueError(f"horizon_days must be between {MIN_HORIZON_DAYS} and {MAX_HORIZON_DAYS}")
    if isinstance(runs, bool) or not isinstance(runs, int):
        raise ValueError("runs must be an integer")
    if not MIN_RUNS <= runs <= MAX_RUNS:
        raise ValueError(f"runs must be between {MIN_RUNS} and {MAX_RUNS}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")


def _historical_returns(
    portfolio: Portfolio, series: Sequence[PriceSeries]
) -> npt.NDArray[np.float64]:
    """The portfolio's historical daily returns, revalidated by the portfolio."""

    return np.array(portfolio.return_series(series), dtype=np.float64)


def _summarize(
    mode: SimulationMode,
    seed: int,
    horizon_days: int,
    terminal: npt.NDArray[np.float64],
) -> MonteCarloResult:
    runs = len(terminal)
    ordered = np.sort(terminal)
    p5, p25, p50, p75, p95 = (float(value) for value in np.percentile(ordered, _PERCENTILE_LEVELS))
    return MonteCarloResult(
        mode=mode,
        seed=seed,
        runs=runs,
        horizon_days=horizon_days,
        terminal_mean=float(np.mean(ordered)),
        terminal_stddev=float(np.std(ordered, ddof=1)),
        terminal_percentiles=Percentiles(p5=p5, p25=p25, p50=p50, p75=p75, p95=p95),
        probability_below_initial=float(np.count_nonzero(ordered < 1.0)) / runs,
        terminal_values=tuple(float(value) for value in ordered),
    )


def run_bootstrap(
    portfolio: Portfolio,
    series: Sequence[PriceSeries],
    *,
    horizon_days: int,
    runs: int,
    seed: int,
) -> MonteCarloResult:
    """Bootstrap simulation: i.i.d. resampling of the historical daily returns.

    Each simulated day draws one historical daily return uniformly with
    replacement, so every terminal value is a product of factors the
    portfolio actually printed — it lies inside
    ``[(1 + min r)^h, (1 + max r)^h]`` by construction, and if every
    historical return is the same value ``r`` the distribution collapses
    to the single point ``(1 + r)^h``.

    The known limitation: resampling days independently destroys
    autocorrelation and volatility clustering. In real return series a
    crash day tends to be followed by more turbulent days; an i.i.d.
    bootstrap scatters those days apart, so multi-day drawdown risk is
    understated whenever the history exhibits clustering. A block
    bootstrap — resampling contiguous runs of days to preserve
    short-range dependence — is future work, and this function does not
    silently approximate it.
    """

    _validate_arguments(horizon_days, runs, seed)
    returns = _historical_returns(portfolio, series)
    # An explicitly seeded PCG64 generator: the draws steer a simulation,
    # nothing security-sensitive, and the same seed must reproduce the
    # same result number for number.
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(returns), size=(runs, horizon_days))
    terminal = np.prod(1.0 + returns[indices], axis=1)
    return _summarize("bootstrap", seed, horizon_days, terminal)


def run_parametric_normal(
    portfolio: Portfolio,
    series: Sequence[PriceSeries],
    *,
    horizon_days: int,
    runs: int,
    seed: int,
) -> MonteCarloResult:
    """Parametric simulation: normal draws fitted to the historical returns.

    Each simulated day draws from ``N(mean, stddev)`` with the sample
    mean and sample standard deviation (denominator ``n - 1``) of the
    portfolio's historical daily returns. The normal assumption is the
    same one behind :func:`quantrisk.metrics.parametric_var`, and it
    fails the same way: daily equity returns have fatter tails than a
    normal distribution, so the simulated extremes are too mild and the
    tail percentiles understate risk exactly where they matter most.
    The normal also has unbounded support, so a draw below -100% —
    impossible for a real asset — can occur and flip a run's compounded
    value negative; treat this mode as a smooth cross-check on the
    bootstrap, not a replacement for it.
    """

    _validate_arguments(horizon_days, runs, seed)
    returns = _historical_returns(portfolio, series)
    mean = float(np.mean(returns))
    stddev = float(np.std(returns, ddof=1))
    # Same seeding discipline as the bootstrap above.
    rng = np.random.default_rng(seed)
    draws = rng.normal(loc=mean, scale=stddev, size=(runs, horizon_days))
    terminal = np.prod(1.0 + draws, axis=1)
    return _summarize("parametric_normal", seed, horizon_days, terminal)
