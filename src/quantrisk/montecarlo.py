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

The draws are never materialized as one ``runs x horizon`` matrix:
both modes draw and compound in blocks of at most :data:`BLOCK_RUNS`
runs, so the peak footprint is fixed by the block size and the horizon
cap while the results stay digit-for-digit identical to a whole-matrix
draw for the same seed — the tests assert both.

A daily draw at or below -100% — reachable only in the parametric
mode, whose normal has unbounded support — takes a run's value path
to or through zero. Such a run is absorbed at exactly 0.0 (bankruptcy
is absorbing: a portfolio worth nothing cannot compound back) and the
result counts it in :attr:`MonteCarloResult.bankruptcies`. Runs whose
draws all stay above -100% are untouched, digit for digit.

A simulated distribution is not a forecast. Both modes assume the
future resembles the sampled history; the docstrings below state when
each assumption misleads, and the numbers always describe the
assumption, not the world.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
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
#: years. The cap bounds the width of every draw block and keeps a
#: typo (a calendar-year count, a milliseconds value) from silently
#: inflating each block by orders of magnitude.
MIN_HORIZON_DAYS = 1
MAX_HORIZON_DAYS = 2_520

#: Draws are processed in blocks of at most this many runs: each block
#: materializes its own ``(rows, horizon)`` draw matrix, is compounded
#: into that block's terminal values, and is discarded before the next
#: block is drawn. The peak footprint is therefore
#: O(``BLOCK_RUNS`` x horizon) regardless of ``runs`` — at the horizon
#: cap, one 512 x 2,520 float64 matrix is about 10 MB — where a whole
#: ``runs x horizon`` matrix at both caps would put over a gigabyte
#: across the draw matrix and its compounding temporaries.
BLOCK_RUNS = 512

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

    ``bankruptcies`` counts the runs absorbed at 0.0 because a daily
    draw took the value path to or through zero (any draw at or below
    -100%). Those runs sit at the front of the sorted terminal values
    as exact zeros and are included in every summary figure; the count
    is carried explicitly because a terminal 0.0 could in principle
    also arise from floating-point underflow of a long product of tiny
    positive factors, so counting zeros after the fact is not the same
    statement. Bootstrap runs redraw historical returns, which strictly
    positive prices keep above -100%, so their count is always 0.
    """

    mode: SimulationMode
    seed: int
    runs: int
    horizon_days: int
    terminal_mean: float
    terminal_stddev: float
    terminal_percentiles: Percentiles
    probability_below_initial: float
    bankruptcies: int
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


def _block_row_counts(runs: int) -> tuple[int, ...]:
    """Row counts of the draw blocks covering ``runs`` rows.

    Full blocks of :data:`BLOCK_RUNS` rows first, then the remainder if
    any, so the peak allocation is reached in the first block and never
    exceeded, and the counts always sum to exactly ``runs``.
    """

    full_blocks, remainder = divmod(runs, BLOCK_RUNS)
    return (BLOCK_RUNS,) * full_blocks + ((remainder,) if remainder else ())


def _compound_in_blocks(
    runs: int, draw_block: Callable[[int], npt.NDArray[np.float64]]
) -> tuple[npt.NDArray[np.float64], int]:
    """Terminal values of ``runs`` runs, drawn and compounded per block.

    ``draw_block(rows)`` returns the next ``(rows, horizon)`` matrix of
    daily returns; each block is compounded into its runs' terminal
    values and released before the next call, so only one block's
    matrix (and its compounding temporary) is ever alive.

    Blockwise drawing is stream-identical to one whole-matrix call:
    NumPy's ``Generator`` fills arrays in C (row-major) order and both
    ``integers`` and ``normal`` consume the underlying PCG64 stream
    value by value, so drawing the same rows across consecutive calls
    yields the same numbers in the same positions. The tests assert
    this digit for digit against an inline whole-matrix reference, so
    a regression cannot land silently.

    Bankruptcy is absorbing: a day's factor ``1 + r`` at or below zero
    means the value path touched (r = -1 exactly) or crossed (r < -1)
    zero that day, and a portfolio worth nothing cannot compound back,
    so the run's terminal value is exactly 0.0 and the run is counted.
    Because only terminal values are reported, masking whole rows on
    ``any(factor <= 0)`` is equivalent to clamping a cumulative
    product at its first nonpositive step: a row with every factor
    positive never touches zero and keeps its plain ``np.prod`` — the
    mask assigns nothing, so surviving runs stay bit-identical to the
    pre-policy result — while a row with any factor <= 0 is absorbed
    on the day that factor applies, so whatever follows, its terminal
    is 0.0. The naive product only agrees by accident: a lone zero
    factor, or an odd count of negative factors (a nonsense negative
    terminal). An even count of negative factors yields a
    plausible-looking POSITIVE product that no sign check on the
    output could catch — which is why the rule tests the factors,
    never the product. The second returned value is the number of
    absorbed rows; the draws consumed from the generator are the same
    in every case.
    """

    terminal = np.empty(runs, dtype=np.float64)
    bankruptcies = 0
    start = 0
    for rows in _block_row_counts(runs):
        factors = 1.0 + draw_block(rows)
        bankrupt = np.any(factors <= 0.0, axis=1)
        block_terminal = np.prod(factors, axis=1)
        block_terminal[bankrupt] = 0.0
        terminal[start : start + rows] = block_terminal
        bankruptcies += int(np.count_nonzero(bankrupt))
        start += rows
    return terminal, bankruptcies


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
    bankruptcies: int,
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
        bankruptcies=bankruptcies,
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

    Bankruptcy cannot occur here: :class:`~quantrisk.series.PriceSeries`
    rejects nonpositive prices, so every historical return exceeds
    -100%, a bootstrap can only redraw those values, and the result's
    ``bankruptcies`` count is always 0. The tests assert it.
    """

    _validate_arguments(horizon_days, runs, seed)
    returns = _historical_returns(portfolio, series)
    # An explicitly seeded PCG64 generator: the draws steer a simulation,
    # nothing security-sensitive, and the same seed must reproduce the
    # same result number for number.
    rng = np.random.default_rng(seed)

    def draw_block(rows: int) -> npt.NDArray[np.float64]:
        indices = rng.integers(0, len(returns), size=(rows, horizon_days))
        return returns[indices]

    terminal, bankruptcies = _compound_in_blocks(runs, draw_block)
    return _summarize("bootstrap", seed, horizon_days, terminal, bankruptcies)


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

    The normal also has unbounded support, so a draw at or below -100%
    — impossible for a real asset — can occur. Such a run is absorbed:
    its value path touches or crosses zero, its terminal value is
    exactly 0.0, and the result counts it in ``bankruptcies``. For a
    history like the sample data (daily sigma under 1%) the per-draw
    probability is below 1e-3000 — unobservable — but a caller fitting
    a violently volatile history can make it material: at a daily
    sigma of 0.25 about 3e-5 of draws cross, which at the run and
    horizon caps is over a thousand absorbed runs. Treat this mode as
    a smooth cross-check on the bootstrap, not a replacement for it.
    """

    _validate_arguments(horizon_days, runs, seed)
    returns = _historical_returns(portfolio, series)
    mean = float(np.mean(returns))
    stddev = float(np.std(returns, ddof=1))
    # Same seeding discipline as the bootstrap above.
    rng = np.random.default_rng(seed)

    def draw_block(rows: int) -> npt.NDArray[np.float64]:
        return rng.normal(loc=mean, scale=stddev, size=(rows, horizon_days))

    terminal, bankruptcies = _compound_in_blocks(runs, draw_block)
    return _summarize("parametric_normal", seed, horizon_days, terminal, bankruptcies)
