"""Look-ahead-safe walk-forward backtesting of weight policies.

The design's core claim: the policy interface exposes only
observations strictly preceding the decision date. At each rebalance
the engine hands the policy a :class:`PolicyWindow` built by
*slicing* the aligned data at the decision date — the object
physically contains the dates and prices strictly before that date
and nothing else, so a policy that works from its window alone has no
future data to read, and indexing past the window's end raises
instead of returning a number. The window re-checks the invariant on
construction, and the test suite asserts, for every rebalance of a
run, that each window ends exactly one trading day before its
decision date. The boundary stops at the interface: a Python callable
can still reach future data through closures, external state, files,
or APIs, so caller-supplied policies remain responsible for not
accessing future data through anything outside their window.

Between rebalances the portfolio is buy-and-hold: weights drift with
prices. This is deliberately the opposite convention from
:meth:`quantrisk.portfolio.Portfolio.return_series`, which weights
every day's returns and is therefore implicitly rebalanced daily.
Daily rebalancing is exact for risk description, but as a backtest it
would trade every day while charging costs only on the schedule,
reporting a value path no real portfolio could achieve. Here trades
happen only on scheduled rebalance days, and every one — including the
first, out of cash — is charged a proportional cost: ``cost_rate``
times the turnover (the weight-space distance ``sum |target -
drifted|``) times the pre-trade portfolio value. With a daily schedule
and zero costs the two conventions coincide, and a test asserts that.

Days before the policy's declared ``min_history_days`` are held in
cash, and cash earns exactly zero in this version — no money-market
return, no interest convention smuggled in. The value path stays flat
at the initial value until the first rebalance day.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Protocol

from quantrisk._validation import require_finite_number, require_finite_numbers
from quantrisk.metrics import (
    TRADING_DAYS_PER_YEAR,
    Drawdown,
    annualized_volatility,
    max_drawdown,
)
from quantrisk.portfolio import WEIGHT_SUM_TOLERANCE
from quantrisk.series import PriceSeries, _parse_iso_date


@dataclass(frozen=True, slots=True)
class PolicyWindow:
    """Everything a policy may see when deciding weights: the past.

    The engine builds each window by slicing the aligned series at the
    decision date, so the object holds the dates and prices strictly
    before ``decision_date`` and physically nothing else — no field,
    method, or index reaches data on or after it. That absence is the
    look-ahead boundary: the window exposes only observations strictly
    preceding the decision date, and a policy that reads anything else
    — closures, external state, files, APIs — is reaching around the
    interface, which remains the caller's responsibility to avoid.
    Construction re-checks the invariant and rejects any history that
    touches or passes its own decision date.

    Dates are session labels under the session model
    (``docs/methodology.md``): "strictly before the decision date"
    means earlier sessions, and the window's last date is the session
    immediately preceding the decision — a Friday before a Monday
    decision is one session back, not three days of missing data.
    ``prices[i]`` is the price history of ``names[i]`` on ``dates``,
    taken verbatim from series already validated by
    :class:`~quantrisk.series.PriceSeries`. A directly constructed
    window is re-checked all the same: every price must be a finite
    number (booleans rejected), through the engine's shared
    validation path.
    """

    decision_date: str
    names: tuple[str, ...]
    dates: tuple[str, ...]
    prices: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if not self.names:
            raise ValueError("a policy window needs at least one asset")
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"window names must be unique, got {list(self.names)}")
        if len(self.prices) != len(self.names):
            raise ValueError(
                f"window has {len(self.names)} names but {len(self.prices)} price rows"
            )
        if not self.dates:
            raise ValueError("a policy window needs at least one observed day")
        owner = f"policy window for {self.decision_date}"
        parsed = [_parse_iso_date(text, owner) for text in (*self.dates, self.decision_date)]
        for previous, current in pairwise(parsed):
            if current <= previous:
                raise ValueError(
                    "window dates must be strictly increasing and strictly before "
                    f"the decision date {self.decision_date}; "
                    f"{current.isoformat()} follows {previous.isoformat()}"
                )
        for name, row in zip(self.names, self.prices, strict=True):
            if len(row) != len(self.dates):
                raise ValueError(
                    f"window prices for {name!r} have {len(row)} entries "
                    f"for {len(self.dates)} dates"
                )
            require_finite_numbers(row, f"window price for {name!r}")

    def __len__(self) -> int:
        """The number of observed days, all strictly before the decision date."""

        return len(self.dates)

    def simple_returns(self) -> tuple[tuple[float, ...], ...]:
        """Each asset's simple returns over the window, in ``names`` order.

        A one-day window has no returns yet and yields empty tuples;
        policies that need returns should declare ``min_history_days``
        of at least two.
        """

        return tuple(
            tuple(current / previous - 1.0 for previous, current in pairwise(row))
            for row in self.prices
        )


class WeightPolicy(Protocol):
    """A rebalancing decision rule: past prices in, target weights out.

    ``min_history_days`` is the number of observed days the policy
    needs before its first decision; earlier days are held in cash.
    The call receives only a :class:`PolicyWindow` — data strictly
    before the decision date — and returns target weights in the
    window's asset order, summing to one within
    :data:`~quantrisk.portfolio.WEIGHT_SUM_TOLERANCE`.
    """

    @property
    def min_history_days(self) -> int:
        """Observed days required before the first decision (at least one)."""

    def __call__(self, window: PolicyWindow) -> tuple[float, ...]:
        """Target weights for the window's assets, summing to one."""


@dataclass(frozen=True, slots=True)
class RebalanceRecord:
    """One executed rebalance.

    ``drifted_weights`` are the pre-trade weights the portfolio had
    drifted to (all zero on the first rebalance, out of cash) and
    ``target_weights`` the policy's decision. ``turnover`` is the
    weight-space distance ``sum |target - drifted|`` and ``cost`` the
    currency amount charged: ``cost_rate * turnover * pre-trade
    value``.
    """

    date: str
    drifted_weights: tuple[float, ...]
    target_weights: tuple[float, ...]
    turnover: float
    cost: float


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """A finished walk-forward run.

    ``values`` is the daily portfolio value path on the aligned date
    grid, packaged as a :class:`~quantrisk.series.PriceSeries` so
    every metric in the engine applies to it directly. The return
    figures cover the whole path, cash warmup included:
    ``total_return`` is ``final / initial - 1`` and
    ``annualized_return`` compounds it geometrically over the path's
    return days with the 252-day convention,
    ``(1 + total) ** (252 / (n - 1)) - 1``. ``annualized_volatility``
    and ``max_drawdown`` are :mod:`quantrisk.metrics` applied to the
    same path, and ``total_costs`` sums every rebalance's cost.
    """

    values: PriceSeries
    rebalances: tuple[RebalanceRecord, ...]
    total_return: float
    annualized_return: float
    annualized_volatility: float
    max_drawdown: Drawdown
    total_costs: float


def _require_int(value: int, description: str, *, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{description} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{description} must be at least {minimum}, got {value}")
    return value


def _validate_backtest_series(series: Sequence[PriceSeries]) -> tuple[PriceSeries, ...]:
    """One shared date grid, unique names — but a single series is legal.

    Unlike the cross-asset modules, a backtest of one asset is a
    meaningful (buy-and-hold) run, so only emptiness is rejected.
    """

    items = tuple(series)
    if not items:
        raise ValueError("a backtest needs at least one price series")
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


def _validate_target_weights(weights: Sequence[float], count: int) -> tuple[float, ...]:
    values = tuple(weights)
    if len(values) != count:
        raise ValueError(f"policy returned {len(values)} weights for {count} assets")
    converted = require_finite_numbers(values, "policy weight")
    total = math.fsum(converted)
    if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
        raise ValueError(
            f"policy weights sum to {total!r}; they must sum to 1 within {WEIGHT_SUM_TOLERANCE}"
        )
    return converted


def run_backtest(
    aligned_series: Sequence[PriceSeries],
    policy: WeightPolicy,
    *,
    schedule: int,
    cost_rate: float,
    initial_value: float = 1.0,
) -> BacktestResult:
    """Walk a weight policy forward through the aligned series.

    ``schedule`` is the rebalancing interval in trading sessions —
    rows of the aligned grid, never calendar days (the session model
    of ``docs/methodology.md``), so a weekend between two rows does
    not stretch the interval. The first decision falls on the first
    day with ``policy.min_history_days`` observed days behind it, and
    further decisions come every ``schedule`` sessions after that. Days
    before the first decision are held in cash, which earns exactly
    zero in this version. Between decisions the portfolio is
    buy-and-hold — weights drift with prices — and at each decision
    the policy is handed a :class:`PolicyWindow` sliced strictly
    before the decision date; the interface exposes nothing later, and
    caller-supplied policies remain responsible for not reaching
    future data through external state or external data sources.
    Every rebalance, including the first out of cash, is
    charged ``cost_rate * turnover * pre-trade value`` with turnover
    ``sum |target - drifted|``.

    Raises when a rebalance's cost would consume the whole portfolio,
    or when drifting (possible only with short weights) drives the
    value to zero or below.
    """

    items = _validate_backtest_series(aligned_series)
    _require_int(schedule, "schedule", minimum=1)
    min_history = _require_int(policy.min_history_days, "policy min_history_days", minimum=1)
    rate = require_finite_number(cost_rate, "cost_rate")
    if not 0.0 <= rate < 1.0:
        raise ValueError(f"cost_rate must lie in [0, 1), got {cost_rate}")
    value = require_finite_number(initial_value, "initial_value")
    if value <= 0.0:
        raise ValueError(f"initial_value must be positive, got {initial_value}")
    grid = items[0].dates
    if min_history >= len(grid):
        raise ValueError(
            f"policy needs {min_history} days of history but the series have only "
            f"{len(grid)} days; the warmup leaves nothing to trade"
        )
    names = tuple(item.name for item in items)
    rebalance_days = set(range(min_history, len(grid), schedule))
    values: list[float] = []
    records: list[RebalanceRecord] = []
    holdings: list[float] | None = None
    for today, day in enumerate(grid):
        if holdings is not None:
            holdings = [
                amount * item.prices[today] / item.prices[today - 1]
                for amount, item in zip(holdings, items, strict=True)
            ]
            value = math.fsum(holdings)
            if value <= 0.0:
                raise ValueError(
                    f"portfolio value drifted to {value!r} on {day}; "
                    "short positions drove it to zero or below"
                )
        if today in rebalance_days:
            window = PolicyWindow(
                decision_date=day,
                names=names,
                dates=grid[:today],
                prices=tuple(item.prices[:today] for item in items),
            )
            target = _validate_target_weights(policy(window), len(items))
            drifted = (
                tuple(amount / value for amount in holdings)
                if holdings is not None
                else (0.0,) * len(items)
            )
            turnover = math.fsum(
                abs(goal - held) for goal, held in zip(target, drifted, strict=True)
            )
            charge = rate * turnover * value
            if charge >= value:
                raise ValueError(
                    f"transaction cost {charge!r} on {day} would consume the whole "
                    f"portfolio (value {value!r})"
                )
            value -= charge
            holdings = [weight * value for weight in target]
            records.append(
                RebalanceRecord(
                    date=day,
                    drifted_weights=drifted,
                    target_weights=target,
                    turnover=turnover,
                    cost=charge,
                )
            )
        values.append(value)
    path = PriceSeries(name="backtest", dates=grid, prices=tuple(values))
    total_return = values[-1] / values[0] - 1.0
    periods = len(values) - 1
    return BacktestResult(
        values=path,
        rebalances=tuple(records),
        total_return=total_return,
        annualized_return=(1.0 + total_return) ** (TRADING_DAYS_PER_YEAR / periods) - 1.0,
        annualized_volatility=annualized_volatility(path.simple_returns()),
        max_drawdown=max_drawdown(path),
        total_costs=math.fsum(record.cost for record in records),
    )
