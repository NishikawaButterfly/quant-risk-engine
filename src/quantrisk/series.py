"""Validated daily price series and their return transforms.

A :class:`PriceSeries` is the only way price data enters the engine. Its
constructor rejects anything that would silently poison a risk number:
unsorted or duplicate dates, non-positive or non-finite prices, and
series too short to produce a sample statistic.

Alignment of several series keeps only the dates every series shares.
Forward-filling missing dates is a deliberate non-feature of this
version: a filled price is a fabricated observation, it deflates
volatility and correlation estimates, and the caller cannot see it
happened. Gaps must be resolved upstream, where they are visible.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from itertools import pairwise

#: A series must produce at least two returns, so a sample standard
#: deviation (denominator n - 1) is defined. Two returns need three prices.
MIN_PRICES = 3

#: Aligning fewer than two series is a sign of a caller bug.
MIN_ALIGN_SERIES = 2


def _parse_iso_date(text: str) -> date:
    """Return the calendar date for an ISO ``YYYY-MM-DD`` string."""

    try:
        parsed = date.fromisoformat(text)
    except (TypeError, ValueError) as error:
        raise ValueError(f"date {text!r} is not an ISO YYYY-MM-DD date") from error
    if text != parsed.isoformat():
        raise ValueError(f"date {text!r} is not in canonical YYYY-MM-DD form")
    return parsed


@dataclass(frozen=True, slots=True)
class PriceSeries:
    """One security's closing prices on strictly increasing ISO dates."""

    name: str
    dates: tuple[str, ...]
    prices: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("series name must be a nonempty string")
        if len(self.dates) != len(self.prices):
            raise ValueError(
                f"series {self.name!r} has {len(self.dates)} dates but {len(self.prices)} prices"
            )
        if len(self.prices) < MIN_PRICES:
            raise ValueError(
                f"series {self.name!r} has {len(self.prices)} prices; "
                f"at least {MIN_PRICES} are required"
            )
        parsed = [_parse_iso_date(text) for text in self.dates]
        for previous, current in pairwise(parsed):
            if current <= previous:
                raise ValueError(
                    f"series {self.name!r} dates must be strictly increasing; "
                    f"{current.isoformat()} follows {previous.isoformat()}"
                )
        for text, price in zip(self.dates, self.prices, strict=True):
            if not isinstance(price, int | float) or isinstance(price, bool):
                raise ValueError(f"series {self.name!r} price on {text} is not a number")
            if not math.isfinite(price) or price <= 0:
                raise ValueError(
                    f"series {self.name!r} price on {text} is {price!r}; "
                    "prices must be finite and positive"
                )

    def __len__(self) -> int:
        return len(self.prices)

    def simple_returns(self) -> tuple[float, ...]:
        """Return the simple returns ``p[t] / p[t-1] - 1``."""

        return tuple(current / previous - 1.0 for previous, current in pairwise(self.prices))

    def log_returns(self) -> tuple[float, ...]:
        """Return the log returns ``ln(p[t] / p[t-1])``."""

        return tuple(math.log(current / previous) for previous, current in pairwise(self.prices))


def align(series: Sequence[PriceSeries]) -> tuple[PriceSeries, ...]:
    """Restrict every series to the dates they all share.

    The policy is intersection only: a date survives only if every input
    series has a price on it, and no price is ever invented for a missing
    date. The result preserves the input order of the series, and every
    output series carries the identical date grid.
    """

    if len(series) < MIN_ALIGN_SERIES:
        raise ValueError("alignment needs at least two series")
    names = [item.name for item in series]
    if len(set(names)) != len(names):
        raise ValueError(f"series names must be unique, got {names}")
    common = set(series[0].dates)
    for item in series[1:]:
        common &= set(item.dates)
    if len(common) < MIN_PRICES:
        raise ValueError(
            f"series share only {len(common)} common dates; at least {MIN_PRICES} are required"
        )
    kept = sorted(common)
    aligned = []
    for item in series:
        by_date = dict(zip(item.dates, item.prices, strict=True))
        aligned.append(
            PriceSeries(
                name=item.name,
                dates=tuple(kept),
                prices=tuple(by_date[day] for day in kept),
            )
        )
    return tuple(aligned)
