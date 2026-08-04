"""Validated daily price series and their return transforms.

A :class:`PriceSeries` is the only way price data enters the engine. Its
constructor rejects anything that would silently poison a risk number:
malformed, unsorted, or duplicate dates, non-positive or non-finite
prices, and series too short to produce a sample statistic.

The engine's calendar is the session model of ``docs/methodology.md``:
each row is one trading session, consecutive rows are consecutive
sessions, and a calendar gap between successive dates — a weekend, a
holiday, a halt — carries no information. No return accrues across a
gap, the 252 annualization counts sessions rather than calendar days,
and a date on a Saturday or Sunday is as legal as any other, because
some markets trade every day and the model never reads the weekday.
Validation therefore enforces exactly what the model needs and nothing
more: canonical ISO dates, strictly increasing — and never rejects a
gap, because under the model gaps are the norm.

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

from quantrisk._validation import require_finite_number

#: A series must produce at least two returns, so a sample standard
#: deviation (denominator n - 1) is defined. Two returns need three prices.
MIN_PRICES = 3

#: Aligning fewer than two series is a sign of a caller bug.
MIN_ALIGN_SERIES = 2


def _parse_iso_date(text: str, owner: str) -> date:
    """Return the calendar date for an ISO ``YYYY-MM-DD`` string.

    ``owner`` names the object the date belongs to (``"series 'AAA'"``,
    ``"stress window 'gfc'"``), so every rejection identifies both the
    holder and the offending value. Only the canonical zero-padded form
    passes: real parsing catches impossible dates like ``2026-13-01``,
    and the canonical-form check catches spellings like ``2026-1-5``
    that would break the engine's reliance on lexicographic date order
    equaling chronological order.
    """

    try:
        parsed = date.fromisoformat(text)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{owner} date {text!r} is not an ISO YYYY-MM-DD date") from error
    if text != parsed.isoformat():
        raise ValueError(f"{owner} date {text!r} is not in canonical YYYY-MM-DD form")
    return parsed


@dataclass(frozen=True, slots=True)
class PriceSeries:
    """One security's closing prices, one row per trading session.

    Dates are canonical ISO ``YYYY-MM-DD`` strings, strictly
    increasing. Under the session model (``docs/methodology.md``) they
    are labels for sessions, not quantities: the calendar gap between
    two successive rows carries no information, returns compound
    session over session regardless of it, and a weekend date is legal
    because the model never reads the weekday.
    """

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
        parsed = [_parse_iso_date(text, f"series {self.name!r}") for text in self.dates]
        for previous, current in pairwise(parsed):
            if current <= previous:
                raise ValueError(
                    f"series {self.name!r} dates must be strictly increasing; "
                    f"{current.isoformat()} follows {previous.isoformat()}"
                )
        for text, price in zip(self.dates, self.prices, strict=True):
            number = require_finite_number(price, f"series {self.name!r} price on {text}")
            if number <= 0:
                raise ValueError(
                    f"series {self.name!r} price on {text} is {number!r}; prices must be positive"
                )

    def __len__(self) -> int:
        return len(self.prices)

    def simple_returns(self) -> tuple[float, ...]:
        """Return the simple returns ``p[t] / p[t-1] - 1``.

        One return per pair of consecutive sessions, whatever the
        calendar distance between their dates: a Friday-to-Monday pair
        yields exactly one session return with no weekend accrual.
        """

        return tuple(current / previous - 1.0 for previous, current in pairwise(self.prices))

    def log_returns(self) -> tuple[float, ...]:
        """Return the log returns ``ln(p[t] / p[t-1])``.

        Session over session, like :meth:`simple_returns`; the
        calendar gap between two rows never enters the arithmetic.
        """

        return tuple(math.log(current / previous) for previous, current in pairwise(self.prices))


def align(series: Sequence[PriceSeries]) -> tuple[PriceSeries, ...]:
    """Restrict every series to the dates they all share.

    The policy is intersection only: a date survives only if every input
    series has a price on it, and no price is ever invented for a missing
    date. The result preserves the input order of the series, and every
    output series carries the identical date grid. The shared grid is a
    new session sequence under the session model: dates dropped by the
    intersection become ordinary gaps, and the returns that remain
    compound across them like any other session pair.
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
