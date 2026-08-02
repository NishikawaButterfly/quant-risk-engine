"""Deterministic generator for the synthetic daily price fixture.

The generator invents one seeded trading year of daily closes for four
fictional tickers, so the committed CSV fixture can be reproduced byte
for byte:

    python tools/make_synthetic_prices.py --seed 2026 --output sample-data/prices.csv

Each ticker follows a geometric-brownian-ish walk with its own invented
drift and volatility — AAA is a calm large cap, BBB a choppier mid cap,
CCC a volatile growth name, DDD a sleepy defensive. No real market data
is used anywhere; every constant is a round, invented number and the
only randomness is the explicitly seeded generator.
"""

from __future__ import annotations

import argparse
import math
import random
from datetime import date, timedelta
from pathlib import Path

YEAR = 2025
TRADING_DAYS_PER_YEAR = 252
CSV_HEADER = "Date,AAA,BBB,CCC,DDD"

#: (ticker, initial price, annual drift, annual volatility) — all invented.
TICKERS = (
    ("AAA", 100.0, 0.08, 0.15),
    ("BBB", 50.0, 0.05, 0.25),
    ("CCC", 200.0, 0.12, 0.35),
    ("DDD", 80.0, 0.02, 0.10),
)


def weekdays(year: int) -> list[date]:
    """Return every Monday-to-Friday date of the year, in order."""

    day = date(year, 1, 1)
    days = []
    while day.year == year:
        if day.isoweekday() <= 5:  # noqa: PLR2004 - ISO weekdays 1..5 are Mon..Fri
            days.append(day)
        day += timedelta(days=1)
    return days


def synthetic_prices(seed: int) -> list[tuple[date, tuple[float, ...]]]:
    """Return one weekday year of (date, prices-per-ticker) rows.

    Prices follow the discretized geometric walk
    ``p[t] = p[t-1] * exp((mu - sigma^2 / 2) * dt + sigma * sqrt(dt) * z)``
    with ``dt = 1/252`` and one standard-normal draw per ticker per day,
    drawn in ticker order from a single seeded generator.
    """

    rng = random.Random(seed)  # noqa: S311 - reproducible fixture, not cryptography
    delta_t = 1.0 / TRADING_DAYS_PER_YEAR
    levels = [initial for _, initial, _, _ in TICKERS]
    rows: list[tuple[date, tuple[float, ...]]] = []
    for day in weekdays(YEAR):
        for index, (_, _, drift, volatility) in enumerate(TICKERS):
            shock = rng.gauss(0.0, 1.0)
            growth = (drift - volatility**2 / 2.0) * delta_t
            diffusion = volatility * math.sqrt(delta_t) * shock
            levels[index] *= math.exp(growth + diffusion)
        rows.append((day, tuple(levels)))
    return rows


def render_csv(rows: list[tuple[date, tuple[float, ...]]]) -> str:
    """Render rows as the committed CSV text, prices at two decimals."""

    lines = [CSV_HEADER]
    for day, prices in rows:
        cells = ",".join(f"{price:.2f}" for price in prices)
        lines.append(f"{day.isoformat()},{cells}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("sample-data/prices.csv"))
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        render_csv(synthetic_prices(arguments.seed)), encoding="utf-8", newline="\n"
    )
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
