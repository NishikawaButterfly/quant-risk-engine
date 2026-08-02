from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import ModuleType

from quantrisk.series import PriceSeries, align

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = _REPOSITORY_ROOT / "sample-data" / "prices.csv"


def _load_generator() -> ModuleType:
    """Import the fixture generator script, which lives outside the package."""

    path = _REPOSITORY_ROOT / "tools" / "make_synthetic_prices.py"
    spec = importlib.util.spec_from_file_location("make_synthetic_prices", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture_series() -> list[PriceSeries]:
    lines = _FIXTURE.read_text(encoding="utf-8").splitlines()
    names = lines[0].split(",")[1:]
    dates = []
    columns: list[list[float]] = [[] for _ in names]
    for line in lines[1:]:
        cells = line.split(",")
        dates.append(cells[0])
        for column, cell in zip(columns, cells[1:], strict=True):
            column.append(float(cell))
    return [
        PriceSeries(name=name, dates=tuple(dates), prices=tuple(column))
        for name, column in zip(names, columns, strict=True)
    ]


class SampleDataTests(unittest.TestCase):
    def test_committed_csv_matches_the_seeded_generator(self) -> None:
        generator = _load_generator()
        expected = generator.render_csv(generator.synthetic_prices(2026))
        committed = _FIXTURE.read_text(encoding="utf-8")
        self.assertEqual(committed, expected)

    def test_fixture_has_the_documented_shape(self) -> None:
        lines = _FIXTURE.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0], "Date,AAA,BBB,CCC,DDD")
        self.assertEqual(len(lines), 1 + 261)  # 261 weekdays in 2025

    def test_every_column_is_a_valid_price_series(self) -> None:
        series = _fixture_series()
        self.assertEqual([item.name for item in series], ["AAA", "BBB", "CCC", "DDD"])
        for item in series:
            self.assertEqual(len(item), 261)
        # The columns share one date grid, so alignment changes nothing.
        aligned = align(series)
        for original, after in zip(series, aligned, strict=True):
            self.assertEqual(original, after)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
