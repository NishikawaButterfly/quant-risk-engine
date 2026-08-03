from __future__ import annotations

import unittest
from collections.abc import Sequence

from quantrisk.portfolio import Portfolio
from quantrisk.series import PriceSeries
from quantrisk.stress import (
    MIN_WINDOW_OBSERVATIONS,
    StressWindow,
    historical_stress,
    shock_stress,
)

DATES = (
    "2026-01-05",
    "2026-01-06",
    "2026-01-07",
    "2026-01-08",
    "2026-01-09",
    "2026-01-12",
    "2026-01-13",
)

# The hand-worked two-asset fixture from docs/methodology.md.
RETURNS_A = (0.02, -0.01, -0.02, 0.02, 0.03, 0.02)
RETURNS_B = (0.03, 0.03, -0.03, 0.035, -0.01, -0.025)


def series_from_returns(name: str, start: float, returns: Sequence[float]) -> PriceSeries:
    prices = [start]
    for value in returns:
        prices.append(prices[-1] * (1.0 + value))
    return PriceSeries(name=name, dates=DATES, prices=tuple(prices))


SERIES = (
    series_from_returns("AAA", 100.0, RETURNS_A),
    series_from_returns("BBB", 50.0, RETURNS_B),
)
PORTFOLIO = Portfolio(names=("AAA", "BBB"), weights=(0.6, 0.4))

# The window worked by hand in docs/methodology.md: five observations,
# 2026-01-06 through 2026-01-12, holding the portfolio returns
# (0.006, -0.024, 0.026, 0.014).
HAND_WINDOW = StressWindow(name="hand-worked", start="2026-01-06", end="2026-01-12")


class WindowValidationTests(unittest.TestCase):
    def test_reversed_window_is_rejected_at_construction(self) -> None:
        with self.assertRaisesRegex(ValueError, "reversed"):
            StressWindow(name="backwards", start="2026-01-09", end="2026-01-06")

    def test_non_iso_dates_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            StressWindow(name="sloppy", start="2026-1-6", end="2026-01-09")
        with self.assertRaises(ValueError):
            StressWindow(name="sloppy", start="06/01/2026", end="2026-01-09")

    def test_blank_name_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            StressWindow(name="  ", start="2026-01-06", end="2026-01-09")

    def test_window_outside_the_grid_is_rejected(self) -> None:
        early = StressWindow(name="early", start="2025-12-29", end="2026-01-09")
        with self.assertRaisesRegex(ValueError, "outside the aligned grid"):
            historical_stress(PORTFOLIO, SERIES, [early])
        late = StressWindow(name="late", start="2026-01-06", end="2026-02-27")
        with self.assertRaisesRegex(ValueError, "outside the aligned grid"):
            historical_stress(PORTFOLIO, SERIES, [late])

    def test_window_with_too_few_observations_is_rejected(self) -> None:
        short = StressWindow(name="short", start="2026-01-06", end="2026-01-09")
        with self.assertRaisesRegex(ValueError, "4 observations"):
            historical_stress(PORTFOLIO, SERIES, [short])

    def test_no_windows_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            historical_stress(PORTFOLIO, SERIES, [])

    def test_duplicate_window_names_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            historical_stress(PORTFOLIO, SERIES, [HAND_WINDOW, HAND_WINDOW])

    def test_minimum_window_observations_is_five(self) -> None:
        self.assertEqual(MIN_WINDOW_OBSERVATIONS, 5)


class HistoricalStressTests(unittest.TestCase):
    def test_the_hand_worked_window_digit_for_digit(self) -> None:
        (result,) = historical_stress(PORTFOLIO, SERIES, [HAND_WINDOW])
        self.assertEqual(result.window, HAND_WINDOW)
        self.assertEqual(result.observations, 5)
        # (1.006)(0.976)(1.026)(1.014) - 1, worked in docs/methodology.md.
        self.assertAlmostEqual(result.total_return, 0.021487635584, places=12)
        # sqrt(0.001363 / 3 * 252) = sqrt(0.114492).
        self.assertAlmostEqual(result.annualized_volatility, 0.338366665025, places=12)
        # 1 - 0.981856 / 1.006 = 1 - 0.976 = 0.024 exactly (in decimal).
        self.assertAlmostEqual(result.max_drawdown.depth, 0.024, places=12)
        self.assertEqual(result.max_drawdown.peak_date, "2026-01-07")
        self.assertEqual(result.max_drawdown.trough_date, "2026-01-08")

    def test_the_full_grid_window_replays_the_whole_series(self) -> None:
        window = StressWindow(name="everything", start="2026-01-05", end="2026-01-13")
        (result,) = historical_stress(PORTFOLIO, SERIES, [window])
        self.assertEqual(result.observations, len(DATES))
        expected = 1.0
        for value in PORTFOLIO.return_series(SERIES):
            expected *= 1.0 + value
        self.assertAlmostEqual(result.total_return, expected - 1.0, places=12)

    def test_window_bounds_need_not_be_trading_days(self) -> None:
        # 2026-01-10 is a Saturday off the grid; a window ending inside
        # the weekend selects exactly the same dates as one ending on
        # the preceding Friday.
        tight = StressWindow(name="tight", start="2026-01-05", end="2026-01-09")
        loose = StressWindow(name="loose", start="2026-01-05", end="2026-01-10")
        (first,) = historical_stress(PORTFOLIO, SERIES, [tight])
        (second,) = historical_stress(PORTFOLIO, SERIES, [loose])
        self.assertEqual(first.observations, second.observations)
        self.assertEqual(first.total_return, second.total_return)

    def test_results_preserve_window_order(self) -> None:
        other = StressWindow(name="tail", start="2026-01-07", end="2026-01-13")
        results = historical_stress(PORTFOLIO, SERIES, [other, HAND_WINDOW])
        self.assertEqual([item.window.name for item in results], ["tail", "hand-worked"])

    def test_short_positions_that_wipe_out_the_window_path_are_rejected(self) -> None:
        # A 2x long / 1x short portfolio against an asset that rises 150%
        # in one day prints a -230% daily return; the compounded path
        # inside the window turns negative and is rejected, not reported.
        flat = PriceSeries(
            name="AAA",
            dates=DATES[:5],
            prices=(100.0, 100.0, 100.0, 100.0, 100.0),
        )
        rocket = PriceSeries(
            name="BBB",
            dates=DATES[:5],
            prices=(100.0, 100.0, 250.0, 250.0, 250.0),
        )
        shorted = Portfolio(names=("AAA", "BBB"), weights=(2.0, -1.0))
        window = StressWindow(name="wipeout", start="2026-01-05", end="2026-01-09")
        with self.assertRaisesRegex(ValueError, "zero or below"):
            historical_stress(shorted, (flat, rocket), [window])

    def test_portfolio_validation_still_applies(self) -> None:
        stranger = Portfolio(names=("AAA", "ZZZ"), weights=(0.6, 0.4))
        with self.assertRaises(ValueError):
            historical_stress(stranger, SERIES, [HAND_WINDOW])


class ShockValidationTests(unittest.TestCase):
    def test_no_shocks_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            shock_stress(PORTFOLIO, {})

    def test_missing_asset_is_rejected_not_defaulted(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing a value for \\['BBB'\\]"):
            shock_stress(PORTFOLIO, {"half": {"AAA": -0.10}})

    def test_unknown_asset_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not hold"):
            shock_stress(PORTFOLIO, {"stray": {"AAA": -0.10, "BBB": -0.10, "ZZZ": -0.10}})

    def test_shock_at_or_below_minus_one_is_rejected(self) -> None:
        for value in (-1.0, -1.5):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "must exceed -1"):
                    shock_stress(PORTFOLIO, {"abyss": {"AAA": value, "BBB": 0.0}})

    def test_non_numbers_are_rejected(self) -> None:
        values: tuple[object, ...] = ("-0.2", True, None, float("nan"), float("inf"))
        for value in values:
            with self.subTest(value=value):
                vector = {"bad": {"AAA": value, "BBB": 0.0}}
                with self.assertRaises(ValueError):
                    shock_stress(PORTFOLIO, vector)  # type: ignore[arg-type]

    def test_blank_shock_name_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            shock_stress(PORTFOLIO, {" ": {"AAA": -0.1, "BBB": -0.1}})


class ShockStressTests(unittest.TestCase):
    def test_the_hand_worked_shock_is_exact(self) -> None:
        # 0.6 x (-0.20) + 0.4 x (-0.05) = -0.12 - 0.02 = -0.14, exactly,
        # as worked in docs/methodology.md.
        (result,) = shock_stress(PORTFOLIO, {"hand-worked": {"AAA": -0.20, "BBB": -0.05}})
        self.assertEqual(result.portfolio_return, -0.14)
        self.assertEqual(result.asset_shocks, (-0.20, -0.05))
        self.assertEqual(result.name, "hand-worked")

    def test_an_all_zero_shock_returns_exactly_zero(self) -> None:
        (result,) = shock_stress(PORTFOLIO, {"calm": {"AAA": 0.0, "BBB": 0.0}})
        self.assertEqual(result.portfolio_return, 0.0)

    def test_asset_shocks_come_back_in_portfolio_order(self) -> None:
        # The vector may be written in any order; the echo follows the
        # portfolio's asset order.
        (result,) = shock_stress(PORTFOLIO, {"swapped": {"BBB": -0.05, "AAA": -0.20}})
        self.assertEqual(result.asset_shocks, (-0.20, -0.05))
        self.assertEqual(result.portfolio_return, -0.14)

    def test_results_preserve_shock_order(self) -> None:
        results = shock_stress(
            PORTFOLIO,
            {
                "second-listed": {"AAA": 0.01, "BBB": 0.01},
                "first-listed": {"AAA": -0.01, "BBB": -0.01},
            },
        )
        self.assertEqual([item.name for item in results], ["second-listed", "first-listed"])

    def test_a_short_position_flips_the_shock_sign(self) -> None:
        shorted = Portfolio(names=("AAA", "BBB"), weights=(1.5, -0.5))
        (result,) = shock_stress(shorted, {"crash": {"AAA": 0.0, "BBB": -0.10}})
        self.assertAlmostEqual(result.portfolio_return, 0.05, places=15)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
