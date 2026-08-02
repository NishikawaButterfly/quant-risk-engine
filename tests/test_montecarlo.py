from __future__ import annotations

import math
import unittest
from collections.abc import Sequence

from quantrisk.montecarlo import (
    MAX_HORIZON_DAYS,
    MAX_RUNS,
    MIN_RUNS,
    MonteCarloResult,
    run_bootstrap,
    run_parametric_normal,
)
from quantrisk.portfolio import Portfolio
from quantrisk.series import PriceSeries

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


SERIES_A = series_from_returns("AAA", 100.0, RETURNS_A)
SERIES_B = series_from_returns("BBB", 50.0, RETURNS_B)
SERIES = (SERIES_A, SERIES_B)
PORTFOLIO = Portfolio(names=("AAA", "BBB"), weights=(0.6, 0.4))

# Every daily return exactly +100%: prices double each day, so a
# bootstrap draw can only ever pick the value 1.0.
DOUBLING_A = PriceSeries(name="AAA", dates=DATES[:4], prices=(100.0, 200.0, 400.0, 800.0))
DOUBLING_B = PriceSeries(name="BBB", dates=DATES[:4], prices=(50.0, 100.0, 200.0, 400.0))


def interpolated_percentile(sorted_values: tuple[float, ...], level: float) -> float:
    """The documented convention, recomputed independently of the engine."""

    rank = level / 100.0 * (len(sorted_values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    fraction = rank - lower
    return sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])


class DeterminismTests(unittest.TestCase):
    def assert_identical(self, first: MonteCarloResult, second: MonteCarloResult) -> None:
        self.assertEqual(first.terminal_percentiles, second.terminal_percentiles)
        self.assertEqual(first.terminal_values, second.terminal_values)
        self.assertEqual(first.terminal_mean, second.terminal_mean)
        self.assertEqual(first.terminal_stddev, second.terminal_stddev)
        self.assertEqual(first.probability_below_initial, second.probability_below_initial)

    def test_bootstrap_same_seed_reproduces_every_number(self) -> None:
        first = run_bootstrap(PORTFOLIO, SERIES, horizon_days=63, runs=500, seed=2026)
        second = run_bootstrap(PORTFOLIO, SERIES, horizon_days=63, runs=500, seed=2026)
        self.assert_identical(first, second)

    def test_parametric_same_seed_reproduces_every_number(self) -> None:
        first = run_parametric_normal(PORTFOLIO, SERIES, horizon_days=63, runs=500, seed=2026)
        second = run_parametric_normal(PORTFOLIO, SERIES, horizon_days=63, runs=500, seed=2026)
        self.assert_identical(first, second)

    def test_different_seeds_differ(self) -> None:
        for run in (run_bootstrap, run_parametric_normal):
            with self.subTest(mode=run.__name__):
                first = run(PORTFOLIO, SERIES, horizon_days=63, runs=500, seed=1)
                second = run(PORTFOLIO, SERIES, horizon_days=63, runs=500, seed=2)
                self.assertNotEqual(first.terminal_values, second.terminal_values)


class DegenerateCollapseTests(unittest.TestCase):
    def test_bootstrap_collapses_when_every_return_is_identical(self) -> None:
        # Every historical return is exactly +100%, so each of the ten
        # compounded factors is exactly 2.0 and every run must land on
        # 2**10 = 1024 exactly — no tolerance.
        result = run_bootstrap(
            Portfolio(names=("AAA", "BBB"), weights=(0.5, 0.5)),
            (DOUBLING_A, DOUBLING_B),
            horizon_days=10,
            runs=100,
            seed=7,
        )
        self.assertEqual(result.terminal_values, ((1.0 + 1.0) ** 10,) * 100)
        self.assertEqual(result.terminal_percentiles.p5, 1024.0)
        self.assertEqual(result.terminal_percentiles.p50, 1024.0)
        self.assertEqual(result.terminal_percentiles.p95, 1024.0)
        self.assertEqual(result.terminal_mean, 1024.0)
        self.assertEqual(result.terminal_stddev, 0.0)
        self.assertEqual(result.probability_below_initial, 0.0)

    def test_parametric_collapses_when_the_stddev_is_zero(self) -> None:
        # Identical returns fit N(1.0, 0.0); every draw is the mean.
        result = run_parametric_normal(
            Portfolio(names=("AAA", "BBB"), weights=(0.5, 0.5)),
            (DOUBLING_A, DOUBLING_B),
            horizon_days=10,
            runs=100,
            seed=7,
        )
        self.assertEqual(result.terminal_values, (1024.0,) * 100)
        self.assertEqual(result.terminal_stddev, 0.0)


class EnvelopeTests(unittest.TestCase):
    def test_bootstrap_terminal_values_stay_inside_the_compounding_envelope(self) -> None:
        horizon = 63
        result = run_bootstrap(PORTFOLIO, SERIES, horizon_days=horizon, runs=2000, seed=11)
        daily = PORTFOLIO.return_series(SERIES)
        lower = (1.0 + min(daily)) ** horizon
        upper = (1.0 + max(daily)) ** horizon
        self.assertGreaterEqual(result.terminal_values[0], lower)
        self.assertLessEqual(result.terminal_values[-1], upper)
        self.assertLess(result.terminal_values[0], result.terminal_values[-1])


class SummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.result = run_bootstrap(PORTFOLIO, SERIES, horizon_days=21, runs=400, seed=3)

    def test_result_echoes_its_inputs(self) -> None:
        self.assertEqual(self.result.mode, "bootstrap")
        self.assertEqual(self.result.seed, 3)
        self.assertEqual(self.result.runs, 400)
        self.assertEqual(self.result.horizon_days, 21)
        parametric = run_parametric_normal(PORTFOLIO, SERIES, horizon_days=21, runs=400, seed=3)
        self.assertEqual(parametric.mode, "parametric_normal")

    def test_terminal_values_are_complete_and_sorted(self) -> None:
        values = self.result.terminal_values
        self.assertEqual(len(values), 400)
        self.assertEqual(values, tuple(sorted(values)))

    def test_summary_figures_recompute_from_the_retained_array(self) -> None:
        values = self.result.terminal_values
        mean = sum(values) / len(values)
        self.assertAlmostEqual(self.result.terminal_mean, mean, places=12)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        self.assertAlmostEqual(self.result.terminal_stddev, math.sqrt(variance), places=12)
        below = sum(value < 1.0 for value in values) / len(values)
        self.assertEqual(self.result.probability_below_initial, below)

    def test_percentiles_follow_the_documented_interpolation(self) -> None:
        values = self.result.terminal_values
        reported = self.result.terminal_percentiles
        for level, value in (
            (5.0, reported.p5),
            (25.0, reported.p25),
            (50.0, reported.p50),
            (75.0, reported.p75),
            (95.0, reported.p95),
        ):
            with self.subTest(level=level):
                self.assertAlmostEqual(value, interpolated_percentile(values, level), places=12)
        self.assertLessEqual(reported.p5, reported.p25)
        self.assertLessEqual(reported.p25, reported.p50)
        self.assertLessEqual(reported.p50, reported.p75)
        self.assertLessEqual(reported.p75, reported.p95)


class ValidationTests(unittest.TestCase):
    def run_both(self, **overrides: int) -> None:
        arguments = {"horizon_days": 21, "runs": 200, "seed": 1}
        arguments.update(overrides)
        for run in (run_bootstrap, run_parametric_normal):
            with self.subTest(mode=run.__name__):
                with self.assertRaises(ValueError):
                    run(PORTFOLIO, SERIES, **arguments)

    def test_runs_outside_the_bounds_are_rejected(self) -> None:
        self.run_both(runs=MIN_RUNS - 1)
        self.run_both(runs=MAX_RUNS + 1)

    def test_non_integer_runs_are_rejected(self) -> None:
        self.run_both(runs=200.0)  # type: ignore[arg-type]
        self.run_both(runs=True)

    def test_seed_must_be_a_real_integer(self) -> None:
        self.run_both(seed=1.5)  # type: ignore[arg-type]
        self.run_both(seed=True)

    def test_horizon_outside_the_bounds_is_rejected(self) -> None:
        self.run_both(horizon_days=0)
        self.run_both(horizon_days=MAX_HORIZON_DAYS + 1)
        self.run_both(horizon_days=21.0)  # type: ignore[arg-type]
        self.run_both(horizon_days=True)

    def test_boundary_run_counts_are_accepted(self) -> None:
        result = run_bootstrap(PORTFOLIO, SERIES, horizon_days=1, runs=MIN_RUNS, seed=1)
        self.assertEqual(result.runs, MIN_RUNS)

    def test_portfolio_validation_still_applies(self) -> None:
        stranger = Portfolio(names=("AAA", "ZZZ"), weights=(0.6, 0.4))
        with self.assertRaises(ValueError):
            run_bootstrap(stranger, SERIES, horizon_days=21, runs=200, seed=1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
