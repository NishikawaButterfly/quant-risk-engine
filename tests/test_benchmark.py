from __future__ import annotations

import math
import unittest
from collections.abc import Sequence

from quantrisk.benchmark import (
    MIN_COMPARISON_RETURNS,
    BenchmarkComparison,
    compare_to_benchmark,
)
from quantrisk.portfolio import Portfolio
from quantrisk.series import PriceSeries, align

# Seven consecutive weekdays: the full grid the portfolio's assets trade on.
GRID = (
    "2026-01-05",
    "2026-01-06",
    "2026-01-07",
    "2026-01-08",
    "2026-01-09",
    "2026-01-12",
    "2026-01-13",
)

# The hand-worked six-pair fixture from docs/methodology.md, as *returns*:
# beta 3/2, daily alpha 0.005, tracking error sqrt(0.0504), information
# ratio sqrt(126), up capture 5/3 over four rising days, and down capture
# exactly 1 over exactly two falling days.
PORTFOLIO_RETURNS = (0.04, 0.00, 0.06, -0.03, 0.03, 0.02)
BENCHMARK_RETURNS = (0.02, -0.01, 0.03, -0.02, 0.02, 0.02)


def _prices(returns: Sequence[float], start: float = 64.0) -> tuple[float, ...]:
    """Compound a price path whose simple returns are ``returns``.

    For dyadic returns (0.25, -0.125, ...) every product below is exact in
    binary floating point and ``simple_returns`` recovers each return to
    the last bit. For decimal returns like 0.04 the compounding carries
    sub-ULP rounding, so the derived returns sit within one ulp of the
    literals; the hand-table tests state the tolerance they assert.
    """

    values = [start]
    for value in returns:
        values.append(values[-1] * (1.0 + value))
    return tuple(values)


def _twin_portfolio(
    prices: Sequence[float], dates: Sequence[str] = GRID
) -> tuple[Portfolio, tuple[PriceSeries, PriceSeries]]:
    """A half-and-half portfolio of two sleeves holding the same prices.

    With weights (0.5, 0.5) over identical sleeves the weighted daily
    return is ``0.5 r + 0.5 r``: both halves are exact in binary floating
    point, so the portfolio's return series equals the sleeve's own
    returns bit for bit and the fixtures below stay hand-checkable.
    """

    sleeves = (
        PriceSeries(name="P1", dates=tuple(dates), prices=tuple(prices)),
        PriceSeries(name="P2", dates=tuple(dates), prices=tuple(prices)),
    )
    return Portfolio(names=("P1", "P2"), weights=(0.5, 0.5)), sleeves


def _benchmark(prices: Sequence[float], dates: Sequence[str] = GRID) -> PriceSeries:
    return PriceSeries(name="NNN", dates=tuple(dates), prices=tuple(prices))


def _compare(
    portfolio_returns: Sequence[float],
    benchmark_returns: Sequence[float],
    *,
    risk_free_rate_daily: float = 0.0,
) -> BenchmarkComparison:
    """Compare two return streams laid over the shared seven-day grid."""

    portfolio, sleeves = _twin_portfolio(_prices(portfolio_returns, start=100.0))
    benchmark = _benchmark(_prices(benchmark_returns, start=100.0))
    return compare_to_benchmark(
        portfolio, sleeves, benchmark, risk_free_rate_daily=risk_free_rate_daily
    )


class HandTableTests(unittest.TestCase):
    """The methodology fixture, rebuilt from dated prices.

    The returns derive from compounded price paths, so each one sits
    within an ulp of the documented literal; every digit the methodology
    table displays (six decimals) is still asserted exactly, and the
    absolute drift of each statistic is bounded by the places stated.
    """

    def test_beta_matches_the_hand_table(self) -> None:
        # cov = 0.003 / 5 = 0.0006, var(b) = 0.002 / 5 = 0.0004.
        result = _compare(PORTFOLIO_RETURNS, BENCHMARK_RETURNS)
        self.assertAlmostEqual(result.beta, 1.5, places=15)

    def test_alpha_matches_the_hand_table(self) -> None:
        # Daily: 0.02 - 1.5 * 0.01 = 0.005. Annualized arithmetically:
        # 0.005 * 252 = 1.26.
        result = _compare(PORTFOLIO_RETURNS, BENCHMARK_RETURNS)
        self.assertAlmostEqual(result.alpha_daily, 0.005, places=15)
        self.assertAlmostEqual(result.alpha_annualized, 1.26, places=12)
        self.assertEqual(round(result.alpha_annualized, 6), 1.26)

    def test_tracking_error_matches_the_hand_table(self) -> None:
        # Active returns (0.02, 0.01, 0.03, -0.01, 0.01, 0.00): sample
        # variance 0.001 / 5 = 0.0002, annualized sqrt(0.0504).
        result = _compare(PORTFOLIO_RETURNS, BENCHMARK_RETURNS)
        self.assertAlmostEqual(result.tracking_error, math.sqrt(0.0504), places=15)
        self.assertEqual(round(result.tracking_error, 6), 0.224499)

    def test_information_ratio_matches_the_hand_table(self) -> None:
        # 0.01 * 252 / sqrt(0.0504) collapses to sqrt(126).
        result = _compare(PORTFOLIO_RETURNS, BENCHMARK_RETURNS)
        assert result.information_ratio is not None
        self.assertAlmostEqual(result.information_ratio, math.sqrt(126), places=12)
        self.assertEqual(round(result.information_ratio, 6), 11.224972)

    def test_capture_ratios_match_the_hand_table(self) -> None:
        # Up: (0.15 / 4) / (0.09 / 4) = 5/3 over four rising days.
        # Down: (-0.03 / 2) / (-0.03 / 2) = 1 over exactly two days.
        result = _compare(PORTFOLIO_RETURNS, BENCHMARK_RETURNS)
        assert result.up_capture is not None
        assert result.down_capture is not None
        self.assertAlmostEqual(result.up_capture, 5.0 / 3.0, places=15)
        self.assertEqual(round(result.up_capture, 6), 1.666667)
        self.assertAlmostEqual(result.down_capture, 1.0, places=15)

    def test_risk_free_rate_moves_alpha_by_beta_minus_one(self) -> None:
        # alpha(rf) = (mean_p - rf) - beta * (mean_b - rf), so a daily
        # rate shifts the daily alpha by (beta - 1) * rf and leaves
        # beta itself untouched: constants drop out of covariances.
        base = _compare(PORTFOLIO_RETURNS, BENCHMARK_RETURNS)
        shifted = _compare(PORTFOLIO_RETURNS, BENCHMARK_RETURNS, risk_free_rate_daily=0.0001)
        self.assertEqual(shifted.beta, base.beta)
        self.assertAlmostEqual(shifted.alpha_daily - base.alpha_daily, 0.00005, places=15)


class DateGridTests(unittest.TestCase):
    """Misaligned inputs are refused; the aligned intersection is exact.

    The fixture: two assets on all seven grid dates and a benchmark that
    skips 2026-01-07 and 2026-01-09. Everything is dyadic, so every price,
    return, and hand-derived statistic below is exact in binary floating
    point.
    """

    ASSET_A = PriceSeries(
        name="AAA", dates=GRID, prices=(64.0, 80.0, 88.0, 60.0, 66.0, 90.0, 67.5)
    )
    ASSET_B = PriceSeries(
        name="BBB", dates=GRID, prices=(128.0, 160.0, 150.0, 200.0, 190.0, 200.0, 150.0)
    )
    PORTFOLIO = Portfolio(names=("AAA", "BBB"), weights=(0.5, 0.5))
    BENCHMARK = PriceSeries(
        name="NNN",
        dates=("2026-01-05", "2026-01-06", "2026-01-08", "2026-01-12", "2026-01-13"),
        prices=(64.0, 72.0, 63.0, 70.875, 62.015625),
    )

    def test_a_benchmark_on_its_own_grid_is_rejected(self) -> None:
        # Five benchmark dates against seven asset dates. The raw-returns
        # API accepted the truncation pairing (first four portfolio
        # returns against the benchmark's four) and reported beta
        # 0.4276515151515149 — quiet nonsense; the true intersection beta
        # is 1.5, asserted below. Now the grids are checked and the
        # mismatch is loud.
        with self.assertRaisesRegex(ValueError, r"run align\(\) first"):
            compare_to_benchmark(self.PORTFOLIO, (self.ASSET_A, self.ASSET_B), self.BENCHMARK)

    def test_equal_length_offset_grids_are_rejected(self) -> None:
        # Seven benchmark dates, but two differ from the asset grid, so
        # the return counts match (six against six) while the dates do
        # not. The raw-returns API accepted exactly this pair and
        # reported beta 0.2799199308454483 belonging to no date grid at
        # all; equal lengths prove nothing.
        shifted = PriceSeries(
            name="NNN",
            dates=(
                "2026-01-05",
                "2026-01-06",
                "2026-01-07",
                "2026-01-09",
                "2026-01-12",
                "2026-01-13",
                "2026-01-14",
            ),
            prices=(64.0, 72.0, 63.0, 70.875, 62.015625, 66.0, 70.125),
        )
        with self.assertRaisesRegex(ValueError, r"run align\(\) first"):
            compare_to_benchmark(self.PORTFOLIO, (self.ASSET_A, self.ASSET_B), shifted)

    def test_the_aligned_intersection_lands_on_the_hand_derived_statistics(self) -> None:
        # align() keeps the five shared dates; the two dates only the
        # assets carry are dropped, and the returns spanning the gaps
        # compound across them. On the intersection (n = 4 paired
        # returns; every value dyadic, so exact):
        #
        #   date pair       AAA    BBB    portfolio  NNN
        #   01-05 -> 01-06  0.25   0.25   0.25       0.125
        #   01-06 -> 01-08 -0.25   0.25   0.0       -0.125
        #   01-08 -> 01-12  0.5    0.0    0.25       0.125
        #   01-12 -> 01-13 -0.25  -0.25  -0.25      -0.125
        #
        # means: p 0.0625, b 0. devs p (0.1875, -0.0625, 0.1875, -0.3125),
        # devs b (0.125, -0.125, 0.125, -0.125). Sum of cross products
        # 0.09375, sum of b squares 0.0625; the n - 1 denominators cancel:
        # beta = 0.09375 / 0.0625 = 1.5. alpha = 0.0625 - 1.5 * 0 = 0.0625.
        # Active returns (0.125, 0.125, 0.125, -0.125): sample variance
        # 0.046875 / 3 = 0.015625, stddev 0.125, TE = 0.125 * sqrt(252);
        # IR = (0.0625 * 252) / TE = sqrt(63). The benchmark rose on the
        # first and third pairs (up = 0.25 / 0.125 = 2) and fell on the
        # other two (down = -0.125 / -0.125 = 1).
        aligned_a, aligned_b, aligned_n = align((self.ASSET_A, self.ASSET_B, self.BENCHMARK))
        result = compare_to_benchmark(self.PORTFOLIO, (aligned_a, aligned_b), aligned_n)
        self.assertAlmostEqual(result.beta, 1.5, places=15)
        self.assertEqual(result.alpha_daily, 0.0625)
        self.assertAlmostEqual(result.alpha_annualized, 15.75, places=12)
        self.assertEqual(result.tracking_error, 0.125 * math.sqrt(252))
        assert result.information_ratio is not None
        self.assertAlmostEqual(result.information_ratio, math.sqrt(63), places=12)
        self.assertEqual(result.up_capture, 2.0)
        self.assertEqual(result.down_capture, 1.0)

    def test_a_benchmark_sharing_an_asset_name_is_rejected(self) -> None:
        # A benchmark named like a holding is the spec-level "benchmark
        # inside the portfolio" confusion at the library boundary.
        imposter = PriceSeries(name="AAA", dates=GRID, prices=self.ASSET_B.prices)
        with self.assertRaisesRegex(ValueError, "unique"):
            compare_to_benchmark(self.PORTFOLIO, (self.ASSET_A, self.ASSET_B), imposter)


class PropertyTests(unittest.TestCase):
    def test_a_portfolio_identical_to_its_benchmark(self) -> None:
        # Both sleeves and the benchmark hold the same prices, so the
        # paired returns are equal bit for bit: beta 1, alpha 0, zero
        # tracking error (so no information ratio), both captures 1.
        prices = _prices(BENCHMARK_RETURNS, start=100.0)
        portfolio, sleeves = _twin_portfolio(prices)
        result = compare_to_benchmark(portfolio, sleeves, _benchmark(prices))
        self.assertEqual(result.beta, 1.0)
        self.assertEqual(result.alpha_daily, 0.0)
        self.assertEqual(result.alpha_annualized, 0.0)
        self.assertEqual(result.tracking_error, 0.0)
        self.assertIsNone(result.information_ratio)
        self.assertEqual(result.up_capture, 1.0)
        self.assertEqual(result.down_capture, 1.0)

    def test_a_doubled_benchmark_has_beta_two(self) -> None:
        # Dyadic benchmark returns and a portfolio returning exactly
        # double each day: every deviation doubles, so the covariance is
        # exactly twice the benchmark variance and beta is exactly 2.
        benchmark_returns = (0.25, -0.125, 0.5, -0.25, 0.125, 0.25)
        doubled = tuple(2.0 * value for value in benchmark_returns)
        portfolio, sleeves = _twin_portfolio(_prices(doubled))
        result = compare_to_benchmark(portfolio, sleeves, _benchmark(_prices(benchmark_returns)))
        self.assertEqual(result.beta, 2.0)
        self.assertEqual(result.up_capture, 2.0)
        self.assertEqual(result.down_capture, 2.0)

    def test_a_constant_shift_moves_alpha_and_nothing_else(self) -> None:
        # Adding c to every portfolio return moves the daily alpha by
        # exactly c and the annual alpha by exactly 252 c — the
        # arithmetic-annualization identity — while beta and the
        # tracking error stay put (the shift is deterministic). Dyadic
        # values keep the identity exact through the price round trip.
        benchmark_returns = (0.25, -0.125, 0.5, -0.25, 0.125, 0.25)
        base_returns = (0.5, -0.25, 1.0, -0.5, 0.25, 0.5)
        shift = 0.03125
        moved_returns = tuple(value + shift for value in base_returns)
        benchmark = _benchmark(_prices(benchmark_returns))
        base_portfolio, base_sleeves = _twin_portfolio(_prices(base_returns))
        moved_portfolio, moved_sleeves = _twin_portfolio(_prices(moved_returns))
        base = compare_to_benchmark(base_portfolio, base_sleeves, benchmark)
        moved = compare_to_benchmark(moved_portfolio, moved_sleeves, benchmark)
        self.assertEqual(moved.alpha_daily - base.alpha_daily, shift)
        self.assertEqual(moved.alpha_annualized - base.alpha_annualized, shift * 252)
        self.assertEqual(moved.beta, base.beta)
        self.assertEqual(moved.tracking_error, base.tracking_error)

    def test_a_constant_active_return_has_zero_tracking_error_and_no_ratio(self) -> None:
        # Benchmark plus a fixed daily offset: the comparison is legal
        # (the benchmark itself varies) but the active return is
        # constant, so the information ratio's denominator is zero and
        # the field is None rather than a number or an exception. The
        # values are dyadic (exact in binary floating point) so the
        # offset survives the price round trip and the subtraction to
        # the last bit.
        benchmark_returns = (0.25, -0.125, 0.5, -0.25, 0.125, 0.25)
        offset_returns = tuple(value + 0.25 for value in benchmark_returns)
        portfolio, sleeves = _twin_portfolio(_prices(offset_returns))
        result = compare_to_benchmark(portfolio, sleeves, _benchmark(_prices(benchmark_returns)))
        self.assertEqual(result.beta, 1.0)
        self.assertEqual(result.alpha_daily, 0.25)
        self.assertEqual(result.tracking_error, 0.0)
        self.assertIsNone(result.information_ratio)


class CaptureSideTests(unittest.TestCase):
    def test_a_benchmark_that_never_fell_has_no_down_capture(self) -> None:
        dates = GRID[:4]
        portfolio, sleeves = _twin_portfolio((64.0, 80.0, 64.0, 80.0), dates)
        result = compare_to_benchmark(
            portfolio, sleeves, _benchmark((64.0, 80.0, 96.0, 120.0), dates)
        )
        self.assertIsNone(result.down_capture)
        self.assertIsNotNone(result.up_capture)

    def test_a_benchmark_that_never_rose_has_no_up_capture(self) -> None:
        dates = GRID[:4]
        portfolio, sleeves = _twin_portfolio((80.0, 64.0, 80.0, 64.0), dates)
        result = compare_to_benchmark(
            portfolio, sleeves, _benchmark((120.0, 96.0, 80.0, 64.0), dates)
        )
        self.assertIsNone(result.up_capture)
        self.assertIsNotNone(result.down_capture)

    def test_zero_benchmark_days_belong_to_neither_side(self) -> None:
        # Benchmark returns (0.125, -0.125, 0.0, 0.25) against portfolio
        # returns (0.25, -0.375, 0.5, 0.5), all dyadic. The 0.0 benchmark
        # day (portfolio 0.5) is excluded from both captures: up over
        # days 1 and 4 ((0.25 + 0.5) / 2 over (0.125 + 0.25) / 2 = 2),
        # down over day 2 alone (-0.375 / -0.125 = 3).
        dates = GRID[:5]
        portfolio, sleeves = _twin_portfolio((64.0, 80.0, 50.0, 75.0, 112.5), dates)
        result = compare_to_benchmark(
            portfolio, sleeves, _benchmark((64.0, 72.0, 63.0, 63.0, 78.75), dates)
        )
        self.assertEqual(result.up_capture, 2.0)
        self.assertEqual(result.down_capture, 3.0)


class ValidationTests(unittest.TestCase):
    def test_too_few_paired_returns_are_rejected(self) -> None:
        # Three shared dates yield two paired returns; a straight line
        # fits any two points exactly, so the comparison needs a third.
        self.assertEqual(MIN_COMPARISON_RETURNS, 3)
        dates = GRID[:3]
        portfolio, sleeves = _twin_portfolio((64.0, 80.0, 64.0), dates)
        with self.assertRaisesRegex(ValueError, "at least 3"):
            compare_to_benchmark(portfolio, sleeves, _benchmark((64.0, 72.0, 63.0), dates))

    def test_nonfinite_returns_are_rejected(self) -> None:
        # Prices are validated positive and finite, but a ratio can still
        # overflow: 1e308 over 1e-308 is infinite. Both derived return
        # series re-check through the shared validation path, which
        # names the offending index.
        dates = GRID[:4]
        steady = (64.0, 80.0, 64.0, 80.0)
        overflow = (64.0, 1e-308, 1e308, 80.0)
        portfolio, sleeves = _twin_portfolio(overflow, dates)
        with self.assertRaisesRegex(ValueError, r"portfolio return \[1\] is inf; it must be"):
            compare_to_benchmark(portfolio, sleeves, _benchmark(steady, dates))
        portfolio, sleeves = _twin_portfolio(steady, dates)
        with self.assertRaisesRegex(ValueError, r"benchmark return \[1\] is inf; it must be"):
            compare_to_benchmark(portfolio, sleeves, _benchmark(overflow, dates))

    def test_a_constant_benchmark_is_rejected_for_beta(self) -> None:
        dates = GRID[:4]
        portfolio, sleeves = _twin_portfolio((64.0, 80.0, 64.0, 80.0), dates)
        with self.assertRaisesRegex(ValueError, "zero variance"):
            compare_to_benchmark(portfolio, sleeves, _benchmark((64.0, 64.0, 64.0, 64.0), dates))

    def test_a_bad_risk_free_rate_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be finite"):
            _compare(PORTFOLIO_RETURNS, BENCHMARK_RETURNS, risk_free_rate_daily=math.nan)
        with self.assertRaisesRegex(ValueError, "not a number"):
            _compare(PORTFOLIO_RETURNS, BENCHMARK_RETURNS, risk_free_rate_daily=True)

    def test_series_without_a_weight_are_rejected(self) -> None:
        # An extra asset series that carries no portfolio weight cannot
        # ride along: the portfolio's names must match the supplied
        # series exactly.
        portfolio, sleeves = _twin_portfolio(_prices(PORTFOLIO_RETURNS, start=100.0))
        benchmark = _benchmark(_prices(BENCHMARK_RETURNS, start=100.0))
        extra = PriceSeries(name="XTR", dates=GRID, prices=sleeves[0].prices)
        with self.assertRaisesRegex(ValueError, "have no weight"):
            compare_to_benchmark(portfolio, (*sleeves, extra), benchmark)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
