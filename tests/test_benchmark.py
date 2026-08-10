from __future__ import annotations

import math
import unittest
from collections.abc import Sequence

from quantrisk.benchmark import (
    MIN_COMPARISON_RETURNS,
    BenchmarkComparison,
    _capture_ratio,
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
# ratio sqrt(126), geometric up capture 1.700067 compounded over four
# rising days, and geometric down capture 150/149 over two falling days.
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
        # Geometric: each side's returns compound into one cumulative
        # move before the ratio. Up, over the four rising days 1, 3,
        # 5, 6:
        #   p: 1.04 * 1.06 * 1.03 * 1.02 - 1
        #      = 1.1024 * 1.03 * 1.02 - 1 = 1.135472 * 1.02 - 1
        #      = 0.15818144
        #   b: 1.02 * 1.03 * 1.02 * 1.02 - 1
        #      = 1.0506 * 1.02 * 1.02 - 1 = 1.071612 * 1.02 - 1
        #      = 0.09304424
        #   up = 0.15818144 / 0.09304424 = 1.700067
        # Down, over the two falling days 2 and 4:
        #   p: 1.00 * 0.97 - 1 = -0.03
        #   b: 0.99 * 0.98 - 1 = -0.0298
        #   down = -0.03 / -0.0298 = 150/149 = 1.006711
        # (Arithmetic means would say 5/3 and exactly 1.)
        result = _compare(PORTFOLIO_RETURNS, BENCHMARK_RETURNS)
        assert result.up_capture is not None
        assert result.down_capture is not None
        self.assertAlmostEqual(result.up_capture, 0.15818144 / 0.09304424, places=13)
        self.assertEqual(round(result.up_capture, 6), 1.700067)
        self.assertAlmostEqual(result.down_capture, 150.0 / 149.0, places=13)
        self.assertEqual(round(result.down_capture, 6), 1.006711)

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
        # IR = (0.0625 * 252) / TE = sqrt(63). Capture compounds each
        # side: the benchmark rose on the first and third pairs
        # (up = (1.25² - 1) / (1.125² - 1) = 0.5625 / 0.265625 = 36/17)
        # and fell on the other two
        # (down = (1.0 * 0.75 - 1) / (0.875² - 1) = -0.25 / -0.234375
        # = 16/15). Every compounded product is dyadic and exact, so
        # each ratio is one correctly rounded division.
        aligned_a, aligned_b, aligned_n = align((self.ASSET_A, self.ASSET_B, self.BENCHMARK))
        result = compare_to_benchmark(self.PORTFOLIO, (aligned_a, aligned_b), aligned_n)
        self.assertAlmostEqual(result.beta, 1.5, places=15)
        self.assertEqual(result.alpha_daily, 0.0625)
        self.assertAlmostEqual(result.alpha_annualized, 15.75, places=12)
        self.assertEqual(result.tracking_error, 0.125 * math.sqrt(252))
        assert result.information_ratio is not None
        self.assertAlmostEqual(result.information_ratio, math.sqrt(63), places=12)
        self.assertEqual(result.up_capture, 36.0 / 17.0)
        self.assertEqual(result.down_capture, 16.0 / 15.0)

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
        # The compounded captures are *not* 2 — doubling every daily
        # return is a linear act and compounding is not. Up, over the
        # rising days (0.25, 0.5, 0.125, 0.25):
        #   p: 1.5 * 2.0 * 1.25 * 1.5 - 1 = 5.625 - 1 = 4.625
        #   b: 1.25 * 1.5 * 1.125 * 1.25 - 1 = 2.63671875 - 1
        #      = 1.63671875
        #   up = 4.625 / 1.63671875 = (37/8) / (419/256) = 1184/419
        # Down, over the falling days (-0.125, -0.25):
        #   p: 0.75 * 0.5 - 1 = -0.625
        #   b: 0.875 * 0.75 - 1 = -0.34375
        #   down = -0.625 / -0.34375 = (5/8) / (11/32) = 20/11
        # Doubled daily gains compound to more than double the rise
        # (2.825895 > 2); doubled daily losses compound to less than
        # double the fall (1.818182 < 2). All intermediates are dyadic
        # and exact, so each ratio is one correctly rounded division.
        benchmark_returns = (0.25, -0.125, 0.5, -0.25, 0.125, 0.25)
        doubled = tuple(2.0 * value for value in benchmark_returns)
        portfolio, sleeves = _twin_portfolio(_prices(doubled))
        result = compare_to_benchmark(portfolio, sleeves, _benchmark(_prices(benchmark_returns)))
        self.assertEqual(result.beta, 2.0)
        self.assertEqual(result.up_capture, 1184.0 / 419.0)
        self.assertEqual(result.down_capture, 20.0 / 11.0)

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
        # day (portfolio 0.5) is excluded from both captures: up
        # compounds days 1 and 4
        # ((1.25 * 1.5 - 1) / (1.125 * 1.25 - 1) = 0.875 / 0.40625
        # = 28/13), down is day 2 alone (-0.375 / -0.125 = 3 — a
        # single-day side has nothing to compound, so both conventions
        # coincide there).
        dates = GRID[:5]
        portfolio, sleeves = _twin_portfolio((64.0, 80.0, 50.0, 75.0, 112.5), dates)
        result = compare_to_benchmark(
            portfolio, sleeves, _benchmark((64.0, 72.0, 63.0, 63.0, 78.75), dates)
        )
        self.assertEqual(result.up_capture, 28.0 / 13.0)
        self.assertEqual(result.down_capture, 3.0)


class CaptureConventionTests(unittest.TestCase):
    """The geometric convention, pinned where the conventions split.

    Capture compounds each side's returns into one cumulative move
    before dividing. An arithmetic mean of the same days disagrees
    with the compounded move exactly over runs of same-signed returns
    — when capture matters most — so the fixture here is a run of
    gains and a run of losses. Everything is dyadic: every compounded
    product is exact in binary floating point and each asserted ratio
    is one correctly rounded division.
    """

    def test_runs_of_gains_and_losses_compound_geometrically(self) -> None:
        # Benchmark: three rising days, then three falling days.
        #   b = (0.25, 0.25, 0.25, -0.25, -0.25, -0.25)
        #   p = (0.50, 0.50, 0.50, -0.125, -0.125, -0.125)
        # Up, compounded over days 1-3:
        #   p: 1.5³ - 1 = 3.375 - 1 = 2.375            (= 19/8)
        #   b: 1.25³ - 1 = 1.953125 - 1 = 0.953125     (= 61/64)
        #   up = 2.375 / 0.953125 = 152/61 = 2.491803...
        # Down, compounded over days 4-6:
        #   p: 0.875³ - 1 = 0.669921875 - 1
        #      = -0.330078125                          (= -169/512)
        #   b: 0.75³ - 1 = 0.421875 - 1 = -0.578125    (= -37/64)
        #   down = -0.330078125 / -0.578125 = 169/296 = 0.570945...
        # Arithmetic means would say up = 0.5 / 0.25 = 2 and
        # down = -0.125 / -0.25 = 0.5: the same six days, a different
        # story on both sides.
        portfolio_returns = (0.5, 0.5, 0.5, -0.125, -0.125, -0.125)
        benchmark_returns = (0.25, 0.25, 0.25, -0.25, -0.25, -0.25)
        result = _compare(portfolio_returns, benchmark_returns)
        assert result.up_capture is not None
        assert result.down_capture is not None
        self.assertEqual(result.up_capture, 152.0 / 61.0)
        self.assertEqual(round(result.up_capture, 6), 2.491803)
        self.assertEqual(result.down_capture, 169.0 / 296.0)
        self.assertEqual(round(result.down_capture, 6), 0.570946)
        # The arithmetic values are ruled out to a coarse tolerance:
        # this test fails against a mean-based implementation long
        # before the digit assertions above are reached.
        self.assertNotAlmostEqual(result.up_capture, 2.0, places=1)
        self.assertNotAlmostEqual(result.down_capture, 0.5, places=1)

    def test_a_benchmark_move_that_compounds_to_zero_is_rejected(self) -> None:
        # Mathematically unreachable: rising factors all exceed 1 and
        # falling factors all sit strictly inside (0, 1), so a nonempty
        # side never compounds its benchmark move to a true zero. In
        # floating point a sub-ulp return like ±1e-17 is nonzero — it
        # selects a side — yet 1.0 ± 1e-17 rounds to exactly 1.0, so
        # the side's move compounds to 0.0. Validated prices cannot
        # emit such a return (the smallest nonzero move out of
        # simple_returns keeps 1 + r representable), so the guard is
        # exercised at the function boundary rather than through the
        # public API.
        with self.assertRaisesRegex(ValueError, "compound to exactly zero"):
            _capture_ratio((0.01, -0.02), (1e-17, -0.5), rising=True)
        with self.assertRaisesRegex(ValueError, "compound to exactly zero"):
            _capture_ratio((0.01, -0.02), (0.5, -1e-17), rising=False)


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

    def test_a_benchmark_just_below_the_variance_tolerance_is_rejected(self) -> None:
        # Beta divides by the benchmark *variance*, so the shared
        # volatility tolerance applies squared: benchmark returns
        # alternating ±5e-13 have stddev ~5.8e-13 < MIN_VOLATILITY,
        # i.e. variance below MIN_VOLATILITY². Before the tolerance
        # this benchmark passed the exact-zero guard and produced a
        # beta of order 1e+11 out of pure rounding noise.
        dates = GRID[:5]
        portfolio, sleeves = _twin_portfolio((64.0, 80.0, 64.0, 80.0, 64.0), dates)
        wobble = tuple(10.0 if index % 2 == 0 else 10.0 * (1.0 + 5e-13) for index in range(5))
        with self.assertRaisesRegex(ValueError, "zero variance.*MIN_VOLATILITY"):
            compare_to_benchmark(portfolio, sleeves, _benchmark(wobble, dates))

    def test_a_benchmark_just_above_the_variance_tolerance_computes(self) -> None:
        dates = GRID[:5]
        portfolio, sleeves = _twin_portfolio((64.0, 80.0, 64.0, 80.0, 64.0), dates)
        wobble = tuple(10.0 if index % 2 == 0 else 10.0 * (1.0 + 5e-12) for index in range(5))
        result = compare_to_benchmark(portfolio, sleeves, _benchmark(wobble, dates))
        self.assertTrue(math.isfinite(result.beta))

    def test_information_ratio_degrades_at_the_same_tolerance(self) -> None:
        # The one stated degrade site: an active return whose daily
        # stddev sits below MIN_VOLATILITY yields information_ratio
        # None — same constant, same trigger, but the comparison's
        # other statistics stand, exactly as with an exactly-zero
        # tracking error. Ten times the wiggle and the ratio is back.
        dates = GRID[:5]
        base = (64.0, 80.0, 64.0, 80.0, 64.0)
        benchmark = _benchmark(base, dates)
        for wiggle, expect_none in ((5e-13, True), (5e-12, False)):
            offset = tuple(
                price if index % 2 == 0 else price * (1.0 + wiggle)
                for index, price in enumerate(base)
            )
            portfolio, sleeves = _twin_portfolio(offset, dates)
            result = compare_to_benchmark(portfolio, sleeves, benchmark)
            self.assertTrue(math.isfinite(result.beta))
            self.assertIsNotNone(result.up_capture)
            if expect_none:
                self.assertIsNone(result.information_ratio)
            else:
                self.assertIsNotNone(result.information_ratio)

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
