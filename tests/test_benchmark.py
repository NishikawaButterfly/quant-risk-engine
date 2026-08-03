from __future__ import annotations

import math
import unittest

from quantrisk.benchmark import MIN_COMPARISON_RETURNS, compare_to_benchmark

# The hand-worked six-pair fixture from docs/methodology.md. The returns
# are chosen so every statistic lands on a clean number: beta 3/2, daily
# alpha 0.005, tracking error sqrt(0.0504), information ratio sqrt(126),
# up capture 5/3 over four rising days, and down capture exactly 1 over
# exactly two falling days.
PORTFOLIO = (0.04, 0.00, 0.06, -0.03, 0.03, 0.02)
BENCHMARK = (0.02, -0.01, 0.03, -0.02, 0.02, 0.02)


class HandTableTests(unittest.TestCase):
    def test_beta_matches_the_hand_table(self) -> None:
        # cov = 0.003 / 5 = 0.0006, var(b) = 0.002 / 5 = 0.0004.
        result = compare_to_benchmark(PORTFOLIO, BENCHMARK)
        self.assertAlmostEqual(result.beta, 1.5, places=15)

    def test_alpha_matches_the_hand_table(self) -> None:
        # Daily: 0.02 - 1.5 * 0.01 = 0.005. Annualized arithmetically:
        # 0.005 * 252 = 1.26.
        result = compare_to_benchmark(PORTFOLIO, BENCHMARK)
        self.assertAlmostEqual(result.alpha_daily, 0.005, places=15)
        self.assertAlmostEqual(result.alpha_annualized, 1.26, places=12)
        self.assertEqual(round(result.alpha_annualized, 6), 1.26)

    def test_tracking_error_matches_the_hand_table(self) -> None:
        # Active returns (0.02, 0.01, 0.03, -0.01, 0.01, 0.00): sample
        # variance 0.001 / 5 = 0.0002, annualized sqrt(0.0504).
        result = compare_to_benchmark(PORTFOLIO, BENCHMARK)
        self.assertAlmostEqual(result.tracking_error, math.sqrt(0.0504), places=15)
        self.assertEqual(round(result.tracking_error, 6), 0.224499)

    def test_information_ratio_matches_the_hand_table(self) -> None:
        # 0.01 * 252 / sqrt(0.0504) collapses to sqrt(126).
        result = compare_to_benchmark(PORTFOLIO, BENCHMARK)
        assert result.information_ratio is not None
        self.assertAlmostEqual(result.information_ratio, math.sqrt(126), places=12)
        self.assertEqual(round(result.information_ratio, 6), 11.224972)

    def test_capture_ratios_match_the_hand_table(self) -> None:
        # Up: (0.15 / 4) / (0.09 / 4) = 5/3 over four rising days.
        # Down: (-0.03 / 2) / (-0.03 / 2) = 1 over exactly two days.
        result = compare_to_benchmark(PORTFOLIO, BENCHMARK)
        assert result.up_capture is not None
        self.assertAlmostEqual(result.up_capture, 5.0 / 3.0, places=15)
        self.assertEqual(round(result.up_capture, 6), 1.666667)
        self.assertEqual(result.down_capture, 1.0)

    def test_risk_free_rate_moves_alpha_by_beta_minus_one(self) -> None:
        # alpha(rf) = (mean_p - rf) - beta * (mean_b - rf), so a daily
        # rate shifts the daily alpha by (beta - 1) * rf and leaves
        # beta itself untouched: constants drop out of covariances.
        base = compare_to_benchmark(PORTFOLIO, BENCHMARK)
        shifted = compare_to_benchmark(PORTFOLIO, BENCHMARK, risk_free_rate_daily=0.0001)
        self.assertEqual(shifted.beta, base.beta)
        self.assertAlmostEqual(shifted.alpha_daily - base.alpha_daily, 0.00005, places=15)


class PropertyTests(unittest.TestCase):
    def test_a_portfolio_identical_to_its_benchmark(self) -> None:
        # Beta 1, alpha 0, zero tracking error (so no information
        # ratio), and both captures exactly 1.
        result = compare_to_benchmark(BENCHMARK, BENCHMARK)
        self.assertAlmostEqual(result.beta, 1.0, places=15)
        self.assertAlmostEqual(result.alpha_daily, 0.0, places=15)
        self.assertAlmostEqual(result.alpha_annualized, 0.0, places=12)
        self.assertEqual(result.tracking_error, 0.0)
        self.assertIsNone(result.information_ratio)
        self.assertEqual(result.up_capture, 1.0)
        self.assertEqual(result.down_capture, 1.0)

    def test_a_doubled_benchmark_has_beta_two(self) -> None:
        doubled = tuple(2.0 * value for value in BENCHMARK)
        result = compare_to_benchmark(doubled, BENCHMARK)
        self.assertAlmostEqual(result.beta, 2.0, places=15)
        self.assertEqual(result.up_capture, 2.0)
        self.assertEqual(result.down_capture, 2.0)

    def test_a_constant_shift_moves_alpha_and_nothing_else(self) -> None:
        # Adding c to every portfolio return moves the daily alpha by
        # exactly c and the annual alpha by exactly 252 c — the
        # arithmetic-annualization identity — while beta and the
        # tracking error stay put (the shift is deterministic).
        shift = 0.001
        base = compare_to_benchmark(PORTFOLIO, BENCHMARK)
        moved = compare_to_benchmark(tuple(value + shift for value in PORTFOLIO), BENCHMARK)
        self.assertAlmostEqual(moved.alpha_daily - base.alpha_daily, shift, places=15)
        self.assertAlmostEqual(
            moved.alpha_annualized - base.alpha_annualized, shift * 252, places=12
        )
        self.assertAlmostEqual(moved.beta, base.beta, places=15)
        self.assertAlmostEqual(moved.tracking_error, base.tracking_error, places=12)

    def test_a_constant_active_return_has_zero_tracking_error_and_no_ratio(self) -> None:
        # Benchmark plus a fixed daily offset: the comparison is legal
        # (the benchmark itself varies) but the active return is
        # constant, so the information ratio's denominator is zero and
        # the field is None rather than a number or an exception. The
        # values are dyadic (exact in binary floating point) so the
        # offset survives the subtraction to the last bit.
        benchmark = (0.25, -0.125, 0.5, -0.25, 0.125, 0.25)
        offset = tuple(value + 0.25 for value in benchmark)
        result = compare_to_benchmark(offset, benchmark)
        self.assertEqual(result.beta, 1.0)
        self.assertEqual(result.alpha_daily, 0.25)
        self.assertEqual(result.tracking_error, 0.0)
        self.assertIsNone(result.information_ratio)


class CaptureSideTests(unittest.TestCase):
    def test_a_benchmark_that_never_fell_has_no_down_capture(self) -> None:
        result = compare_to_benchmark((0.02, 0.01, 0.03), (0.01, 0.02, 0.03))
        self.assertIsNone(result.down_capture)
        self.assertIsNotNone(result.up_capture)

    def test_a_benchmark_that_never_rose_has_no_up_capture(self) -> None:
        result = compare_to_benchmark((-0.02, -0.01, -0.03), (-0.01, -0.02, -0.03))
        self.assertIsNone(result.up_capture)
        self.assertIsNotNone(result.down_capture)

    def test_zero_benchmark_days_belong_to_neither_side(self) -> None:
        # The 0.0 benchmark day (portfolio 0.05) is excluded from both
        # captures: up over days 1 and 4, down over day 2 alone.
        portfolio = (0.02, -0.02, 0.05, 0.04)
        benchmark = (0.01, -0.01, 0.00, 0.03)
        result = compare_to_benchmark(portfolio, benchmark)
        assert result.up_capture is not None
        self.assertAlmostEqual(result.up_capture, 0.03 / 0.02, places=15)
        self.assertEqual(result.down_capture, 2.0)


class ValidationTests(unittest.TestCase):
    def test_mismatched_lengths_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "same date grid"):
            compare_to_benchmark((0.01, 0.02, 0.03), (0.01, 0.02))

    def test_too_few_observations_are_rejected(self) -> None:
        self.assertEqual(MIN_COMPARISON_RETURNS, 3)
        with self.assertRaisesRegex(ValueError, "at least 3"):
            compare_to_benchmark((0.01, 0.02), (0.01, 0.02))

    def test_nonfinite_returns_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, r"portfolio return \[1\] is nan; it must be"):
            compare_to_benchmark((0.01, math.nan, 0.02), BENCHMARK[:3])
        with self.assertRaisesRegex(ValueError, r"benchmark return \[1\] is inf; it must be"):
            compare_to_benchmark(PORTFOLIO[:3], (0.01, math.inf, 0.02))

    def test_a_constant_benchmark_is_rejected_for_beta(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero variance"):
            compare_to_benchmark((0.01, 0.02, 0.03), (0.01, 0.01, 0.01))

    def test_a_bad_risk_free_rate_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be finite"):
            compare_to_benchmark(PORTFOLIO, BENCHMARK, risk_free_rate_daily=math.nan)
        with self.assertRaisesRegex(ValueError, "not a number"):
            compare_to_benchmark(PORTFOLIO, BENCHMARK, risk_free_rate_daily=True)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
