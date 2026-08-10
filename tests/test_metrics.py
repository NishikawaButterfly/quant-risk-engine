from __future__ import annotations

import math
import random
import unittest

from quantrisk._validation import MIN_VOLATILITY
from quantrisk.metrics import (
    TRADING_DAYS_PER_YEAR,
    annualized_volatility,
    historical_cvar,
    historical_var,
    max_drawdown,
    parametric_var,
    sharpe_ratio,
    sortino_ratio,
)
from quantrisk.series import PriceSeries

# The hand-worked fixture from docs/methodology.md. Every asserted digit
# below appears in that document and can be recomputed by hand.
RETURNS = (0.01, 0.02, -0.03, 0.04, -0.02)
SERIES = PriceSeries(
    name="AAA",
    dates=("2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09", "2026-01-12"),
    prices=(100.0, 102.0, 99.0, 101.0, 98.0, 100.0),
)


class HandTableTests(unittest.TestCase):
    def test_annualized_volatility_matches_the_hand_table(self) -> None:
        # mean 0.004, squared deviations sum 0.00332, sample variance
        # 0.00083, stddev 0.02880972..., times sqrt(252).
        value = annualized_volatility(RETURNS)
        self.assertAlmostEqual(value, math.sqrt(0.00083) * math.sqrt(252), places=15)
        self.assertEqual(round(value, 6), 0.45734)

    def test_sharpe_ratio_matches_the_hand_table(self) -> None:
        # Annual risk-free 2.52% is exactly 0.0001 per day; excess mean
        # 0.0039 over stddev 0.02880972..., annualized with sqrt(252).
        value = sharpe_ratio(RETURNS, risk_free_rate_annual=0.0252)
        self.assertAlmostEqual(value, 0.0039 / math.sqrt(0.00083) * math.sqrt(252), places=12)
        self.assertEqual(round(value, 6), 2.148948)

    def test_sortino_ratio_matches_the_hand_table(self) -> None:
        # Shortfalls below the zero target: -0.03 and -0.02. Downside
        # deviation sqrt((0.0009 + 0.0004) / 5) = sqrt(0.00026).
        value = sortino_ratio(RETURNS)
        self.assertAlmostEqual(value, 0.004 / math.sqrt(0.00026) * math.sqrt(252), places=12)
        self.assertEqual(round(value, 6), 3.937981)

    def test_max_drawdown_matches_the_hand_table(self) -> None:
        # Running peak 102 from 2026-01-06; trough 98 on 2026-01-09;
        # depth 4 / 102.
        result = max_drawdown(SERIES)
        self.assertAlmostEqual(result.depth, 4.0 / 102.0, places=15)
        self.assertEqual(round(result.depth, 6), 0.039216)
        self.assertEqual(result.peak_date, "2026-01-06")
        self.assertEqual(result.trough_date, "2026-01-09")

    def test_historical_var_matches_the_hand_table(self) -> None:
        # Sorted returns -0.03, -0.02, 0.01, 0.02, 0.04. At 95% the
        # target rank is 0.05 * 4 = 0.2, so the quantile interpolates
        # to -0.03 + 0.2 * 0.01 = -0.028.
        self.assertAlmostEqual(historical_var(RETURNS, confidence=0.95), 0.028, places=15)
        # At 90% the rank is 0.4, giving -0.026.
        self.assertAlmostEqual(historical_var(RETURNS, confidence=0.90), 0.026, places=15)
        # At 75% the rank is exactly 1.0, so no interpolation happens and
        # the quantile is the second-worst return, -0.02.
        self.assertEqual(historical_var(RETURNS, confidence=0.75), 0.02)

    def test_historical_cvar_matches_the_hand_table(self) -> None:
        # Only -0.03 lies at or below the -0.028 quantile, so the
        # 95% CVaR is 0.03.
        self.assertAlmostEqual(historical_cvar(RETURNS, confidence=0.95), 0.03, places=15)

    def test_parametric_var_matches_the_hand_table(self) -> None:
        # -(0.004 + z * 0.02880972...) with z = norm.ppf(0.05).
        value = parametric_var(RETURNS, confidence=0.95)
        expected = -(0.004 + (-1.6448536269514729) * math.sqrt(0.00083))
        self.assertAlmostEqual(value, expected, places=12)
        self.assertEqual(round(value, 6), 0.043388)


class ConventionTests(unittest.TestCase):
    def test_the_annualization_constant_is_252(self) -> None:
        self.assertEqual(TRADING_DAYS_PER_YEAR, 252)

    def test_volatility_and_ratios_agree_on_the_returns_of_the_price_series(self) -> None:
        # The metrics accept plain sequences, so the series returns feed
        # straight in.
        returns = SERIES.simple_returns()
        self.assertGreater(annualized_volatility(returns), 0.0)
        self.assertIsInstance(sharpe_ratio(returns, risk_free_rate_annual=0.0), float)

    def test_var_is_monotone_in_confidence(self) -> None:
        confidences = (0.80, 0.90, 0.95, 0.975, 0.99)
        historical = [historical_var(RETURNS, confidence=c) for c in confidences]
        self.assertEqual(historical, sorted(historical))
        parametric = [parametric_var(RETURNS, confidence=c) for c in confidences]
        self.assertEqual(parametric, sorted(parametric))

    def test_cvar_never_falls_below_var(self) -> None:
        for confidence in (0.80, 0.90, 0.95, 0.99):
            self.assertGreaterEqual(
                historical_cvar(RETURNS, confidence=confidence),
                historical_var(RETURNS, confidence=confidence),
            )

    def test_var_can_be_negative_when_the_tail_is_a_gain(self) -> None:
        gains = (0.01, 0.02, 0.03, 0.04, 0.05)
        self.assertLess(historical_var(gains, confidence=0.80), 0.0)

    def test_a_flat_price_path_has_zero_drawdown_on_the_first_date(self) -> None:
        rising = PriceSeries("UP", SERIES.dates, (100.0, 101.0, 102.0, 103.0, 104.0, 105.0))
        result = max_drawdown(rising)
        self.assertEqual(result.depth, 0.0)
        self.assertEqual(result.peak_date, "2026-01-05")
        self.assertEqual(result.trough_date, "2026-01-05")

    def test_ties_keep_the_earliest_trough(self) -> None:
        double_dip = PriceSeries("VV", SERIES.dates, (100.0, 90.0, 100.0, 90.0, 100.0, 100.0))
        result = max_drawdown(double_dip)
        self.assertAlmostEqual(result.depth, 0.1, places=15)
        self.assertEqual(result.peak_date, "2026-01-05")
        self.assertEqual(result.trough_date, "2026-01-06")

    def test_drawdown_dates_are_session_labels_across_a_weekend(self) -> None:
        # Peak on Friday 2026-01-09, trough on Monday 2026-01-12: the
        # dates are session labels under the session model
        # (docs/methodology.md), so this drawdown is one session deep
        # even though the calendar span is three days, and its depth is
        # the plain session-over-session decline with no weekend
        # accrual.
        weekend = PriceSeries(
            "WKND",
            ("2026-01-08", "2026-01-09", "2026-01-12", "2026-01-13"),
            (100.0, 110.0, 99.0, 111.0),
        )
        result = max_drawdown(weekend)
        self.assertEqual(result.peak_date, "2026-01-09")
        self.assertEqual(result.trough_date, "2026-01-12")
        self.assertEqual(result.depth, 1.0 - 99.0 / 110.0)


class TailBoundaryConventionTests(unittest.TestCase):
    """Pin the sign and tail-boundary conventions digit for digit.

    Textbooks and libraries disagree on two silent choices: the sign a
    reported VaR carries, and what happens at the quantile boundary
    when the tail does not divide evenly into the sample. Each fixture
    below is hand-calculated under the competing conventions, which
    give different answers, and the assertions pin exactly which one
    this engine produces. The conventions are stated in prose in
    docs/methodology.md ("VaR and CVaR conventions").
    """

    def test_the_boundary_observation_is_included_when_the_rank_is_exact(self) -> None:
        # Doc fixture, sorted: -0.03, -0.02, 0.01, 0.02, 0.04. At 75%
        # confidence the target rank 0.25 * 4 = 1.0 lands exactly on
        # the order statistic -0.02, so the quantile is -0.02 (VaR
        # 0.020) and the boundary observation itself is the whole
        # disagreement. Three textbook treatments of the 25% tail:
        #   include the boundary: mean(-0.03, -0.02) = -0.025 -> CVaR 0.025
        #   exclude the boundary: mean(-0.03)        = -0.030 -> CVaR 0.030
        #   Acerbi-Tasche tail expectation, n * alpha = 5 * 0.25 = 1.25:
        #     (0.03 + 0.25 * 0.02) / 1.25 = 0.035 / 1.25     -> CVaR 0.028
        # This engine includes the boundary observation.
        self.assertEqual(historical_var(RETURNS, confidence=0.75), 0.02)
        value = historical_cvar(RETURNS, confidence=0.75)
        self.assertAlmostEqual(value, 0.025, places=15)
        self.assertNotAlmostEqual(value, 0.030, places=9)
        self.assertNotAlmostEqual(value, 0.028, places=9)

    def test_an_uneven_tail_is_cut_at_the_interpolated_quantile(self) -> None:
        # Six returns, sorted: -0.04, -0.02, -0.01, 0.01, 0.02, 0.03.
        # At 75% confidence the tail holds n * alpha = 6 * 0.25 = 1.5
        # observations — it does not divide evenly into the sample. The
        # rank is 0.25 * 5 = 1.25, so the quantile interpolates to
        # -0.02 + 0.25 * (-0.01 - -0.02) = -0.0175 and VaR is 0.0175.
        # The competing conventions:
        #   this engine, mean of returns at or below -0.0175:
        #     mean(-0.04, -0.02) = -0.03                     -> CVaR 0.030
        #   worst floor(n * alpha) = 1 observation only:
        #     mean(-0.04)                                    -> CVaR 0.040
        #   Acerbi-Tasche tail expectation:
        #     (0.04 + 0.5 * 0.02) / 1.5 = 0.05 / 1.5         -> CVaR 0.0333...
        returns = (0.01, -0.04, 0.02, -0.02, 0.03, -0.01)
        self.assertAlmostEqual(historical_var(returns, confidence=0.75), 0.0175, places=15)
        value = historical_cvar(returns, confidence=0.75)
        self.assertAlmostEqual(value, 0.03, places=15)
        self.assertNotAlmostEqual(value, 0.04, places=9)
        self.assertNotAlmostEqual(value, 0.05 / 1.5, places=9)

    def test_the_tail_is_a_quantile_cut_not_a_rounded_count_of_observations(self) -> None:
        # Eight returns, sorted: -0.05, -0.03, -0.01, 0.0, 0.01, 0.02,
        # 0.03, 0.04, at 72% confidence (alpha = 0.28). A count-based
        # scheme averaging the worst ceil(n * alpha) = ceil(2.24) = 3
        # observations reports mean(-0.05, -0.03, -0.01) = -0.03, CVaR
        # 0.030. This engine instead cuts at the interpolated quantile:
        # rank 0.28 * 7 = 1.96, quantile -0.03 + 0.96 * 0.02 = -0.0108,
        # and only -0.05 and -0.03 lie at or below it, so CVaR is
        # mean(-0.05, -0.03) negated: 0.040.
        returns = (0.01, -0.05, 0.03, -0.01, 0.04, 0.0, -0.03, 0.02)
        self.assertAlmostEqual(historical_var(returns, confidence=0.72), 0.0108, places=15)
        value = historical_cvar(returns, confidence=0.72)
        self.assertAlmostEqual(value, 0.04, places=15)
        self.assertNotAlmostEqual(value, 0.03, places=9)

    def test_losses_carry_positive_sign_and_gains_negative(self) -> None:
        # Sign convention: a loss is reported as a positive number, so
        # a negative VaR or CVaR means even the tail gained. All-gain
        # fixture at 80%: rank 0.2 * 4 = 0.8, quantile 0.01 + 0.8 *
        # 0.01 = 0.018, VaR = -0.018. The tail at or below 0.018 is
        # {0.01} alone, so CVaR = -0.01.
        gains = (0.01, 0.02, 0.03, 0.04, 0.05)
        self.assertAlmostEqual(historical_var(gains, confidence=0.80), -0.018, places=15)
        self.assertAlmostEqual(historical_cvar(gains, confidence=0.80), -0.01, places=15)
        # The doc fixture's 95% tail is a loss, so every VaR flavour
        # reports a positive number there.
        self.assertGreater(historical_var(RETURNS, confidence=0.95), 0.0)
        self.assertGreater(historical_cvar(RETURNS, confidence=0.95), 0.0)
        self.assertGreater(parametric_var(RETURNS, confidence=0.95), 0.0)

    def test_cvar_never_falls_below_var_on_seeded_random_samples(self) -> None:
        # Property guard for boundary consistency between the two
        # historical functions: both cut at the same interpolated
        # quantile, the CVaR tail is bounded above by that quantile, so
        # its mean cannot exceed the quantile and the reported CVaR can
        # never fall below the reported VaR.
        rng = random.Random(20260809)  # noqa: S311 - reproducible fixture, not cryptography
        for _ in range(25):
            count = rng.randint(2, 40)
            returns = tuple(rng.uniform(-0.1, 0.1) for _ in range(count))
            for confidence in (0.5, 0.75, 0.9, 0.95, 0.99):
                self.assertGreaterEqual(
                    historical_cvar(returns, confidence=confidence),
                    historical_var(returns, confidence=confidence),
                )


class ValidationTests(unittest.TestCase):
    def test_too_few_returns_are_rejected(self) -> None:
        for func in (annualized_volatility,):
            with self.assertRaisesRegex(ValueError, "at least 2"):
                func((0.01,))
        with self.assertRaisesRegex(ValueError, "at least 2"):
            historical_var((0.01,), confidence=0.95)

    def test_nonfinite_returns_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            annualized_volatility((0.01, math.nan))
        with self.assertRaisesRegex(ValueError, "finite"):
            parametric_var((0.01, math.inf), confidence=0.95)

    def test_confidence_bounds_are_enforced(self) -> None:
        for confidence in (0.0, 1.0, -0.5, 1.5):
            with self.assertRaisesRegex(ValueError, "strictly between"):
                historical_var(RETURNS, confidence=confidence)
            with self.assertRaisesRegex(ValueError, "strictly between"):
                historical_cvar(RETURNS, confidence=confidence)
            with self.assertRaisesRegex(ValueError, "strictly between"):
                parametric_var(RETURNS, confidence=confidence)

    def test_sharpe_ratio_rejects_constant_returns(self) -> None:
        with self.assertRaisesRegex(ValueError, "constant"):
            sharpe_ratio((0.01, 0.01, 0.01), risk_free_rate_annual=0.0)

    def test_sortino_ratio_rejects_series_with_no_downside(self) -> None:
        with self.assertRaisesRegex(ValueError, "below the target"):
            sortino_ratio((0.01, 0.02, 0.03))


class VarianceToleranceTests(unittest.TestCase):
    """One named tolerance guards every volatility denominator.

    Before the tolerance every guard in the engine compared its
    denominator against exactly ``0.0``, so a volatility produced by
    float64 rounding alone (each return carries absolute noise of
    order 1e-16) sailed through the guard and came back as a huge,
    meaningless ratio. The fixtures here sit strictly between the old
    threshold (exact zero) and the new one (``MIN_VOLATILITY``) —
    exactly the region every old guard accepted and the unified guard
    rejects.
    """

    # Alternating ±5e-13: mean exactly 0, sample stddev ~5.8e-13,
    # just below the 1e-12 tolerance. Ten times larger: just above.
    BELOW = (5e-13, -5e-13, 5e-13, -5e-13)
    ABOVE = (5e-12, -5e-12, 5e-12, -5e-12)

    def test_the_tolerance_is_one_named_constant(self) -> None:
        self.assertEqual(MIN_VOLATILITY, 1e-12)

    def test_sharpe_rejects_volatility_just_below_the_tolerance(self) -> None:
        with self.assertRaisesRegex(ValueError, "MIN_VOLATILITY"):
            sharpe_ratio(self.BELOW, risk_free_rate_annual=0.0)

    def test_sharpe_computes_just_above_the_tolerance(self) -> None:
        # Mean exactly zero, so the ratio is exactly 0.0 — the point is
        # that it computes rather than raising.
        self.assertEqual(sharpe_ratio(self.ABOVE, risk_free_rate_annual=0.0), 0.0)

    def test_sortino_rejects_downside_just_below_the_tolerance(self) -> None:
        # One shortfall of 5e-13 gives a downside deviation of 2.5e-13.
        with self.assertRaisesRegex(ValueError, "MIN_VOLATILITY"):
            sortino_ratio((0.0, -5e-13, 0.0, 5e-13))

    def test_sortino_computes_just_above_the_tolerance(self) -> None:
        # One shortfall of 5e-12: downside deviation 2.5e-12, mean 0.
        self.assertEqual(sortino_ratio((0.0, -5e-12, 0.0, 5e-12)), 0.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
