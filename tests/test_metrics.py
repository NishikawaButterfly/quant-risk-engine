from __future__ import annotations

import math
import unittest

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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
