from __future__ import annotations

import math
import unittest
from collections.abc import Sequence

from quantrisk.metrics import annualized_volatility
from quantrisk.portfolio import Portfolio, correlation_matrix, covariance_matrix
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

# The hand-worked two-asset fixture from docs/methodology.md. The returns
# are chosen so the sample covariance matrix is exactly
# [[0.0004, 0.0001], [0.0001, 0.0009]]: correlation 1/6, and with weights
# 60/40 both assets contribute exactly half the portfolio variance.
RETURNS_A = (0.02, -0.01, -0.02, 0.02, 0.03, 0.02)
RETURNS_B = (0.03, 0.03, -0.03, 0.035, -0.01, -0.025)


def series_from_returns(name: str, start: float, returns: Sequence[float]) -> PriceSeries:
    prices = [start]
    for value in returns:
        prices.append(prices[-1] * (1.0 + value))
    return PriceSeries(name=name, dates=DATES, prices=tuple(prices))


SERIES_A = series_from_returns("AAA", 100.0, RETURNS_A)
SERIES_B = series_from_returns("BBB", 50.0, RETURNS_B)
SERIES_C = series_from_returns("CCC", 75.0, (0.01, -0.02, 0.03, -0.01, 0.02, -0.03))
PORTFOLIO = Portfolio(names=("AAA", "BBB"), weights=(0.6, 0.4))


class HandTableTests(unittest.TestCase):
    def test_covariance_matrix_matches_the_hand_table(self) -> None:
        # Squared deviations sum to 0.002 (A) and 0.0045 (B), cross
        # products to 0.0005; divided by n - 1 = 5.
        matrix = covariance_matrix((SERIES_A, SERIES_B))
        self.assertAlmostEqual(matrix[0][0], 0.0004, places=15)
        self.assertAlmostEqual(matrix[1][1], 0.0009, places=15)
        self.assertAlmostEqual(matrix[0][1], 0.0001, places=15)
        self.assertAlmostEqual(matrix[1][0], 0.0001, places=15)

    def test_correlation_matrix_matches_the_hand_table(self) -> None:
        # 0.0001 / (0.02 * 0.03) = 1/6.
        matrix = correlation_matrix((SERIES_A, SERIES_B))
        self.assertEqual(matrix[0][0], 1.0)
        self.assertEqual(matrix[1][1], 1.0)
        self.assertAlmostEqual(matrix[0][1], 1.0 / 6.0, places=12)
        self.assertEqual(round(matrix[0][1], 6), 0.166667)
        self.assertEqual(matrix[0][1], matrix[1][0])

    def test_portfolio_return_series_matches_the_hand_table(self) -> None:
        combined = PORTFOLIO.return_series((SERIES_A, SERIES_B))
        expected = (0.024, 0.006, -0.024, 0.026, 0.014, 0.002)
        self.assertEqual(len(combined), len(expected))
        for value, target in zip(combined, expected, strict=True):
            self.assertAlmostEqual(value, target, places=15)

    def test_portfolio_volatility_matches_the_hand_table(self) -> None:
        # w'Sw = 0.000144 + 0.000048 + 0.000144 = 0.000336, times 252.
        value = PORTFOLIO.annualized_volatility((SERIES_A, SERIES_B))
        self.assertAlmostEqual(value, math.sqrt(0.000336 * 252), places=12)
        self.assertEqual(round(value, 6), 0.290985)

    def test_risk_contributions_match_the_hand_table(self) -> None:
        # (Sw) = (0.00028, 0.00042); 0.6*0.00028 and 0.4*0.00042 are both
        # 0.000168, so each asset contributes exactly half.
        contributions = PORTFOLIO.risk_contributions((SERIES_A, SERIES_B))
        self.assertEqual(set(contributions), {"AAA", "BBB"})
        self.assertAlmostEqual(contributions["AAA"], 0.5, places=12)
        self.assertAlmostEqual(contributions["BBB"], 0.5, places=12)

    def test_diversification_benefit_matches_the_hand_table(self) -> None:
        # Weighted vols: 0.6*0.02 + 0.4*0.03 = 0.024 daily, annualized,
        # minus the portfolio's 0.290985.
        value = PORTFOLIO.diversification_benefit((SERIES_A, SERIES_B))
        self.assertAlmostEqual(value, 0.024 * math.sqrt(252) - math.sqrt(0.084672), places=12)
        self.assertEqual(round(value, 6), 0.090004)


class ConventionTests(unittest.TestCase):
    def test_covariance_matrix_is_symmetric_with_sample_variances_on_the_diagonal(self) -> None:
        trio = (SERIES_A, SERIES_B, SERIES_C)
        matrix = covariance_matrix(trio)
        for i in range(3):
            for j in range(3):
                self.assertEqual(matrix[i][j], matrix[j][i])
        for item, diagonal in zip(trio, (matrix[0][0], matrix[1][1], matrix[2][2]), strict=True):
            returns = item.simple_returns()
            mean = sum(returns) / len(returns)
            variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
            self.assertAlmostEqual(diagonal, variance, places=15)

    def test_correlation_matrix_has_unit_diagonal_and_bounded_entries(self) -> None:
        matrix = correlation_matrix((SERIES_A, SERIES_B, SERIES_C))
        for i in range(3):
            self.assertEqual(matrix[i][i], 1.0)
            for j in range(3):
                self.assertEqual(matrix[i][j], matrix[j][i])
                self.assertGreaterEqual(matrix[i][j], -1.0)
                self.assertLessEqual(matrix[i][j], 1.0)

    def test_portfolio_volatility_agrees_with_its_return_series(self) -> None:
        # var(w . r) = w'Sw, so the metrics-module volatility of the
        # portfolio return series must match the matrix route.
        direct = PORTFOLIO.annualized_volatility((SERIES_A, SERIES_B))
        via_returns = annualized_volatility(PORTFOLIO.return_series((SERIES_A, SERIES_B)))
        self.assertAlmostEqual(direct, via_returns, places=12)

    def test_a_weight_of_one_recovers_the_single_asset_volatility(self) -> None:
        solo = Portfolio(names=("AAA", "BBB"), weights=(1.0, 0.0))
        value = solo.annualized_volatility((SERIES_A, SERIES_B))
        self.assertAlmostEqual(value, annualized_volatility(SERIES_A.simple_returns()), places=12)

    def test_risk_contributions_sum_to_one(self) -> None:
        for weights in ((0.6, 0.4), (0.5, 0.5), (0.9, 0.1), (1.5, -0.5), (-0.25, 1.25)):
            portfolio = Portfolio(names=("AAA", "BBB"), weights=weights)
            contributions = portfolio.risk_contributions((SERIES_A, SERIES_B))
            self.assertAlmostEqual(math.fsum(contributions.values()), 1.0, places=15)

    def test_diversification_is_strict_below_perfect_correlation(self) -> None:
        # Correlation is 1/6 < 1, so the portfolio volatility must sit
        # strictly below the weighted sum of individual volatilities.
        weighted_sum = 0.6 * annualized_volatility(SERIES_A.simple_returns()) + (
            0.4 * annualized_volatility(SERIES_B.simple_returns())
        )
        value = PORTFOLIO.annualized_volatility((SERIES_A, SERIES_B))
        self.assertLess(value, weighted_sum)
        self.assertGreater(PORTFOLIO.diversification_benefit((SERIES_A, SERIES_B)), 0.0)

    def test_diversification_vanishes_under_perfect_correlation(self) -> None:
        # A series whose returns are exactly twice A's is perfectly
        # correlated with it; the inequality collapses to equality.
        doubled = series_from_returns(
            "DDD", 80.0, tuple(2.0 * value for value in SERIES_A.simple_returns())
        )
        matrix = correlation_matrix((SERIES_A, doubled))
        self.assertAlmostEqual(matrix[0][1], 1.0, places=12)
        self.assertLessEqual(matrix[0][1], 1.0)
        portfolio = Portfolio(names=("AAA", "DDD"), weights=(0.7, 0.3))
        self.assertAlmostEqual(
            portfolio.diversification_benefit((SERIES_A, doubled)), 0.0, places=12
        )

    def test_has_short_positions_flags_negative_weights_only(self) -> None:
        self.assertFalse(PORTFOLIO.has_short_positions)
        self.assertFalse(Portfolio(names=("AAA", "BBB"), weights=(1.0, 0.0)).has_short_positions)
        self.assertTrue(Portfolio(names=("AAA", "BBB"), weights=(1.5, -0.5)).has_short_positions)

    def test_short_portfolios_still_decompose(self) -> None:
        # Shorts are flagged, not forbidden: every number still computes
        # and the contributions still sum to one.
        short = Portfolio(names=("AAA", "BBB"), weights=(1.5, -0.5))
        self.assertGreater(short.annualized_volatility((SERIES_A, SERIES_B)), 0.0)
        contributions = short.risk_contributions((SERIES_A, SERIES_B))
        self.assertAlmostEqual(math.fsum(contributions.values()), 1.0, places=15)


class ValidationTests(unittest.TestCase):
    def test_matrices_require_at_least_two_series(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two"):
            covariance_matrix((SERIES_A,))
        with self.assertRaisesRegex(ValueError, "at least two"):
            correlation_matrix((SERIES_A,))

    def test_matrices_reject_duplicate_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            covariance_matrix((SERIES_A, SERIES_A))

    def test_matrices_reject_series_on_different_date_grids(self) -> None:
        shifted_dates = (*DATES[:-1], "2026-01-14")
        shifted = PriceSeries("BBB", shifted_dates, SERIES_B.prices)
        with self.assertRaisesRegex(ValueError, "not on the same date grid"):
            covariance_matrix((SERIES_A, shifted))
        shorter = PriceSeries("BBB", DATES[:4], SERIES_B.prices[:4])
        with self.assertRaisesRegex(ValueError, "not on the same date grid"):
            covariance_matrix((SERIES_A, shorter))

    def test_correlation_rejects_constant_returns(self) -> None:
        flat = PriceSeries("FLT", DATES, (100.0,) * 7)
        matrix = covariance_matrix((SERIES_A, flat))
        self.assertEqual(matrix[1][1], 0.0)
        with self.assertRaisesRegex(ValueError, "constant returns"):
            correlation_matrix((SERIES_A, flat))

    def test_portfolio_requires_at_least_two_weights(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two"):
            Portfolio(names=("AAA",), weights=(1.0,))

    def test_portfolio_rejects_duplicate_and_blank_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            Portfolio(names=("AAA", "AAA"), weights=(0.5, 0.5))
        with self.assertRaisesRegex(ValueError, "nonempty"):
            Portfolio(names=("AAA", "  "), weights=(0.5, 0.5))

    def test_portfolio_rejects_mismatched_lengths(self) -> None:
        with self.assertRaisesRegex(ValueError, "2 names but 3 weights"):
            Portfolio(names=("AAA", "BBB"), weights=(0.5, 0.3, 0.2))

    def test_portfolio_rejects_nonfinite_and_boolean_weights(self) -> None:
        for bad in (math.nan, math.inf, -math.inf):
            with self.assertRaisesRegex(ValueError, "finite"):
                Portfolio(names=("AAA", "BBB"), weights=(bad, 0.5))
        with self.assertRaisesRegex(ValueError, "not a number"):
            Portfolio(names=("AAA", "BBB"), weights=(True, 0.0))

    def test_portfolio_rejects_weights_that_do_not_sum_to_one(self) -> None:
        for weights in ((0.6, 0.3), (0.6, 0.5), (60.0, 40.0), (0.0, 0.0)):
            with self.assertRaisesRegex(ValueError, "must sum to 1"):
                Portfolio(names=("AAA", "BBB"), weights=weights)

    def test_weight_sums_inside_the_tolerance_are_accepted(self) -> None:
        thirds = Portfolio(names=("AAA", "BBB", "CCC"), weights=(1 / 3, 1 / 3, 1 / 3))
        self.assertFalse(thirds.has_short_positions)

    def test_portfolio_rejects_unknown_series_names(self) -> None:
        stranger = Portfolio(names=("AAA", "ZZZ"), weights=(0.6, 0.4))
        with self.assertRaisesRegex(ValueError, r"\['ZZZ'\] match no supplied series"):
            stranger.annualized_volatility((SERIES_A, SERIES_B))

    def test_portfolio_rejects_series_it_has_no_weight_for(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\['CCC'\] have no weight"):
            PORTFOLIO.risk_contributions((SERIES_A, SERIES_B, SERIES_C))

    def test_portfolio_methods_reject_misaligned_series(self) -> None:
        shorter = PriceSeries("BBB", DATES[:4], SERIES_B.prices[:4])
        with self.assertRaisesRegex(ValueError, "not on the same date grid"):
            PORTFOLIO.return_series((SERIES_A, shorter))

    def test_zero_variance_portfolios_have_no_risk_contributions(self) -> None:
        flat_one = PriceSeries("FL1", DATES, (100.0,) * 7)
        flat_two = PriceSeries("FL2", DATES, (50.0,) * 7)
        still = Portfolio(names=("FL1", "FL2"), weights=(0.5, 0.5))
        self.assertEqual(still.annualized_volatility((flat_one, flat_two)), 0.0)
        with self.assertRaisesRegex(ValueError, "zero-variance"):
            still.risk_contributions((flat_one, flat_two))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
