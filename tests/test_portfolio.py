from __future__ import annotations

import math
import unittest
from collections.abc import Sequence

from quantrisk.metrics import annualized_volatility
from quantrisk.portfolio import (
    CONDITION_NUMBER_REFUSE_LIMIT,
    CONDITION_NUMBER_WARN_LIMIT,
    PSD_TOLERANCE,
    Portfolio,
    correlation_matrix,
    covariance_matrix,
    validate_covariance,
)
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

# The hand fixture's covariance matrix: eigenvalues (13 ± sqrt(29))e-4 / 2,
# so its 2-norm condition number is (13 + sqrt(29)) / (13 - sqrt(29))
# = (99 + 13 sqrt(29)) / 70 = 2.414388 — comfortably well conditioned.
SIGMA = ((0.0004, 0.0001), (0.0001, 0.0009))
SIGMA_CONDITION_NUMBER = (99.0 + 13.0 * math.sqrt(29.0)) / 70.0

# Indefinite: eigenvalues -6.5685e-5 and 1.0657e-3. No return series can
# produce this matrix; it must be rejected, not fed to any solver.
INDEFINITE = ((0.0001, -0.0004), (-0.0004, 0.0009))

# Barely indefinite: eigenvalues -5.0e-11 and 8.0e-4. The negative
# eigenvalue sits inside the PSD tolerance floor of
# -PSD_TOLERANCE * max(1, max_eig) = -1e-10. This is the shape a sample
# covariance of near-collinear return series takes after floating-point
# rounding: the true matrix is PSD by construction, but the computed one
# can carry an eigenvalue a hair below zero. Rejecting it would reject
# legitimate data, which is exactly why the tolerance exists.
NEARLY_PSD = ((0.0004, 0.0004), (0.0004, 0.0004 - 1e-10))


class CovarianceValidationTests(unittest.TestCase):
    def test_the_hand_fixture_passes_with_positive_eigenvalues(self) -> None:
        diagnostics = validate_covariance(SIGMA)
        self.assertGreater(diagnostics.smallest_eigenvalue, 0.0)
        self.assertGreater(diagnostics.largest_eigenvalue, diagnostics.smallest_eigenvalue)
        self.assertAlmostEqual(diagnostics.condition_number, SIGMA_CONDITION_NUMBER, places=12)
        self.assertIsNone(diagnostics.conditioning_warning)

    def test_an_indefinite_matrix_is_rejected_naming_its_eigenvalue(self) -> None:
        with self.assertRaisesRegex(ValueError, "not positive semidefinite.*smallest eigenvalue"):
            validate_covariance(INDEFINITE)

    def test_an_asymmetric_matrix_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "symmetric"):
            validate_covariance(((0.0004, 0.0002), (0.0001, 0.0009)))

    def test_a_barely_negative_eigenvalue_inside_the_tolerance_passes(self) -> None:
        # Floating-point covariance of near-collinear series: the true
        # matrix is PSD, the computed eigenvalue is rounding noise below
        # zero. The tolerance floor accepts it instead of rejecting
        # legitimate data; near-singularity is still disclosed through
        # the condition number, never hidden.
        diagnostics = validate_covariance(NEARLY_PSD)
        self.assertLess(diagnostics.smallest_eigenvalue, 0.0)
        floor = PSD_TOLERANCE * max(1.0, diagnostics.largest_eigenvalue)
        self.assertGreaterEqual(diagnostics.smallest_eigenvalue, -floor)

    def test_the_same_matrix_fails_outside_a_tighter_tolerance(self) -> None:
        with self.assertRaisesRegex(ValueError, "smallest eigenvalue"):
            validate_covariance(NEARLY_PSD, tolerance=1e-12)


class ConditioningTests(unittest.TestCase):
    def test_the_fixture_portfolio_is_well_conditioned(self) -> None:
        diagnostics = PORTFOLIO.covariance_diagnostics((SERIES_A, SERIES_B))
        self.assertAlmostEqual(diagnostics.condition_number, SIGMA_CONDITION_NUMBER, places=12)
        self.assertIsNone(diagnostics.conditioning_warning)

    def test_near_collinear_series_carry_a_conditioning_warning(self) -> None:
        # The echo series repeats AAA's returns plus a +/-1e-6 wiggle:
        # correlation 1 - 7.5e-10, condition number 1.33e9 — inside the
        # warn band, outside the refuse band. The numbers still compute
        # (quadratic forms never invert the matrix) but the warning is
        # carried, not hidden.
        echo = series_from_returns(
            "ECH",
            100.0,
            tuple(
                value + wiggle
                for value, wiggle in zip(
                    RETURNS_A, (1e-6, -1e-6, 1e-6, -1e-6, 1e-6, -1e-6), strict=True
                )
            ),
        )
        pair = Portfolio(names=("AAA", "ECH"), weights=(0.5, 0.5))
        diagnostics = pair.covariance_diagnostics((SERIES_A, echo))
        self.assertGreater(diagnostics.condition_number, CONDITION_NUMBER_WARN_LIMIT)
        self.assertLess(diagnostics.condition_number, CONDITION_NUMBER_REFUSE_LIMIT)
        self.assertIsNotNone(diagnostics.conditioning_warning)
        self.assertIn("condition number", diagnostics.conditioning_warning or "")
        self.assertGreater(pair.annualized_volatility((SERIES_A, echo)), 0.0)

    def test_perfectly_collinear_series_report_infinite_conditioning(self) -> None:
        # A series whose returns are exactly twice AAA's makes the
        # covariance exactly singular. Portfolio arithmetic (quadratic
        # forms) still works — the existing diversification test relies
        # on that — so the diagnosis reports an infinite condition
        # number with a warning rather than refusing.
        doubled = series_from_returns(
            "DDD", 80.0, tuple(2.0 * value for value in SERIES_A.simple_returns())
        )
        pair = Portfolio(names=("AAA", "DDD"), weights=(0.7, 0.3))
        diagnostics = pair.covariance_diagnostics((SERIES_A, doubled))
        self.assertEqual(diagnostics.condition_number, math.inf)
        self.assertIsNotNone(diagnostics.conditioning_warning)
        self.assertAlmostEqual(pair.diversification_benefit((SERIES_A, doubled)), 0.0, places=12)


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


def _wiggle_series(name: str, start: float, wiggle: float) -> PriceSeries:
    """Prices alternating between ``start`` and ``start * (1 + wiggle)``.

    The six returns alternate ±wiggle (to a relative float error of
    ~1e-16), so the daily sample stddev is ``wiggle * sqrt(6/5)`` —
    about ``1.1 * wiggle`` — putting the series precisely on either
    side of MIN_VOLATILITY by choice of ``wiggle``.
    """

    prices = tuple(start if index % 2 == 0 else start * (1.0 + wiggle) for index in range(7))
    return PriceSeries(name=name, dates=DATES, prices=prices)


class VarianceToleranceTests(unittest.TestCase):
    """The stddev and variance denominators share one constant.

    The old guards fired only at exactly zero; a sub-tolerance wiggle
    (stddev ~5.5e-13, below MIN_VOLATILITY = 1e-12) was accepted by
    every one of them and produced correlations and risk contributions
    made of float rounding noise. These fixtures sit in that formerly
    accepted region and must now be rejected — and their ten-times
    larger twins must still compute, at every site.
    """

    def test_correlation_rejects_a_series_just_below_the_tolerance(self) -> None:
        nearly_flat = _wiggle_series("NRF", 100.0, 5e-13)
        with self.assertRaisesRegex(ValueError, "'NRF'.*MIN_VOLATILITY"):
            correlation_matrix((SERIES_A, nearly_flat))

    def test_correlation_computes_just_above_the_tolerance(self) -> None:
        barely_alive = _wiggle_series("BRL", 100.0, 5e-12)
        matrix = correlation_matrix((SERIES_A, barely_alive))
        self.assertEqual(matrix[0][0], 1.0)
        self.assertEqual(matrix[1][1], 1.0)
        self.assertTrue(all(math.isfinite(value) for row in matrix for value in row))

    def test_risk_contributions_reject_portfolio_variance_below_the_squared_tolerance(
        self,
    ) -> None:
        # A variance site: the same constant applies squared. Portfolio
        # stddev ~5.5e-13, so its variance ~3e-25 < MIN_VOLATILITY².
        first = _wiggle_series("FL1", 100.0, 5e-13)
        second = _wiggle_series("FL2", 50.0, 5e-13)
        still = Portfolio(names=("FL1", "FL2"), weights=(0.5, 0.5))
        with self.assertRaisesRegex(ValueError, "zero-variance.*MIN_VOLATILITY"):
            still.risk_contributions((first, second))

    def test_risk_contributions_compute_just_above_the_squared_tolerance(self) -> None:
        first = _wiggle_series("FL1", 100.0, 5e-12)
        second = _wiggle_series("FL2", 50.0, 5e-12)
        still = Portfolio(names=("FL1", "FL2"), weights=(0.5, 0.5))
        contributions = still.risk_contributions((first, second))
        self.assertAlmostEqual(math.fsum(contributions.values()), 1.0, places=12)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
