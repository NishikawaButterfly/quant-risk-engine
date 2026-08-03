from __future__ import annotations

import math
import unittest
from collections.abc import Sequence
from itertools import pairwise

from quantrisk.frontier import (
    annualized_mean_returns,
    efficient_frontier,
    minimum_variance_portfolio,
)
from quantrisk.portfolio import (
    CONDITION_NUMBER_REFUSE_LIMIT,
    CONDITION_NUMBER_WARN_LIMIT,
    covariance_matrix,
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

# The hand-worked fixture from docs/methodology.md: the daily covariance
# matrix [[0.0004, 0.0001], [0.0001, 0.0009]] with supplied expected
# annual returns (0.05, 0.10). By hand, Sigma^-1 1 is proportional to
# (0.0008, 0.0003), so the unconstrained minimum-variance weights are
# (8/11, 3/11), and the target-0.08 frontier point is pinned by the two
# equality constraints at weights (0.4, 0.6).
SIGMA = ((0.0004, 0.0001), (0.0001, 0.0009))
MU = (0.05, 0.10)

RETURNS_A = (0.02, -0.01, -0.02, 0.02, 0.03, 0.02)
RETURNS_B = (0.03, 0.03, -0.03, 0.035, -0.01, -0.025)

# Three assets, the third with a deeply negative expected return.
SIGMA_THREE = (
    (0.0004, 0.0001, 0.00005),
    (0.0001, 0.0009, 0.0001),
    (0.00005, 0.0001, 0.0016),
)
MU_THREE = (0.05, 0.10, -0.50)

# The middle asset is highly correlated with the first but much more
# volatile, so the unconstrained minimum variance shorts it. Excluding
# it leaves exactly the two-asset hand fixture in rows/columns 0 and 2.
SIGMA_SHORT = (
    (0.0004, 0.00055, 0.0001),
    (0.00055, 0.0009, 0.0001),
    (0.0001, 0.0001, 0.0009),
)


def series_from_returns(name: str, start: float, returns: Sequence[float]) -> PriceSeries:
    prices = [start]
    for value in returns:
        prices.append(prices[-1] * (1.0 + value))
    return PriceSeries(name=name, dates=DATES, prices=tuple(prices))


SERIES_A = series_from_returns("AAA", 100.0, RETURNS_A)
SERIES_B = series_from_returns("BBB", 50.0, RETURNS_B)


class HandTableTests(unittest.TestCase):
    def test_minimum_variance_weights_match_the_hand_derivation(self) -> None:
        # Sigma^-1 1 = (16000/7, 6000/7); normalized, (8/11, 3/11).
        weights = minimum_variance_portfolio(SIGMA)
        self.assertEqual(len(weights), 2)
        self.assertAlmostEqual(weights[0], 8.0 / 11.0, places=12)
        self.assertAlmostEqual(weights[1], 3.0 / 11.0, places=12)
        self.assertEqual(round(weights[0], 6), 0.727273)
        self.assertEqual(round(weights[1], 6), 0.272727)

    def test_long_only_solver_agrees_with_the_closed_form(self) -> None:
        # The unconstrained solution is already long-only, so SLSQP must
        # land on the closed-form weights.
        closed = minimum_variance_portfolio(SIGMA)
        solved = minimum_variance_portfolio(SIGMA, long_only=True)
        for closed_value, solved_value in zip(closed, solved, strict=True):
            self.assertAlmostEqual(closed_value, solved_value, places=10)

    def test_frontier_point_at_the_hand_target(self) -> None:
        # Two assets and two equality constraints pin the weights:
        # w_A = (0.10 - 0.08) / (0.10 - 0.05) = 0.4, variance 0.000436.
        (point,) = efficient_frontier(MU, SIGMA, (0.08,))
        self.assertAlmostEqual(point.weights[0], 0.4, places=10)
        self.assertAlmostEqual(point.weights[1], 0.6, places=10)
        self.assertAlmostEqual(point.expected_return, 0.08, places=12)
        self.assertAlmostEqual(point.volatility, math.sqrt(0.000436 * 252), places=12)
        self.assertEqual(round(point.volatility, 6), 0.331469)

    def test_frontier_at_the_minimum_variance_return_recovers_the_minimum(self) -> None:
        # The minimum-variance portfolio returns (8*0.05 + 3*0.10)/11
        # = 7/110; targeting it must reproduce the closed form, with
        # daily variance 7/22000.
        (point,) = efficient_frontier(MU, SIGMA, (7.0 / 110.0,))
        self.assertAlmostEqual(point.weights[0], 8.0 / 11.0, places=9)
        self.assertAlmostEqual(point.weights[1], 3.0 / 11.0, places=9)
        self.assertAlmostEqual(point.volatility, math.sqrt(7.0 / 22000.0 * 252.0), places=12)
        self.assertEqual(round(point.volatility, 6), 0.283164)

    def test_annualized_mean_returns_match_the_hand_fixture(self) -> None:
        # Mean daily returns 0.01 and 0.005 times 252 — annualized to
        # 252% and 126%, absurd numbers from six days of data, which is
        # the module's own argument for never defaulting to them.
        means = annualized_mean_returns((SERIES_A, SERIES_B))
        self.assertAlmostEqual(means[0], 2.52, places=12)
        self.assertAlmostEqual(means[1], 1.26, places=12)

    def test_the_covariance_pipeline_feeds_the_frontier(self) -> None:
        # covariance_matrix over the fixture series is exactly SIGMA, so
        # the whole pipeline reproduces the hand-derived weights.
        weights = minimum_variance_portfolio(covariance_matrix((SERIES_A, SERIES_B)))
        self.assertAlmostEqual(weights[0], 8.0 / 11.0, places=12)
        self.assertAlmostEqual(weights[1], 3.0 / 11.0, places=12)


class PropertyTests(unittest.TestCase):
    def test_frontier_volatility_is_convex_in_the_target(self) -> None:
        # Volatility falls until the global minimum-variance return
        # (7/110 = 0.0636...) and rises after it: the differences change
        # sign at most once, from negative to positive.
        targets = tuple(0.05 + 0.005 * step for step in range(11))
        points = efficient_frontier(MU, SIGMA, targets)
        differences = [
            later.volatility - earlier.volatility for earlier, later in pairwise(points)
        ]
        first_rise = next(index for index, diff in enumerate(differences) if diff > 0.0)
        for diff in differences[:first_rise]:
            self.assertLessEqual(diff, 0.0)
        for diff in differences[first_rise:]:
            self.assertGreaterEqual(diff, 0.0)

    def test_the_minimum_variance_point_has_the_lowest_volatility(self) -> None:
        weights = minimum_variance_portfolio(SIGMA)
        minimum = math.sqrt(7.0 / 22000.0 * 252.0)
        targets = (*(0.05 + 0.005 * step for step in range(11)), 7.0 / 110.0)
        points = efficient_frontier(MU, SIGMA, targets)
        self.assertAlmostEqual(points[-1].volatility, minimum, places=12)
        for point in points[:-1]:
            self.assertGreaterEqual(point.volatility, points[-1].volatility)
        self.assertEqual(len(weights), len(points[-1].weights))

    def test_two_asset_frontier_weights_are_pinned_by_the_constraints(self) -> None:
        # With two assets the equality constraints leave nothing to
        # minimize: w_A = (0.10 - target) / 0.05 analytically.
        targets = (0.055, 0.07, 0.095)
        points = efficient_frontier(MU, SIGMA, targets)
        for target, point in zip(targets, points, strict=True):
            self.assertAlmostEqual(point.weights[0], (0.10 - target) / 0.05, places=9)
            self.assertAlmostEqual(point.expected_return, target, places=12)

    def test_unconstrained_targets_beyond_the_asset_range_use_leverage(self) -> None:
        # 0.05 w_A + 0.10 w_B = 0.12 with w_A + w_B = 1 needs
        # w = (-0.4, 1.4): a short position, legal without bounds.
        (point,) = efficient_frontier(MU, SIGMA, (0.12,))
        self.assertAlmostEqual(point.weights[0], -0.4, places=9)
        self.assertAlmostEqual(point.weights[1], 1.4, places=9)
        self.assertAlmostEqual(math.fsum(point.weights), 1.0, places=12)

    def test_long_only_weights_are_never_negative_and_sum_to_one(self) -> None:
        targets = (-0.4, -0.2, 0.0, 0.05, 0.09)
        points = efficient_frontier(MU_THREE, SIGMA_THREE, targets, long_only=True)
        for target, point in zip(targets, points, strict=True):
            for weight in point.weights:
                self.assertGreaterEqual(weight, 0.0)
                self.assertLessEqual(weight, 1.0 + 1e-9)
            self.assertAlmostEqual(math.fsum(point.weights), 1.0, places=9)
            self.assertAlmostEqual(point.expected_return, target, places=9)

    def test_a_deeply_negative_asset_gets_no_weight_at_high_targets(self) -> None:
        # At a target near the best asset's 0.10, holding the -50% asset
        # would only force more leverage elsewhere; long-only it drops
        # to exactly zero.
        (point,) = efficient_frontier(MU_THREE, SIGMA_THREE, (0.09,), long_only=True)
        self.assertEqual(point.weights[2], 0.0)
        self.assertAlmostEqual(point.weights[0] + point.weights[1], 1.0, places=12)

    def test_long_only_never_beats_the_unconstrained_frontier(self) -> None:
        targets = (0.0, 0.05, 0.09)
        bounded = efficient_frontier(MU_THREE, SIGMA_THREE, targets, long_only=True)
        free = efficient_frontier(MU_THREE, SIGMA_THREE, targets)
        for bounded_point, free_point in zip(bounded, free, strict=True):
            self.assertGreaterEqual(bounded_point.volatility + 1e-12, free_point.volatility)

    def test_long_only_exclusion_falls_back_to_the_remaining_assets(self) -> None:
        # Unconstrained, the correlated volatile asset is shorted.
        # Long-only it drops to zero, and what remains is exactly the
        # two-asset hand fixture — weights (8/11, 0, 3/11).
        unconstrained = minimum_variance_portfolio(SIGMA_SHORT)
        self.assertLess(unconstrained[1], 0.0)
        bounded = minimum_variance_portfolio(SIGMA_SHORT, long_only=True)
        self.assertAlmostEqual(bounded[0], 8.0 / 11.0, places=9)
        self.assertAlmostEqual(bounded[1], 0.0, places=9)
        self.assertAlmostEqual(bounded[2], 3.0 / 11.0, places=9)

    def test_equal_expected_returns_admit_their_common_value(self) -> None:
        # The return constraint is redundant, so the frontier point is
        # simply the minimum-variance portfolio.
        (point,) = efficient_frontier((0.07, 0.07), SIGMA, (0.07,))
        self.assertAlmostEqual(point.weights[0], 8.0 / 11.0, places=9)
        self.assertAlmostEqual(point.weights[1], 3.0 / 11.0, places=9)


class ValidationTests(unittest.TestCase):
    def test_covariance_must_have_at_least_two_assets(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two"):
            minimum_variance_portfolio(((0.0004,),))

    def test_covariance_must_be_square(self) -> None:
        with self.assertRaisesRegex(ValueError, "square"):
            minimum_variance_portfolio(((0.0004, 0.0001), (0.0001, 0.0009, 0.0)))

    def test_covariance_entries_must_be_finite_numbers(self) -> None:
        for bad in (math.nan, math.inf):
            with self.assertRaisesRegex(ValueError, "finite"):
                minimum_variance_portfolio(((0.0004, bad), (bad, 0.0009)))
        with self.assertRaisesRegex(ValueError, "not a number"):
            minimum_variance_portfolio(((True, 0.0001), (0.0001, 0.0009)))

    def test_covariance_must_be_symmetric(self) -> None:
        with self.assertRaisesRegex(ValueError, "symmetric"):
            minimum_variance_portfolio(((0.0004, 0.0002), (0.0001, 0.0009)))

    def test_covariance_rejects_negative_variances(self) -> None:
        with self.assertRaisesRegex(ValueError, "negative"):
            minimum_variance_portfolio(((-0.0004, 0.0001), (0.0001, 0.0009)))

    def test_singular_covariance_is_rejected(self) -> None:
        # The second row is exactly half the first: det = 0, condition
        # number infinite. The conditioning gate refuses it at entry,
        # before any solve could run.
        with self.assertRaisesRegex(ValueError, "singular"):
            minimum_variance_portfolio(((0.0004, 0.0002), (0.0002, 0.0001)))

    def test_indefinite_covariance_is_rejected_at_entry(self) -> None:
        # Eigenvalues -6.57e-5 and 1.07e-3: not a covariance matrix.
        # Formerly this reached the closed form and failed only at the
        # 1'Sigma^-1 1 > 0 guard; the eigenvalue validation now rejects
        # it before any solve, naming the offending eigenvalue.
        with self.assertRaisesRegex(ValueError, "not positive semidefinite.*smallest eigenvalue"):
            minimum_variance_portfolio(((0.0001, 0.0004), (0.0004, 0.0009)))

    def test_the_frontier_rejects_an_indefinite_covariance_at_entry(self) -> None:
        # The same eigenvalues with the off-diagonal sign flipped.
        # Formerly this survived until a portfolio variance came out
        # negative near the 0.064 target; it is now rejected before the
        # solver ever sees it. The variance guard stays in the code as
        # defense in depth.
        bad = ((0.0001, -0.0004), (-0.0004, 0.0009))
        with self.assertRaisesRegex(ValueError, "not positive semidefinite.*smallest eigenvalue"):
            efficient_frontier(MU, bad, (0.064,))

    def test_expected_returns_must_match_the_matrix(self) -> None:
        with self.assertRaisesRegex(ValueError, "3 expected returns"):
            efficient_frontier((0.05, 0.10, 0.15), SIGMA, (0.08,))
        with self.assertRaisesRegex(ValueError, "finite"):
            efficient_frontier((0.05, math.nan), SIGMA, (0.08,))

    def test_targets_must_be_finite_and_nonempty(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one target"):
            efficient_frontier(MU, SIGMA, ())
        with self.assertRaisesRegex(ValueError, "finite"):
            efficient_frontier(MU, SIGMA, (0.08, math.inf))

    def test_long_only_rejects_targets_outside_the_asset_range(self) -> None:
        for target in (0.04, 0.11):
            with self.assertRaisesRegex(ValueError, "unreachable long-only"):
                efficient_frontier(MU, SIGMA, (target,), long_only=True)

    def test_equal_expected_returns_reach_only_their_common_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "no combination"):
            efficient_frontier((0.07, 0.07), SIGMA, (0.08,))

    def test_annualized_mean_returns_requires_aligned_series(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two"):
            annualized_mean_returns((SERIES_A,))
        shorter = PriceSeries("BBB", DATES[:4], SERIES_B.prices[:4])
        with self.assertRaisesRegex(ValueError, "not on the same date grid"):
            annualized_mean_returns((SERIES_A, shorter))


# Two equal-variance assets at correlation rho have eigenvalues
# v(1 - rho) and v(1 + rho), so cond2 = (1 + rho) / (1 - rho): 2e9 at
# rho = 1 - 1e-9 (warn band) and 2e13 at rho = 1 - 1e-13 (refuse band).
NEAR_SINGULAR = (
    (0.0004, 0.0004 * (1.0 - 1e-9)),
    (0.0004 * (1.0 - 1e-9), 0.0004),
)
BEYOND_REFUSAL = (
    (0.0004, 0.0004 * (1.0 - 1e-13)),
    (0.0004 * (1.0 - 1e-13), 0.0004),
)


class ConditioningTests(unittest.TestCase):
    def test_a_well_conditioned_frontier_point_carries_its_condition_number(self) -> None:
        # SIGMA's eigenvalues are (13 +/- sqrt(29))e-4 / 2, so cond2 is
        # (13 + sqrt(29)) / (13 - sqrt(29)) = (99 + 13 sqrt(29)) / 70.
        (point,) = efficient_frontier(MU, SIGMA, (0.08,))
        expected = (99.0 + 13.0 * math.sqrt(29.0)) / 70.0
        self.assertAlmostEqual(point.condition_number, expected, places=12)
        self.assertIsNone(point.conditioning_warning)

    def test_every_point_of_one_frontier_shares_the_diagnosis(self) -> None:
        targets = (0.055, 0.07, 0.095)
        points = efficient_frontier(MU, SIGMA, targets)
        for point in points:
            self.assertEqual(point.condition_number, points[0].condition_number)
            self.assertIsNone(point.conditioning_warning)

    def test_a_near_singular_covariance_warns_but_proceeds(self) -> None:
        # cond2 = 2e9 sits above the warn limit and below the refuse
        # limit: the solve proceeds — equal variances pin the
        # minimum-variance weights at (0.5, 0.5) — and the warning is
        # carried on the result instead of being printed or dropped.
        # The loose tolerance is the warning made visible: at cond2
        # = 2e9 the solve really does return 0.5 only to ~8 digits.
        weights = minimum_variance_portfolio(NEAR_SINGULAR)
        self.assertAlmostEqual(weights[0], 0.5, places=6)
        self.assertAlmostEqual(weights[1], 0.5, places=6)
        (point,) = efficient_frontier(MU, NEAR_SINGULAR, (0.08,))
        self.assertGreater(point.condition_number, CONDITION_NUMBER_WARN_LIMIT)
        self.assertLess(point.condition_number, CONDITION_NUMBER_REFUSE_LIMIT)
        self.assertIsNotNone(point.conditioning_warning)
        self.assertIn("condition number", point.conditioning_warning or "")

    def test_an_extreme_condition_number_is_refused(self) -> None:
        # cond2 = 2e13 exceeds the refuse limit: solving would amplify
        # relative input noise by thirteen orders of magnitude, so both
        # entry points refuse instead of returning noise as weights.
        with self.assertRaisesRegex(ValueError, "condition number"):
            minimum_variance_portfolio(BEYOND_REFUSAL)
        with self.assertRaisesRegex(ValueError, "condition number"):
            efficient_frontier(MU, BEYOND_REFUSAL, (0.08,))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
