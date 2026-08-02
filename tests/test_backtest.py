from __future__ import annotations

import math
import unittest
from collections.abc import Sequence

from quantrisk.backtest import PolicyWindow, run_backtest
from quantrisk.metrics import annualized_volatility, max_drawdown
from quantrisk.portfolio import Portfolio
from quantrisk.series import PriceSeries

DATES_5 = ("2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09")
DATES_7 = (*DATES_5, "2026-01-12", "2026-01-13")
DATES_8 = (*DATES_7, "2026-01-14")

# The hand-worked cost fixture from docs/methodology.md: A jumps 50%
# on the third day and B never moves, so a 50/50 policy rebalanced
# every 2 trading days drifts to exactly (0.6, 0.4) before its second
# decision. With a 2% cost rate on turnover the value path is
# 1000, 980, 1225, 1220.1, 1220.1.
COST_A = PriceSeries("AAA", DATES_5, (100.0, 100.0, 150.0, 150.0, 150.0))
COST_B = PriceSeries("BBB", DATES_5, (100.0, 100.0, 100.0, 100.0, 100.0))

# The mean-reverting fixture for the peeking test: A's returns come in
# two-day blocks (+10%, +10%, -10%, -10%, ...) while B never moves, so
# on every rebalance the trailing two-day winner is the forward loser.
MEAN_REVERTING_RETURNS = (0.1, 0.1, -0.1, -0.1, 0.1, 0.1, -0.1)


def series_from_returns(name: str, start: float, returns: Sequence[float]) -> PriceSeries:
    prices = [start]
    for value in returns:
        prices.append(prices[-1] * (1.0 + value))
    return PriceSeries(name=name, dates=DATES_8, prices=tuple(prices))


REVERTING_A = series_from_returns("AAA", 100.0, MEAN_REVERTING_RETURNS)
FLAT_B = PriceSeries("BBB", DATES_8, (100.0,) * 8)


class ConstantWeights:
    """The same target weights at every rebalance, window ignored."""

    def __init__(self, weights: tuple[float, ...], min_history_days: int = 1) -> None:
        self.weights = weights
        self.min_history_days = min_history_days

    def __call__(self, window: PolicyWindow) -> tuple[float, ...]:
        return self.weights


class RecordingEqualWeights:
    """Equal weights; keeps every window it is shown for inspection."""

    def __init__(self, min_history_days: int = 1) -> None:
        self.min_history_days = min_history_days
        self.windows: list[PolicyWindow] = []

    def __call__(self, window: PolicyWindow) -> tuple[float, ...]:
        self.windows.append(window)
        count = len(window.names)
        return (1.0 / count,) * count


class LastTwoDayMomentum:
    """Honest: all-in on the best trailing two-day growth, window only."""

    min_history_days = 2

    def __call__(self, window: PolicyWindow) -> tuple[float, ...]:
        growths = []
        for returns in window.simple_returns():
            growth = 1.0
            for value in returns[-2:]:
                growth *= 1.0 + value
            growths.append(growth)
        best = growths.index(max(growths))
        return tuple(1.0 if index == best else 0.0 for index in range(len(growths)))


class PeekingMomentum:
    """Cheat: handed the full series separately, it reads the future.

    It locates the decision date in the full grid — data the window
    does not carry — and goes all-in on the asset with the best
    *forward* growth over the coming holding period. The real policy
    interface cannot express this: see LookAheadTests.
    """

    min_history_days = 2

    def __init__(self, full_series: tuple[PriceSeries, ...], schedule: int) -> None:
        self.full_series = full_series
        self.schedule = schedule

    def __call__(self, window: PolicyWindow) -> tuple[float, ...]:
        grid = self.full_series[0].dates
        today = grid.index(window.decision_date)
        horizon = min(today + self.schedule, len(grid) - 1)
        forward = [item.prices[horizon] / item.prices[today] for item in self.full_series]
        best = forward.index(max(forward))
        return tuple(1.0 if index == best else 0.0 for index in range(len(forward)))


class HandTableTests(unittest.TestCase):
    def run_cost_fixture(self, cost_rate: float) -> tuple[float, ...]:
        result = run_backtest(
            (COST_A, COST_B),
            ConstantWeights((0.5, 0.5)),
            schedule=2,
            cost_rate=cost_rate,
            initial_value=1000.0,
        )
        return result.values.prices

    def test_the_value_path_matches_the_hand_table(self) -> None:
        # Cash 1000; first rebalance costs 0.02 * 1 * 1000 = 20; A's 50%
        # gain lifts 490 + 490 to 735 + 490 = 1225; the second rebalance
        # costs 0.02 * 0.2 * 1225 = 4.9; nothing moves afterwards.
        values = self.run_cost_fixture(0.02)
        self.assertEqual(values[:3], (1000.0, 980.0, 1225.0))
        self.assertAlmostEqual(values[3], 1220.1, places=9)
        self.assertEqual(values[3], values[4])

    def test_the_rebalance_records_match_the_hand_table(self) -> None:
        result = run_backtest(
            (COST_A, COST_B),
            ConstantWeights((0.5, 0.5)),
            schedule=2,
            cost_rate=0.02,
            initial_value=1000.0,
        )
        first, second = result.rebalances
        self.assertEqual(first.date, "2026-01-06")
        self.assertEqual(first.drifted_weights, (0.0, 0.0))
        self.assertEqual(first.target_weights, (0.5, 0.5))
        self.assertEqual(first.turnover, 1.0)
        self.assertEqual(first.cost, 20.0)
        self.assertEqual(second.date, "2026-01-08")
        self.assertEqual(second.drifted_weights, (0.6, 0.4))
        self.assertEqual(second.target_weights, (0.5, 0.5))
        self.assertAlmostEqual(second.turnover, 0.2, places=12)
        self.assertAlmostEqual(second.cost, 4.9, places=12)
        self.assertEqual(round(second.cost, 6), 4.9)

    def test_totals_match_the_hand_table(self) -> None:
        result = run_backtest(
            (COST_A, COST_B),
            ConstantWeights((0.5, 0.5)),
            schedule=2,
            cost_rate=0.02,
            initial_value=1000.0,
        )
        self.assertAlmostEqual(result.total_costs, 24.9, places=12)
        self.assertEqual(round(result.total_costs, 6), 24.9)
        self.assertAlmostEqual(result.total_return, 0.2201, places=12)
        self.assertEqual(round(result.total_return, 6), 0.2201)

    def test_weights_drift_between_rebalances_rather_than_reset(self) -> None:
        # The second record's drifted weights are (0.6, 0.4): nobody
        # traded between the decisions, prices moved. A daily-rebalanced
        # convention would have reported (0.5, 0.5).
        result = run_backtest(
            (COST_A, COST_B),
            ConstantWeights((0.5, 0.5)),
            schedule=2,
            cost_rate=0.0,
            initial_value=1000.0,
        )
        self.assertEqual(result.rebalances[1].drifted_weights, (0.6, 0.4))


class LookAheadTests(unittest.TestCase):
    def test_every_window_ends_the_day_before_its_decision_date(self) -> None:
        # The dedicated invariant test: for every rebalance of a real
        # run, the window's data is exactly the grid sliced strictly
        # before the decision date — its last date is the immediately
        # preceding trading day, and nothing on or after the decision
        # date is present.
        policy = RecordingEqualWeights(min_history_days=2)
        result = run_backtest(
            (REVERTING_A, FLAT_B), policy, schedule=2, cost_rate=0.0, initial_value=1000.0
        )
        grid = REVERTING_A.dates
        self.assertEqual(len(policy.windows), len(result.rebalances))
        self.assertGreaterEqual(len(policy.windows), 3)
        for window, record in zip(policy.windows, result.rebalances, strict=True):
            self.assertEqual(window.decision_date, record.date)
            today = grid.index(window.decision_date)
            self.assertEqual(window.dates, grid[:today])
            self.assertEqual(window.dates[-1], grid[today - 1])
            self.assertEqual(len(window), today)
            self.assertEqual(window.prices, (REVERTING_A.prices[:today], FLAT_B.prices[:today]))

    def test_peeking_strictly_beats_honest_momentum_when_tomorrow_is_knowable(self) -> None:
        # On the mean-reverting fixture the trailing winner is the
        # forward loser, so honest momentum rides A through its losing
        # blocks (final value ~810) while the cheat — which reads the
        # full series outside the window interface — sidesteps every
        # loss and catches the winning block (~1210).
        honest = run_backtest(
            (REVERTING_A, FLAT_B),
            LastTwoDayMomentum(),
            schedule=2,
            cost_rate=0.0,
            initial_value=1000.0,
        )
        cheat = run_backtest(
            (REVERTING_A, FLAT_B),
            PeekingMomentum((REVERTING_A, FLAT_B), schedule=2),
            schedule=2,
            cost_rate=0.0,
            initial_value=1000.0,
        )
        self.assertAlmostEqual(honest.values.prices[-1], 810.0, places=6)
        self.assertAlmostEqual(cheat.values.prices[-1], 1210.0, places=6)
        self.assertLess(honest.values.prices[-1], cheat.values.prices[-1])

    def test_the_window_cannot_express_the_cheat(self) -> None:
        # The cheat's first move is locating "today" in its own copy of
        # the data. From the PolicyWindow alone that data does not
        # exist: the decision date is absent, looking it up raises, and
        # indexing the day at or after the decision raises — the window
        # physically ends the day before.
        policy = RecordingEqualWeights(min_history_days=2)
        run_backtest(
            (REVERTING_A, FLAT_B), policy, schedule=2, cost_rate=0.0, initial_value=1000.0
        )
        grid = REVERTING_A.dates
        for window in policy.windows:
            today = grid.index(window.decision_date)
            self.assertNotIn(window.decision_date, window.dates)
            with self.assertRaises(ValueError):
                window.dates.index(window.decision_date)
            for row in window.prices:
                with self.assertRaises(IndexError):
                    row[today]
                with self.assertRaises(IndexError):
                    row[today + 1]

    def test_windows_that_touch_their_decision_date_cannot_be_constructed(self) -> None:
        # The constructive half of the guarantee: a window whose history
        # reaches its own decision date is rejected outright.
        for decision in ("2026-01-06", "2026-01-05", "2026-01-02"):
            with self.assertRaisesRegex(ValueError, "strictly before"):
                PolicyWindow(
                    decision_date=decision,
                    names=("AAA",),
                    dates=("2026-01-05", "2026-01-06"),
                    prices=((100.0, 101.0),),
                )


class PolicyWindowTests(unittest.TestCase):
    def test_a_valid_window_reports_length_and_returns(self) -> None:
        window = PolicyWindow(
            decision_date="2026-01-08",
            names=("AAA", "BBB"),
            dates=("2026-01-05", "2026-01-06", "2026-01-07"),
            prices=((100.0, 110.0, 99.0), (50.0, 50.0, 60.0)),
        )
        self.assertEqual(len(window), 3)
        returns_a, returns_b = window.simple_returns()
        self.assertAlmostEqual(returns_a[0], 0.1, places=15)
        self.assertAlmostEqual(returns_a[1], 99.0 / 110.0 - 1.0, places=15)
        self.assertEqual(returns_b[0], 0.0)
        self.assertAlmostEqual(returns_b[1], 0.2, places=15)

    def test_a_one_day_window_has_no_returns_yet(self) -> None:
        window = PolicyWindow(
            decision_date="2026-01-06",
            names=("AAA",),
            dates=("2026-01-05",),
            prices=((100.0,),),
        )
        self.assertEqual(len(window), 1)
        self.assertEqual(window.simple_returns(), ((),))

    def test_windows_reject_empty_or_duplicate_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one asset"):
            PolicyWindow("2026-01-06", (), ("2026-01-05",), ())
        with self.assertRaisesRegex(ValueError, "unique"):
            PolicyWindow("2026-01-06", ("AAA", "AAA"), ("2026-01-05",), ((100.0,), (100.0,)))

    def test_windows_reject_mismatched_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "2 names but 1 price rows"):
            PolicyWindow("2026-01-06", ("AAA", "BBB"), ("2026-01-05",), ((100.0,),))
        with self.assertRaisesRegex(ValueError, "2 entries for 1 dates"):
            PolicyWindow("2026-01-06", ("AAA",), ("2026-01-05",), ((100.0, 101.0),))

    def test_windows_reject_an_empty_history(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one observed day"):
            PolicyWindow("2026-01-06", ("AAA",), (), ((),))

    def test_windows_reject_unsorted_history_and_malformed_dates(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            PolicyWindow(
                "2026-01-08",
                ("AAA",),
                ("2026-01-06", "2026-01-05"),
                ((100.0, 101.0),),
            )
        with self.assertRaisesRegex(ValueError, "ISO|canonical"):
            PolicyWindow("2026-01-06", ("AAA",), ("2026-1-05",), ((100.0,),))


class PropertyTests(unittest.TestCase):
    def test_constant_prices_and_zero_cost_preserve_value_exactly(self) -> None:
        flat_a = PriceSeries("AAA", DATES_7, (100.0,) * 7)
        flat_b = PriceSeries("BBB", DATES_7, (50.0,) * 7)
        result = run_backtest(
            (flat_a, flat_b),
            ConstantWeights((0.5, 0.5)),
            schedule=1,
            cost_rate=0.0,
            initial_value=1000.0,
        )
        self.assertEqual(result.values.prices, (1000.0,) * 7)
        self.assertEqual(result.total_return, 0.0)
        self.assertEqual(result.annualized_return, 0.0)
        self.assertEqual(result.annualized_volatility, 0.0)
        self.assertEqual(result.max_drawdown.depth, 0.0)
        self.assertEqual(result.total_costs, 0.0)
        self.assertEqual(len(result.rebalances), 6)

    def test_costs_strictly_reduce_the_final_value(self) -> None:
        def final_value(cost_rate: float) -> float:
            result = run_backtest(
                (COST_A, COST_B),
                ConstantWeights((0.5, 0.5)),
                schedule=2,
                cost_rate=cost_rate,
                initial_value=1000.0,
            )
            return result.values.prices[-1]

        self.assertEqual(final_value(0.0), 1250.0)
        self.assertLess(final_value(0.001), final_value(0.0))
        self.assertLess(final_value(0.02), final_value(0.001))

    def test_single_asset_buy_and_hold_reproduces_the_asset_total_return(self) -> None:
        # Power-of-two price ratios make every drift step exact, and the
        # first two prices are equal, so the backtest's total return
        # equals the asset's own total return to the digit: 3.0.
        asset = PriceSeries("AAA", DATES_5, (100.0, 100.0, 200.0, 50.0, 400.0))
        result = run_backtest(
            (asset,),
            ConstantWeights((1.0,)),
            schedule=10,
            cost_rate=0.0,
            initial_value=1000.0,
        )
        self.assertEqual(result.values.prices, (1000.0, 1000.0, 2000.0, 500.0, 4000.0))
        self.assertEqual(result.total_return, asset.prices[-1] / asset.prices[0] - 1.0)
        self.assertEqual(result.total_return, 3.0)
        self.assertEqual(len(result.rebalances), 1)

    def test_a_daily_schedule_at_zero_cost_matches_the_daily_rebalanced_convention(
        self,
    ) -> None:
        # With schedule=1 and no costs, drifting for a single day and
        # rebalancing is the same as weighting each day's returns — the
        # portfolio.py convention — so the value path must compound the
        # Portfolio return series exactly.
        series_a = series_from_returns("AAA", 100.0, (0.02, -0.01, -0.02, 0.02, 0.03, 0.02, 0.01))
        series_b = series_from_returns(
            "BBB", 50.0, (0.03, 0.03, -0.03, 0.035, -0.01, -0.025, 0.02)
        )
        result = run_backtest(
            (series_a, series_b),
            ConstantWeights((0.6, 0.4)),
            schedule=1,
            cost_rate=0.0,
            initial_value=1000.0,
        )
        portfolio_returns = Portfolio(names=("AAA", "BBB"), weights=(0.6, 0.4)).return_series(
            (series_a, series_b)
        )
        expected = 1000.0
        # Invested from day 1; the first portfolio return is already
        # spent moving from day 0 to day 1, in cash.
        for value in portfolio_returns[1:]:
            expected *= 1.0 + value
        self.assertAlmostEqual(result.values.prices[-1], expected, places=9)

    def test_warmup_days_hold_the_initial_value_in_cash(self) -> None:
        policy = RecordingEqualWeights(min_history_days=3)
        result = run_backtest(
            (REVERTING_A, FLAT_B), policy, schedule=2, cost_rate=0.0, initial_value=1000.0
        )
        self.assertEqual(result.values.prices[:3], (1000.0, 1000.0, 1000.0))
        self.assertEqual(result.rebalances[0].date, DATES_8[3])

    def test_result_metrics_reuse_the_metrics_module_on_the_value_path(self) -> None:
        result = run_backtest(
            (COST_A, COST_B),
            ConstantWeights((0.5, 0.5)),
            schedule=2,
            cost_rate=0.02,
            initial_value=1000.0,
        )
        self.assertEqual(result.values.name, "backtest")
        self.assertEqual(result.values.dates, DATES_5)
        self.assertEqual(result.max_drawdown, max_drawdown(result.values))
        self.assertEqual(
            result.annualized_volatility, annualized_volatility(result.values.simple_returns())
        )
        periods = len(result.values) - 1
        self.assertEqual(
            result.annualized_return,
            (1.0 + result.total_return) ** (252 / periods) - 1.0,
        )


class ValidationTests(unittest.TestCase):
    def test_a_backtest_needs_at_least_one_series(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one price series"):
            run_backtest((), ConstantWeights((1.0,)), schedule=1, cost_rate=0.0)

    def test_duplicate_names_and_misaligned_grids_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            run_backtest((COST_A, COST_A), ConstantWeights((0.5, 0.5)), schedule=1, cost_rate=0.0)
        shorter = PriceSeries("BBB", DATES_5[:4], COST_B.prices[:4])
        with self.assertRaisesRegex(ValueError, "not on the same date grid"):
            run_backtest((COST_A, shorter), ConstantWeights((0.5, 0.5)), schedule=1, cost_rate=0.0)

    def test_the_schedule_must_be_a_positive_integer(self) -> None:
        policy = ConstantWeights((0.5, 0.5))
        with self.assertRaisesRegex(ValueError, "schedule must be at least 1"):
            run_backtest((COST_A, COST_B), policy, schedule=0, cost_rate=0.0)
        with self.assertRaisesRegex(ValueError, "schedule must be an integer"):
            run_backtest((COST_A, COST_B), policy, schedule=True, cost_rate=0.0)

    def test_the_cost_rate_must_be_a_fraction_below_one(self) -> None:
        policy = ConstantWeights((0.5, 0.5))
        for bad in (-0.01, 1.0, 1.5):
            with self.assertRaisesRegex(ValueError, r"cost_rate must lie in \[0, 1\)"):
                run_backtest((COST_A, COST_B), policy, schedule=1, cost_rate=bad)
        with self.assertRaisesRegex(ValueError, "finite"):
            run_backtest((COST_A, COST_B), policy, schedule=1, cost_rate=math.nan)
        with self.assertRaisesRegex(ValueError, "not a number"):
            run_backtest((COST_A, COST_B), policy, schedule=1, cost_rate=True)

    def test_the_initial_value_must_be_positive_and_finite(self) -> None:
        policy = ConstantWeights((0.5, 0.5))
        for bad in (0.0, -1000.0):
            with self.assertRaisesRegex(ValueError, "initial_value must be positive"):
                run_backtest(
                    (COST_A, COST_B), policy, schedule=1, cost_rate=0.0, initial_value=bad
                )
        with self.assertRaisesRegex(ValueError, "finite"):
            run_backtest(
                (COST_A, COST_B), policy, schedule=1, cost_rate=0.0, initial_value=math.inf
            )

    def test_the_minimum_history_must_be_a_positive_integer(self) -> None:
        with self.assertRaisesRegex(ValueError, "min_history_days must be at least 1"):
            run_backtest(
                (COST_A, COST_B),
                ConstantWeights((0.5, 0.5), min_history_days=0),
                schedule=1,
                cost_rate=0.0,
            )

    def test_a_warmup_consuming_the_whole_series_is_rejected(self) -> None:
        for days in (5, 6):
            with self.assertRaisesRegex(ValueError, "warmup leaves nothing to trade"):
                run_backtest(
                    (COST_A, COST_B),
                    ConstantWeights((0.5, 0.5), min_history_days=days),
                    schedule=1,
                    cost_rate=0.0,
                )

    def test_policy_weights_are_validated_like_portfolio_weights(self) -> None:
        with self.assertRaisesRegex(ValueError, "returned 3 weights for 2 assets"):
            run_backtest(
                (COST_A, COST_B),
                ConstantWeights((0.5, 0.3, 0.2)),
                schedule=1,
                cost_rate=0.0,
            )
        with self.assertRaisesRegex(ValueError, "must sum to 1"):
            run_backtest((COST_A, COST_B), ConstantWeights((0.6, 0.3)), schedule=1, cost_rate=0.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            run_backtest(
                (COST_A, COST_B),
                ConstantWeights((math.nan, 1.0)),
                schedule=1,
                cost_rate=0.0,
            )
        with self.assertRaisesRegex(ValueError, "not a number"):
            run_backtest((COST_A, COST_B), ConstantWeights((True, 0.0)), schedule=1, cost_rate=0.0)

    def test_a_cost_that_would_consume_the_portfolio_is_rejected(self) -> None:
        # Weights (5, -4) sum to one but carry turnover 9 out of cash;
        # at a 20% cost rate the charge is 1.8 times the whole value.
        with self.assertRaisesRegex(ValueError, "consume the whole portfolio"):
            run_backtest(
                (COST_A, COST_B),
                ConstantWeights((5.0, -4.0)),
                schedule=1,
                cost_rate=0.2,
            )

    def test_short_positions_that_drift_the_value_below_zero_are_rejected(self) -> None:
        # Long 3x a flat asset, short 2x a doubling one: the short leg
        # overwhelms the portfolio on the next drift step.
        flat = PriceSeries("AAA", DATES_5[:4], (100.0, 100.0, 100.0, 100.0))
        doubling = PriceSeries("BBB", DATES_5[:4], (100.0, 100.0, 200.0, 400.0))
        with self.assertRaisesRegex(ValueError, "zero or below"):
            run_backtest(
                (flat, doubling),
                ConstantWeights((3.0, -2.0)),
                schedule=10,
                cost_rate=0.0,
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
