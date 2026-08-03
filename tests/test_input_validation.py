"""The shared number-validation path, exercised at every public boundary.

Issue #20: finiteness and boolean rejection were enforced in several
constructors but not uniformly across every public entry point. These
tests pin the contract at each boundary that previously lacked it —
NaN, +inf, -inf, and both booleans each rejected with a message naming
the offending parameter — plus the JSON-spec boundary, where ``true``
and the ``NaN``/``Infinity`` tokens must fail at parse instead of
computing, and the NumPy-scalar acceptance rule (``numbers.Real`` and
not ``bool``, so ``np.float64`` and ``np.int64`` both pass).
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

import numpy as np

from quantrisk._validation import require_finite_number, require_finite_numbers
from quantrisk.backtest import PolicyWindow
from quantrisk.benchmark import compare_to_benchmark
from quantrisk.io import SpecFileError, load_spec
from quantrisk.metrics import (
    annualized_volatility,
    historical_cvar,
    historical_var,
    parametric_var,
    sharpe_ratio,
    sortino_ratio,
)
from quantrisk.portfolio import Portfolio
from quantrisk.series import PriceSeries

RETURNS = (0.01, 0.02, -0.03, 0.04, -0.02)
NON_FINITE = (math.nan, math.inf, -math.inf)
BOOLEANS = (True, False)


class SharedHelperTests(unittest.TestCase):
    def test_plain_numbers_pass_and_normalize_to_float(self) -> None:
        self.assertEqual(require_finite_number(2, "x"), 2.0)
        self.assertIsInstance(require_finite_number(2, "x"), float)
        self.assertEqual(require_finite_number(0.5, "x"), 0.5)

    def test_numpy_float64_and_int64_both_pass(self) -> None:
        # np.float64 subclasses float; np.int64 does NOT subclass int.
        # Acceptance is decided by numbers.Real, so both pass and both
        # come back as built-in floats.
        self.assertEqual(require_finite_number(np.float64(0.25), "x"), 0.25)
        self.assertEqual(require_finite_number(np.int64(3), "x"), 3.0)
        self.assertIsInstance(require_finite_number(np.int64(3), "x"), float)

    def test_booleans_are_rejected_as_non_numbers(self) -> None:
        for value in BOOLEANS:
            with self.assertRaisesRegex(ValueError, "x is not a number"):
                require_finite_number(value, "x")

    def test_non_numbers_are_rejected_with_the_description(self) -> None:
        for value in ("0.5", None, [0.5], object()):
            with self.assertRaisesRegex(ValueError, "x is not a number"):
                require_finite_number(value, "x")

    def test_non_finite_values_are_rejected_with_the_description(self) -> None:
        for value in NON_FINITE:
            with self.assertRaisesRegex(ValueError, "x is .*; it must be finite"):
                require_finite_number(value, "x")
        with self.assertRaisesRegex(ValueError, "must be finite"):
            require_finite_number(np.float64("nan"), "x")

    def test_the_sequence_helper_names_the_offending_index(self) -> None:
        with self.assertRaisesRegex(ValueError, r"level \[2\] is not a number"):
            require_finite_numbers((0.1, 0.2, True), "level")
        with self.assertRaisesRegex(ValueError, r"level \[1\] is inf; it must be finite"):
            require_finite_numbers((0.1, math.inf), "level")
        self.assertEqual(require_finite_numbers((1, np.int64(2)), "level"), (1.0, 2.0))


class MetricsReturnBoundaryTests(unittest.TestCase):
    """Booleans and non-numbers inside a returns sequence must fail."""

    def run_every_metric(self, returns: tuple[Any, ...]) -> list[Any]:
        return [
            lambda: annualized_volatility(returns),
            lambda: sharpe_ratio(returns, risk_free_rate_annual=0.0),
            lambda: sortino_ratio(returns),
            lambda: historical_var(returns, confidence=0.95),
            lambda: historical_cvar(returns, confidence=0.95),
            lambda: parametric_var(returns, confidence=0.95),
        ]

    def test_boolean_returns_are_rejected_by_every_metric(self) -> None:
        for call in self.run_every_metric((0.01, True, -0.03, 0.04, -0.02)):
            with self.assertRaisesRegex(ValueError, r"return \[1\] is not a number"):
                call()

    def test_non_number_returns_raise_value_error_not_type_error(self) -> None:
        for call in self.run_every_metric((0.01, "0.02", -0.03, 0.04, -0.02)):
            with self.assertRaisesRegex(ValueError, r"return \[1\] is not a number"):
                call()

    def test_non_finite_returns_name_their_index(self) -> None:
        for value in NON_FINITE:
            with self.assertRaisesRegex(ValueError, r"return \[2\] is .*; it must be finite"):
                annualized_volatility((0.01, 0.02, value))


class MetricsScalarBoundaryTests(unittest.TestCase):
    def test_sharpe_rejects_non_finite_and_boolean_risk_free_rates(self) -> None:
        for value in NON_FINITE:
            with self.assertRaisesRegex(ValueError, "risk_free_rate_annual is .*; it must be"):
                sharpe_ratio(RETURNS, risk_free_rate_annual=value)
        for value in BOOLEANS:
            with self.assertRaisesRegex(ValueError, "risk_free_rate_annual is not a number"):
                sharpe_ratio(RETURNS, risk_free_rate_annual=value)

    def test_sortino_rejects_non_finite_and_boolean_targets(self) -> None:
        for value in NON_FINITE:
            with self.assertRaisesRegex(ValueError, "target_return_periodic is .*; it must be"):
                sortino_ratio(RETURNS, target_return_periodic=value)
        for value in BOOLEANS:
            with self.assertRaisesRegex(ValueError, "target_return_periodic is not a number"):
                sortino_ratio(RETURNS, target_return_periodic=value)

    def test_confidence_rejects_booleans_as_non_numbers(self) -> None:
        for func in (historical_var, historical_cvar, parametric_var):
            for value in BOOLEANS:
                with self.assertRaisesRegex(ValueError, "confidence is not a number"):
                    func(RETURNS, confidence=value)

    def test_confidence_rejects_nan_as_non_finite(self) -> None:
        for func in (historical_var, historical_cvar, parametric_var):
            with self.assertRaisesRegex(ValueError, "confidence is nan; it must be finite"):
                func(RETURNS, confidence=math.nan)


class BenchmarkBoundaryTests(unittest.TestCase):
    """The dated comparison's remaining number boundary: the rate.

    Boolean/NaN *returns* no longer have a boundary here: the comparison
    derives both return series from PriceSeries and Portfolio inputs
    validated at construction (tests/test_series.py and this file), and
    the derived returns re-check through the shared path
    (tests/test_benchmark.py's nonfinite-returns test).
    """

    DATES = ("2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09")
    PRICES = (64.0, 80.0, 60.0, 90.0, 67.5)
    HELD = Portfolio(names=("P1", "P2"), weights=(0.5, 0.5))
    SLEEVES = (
        PriceSeries(name="P1", dates=DATES, prices=PRICES),
        PriceSeries(name="P2", dates=DATES, prices=PRICES),
    )
    BENCHMARK = PriceSeries(name="NNN", dates=DATES, prices=(64.0, 72.0, 63.0, 70.875, 62.015625))

    def compare(self, risk_free_rate_daily: Any) -> Any:
        return compare_to_benchmark(
            self.HELD, self.SLEEVES, self.BENCHMARK, risk_free_rate_daily=risk_free_rate_daily
        )

    def test_the_risk_free_rate_rejects_non_finite_and_boolean_values(self) -> None:
        for value in NON_FINITE:
            with self.assertRaisesRegex(ValueError, "risk_free_rate_daily is .*; it must be"):
                self.compare(value)
        for value in BOOLEANS:
            with self.assertRaisesRegex(ValueError, "risk_free_rate_daily is not a number"):
                self.compare(value)

    def test_numpy_scalars_pass_through_the_comparison(self) -> None:
        # numpy prices pass PriceSeries validation (numbers.Real), and a
        # numpy rate passes the comparison's own scalar boundary.
        benchmark = PriceSeries(
            name="NNN",
            dates=self.DATES,
            prices=tuple(np.float64(price) for price in self.BENCHMARK.prices),
        )
        result = compare_to_benchmark(
            self.HELD, self.SLEEVES, benchmark, risk_free_rate_daily=np.float64(0.0001)
        )
        self.assertTrue(math.isfinite(result.beta))


class PortfolioNumpyScalarTests(unittest.TestCase):
    def test_numpy_float64_and_int64_weights_are_accepted(self) -> None:
        held = Portfolio(names=("AAA", "BBB"), weights=(np.float64(0.6), np.float64(0.4)))
        self.assertFalse(held.has_short_positions)
        # np.int64 satisfies the runtime contract (numbers.Real) but not
        # the tuple[float, ...] annotation; the cast exercises the path
        # an untyped caller takes.
        whole_weights = cast("tuple[float, ...]", (np.int64(1), np.float64(0.0)))
        whole = Portfolio(names=("AAA", "BBB"), weights=whole_weights)
        self.assertFalse(whole.has_short_positions)


class PolicyWindowBoundaryTests(unittest.TestCase):
    DATES = ("2026-01-05", "2026-01-06", "2026-01-07")

    def window(self, prices: tuple[Any, ...]) -> PolicyWindow:
        return PolicyWindow(
            decision_date="2026-01-08",
            names=("AAA",),
            dates=self.DATES,
            prices=(prices,),
        )

    def test_non_finite_window_prices_are_rejected(self) -> None:
        for value in NON_FINITE:
            message = r"window price for 'AAA' \[1\] is .*; it must be finite"
            with self.assertRaisesRegex(ValueError, message):
                self.window((100.0, value, 101.0))

    def test_boolean_window_prices_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, r"window price for 'AAA' \[2\] is not a number"):
            self.window((100.0, 101.0, True))


class SpecBoundaryTests(unittest.TestCase):
    """JSON specs: true/false and NaN/Infinity tokens must fail at parse."""

    CSV = "\n".join(
        [
            "Date,AAA,BBB",
            "2026-01-05,100.0,50.0",
            "2026-01-06,102.0,51.0",
            "2026-01-07,99.0,50.0",
            "2026-01-08,101.0,52.0",
        ]
    )

    def setUp(self) -> None:
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.directory = Path(holder.name)
        (self.directory / "prices.csv").write_text(self.CSV, encoding="utf-8")

    def write_spec(self, payload: dict[str, Any]) -> Path:
        path = self.directory / "spec.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "prices_csv": "prices.csv",
            "portfolio": {"AAA": 0.6, "BBB": 0.4},
        }

    def test_a_boolean_weight_fails_cleanly_at_parse(self) -> None:
        payload = self.payload()
        payload["portfolio"] = {"AAA": True, "BBB": 0.4}
        with self.assertRaisesRegex(SpecFileError, "portfolio.AAA must be a JSON number"):
            load_spec(self.write_spec(payload))

    def test_a_nan_token_in_the_spec_is_rejected(self) -> None:
        payload = self.payload()
        payload["risk_free_rate_annual"] = math.nan
        # json.dumps writes the literal token NaN, which json.loads
        # would happily parse back into a float; the spec loader must
        # refuse the document instead of computing with it.
        with self.assertRaisesRegex(SpecFileError, "NaN"):
            load_spec(self.write_spec(payload))

    def test_infinity_tokens_in_weights_and_shocks_are_rejected(self) -> None:
        payload = self.payload()
        payload["portfolio"] = {"AAA": math.inf, "BBB": 0.4}
        with self.assertRaisesRegex(SpecFileError, "Infinity"):
            load_spec(self.write_spec(payload))
        payload = self.payload()
        payload["stress"] = {"shocks": {"crash": {"AAA": -math.inf, "BBB": 0.0}}}
        with self.assertRaisesRegex(SpecFileError, "-Infinity"):
            load_spec(self.write_spec(payload))

    def test_a_boolean_shock_fails_cleanly_at_parse(self) -> None:
        payload = self.payload()
        payload["stress"] = {"shocks": {"crash": {"AAA": False, "BBB": 0.0}}}
        message = r"stress.shocks\['crash'\].AAA must be a JSON number"
        with self.assertRaisesRegex(SpecFileError, message):
            load_spec(self.write_spec(payload))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
