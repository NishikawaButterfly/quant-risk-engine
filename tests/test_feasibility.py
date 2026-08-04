"""Validation implies runnability: what validate accepts, run completes.

Before the feasibility pass, ``quantrisk validate`` only proved that a
spec *loaded*: several well-formed specs validated with exit code 0 and
then failed evaluation under ``quantrisk run`` — a benchmark grid of
three dates (two paired returns), a constant series with no Sharpe
ratio, a short position driving the compounded value path to zero, a
price ratio overflowing to a non-finite return. The regression tests
here pin each of those once-latent failures as a validation failure,
assert that validate and run refuse with one identical message, and a
seeded property test sweeps structured random specs asserting the
guarantee itself: every generated spec is either rejected by validate
or run to completion.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import sys
import tempfile
import unittest
import warnings
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from quantrisk.cli import main
from quantrisk.feasibility import SpecFeasibilityError, check_feasibility
from quantrisk.io import load_spec

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE_SPEC = _REPOSITORY_ROOT / "sample-data" / "portfolio-spec.json"


def run_cli(arguments: list[str]) -> tuple[int, str]:
    """Run the CLI in-process; a ``SystemExit`` maps to (1, its message)."""

    stream = io.StringIO()
    try:
        with contextlib.redirect_stdout(stream):
            code = main(arguments)
    except SystemExit as exc:
        return 1, str(exc)
    return code, stream.getvalue()


class SpecDirectory:
    """One temporary directory holding a spec and its prices CSV."""

    def __init__(self, case: unittest.TestCase, csv_text: str, spec: dict[str, Any]) -> None:
        holder = tempfile.TemporaryDirectory()
        case.addCleanup(holder.cleanup)
        self.directory = Path(holder.name)
        (self.directory / "prices.csv").write_text(csv_text, encoding="utf-8")
        self.spec_path = self.directory / "spec.json"
        self.spec_path.write_text(json.dumps(spec), encoding="utf-8")

    def validate(self) -> tuple[int, str]:
        return self._quietly(["validate", "--spec", str(self.spec_path)])

    def run(self) -> tuple[int, str]:
        output = self.directory / "results"
        return self._quietly(["run", "--spec", str(self.spec_path), "--output", str(output)])

    @staticmethod
    def _quietly(arguments: list[str]) -> tuple[int, str]:
        # Pathological fixtures (infinite returns) make numpy warn on
        # the way to a clean refusal; these tests assert exit codes and
        # messages, not warning hygiene.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return run_cli(arguments)


def _base_spec(**extra: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "prices_csv": "prices.csv",
        "portfolio": {"AAA": 0.5, "BBB": 0.5},
        **extra,
    }


class LateFailureRegressionTests(unittest.TestCase):
    """Each spec here validated and then failed at run before the fix.

    The assertions pin the new contract: validate refuses, run refuses,
    and both refuse with the same message, because both call the same
    feasibility check at the same point.
    """

    def assert_validate_and_run_agree_on_refusal(
        self, fixture: SpecDirectory, fragment: str
    ) -> None:
        validate_code, validate_message = fixture.validate()
        run_code, run_message = fixture.run()
        self.assertEqual(validate_code, 1)
        self.assertEqual(run_code, 1)
        self.assertIn("spec cannot be evaluated", validate_message)
        self.assertIn(fragment, validate_message)
        self.assertEqual(validate_message, run_message)

    def test_a_three_date_benchmark_grid_fails_validation(self) -> None:
        # align() accepts a three-date grid (two returns), but the
        # benchmark comparison needs three paired returns; before the
        # fix this spec validated and then failed at run.
        fixture = SpecDirectory(
            self,
            "Date,AAA,BBB,DDD\n"
            "2026-01-05,100.0,50.0,10.0\n"
            "2026-01-06,101.0,51.0,10.1\n"
            "2026-01-07,102.0,50.5,10.3\n",
            _base_spec(benchmark="DDD"),
        )
        self.assert_validate_and_run_agree_on_refusal(
            fixture, "need at least 3 paired returns, got 2"
        )

    def test_a_constant_benchmark_fails_validation(self) -> None:
        fixture = SpecDirectory(
            self,
            "Date,AAA,BBB,DDD\n"
            "2026-01-05,100.0,50.0,10.0\n"
            "2026-01-06,101.0,51.0,10.0\n"
            "2026-01-07,102.0,50.5,10.0\n"
            "2026-01-08,103.0,50.0,10.0\n"
            "2026-01-09,104.0,49.0,10.0\n",
            _base_spec(benchmark="DDD"),
        )
        self.assert_validate_and_run_agree_on_refusal(
            fixture, "Sharpe ratio is undefined for constant returns"
        )

    def test_a_constant_asset_fails_validation(self) -> None:
        fixture = SpecDirectory(
            self,
            "Date,AAA,BBB\n"
            "2026-01-05,100.0,50.0\n"
            "2026-01-06,100.0,51.0\n"
            "2026-01-07,100.0,50.5\n"
            "2026-01-08,100.0,49.0\n",
            _base_spec(),
        )
        self.assert_validate_and_run_agree_on_refusal(
            fixture, "Sharpe ratio is undefined for constant returns"
        )

    def test_a_short_position_reaching_a_nonpositive_path_fails_validation(self) -> None:
        # AAA triples on day two; a -100% short weight on it drives the
        # compounded portfolio value path to -2.0, which the stress
        # engine would refuse inside a window but nothing refused over
        # the whole history before the fix.
        fixture = SpecDirectory(
            self,
            "Date,AAA,BBB\n"
            "2026-01-05,100.0,50.0\n"
            "2026-01-06,300.0,25.0\n"
            "2026-01-07,300.0,25.0\n"
            "2026-01-08,310.0,26.0\n",
            _base_spec(portfolio={"AAA": -1.0, "BBB": 2.0}),
        )
        self.assert_validate_and_run_agree_on_refusal(fixture, "prices must be positive")

    def test_an_overflowing_price_ratio_fails_validation(self) -> None:
        # Every price is finite and positive, but the day-over-day
        # ratio overflows to an infinite return.
        fixture = SpecDirectory(
            self,
            "Date,AAA,BBB\n"
            "2026-01-05,1e-300,50.0\n"
            "2026-01-06,1e300,51.0\n"
            "2026-01-07,1e-300,50.5\n"
            "2026-01-08,1e300,50.0\n",
            _base_spec(),
        )
        self.assert_validate_and_run_agree_on_refusal(fixture, "must be finite")

    def test_check_feasibility_raises_a_value_error_subclass(self) -> None:
        # The CLI catches ValueError; the feasibility refusal must be
        # one, so both commands report it as a clean error message.
        fixture = SpecDirectory(
            self,
            "Date,AAA,BBB\n"
            "2026-01-05,100.0,50.0\n"
            "2026-01-06,100.0,51.0\n"
            "2026-01-07,100.0,50.5\n"
            "2026-01-08,100.0,49.0\n",
            _base_spec(),
        )
        spec = load_spec(fixture.spec_path)
        with self.assertRaises(SpecFeasibilityError) as context:
            check_feasibility(spec)
        self.assertIsInstance(context.exception, ValueError)
        self.assertIn("spec cannot be evaluated", str(context.exception))

    def test_a_stress_window_below_minimum_observations_still_fails_at_load(self) -> None:
        # The inventory's already-closed case: stress windows are
        # checked at load against the final aligned grid, so a window
        # covering fewer than five observations never validated. The
        # test keeps that guarantee pinned alongside the new ones.
        fixture = SpecDirectory(
            self,
            "Date,AAA,BBB\n"
            "2026-01-05,100.0,50.0\n"
            "2026-01-06,101.0,51.0\n"
            "2026-01-07,102.0,50.5\n"
            "2026-01-08,103.0,49.0\n"
            "2026-01-09,104.0,49.5\n"
            "2026-01-12,105.0,50.0\n",
            _base_spec(stress={"windows": {"thin": {"start": "2026-01-05", "end": "2026-01-08"}}}),
        )
        validate_code, validate_message = fixture.validate()
        run_code, run_message = fixture.run()
        self.assertEqual((validate_code, run_code), (1, 1))
        self.assertIn("at least 5 are required", validate_message)
        self.assertEqual(validate_message, run_message)


class MonteCarloFeasibilityTests(unittest.TestCase):
    """The one check that is not a plain dry run: bounded or simulated."""

    def _explosive_csv(self) -> str:
        # AAA roughly doubles every day (with variation, so no Sharpe
        # refusal fires first); BBB drifts normally.
        rows = ["Date,AAA,BBB"]
        price_a, price_b = 1.0, 50.0
        growth = (2.0, 2.1, 1.9, 2.05, 1.95)
        dates = ("2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09")
        moves = (0.0, 0.01, -0.02, 0.015, -0.005)
        for day, factor, move in zip(dates, growth, moves, strict=True):
            rows.append(f"{day},{price_a!r},{price_b!r}")
            price_a *= factor
            price_b *= 1.0 + move
        rows.append(f"2026-01-12,{price_a!r},{price_b!r}")
        return "\n".join(rows) + "\n"

    def test_a_bootstrap_that_must_overflow_fails_validation(self) -> None:
        # Portfolio returns near +50% daily, compounded over the full
        # horizon cap: every resampled path overflows the float range,
        # so the run's own allow_nan=False serialization would refuse.
        # The bound cannot certify this spec, the dry run simulates it,
        # and validation fails exactly like the run used to.
        fixture = SpecDirectory(
            self,
            self._explosive_csv(),
            _base_spec(
                monte_carlo={
                    "mode": "bootstrap",
                    "runs": 100,
                    "horizon_days": 2520,
                    "seed": 7,
                }
            ),
        )
        validate_code, validate_message = fixture.validate()
        run_code, run_message = fixture.run()
        self.assertEqual((validate_code, run_code), (1, 1))
        self.assertIn("spec cannot be evaluated", validate_message)
        self.assertEqual(validate_message, run_message)

    def test_a_dry_run_bootstrap_outside_the_bound_can_still_validate(self) -> None:
        # One +20% day at the full horizon cap defeats the certainty
        # bound (1.2 ** 2520 alone would overflow), but the simulated
        # paths mix that day with flat ones and stay finite: the dry
        # run must accept what the bound cannot prove.
        rows = ["Date,AAA,BBB"]
        price_a, price_b = 100.0, 50.0
        moves_a = (0.001, -0.002, 0.2, 0.001, -0.001, 0.002, -0.002, 0.001, -0.001)
        moves_b = (0.002, 0.001, -0.001, 0.003, -0.002, 0.001, 0.002, -0.003, 0.001)
        days = (5, 6, 7, 8, 9, 12, 13, 14, 15, 16)
        for index, day in enumerate(days):
            rows.append(f"2026-01-{day:02d},{price_a!r},{price_b!r}")
            if index < len(moves_a):
                price_a *= 1.0 + moves_a[index]
                price_b *= 1.0 + moves_b[index]
        fixture = SpecDirectory(
            self,
            "\n".join(rows) + "\n",
            _base_spec(
                monte_carlo={
                    "mode": "bootstrap",
                    "runs": 100,
                    "horizon_days": 2520,
                    "seed": 11,
                }
            ),
        )
        validate_code, _ = fixture.validate()
        self.assertEqual(validate_code, 0)
        run_code, _ = fixture.run()
        self.assertEqual(run_code, 0)

    def test_the_sample_bootstrap_spec_is_certified_without_simulation(self) -> None:
        # The shipped spec's worst daily return is a few percent, so
        # runs * ((1 + max) ** horizon) ** 2 fits in a float with room
        # to spare and validation never needs to simulate. The bound's
        # arithmetic is checked here against the spec's own numbers.
        spec = load_spec(_SAMPLE_SPEC)
        assert spec.monte_carlo is not None
        top = max(spec.portfolio.return_series(spec.asset_series))
        self.assertGreater(top, 0.0)
        budget = (math.log(sys.float_info.max) - math.log(spec.monte_carlo.runs)) / 2.0
        self.assertLessEqual(spec.monte_carlo.horizon_days * math.log1p(top), budget)
        check_feasibility(spec)  # certifies and returns without error


class FeasibilityAcceptanceTests(unittest.TestCase):
    """Specs that must keep validating: the guarantee cannot over-refuse."""

    def test_the_sample_spec_still_validates_and_runs(self) -> None:
        code, output = run_cli(["validate", "--spec", str(_SAMPLE_SPEC)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output)["status"], "valid")

    def test_a_minimal_three_date_spec_without_benchmark_validates_and_runs(self) -> None:
        # Three dates is align()'s floor and produces two returns —
        # enough for every metric except a benchmark comparison, which
        # this spec does not request. Refusing it would over-tighten.
        fixture = SpecDirectory(
            self,
            "Date,AAA,BBB\n2026-01-05,100.0,50.0\n2026-01-06,101.0,51.0\n2026-01-07,102.0,50.5\n",
            _base_spec(),
        )
        self.assertEqual(fixture.validate()[0], 0)
        self.assertEqual(fixture.run()[0], 0)


def _walk_column(rng: np.random.Generator, days: int, scale: float) -> list[float]:
    prices = [float(rng.integers(1, 200))]
    for _ in range(days - 1):
        prices.append(prices[-1] * (1.0 + float(rng.normal(0.0005, scale))))
    return prices


def _column(rng: np.random.Generator, style: int, days: int, anchor: list[float]) -> list[float]:
    if style == 0:
        return _walk_column(rng, days, 0.01)
    if style == 1:
        return _walk_column(rng, days, 0.2)
    if style == 2:  # constant — no Sharpe ratio exists
        return [100.0] * days
    if style == 3:  # near-collinear with the first column
        return [price * (1.0 + float(rng.normal(0.0, 1e-7))) for price in anchor]
    if style == 4:  # explosive growth
        return [float(2.0**index) for index in range(days)]
    # extreme magnitudes — day-over-day ratios can overflow
    return [1e-250 if index % 2 == 0 else 1e250 for index in range(days)]


_WEIGHT_CHOICES: tuple[tuple[float, ...], ...] = (
    (0.5, 0.5, 0.0, 0.0),
    (0.25, 0.25, 0.25, 0.25),
    (1.5, -0.5, 0.0, 0.0),
    (3.0, -2.0, 0.0, 0.0),
    (0.4, 0.3, 0.2, 0.1),
)


def _stress_block(
    rng: np.random.Generator, dates: list[str], portfolio: dict[str, float]
) -> dict[str, Any]:
    """A sometimes-empty stress block whose windows hug the grid edges."""

    days = len(dates)
    stress: dict[str, Any] = {}
    if rng.integers(0, 10) < 3:
        start = int(rng.integers(0, days))
        length = int(rng.integers(0, days))
        end = min(start + length, days - 1)
        window = {"start": dates[start], "end": dates[end]}
        if rng.integers(0, 4) == 0:
            window["end"] = "2100-01-01"  # deliberately outside the grid
        stress["windows"] = {"w": window}
    if rng.integers(0, 10) < 2:
        vector = {ticker: float(rng.uniform(-0.5, 0.3)) for ticker in portfolio}
        if rng.integers(0, 4) == 0 and len(vector) > 2:
            vector.pop(next(iter(vector)))  # incomplete vector: load must refuse
        stress["shocks"] = {"s": vector}
    return stress


def _generate_case(rng: np.random.Generator) -> tuple[str, dict[str, Any]]:
    """One structured random spec plus its CSV, boundary-heavy on purpose."""

    days = int(rng.integers(3, 24))
    dates: list[str] = []
    ordinal = int(rng.integers(738000, 738200))
    for _ in range(days):
        dates.append(date.fromordinal(ordinal).isoformat())
        ordinal += int(rng.integers(1, 4))
    asset_count = int(rng.integers(2, 5))
    with_benchmark = bool(rng.integers(0, 2))
    tickers = [f"A{index}" for index in range(asset_count)] + (["BEN"] if with_benchmark else [])
    columns: list[list[float]] = []
    for index in range(len(tickers)):
        style = int(rng.integers(0, 12))  # styles 6..11 fold onto 0/1 (common cases)
        style = style if style < 6 else int(style % 2)
        if style == 3 and not index:
            style = 0  # the collinear style needs a first column to copy
        anchor = columns[0] if index else []
        columns.append(_column(rng, style, days, anchor))
    rows = ["Date," + ",".join(tickers)]
    for day_index, day in enumerate(dates):
        cells = ",".join(repr(column[day_index]) for column in columns)
        rows.append(f"{day},{cells}")
    csv_text = "\n".join(rows) + "\n"

    weights = _WEIGHT_CHOICES[int(rng.integers(0, len(_WEIGHT_CHOICES)))][:asset_count]
    total = math.fsum(weights)
    if total == 0.0:
        weights = tuple(1.0 / asset_count for _ in range(asset_count))
        total = 1.0
    portfolio = {ticker: weight / total for ticker, weight in zip(tickers, weights, strict=False)}
    spec: dict[str, Any] = {
        "schema_version": 1,
        "prices_csv": "prices.csv",
        "portfolio": portfolio,
    }
    if with_benchmark:
        spec["benchmark"] = "BEN"
    if rng.integers(0, 2):
        spec["risk_free_rate_annual"] = 0.02
    stress = _stress_block(rng, dates, portfolio)
    if stress:
        spec["stress"] = stress
    if rng.integers(0, 10) < 3:
        spec["monte_carlo"] = {
            "mode": "bootstrap" if rng.integers(0, 2) else "parametric_normal",
            "runs": 100,
            "horizon_days": int((1, 10, 60)[int(rng.integers(0, 3))]),
            "seed": int(rng.integers(0, 100000)),
        }
    return csv_text, spec


class ValidateImpliesRunPropertyTests(unittest.TestCase):
    """The issue's property, tested: validate accepts it or run completes.

    A seeded generator sweeps structured specs across the boundaries
    that produced the historical gaps — tiny grids, constant and
    near-collinear and explosive columns, short weights, stress windows
    hugging or escaping the grid edges, incomplete shock vectors, both
    Monte Carlo modes — and asserts, for every case, that a spec
    accepted by ``validate`` is run to completion. The seed pins the
    sweep, so a failure names a reproducible case.
    """

    CASES = 250
    SEED = 20260804

    def test_every_generated_spec_validates_or_runs(self) -> None:
        rng = np.random.default_rng(self.SEED)
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        base = Path(holder.name)
        spec_path = base / "spec.json"
        accepted = 0
        rejected = 0
        with warnings.catch_warnings():
            # Deliberately pathological cases (infinite returns and the
            # like) make numpy warn on the way to a clean refusal; the
            # property is about exit codes, not warning hygiene.
            warnings.simplefilter("ignore", RuntimeWarning)
            for case in range(self.CASES):
                csv_text, spec = _generate_case(rng)
                with self.subTest(case=case):
                    (base / "prices.csv").write_text(csv_text, encoding="utf-8")
                    spec_path.write_text(json.dumps(spec), encoding="utf-8")
                    validate_code, _ = run_cli(["validate", "--spec", str(spec_path)])
                    if validate_code != 0:
                        rejected += 1
                        continue
                    accepted += 1
                    output = base / f"results-{case}"
                    run_code, run_message = run_cli(
                        ["run", "--spec", str(spec_path), "--output", str(output)]
                    )
                    self.assertEqual(
                        run_code,
                        0,
                        f"case {case}: validate accepted but run failed: {run_message!r}\n"
                        f"spec: {json.dumps(spec)}\ncsv:\n{csv_text}",
                    )
        # The sweep must exercise both branches, or the property is vacuous.
        self.assertGreater(accepted, 20)
        self.assertGreater(rejected, 20)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
