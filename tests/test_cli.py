from __future__ import annotations

import contextlib
import io
import json
import math
import re
import tempfile
import unittest
from pathlib import Path
from typing import Any

from quantrisk.cli import main

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SPEC = _REPOSITORY_ROOT / "sample-data" / "portfolio-spec.json"


def run_cli(arguments: list[str]) -> tuple[int, str]:
    """Run the CLI in-process, returning its exit code and stdout."""

    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        code = main(arguments)
    return code, stream.getvalue()


class SampleRunCase(unittest.TestCase):
    """One CLI run over the shipped sample spec, shared by the assertions."""

    results: dict[str, Any]
    report: str

    @classmethod
    def setUpClass(cls) -> None:
        holder = tempfile.TemporaryDirectory()
        cls.addClassCleanup(holder.cleanup)
        output = Path(holder.name) / "results"
        code, _ = run_cli(["run", "--spec", str(_SPEC), "--output", str(output)])
        assert code == 0
        cls.results = json.loads((output / "results.json").read_text(encoding="utf-8"))
        cls.report = (output / "report.md").read_text(encoding="utf-8")


class ResultsJsonTests(SampleRunCase):
    def test_portfolio_metrics_land_on_the_pinned_digits(self) -> None:
        metrics = self.results["portfolio"]["metrics"]
        self.assertAlmostEqual(metrics["total_return"], -0.002264384375557, places=12)
        self.assertAlmostEqual(metrics["annualized_volatility"], 0.129525667932682, places=12)
        self.assertAlmostEqual(metrics["sharpe_ratio"], -0.106873527292221, places=12)
        self.assertAlmostEqual(metrics["var_95"], 0.011873432017429, places=12)
        self.assertAlmostEqual(metrics["cvar_95"], 0.016341900779589, places=12)
        self.assertAlmostEqual(metrics["diversification_benefit"], 0.087603897680849, places=12)
        self.assertEqual(metrics["max_drawdown"]["peak_date"], "2025-01-24")
        self.assertEqual(metrics["max_drawdown"]["trough_date"], "2025-04-25")
        self.assertAlmostEqual(metrics["max_drawdown"]["depth"], 0.110629514347550, places=12)

    def test_risk_contributions_sum_to_one(self) -> None:
        contributions = self.results["portfolio"]["metrics"]["risk_contributions"]
        self.assertEqual(sorted(contributions), ["AAA", "BBB", "CCC"])
        self.assertAlmostEqual(math.fsum(contributions.values()), 1.0, places=12)
        self.assertAlmostEqual(contributions["AAA"], 0.355176414470461, places=12)

    def test_conditioning_diagnostics_are_exposed_and_quiet(self) -> None:
        # The synthetic assets are far from collinear, so the covariance
        # condition number is small and no conditioning warning appears.
        metrics = self.results["portfolio"]["metrics"]
        self.assertGreaterEqual(metrics["condition_number"], 1.0)
        self.assertLess(metrics["condition_number"], 1e8)
        self.assertIsNone(metrics["conditioning_warning"])

    def test_per_asset_metrics_cover_every_holding(self) -> None:
        assets = self.results["assets"]
        self.assertEqual(sorted(assets), ["AAA", "BBB", "CCC"])
        self.assertAlmostEqual(assets["AAA"]["total_return"], 0.017154189390183, places=12)
        self.assertAlmostEqual(
            assets["CCC"]["annualized_volatility"], 0.358093592514695, places=12
        )

    def test_benchmark_comparison_lands_on_the_pinned_digits(self) -> None:
        benchmark = self.results["benchmark"]
        self.assertEqual(benchmark["ticker"], "DDD")
        self.assertAlmostEqual(benchmark["metrics"]["total_return"], 0.160015013136494, places=12)
        comparison = benchmark["comparison"]
        self.assertAlmostEqual(comparison["beta"], 0.002078133650384, places=12)
        self.assertAlmostEqual(comparison["alpha_annualized"], -0.014111801883636, places=12)
        self.assertAlmostEqual(comparison["tracking_error"], 0.166683377688661, places=12)

    def test_stress_results_include_the_window_and_both_shocks(self) -> None:
        stress = self.results["stress"]
        (window,) = stress["windows"]
        self.assertEqual(window["name"], "spring-drawdown")
        self.assertEqual(window["observations"], 43)
        self.assertAlmostEqual(window["total_return"], -0.059185086039227, places=12)
        self.assertAlmostEqual(window["max_drawdown"]["depth"], 0.082563744093248, places=12)
        first, second = stress["shocks"]
        self.assertEqual(first["name"], "broad-selloff")
        self.assertEqual(first["portfolio_return"], -0.1)
        self.assertEqual(second["name"], "concentration-hit")
        self.assertAlmostEqual(second["portfolio_return"], -0.071, places=15)

    def test_a_shock_recomputes_from_the_spec_weights(self) -> None:
        weights = self.results["portfolio"]["weights"]
        (shock,) = [
            entry
            for entry in self.results["stress"]["shocks"]
            if entry["name"] == "concentration-hit"
        ]
        expected = math.fsum(
            weights[name] * value for name, value in shock["asset_shocks"].items()
        )
        self.assertEqual(shock["portfolio_return"], expected)

    def test_monte_carlo_keeps_seed_runs_and_every_terminal_value(self) -> None:
        simulation = self.results["monte_carlo"]
        self.assertEqual(simulation["mode"], "bootstrap")
        self.assertEqual(simulation["seed"], 2026)
        self.assertEqual(simulation["runs"], 1000)
        self.assertEqual(simulation["horizon_days"], 252)
        self.assertEqual(len(simulation["terminal_values"]), 1000)
        self.assertAlmostEqual(simulation["terminal_mean"], 1.002033242129725, places=12)
        percentiles = simulation["terminal_percentiles"]
        self.assertAlmostEqual(percentiles["p5"], 0.808745433533912, places=12)
        self.assertAlmostEqual(percentiles["p50"], 0.988354557854029, places=12)
        self.assertAlmostEqual(percentiles["p95"], 1.221637079886091, places=12)
        self.assertEqual(simulation["probability_below_initial"], 0.532)
        # Bootstrap resamples historical returns, which positive prices
        # bound above -100%, so the count is structurally zero here.
        self.assertEqual(simulation["bankruptcies"], 0)


class ReportTests(SampleRunCase):
    def assert_line(self, line: str) -> None:
        self.assertIn("\n" + line + "\n", self.report)

    def test_the_report_opens_with_its_title_and_source(self) -> None:
        self.assertTrue(self.report.startswith("# Portfolio risk report\n"))
        self.assertIn("Prepared by quantrisk from `portfolio-spec.json`.", self.report)

    def test_the_holdings_sentence_states_weights_grid_and_benchmark(self) -> None:
        self.assert_line(
            "The portfolio holds AAA at 50.00%, BBB at 30.00%, and CCC at 20.00% over "
            "the 261 shared trading days from 2025-01-01 to 2025-12-31 in `prices.csv`, "
            "measured against DDD as the benchmark. The annual risk-free rate is 2.00%."
        )

    def test_the_risk_contribution_rows_render(self) -> None:
        self.assert_line("| AAA | 50.00% | 35.52% |")
        self.assert_line("| CCC | 20.00% | 35.33% |")

    def test_the_stress_tables_render(self) -> None:
        self.assert_line(
            "| spring-drawdown | 2025-03-03 | 2025-04-30 | 43 | -5.92% | 12.54% | 8.26% |"
        )
        self.assert_line("| broad-selloff | -10.00% | -10.00% | -10.00% | -10.00% |")
        self.assert_line("| concentration-hit | -5.00% | -2.00% | -20.00% | -7.10% |")

    def test_the_monte_carlo_section_states_seed_and_runs(self) -> None:
        self.assert_line(
            "The simulation ran 1000 bootstrap (i.i.d. resampling of the portfolio's "
            "daily returns) runs with seed 2026 over a horizon of 252 trading days, "
            "compounding daily returns from an initial value of 1.0. The same seed "
            "reproduces every figure exactly."
        )
        self.assert_line("| P5 | 0.8087 |")
        self.assert_line("| P50 | 0.9884 |")
        self.assertIn(
            "0 runs were absorbed at exactly zero by a daily return at or below -100%.",
            self.report,
        )

    def test_the_caveats_are_present(self) -> None:
        self.assertIn("- This is not a prediction tool.", self.report)
        self.assertIn("volatility clustering", self.report)
        self.assertIn("fatter-tailed than a normal distribution", self.report)
        self.assertIn("fictional", self.report)
        self.assertIn("- Nothing in this report is investment advice.", self.report)

    def test_the_report_carries_no_timestamp(self) -> None:
        # No clock time (hh:mm) and no ISO datetime appears anywhere;
        # the only dates are the price grid's own. Byte-identity across
        # runs is asserted separately.
        self.assertIsNone(re.search(r"\d{2}:\d{2}", self.report))
        self.assertIsNone(re.search(r"\d{4}-\d{2}-\d{2}T", self.report))


class DeterminismTests(unittest.TestCase):
    def test_two_runs_produce_byte_identical_artifacts(self) -> None:
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        first = Path(holder.name) / "first"
        second = Path(holder.name) / "second"
        for output in (first, second):
            code, _ = run_cli(["run", "--spec", str(_SPEC), "--output", str(output)])
            self.assertEqual(code, 0)
        for name in ("results.json", "report.md"):
            with self.subTest(artifact=name):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())


class ForceDisciplineTests(unittest.TestCase):
    def setUp(self) -> None:
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.output = Path(holder.name) / "results"

    def test_a_second_run_requires_force(self) -> None:
        code, _ = run_cli(["run", "--spec", str(_SPEC), "--output", str(self.output)])
        self.assertEqual(code, 0)
        with self.assertRaises(SystemExit) as context:
            run_cli(["run", "--spec", str(_SPEC), "--output", str(self.output)])
        self.assertIn("refusing to overwrite", str(context.exception))
        code, _ = run_cli(["run", "--spec", str(_SPEC), "--output", str(self.output), "--force"])
        self.assertEqual(code, 0)


class ValidateTests(unittest.TestCase):
    def test_validate_summarizes_the_sample_spec(self) -> None:
        code, output = run_cli(["validate", "--spec", str(_SPEC)])
        self.assertEqual(code, 0)
        summary = json.loads(output)
        self.assertEqual(summary["status"], "valid")
        self.assertEqual(summary["weights"], {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2})
        self.assertEqual(summary["benchmark"], "DDD")
        self.assertEqual(summary["trading_days"], 261)
        self.assertEqual(summary["first_date"], "2025-01-01")
        self.assertEqual(summary["last_date"], "2025-12-31")
        self.assertEqual(summary["stress_windows"], ["spring-drawdown"])
        self.assertEqual(summary["stress_shocks"], ["broad-selloff", "concentration-hit"])
        self.assertEqual(
            summary["monte_carlo"],
            {"mode": "bootstrap", "runs": 1000, "horizon_days": 252, "seed": 2026},
        )

    def test_validate_of_a_missing_spec_fails_with_a_message(self) -> None:
        with self.assertRaises(SystemExit) as context:
            run_cli(["validate", "--spec", str(_SPEC.with_name("absent.json"))])
        self.assertTrue(str(context.exception).startswith("error: "))


class MinimalSpecReportTests(unittest.TestCase):
    """A spec with no optional block still renders every report section."""

    def test_optional_sections_state_their_absence(self) -> None:
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        directory = Path(holder.name)
        (directory / "prices.csv").write_text(
            "Date,AAA,BBB\n"
            "2026-01-05,100.0,50.0\n"
            "2026-01-06,102.0,51.0\n"
            "2026-01-07,99.0,50.0\n"
            "2026-01-08,101.0,52.0\n"
            "2026-01-09,98.0,51.0\n",
            encoding="utf-8",
        )
        spec_path = directory / "spec.json"
        spec_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "prices_csv": "prices.csv",
                    "portfolio": {"AAA": 0.6, "BBB": 0.4},
                }
            ),
            encoding="utf-8",
        )
        output = directory / "results"
        code, _ = run_cli(["run", "--spec", str(spec_path), "--output", str(output)])
        self.assertEqual(code, 0)
        report = (output / "report.md").read_text(encoding="utf-8")
        self.assertIn("The spec names no benchmark.", report)
        self.assertIn("The spec requests no stress scenarios.", report)
        self.assertIn("The spec requests no Monte Carlo simulation.", report)
        results = json.loads((output / "results.json").read_text(encoding="utf-8"))
        self.assertIsNone(results["benchmark"])
        self.assertIsNone(results["monte_carlo"])
        self.assertEqual(results["stress"], {"windows": [], "shocks": []})

    def test_an_ill_conditioned_portfolio_is_flagged_under_caveats(self) -> None:
        # Two near-collinear price series: the second repeats the
        # first's returns plus a +/-1e-6 wiggle, putting the covariance
        # condition number at 1.33e9 — inside the warn band. The run
        # succeeds, and the report discloses the conditioning under
        # Caveats instead of failing or staying silent.
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        directory = Path(holder.name)
        returns_a = (0.02, -0.01, -0.02, 0.02, 0.03, 0.02)
        wiggles = (1e-6, -1e-6, 1e-6, -1e-6, 1e-6, -1e-6)
        prices_a = [100.0]
        prices_b = [100.0]
        for value, wiggle in zip(returns_a, wiggles, strict=True):
            prices_a.append(prices_a[-1] * (1.0 + value))
            prices_b.append(prices_b[-1] * (1.0 + value + wiggle))
        dates = (
            "2026-01-05",
            "2026-01-06",
            "2026-01-07",
            "2026-01-08",
            "2026-01-09",
            "2026-01-12",
            "2026-01-13",
        )
        rows = ["Date,AAA,ECH"]
        for date, price_a, price_b in zip(dates, prices_a, prices_b, strict=True):
            rows.append(f"{date},{price_a!r},{price_b!r}")
        (directory / "prices.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        spec_path = directory / "spec.json"
        spec_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "prices_csv": "prices.csv",
                    "portfolio": {"AAA": 0.5, "ECH": 0.5},
                }
            ),
            encoding="utf-8",
        )
        output = directory / "results"
        code, _ = run_cli(["run", "--spec", str(spec_path), "--output", str(output)])
        self.assertEqual(code, 0)
        report = (output / "report.md").read_text(encoding="utf-8")
        self.assertIn("- Numerical conditioning: ", report)
        self.assertIn("ill-conditioned", report)
        results = json.loads((output / "results.json").read_text(encoding="utf-8"))
        metrics = results["portfolio"]["metrics"]
        self.assertGreater(metrics["condition_number"], 1e8)
        self.assertLess(metrics["condition_number"], 1e12)
        self.assertIn("condition number", metrics["conditioning_warning"])

    def test_a_parametric_normal_spec_names_its_mode_in_the_report(self) -> None:
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        directory = Path(holder.name)
        source = json.loads(_SPEC.read_text(encoding="utf-8"))
        source["monte_carlo"]["mode"] = "parametric_normal"
        del source["stress"]
        (directory / "prices.csv").write_text(
            (_SPEC.parent / "prices.csv").read_text(encoding="utf-8"), encoding="utf-8"
        )
        spec_path = directory / "spec.json"
        spec_path.write_text(json.dumps(source), encoding="utf-8")
        output = directory / "results"
        code, _ = run_cli(["run", "--spec", str(spec_path), "--output", str(output)])
        self.assertEqual(code, 0)
        report = (output / "report.md").read_text(encoding="utf-8")
        self.assertIn("parametric normal (draws fitted to the portfolio's daily returns)", report)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
