from __future__ import annotations

import contextlib
import hashlib
import json
import tempfile
import unittest
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import quantrisk.io
from quantrisk.io import SpecFileError, load_spec, write_run_artifacts

CSV = "\n".join(
    [
        "Date,AAA,BBB,DDD",
        "2026-01-05,100.0,50.0,80.0",
        "2026-01-06,102.0,51.0,81.0",
        "2026-01-07,99.0,50.0,80.0",
        "2026-01-08,101.0,52.0,82.0",
        "2026-01-09,98.0,51.0,81.0",
        "2026-01-12,100.0,50.0,80.0",
        "2026-01-13,101.0,49.0,79.0",
    ]
)


def minimal_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "prices_csv": "prices.csv",
        "portfolio": {"AAA": 0.6, "BBB": 0.4},
    }


def full_payload() -> dict[str, Any]:
    payload = minimal_payload()
    payload["risk_free_rate_annual"] = 0.02
    payload["benchmark"] = "DDD"
    payload["stress"] = {
        "windows": {"whole-week": {"start": "2026-01-05", "end": "2026-01-13"}},
        "shocks": {"hand-worked": {"AAA": -0.20, "BBB": -0.05}},
    }
    payload["monte_carlo"] = {
        "mode": "parametric_normal",
        "runs": 200,
        "horizon_days": 21,
        "seed": 7,
    }
    return payload


class SpecCase(unittest.TestCase):
    """A temp directory holding a prices CSV and a spec to load."""

    def setUp(self) -> None:
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.directory = Path(holder.name)
        (self.directory / "prices.csv").write_text(CSV, encoding="utf-8")

    def write_spec(self, payload: dict[str, Any]) -> Path:
        path = self.directory / "spec.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def write_csv(self, content: str) -> None:
        (self.directory / "prices.csv").write_text(content, encoding="utf-8")

    def assert_load_fails(self, payload: dict[str, Any], pattern: str) -> None:
        with self.assertRaisesRegex(SpecFileError, pattern):
            load_spec(self.write_spec(payload))


class RoundTripTests(SpecCase):
    def test_minimal_spec_round_trips_with_defaults(self) -> None:
        spec = load_spec(self.write_spec(minimal_payload()))
        self.assertEqual(spec.prices_csv_name, "prices.csv")
        self.assertEqual(spec.portfolio.names, ("AAA", "BBB"))
        self.assertEqual(spec.portfolio.weights, (0.6, 0.4))
        self.assertEqual([item.name for item in spec.asset_series], ["AAA", "BBB"])
        self.assertIsNone(spec.benchmark)
        self.assertIsNone(spec.benchmark_series)
        self.assertEqual(spec.risk_free_rate_annual, 0.0)
        self.assertEqual(spec.stress_windows, ())
        self.assertEqual(spec.stress_shocks, {})
        self.assertIsNone(spec.monte_carlo)

    def test_full_spec_round_trips_every_block(self) -> None:
        spec = load_spec(self.write_spec(full_payload()))
        self.assertEqual(spec.risk_free_rate_annual, 0.02)
        self.assertEqual(spec.benchmark, "DDD")
        assert spec.benchmark_series is not None
        self.assertEqual(spec.benchmark_series.name, "DDD")
        self.assertEqual(spec.benchmark_series.dates, spec.asset_series[0].dates)
        self.assertEqual(len(spec.stress_windows), 1)
        self.assertEqual(spec.stress_windows[0].name, "whole-week")
        self.assertEqual(spec.stress_windows[0].start, "2026-01-05")
        self.assertEqual(spec.stress_windows[0].end, "2026-01-13")
        self.assertEqual(spec.stress_shocks, {"hand-worked": {"AAA": -0.20, "BBB": -0.05}})
        assert spec.monte_carlo is not None
        self.assertEqual(spec.monte_carlo.mode, "parametric_normal")
        self.assertEqual(spec.monte_carlo.runs, 200)
        self.assertEqual(spec.monte_carlo.horizon_days, 21)
        self.assertEqual(spec.monte_carlo.seed, 7)

    def test_the_benchmark_column_stays_out_of_the_asset_series(self) -> None:
        spec = load_spec(self.write_spec(full_payload()))
        self.assertEqual([item.name for item in spec.asset_series], ["AAA", "BBB"])


class InputHashTests(SpecCase):
    def test_load_records_the_sha256_of_both_raw_input_files(self) -> None:
        spec_path = self.write_spec(minimal_payload())
        spec = load_spec(spec_path)
        self.assertEqual(spec.spec_sha256, hashlib.sha256(spec_path.read_bytes()).hexdigest())
        self.assertEqual(
            spec.prices_csv_sha256,
            hashlib.sha256((self.directory / "prices.csv").read_bytes()).hexdigest(),
        )

    def test_the_hash_covers_the_raw_bytes_including_a_bom(self) -> None:
        # The digest is taken from the bytes as read, before decoding:
        # a UTF-8 BOM changes the file, so it changes the hash, even
        # though the parsed spec is identical either way.
        spec_path = self.write_spec(minimal_payload())
        plain = load_spec(spec_path)
        (self.directory / "prices.csv").write_bytes(b"\xef\xbb\xbf" + CSV.encode("utf-8"))
        with_bom = load_spec(spec_path)
        self.assertEqual(plain.asset_series, with_bom.asset_series)
        self.assertNotEqual(plain.prices_csv_sha256, with_bom.prices_csv_sha256)
        self.assertEqual(plain.spec_sha256, with_bom.spec_sha256)


class DocumentErrorTests(SpecCase):
    def test_duplicate_json_keys_are_rejected(self) -> None:
        path = self.directory / "spec.json"
        path.write_text('{"schema_version": 1, "schema_version": 1}', encoding="utf-8")
        with self.assertRaisesRegex(SpecFileError, "duplicate JSON key"):
            load_spec(path)

    def test_invalid_json_is_rejected(self) -> None:
        path = self.directory / "spec.json"
        path.write_text("{not json", encoding="utf-8")
        with self.assertRaisesRegex(SpecFileError, "invalid spec JSON"):
            load_spec(path)

    def test_an_oversized_spec_is_rejected(self) -> None:
        path = self.directory / "spec.json"
        path.write_text(" " * 1_000_001, encoding="utf-8")
        with self.assertRaisesRegex(SpecFileError, "byte limit"):
            load_spec(path)

    def test_a_non_object_root_is_rejected(self) -> None:
        path = self.directory / "spec.json"
        path.write_text("[1, 2]", encoding="utf-8")
        with self.assertRaisesRegex(SpecFileError, "spec must be a JSON object"):
            load_spec(path)

    def test_unknown_root_keys_are_rejected(self) -> None:
        payload = minimal_payload()
        payload["stress_test"] = {}
        self.assert_load_fails(payload, "unknown spec key\\(s\\): stress_test")

    def test_wrong_schema_version_is_rejected(self) -> None:
        payload = minimal_payload()
        payload["schema_version"] = 2
        self.assert_load_fails(payload, "schema_version must be 1")

    def test_missing_portfolio_is_rejected(self) -> None:
        payload = minimal_payload()
        del payload["portfolio"]
        self.assert_load_fails(payload, "portfolio is required")

    def test_missing_prices_csv_is_rejected(self) -> None:
        payload = minimal_payload()
        del payload["prices_csv"]
        self.assert_load_fails(payload, "spec.prices_csv is required")


class PathRuleTests(SpecCase):
    def test_an_absolute_prices_path_is_rejected(self) -> None:
        payload = minimal_payload()
        payload["prices_csv"] = str(self.directory / "prices.csv")
        self.assert_load_fails(payload, "must be a relative path")

    def test_a_path_escaping_the_spec_directory_is_rejected(self) -> None:
        payload = minimal_payload()
        payload["prices_csv"] = "../prices.csv"
        self.assert_load_fails(payload, "escapes the spec's directory")

    def test_a_missing_csv_names_the_file(self) -> None:
        payload = minimal_payload()
        payload["prices_csv"] = "absent.csv"
        self.assert_load_fails(payload, "cannot access")

    def test_a_subdirectory_path_is_allowed(self) -> None:
        (self.directory / "data").mkdir()
        (self.directory / "data" / "prices.csv").write_text(CSV, encoding="utf-8")
        payload = minimal_payload()
        payload["prices_csv"] = "data/prices.csv"
        spec = load_spec(self.write_spec(payload))
        self.assertEqual(spec.prices_csv_name, "prices.csv")


class PricesCsvTests(SpecCase):
    def test_a_bad_header_is_rejected(self) -> None:
        self.write_csv("Time,AAA,BBB\n2026-01-05,1.0,2.0")
        self.assert_load_fails(minimal_payload(), "header must be 'Date'")

    def test_too_few_columns_are_rejected(self) -> None:
        self.write_csv("Date,AAA\n2026-01-05,1.0")
        self.assert_load_fails(minimal_payload(), "at least two tickers")

    def test_duplicate_tickers_are_rejected(self) -> None:
        self.write_csv("Date,AAA,AAA\n2026-01-05,1.0,2.0")
        self.assert_load_fails(minimal_payload(), "duplicate tickers")

    def test_a_ragged_row_names_its_line(self) -> None:
        self.write_csv("Date,AAA,BBB\n2026-01-05,100.0,50.0\n2026-01-06,101.0")
        self.assert_load_fails(minimal_payload(), "line 3 has 2 cells; expected 3")

    def test_a_non_numeric_price_names_line_and_ticker(self) -> None:
        self.write_csv("Date,AAA,BBB\n2026-01-05,100.0,50.0\n2026-01-06,oops,51.0")
        self.assert_load_fails(minimal_payload(), "line 3: AAA price 'oops'")

    def test_series_validation_errors_are_prefixed(self) -> None:
        self.write_csv(
            "Date,AAA,BBB\n2026-01-05,100.0,50.0\n2026-01-06,101.0,51.0\n2026-01-06,102.0,52.0"
        )
        self.assert_load_fails(minimal_payload(), "prices CSV prices.csv: .*strictly increasing")

    def test_an_empty_csv_is_rejected(self) -> None:
        self.write_csv("")
        self.assert_load_fails(minimal_payload(), "is empty")


class PortfolioBlockTests(SpecCase):
    def test_a_ticker_missing_from_the_csv_is_rejected(self) -> None:
        payload = minimal_payload()
        payload["portfolio"] = {"AAA": 0.6, "ZZZ": 0.4}
        self.assert_load_fails(payload, "\\['ZZZ'\\] are not columns")

    def test_weights_that_do_not_sum_to_one_are_rejected_with_prefix(self) -> None:
        payload = minimal_payload()
        payload["portfolio"] = {"AAA": 0.6, "BBB": 0.6}
        self.assert_load_fails(payload, "portfolio: .*sum to")

    def test_a_non_number_weight_names_the_ticker(self) -> None:
        payload = minimal_payload()
        payload["portfolio"] = {"AAA": 0.6, "BBB": "0.4"}
        self.assert_load_fails(payload, "portfolio.BBB must be a JSON number")

    def test_an_empty_portfolio_is_rejected(self) -> None:
        payload = minimal_payload()
        payload["portfolio"] = {}
        self.assert_load_fails(payload, "at least two tickers")

    def test_a_non_finite_risk_free_rate_is_rejected(self) -> None:
        payload = minimal_payload()
        payload["risk_free_rate_annual"] = "high"
        self.assert_load_fails(payload, "risk_free_rate_annual must be a JSON number")


class BenchmarkBlockTests(SpecCase):
    def test_a_benchmark_missing_from_the_csv_is_rejected(self) -> None:
        payload = minimal_payload()
        payload["benchmark"] = "ZZZ"
        self.assert_load_fails(payload, "benchmark 'ZZZ' is not a column")

    def test_a_benchmark_with_a_weight_is_rejected(self) -> None:
        payload = minimal_payload()
        payload["benchmark"] = "BBB"
        self.assert_load_fails(payload, "also carries a portfolio weight")

    def test_a_blank_benchmark_is_rejected(self) -> None:
        payload = minimal_payload()
        payload["benchmark"] = "  "
        self.assert_load_fails(payload, "spec.benchmark must be a non-blank string")


class StressBlockTests(SpecCase):
    def test_an_empty_stress_block_is_rejected(self) -> None:
        payload = minimal_payload()
        payload["stress"] = {}
        self.assert_load_fails(payload, "windows and/or shocks")

    def test_unknown_stress_keys_are_rejected(self) -> None:
        payload = minimal_payload()
        payload["stress"] = {"scenarios": {}}
        self.assert_load_fails(payload, "unknown stress key\\(s\\): scenarios")

    def test_empty_windows_are_rejected(self) -> None:
        payload = minimal_payload()
        payload["stress"] = {"windows": {}}
        self.assert_load_fails(payload, "at least one named window")

    def test_a_window_missing_its_start_is_rejected(self) -> None:
        payload = minimal_payload()
        payload["stress"] = {"windows": {"broken": {"end": "2026-01-09"}}}
        self.assert_load_fails(payload, "stress.windows\\['broken'\\].start is required")

    def test_a_reversed_window_is_rejected_with_its_field_named(self) -> None:
        payload = minimal_payload()
        payload["stress"] = {
            "windows": {"backwards": {"start": "2026-01-09", "end": "2026-01-05"}}
        }
        self.assert_load_fails(payload, "stress.windows\\['backwards'\\]: .*reversed")

    def test_a_window_outside_the_grid_fails_at_load_time(self) -> None:
        payload = minimal_payload()
        payload["stress"] = {"windows": {"early": {"start": "2025-12-01", "end": "2026-01-09"}}}
        self.assert_load_fails(payload, "stress.windows\\['early'\\]: .*outside the aligned grid")

    def test_empty_shocks_are_rejected(self) -> None:
        payload = minimal_payload()
        payload["stress"] = {"shocks": {}}
        self.assert_load_fails(payload, "at least one named shock vector")

    def test_an_incomplete_shock_fails_at_load_time(self) -> None:
        payload = minimal_payload()
        payload["stress"] = {"shocks": {"half": {"AAA": -0.1}}}
        self.assert_load_fails(payload, "stress.shocks\\['half'\\]: .*missing a value")

    def test_a_non_number_shock_names_its_field(self) -> None:
        payload = minimal_payload()
        payload["stress"] = {"shocks": {"bad": {"AAA": "-0.1", "BBB": 0.0}}}
        self.assert_load_fails(payload, "stress.shocks\\['bad'\\].AAA must be a JSON number")


class MonteCarloBlockTests(SpecCase):
    def payload_with(self, **overrides: Any) -> dict[str, Any]:
        payload = minimal_payload()
        block: dict[str, Any] = {
            "mode": "bootstrap",
            "runs": 200,
            "horizon_days": 21,
            "seed": 1,
        }
        block.update(overrides)
        payload["monte_carlo"] = {key: value for key, value in block.items() if value is not None}
        return payload

    def test_a_missing_seed_states_why_it_is_required(self) -> None:
        self.assert_load_fails(self.payload_with(seed=None), "cannot be reproduced")

    def test_an_unknown_mode_is_rejected(self) -> None:
        self.assert_load_fails(
            self.payload_with(mode="quantum"),
            "monte_carlo.mode must be one of: bootstrap, parametric_normal",
        )

    def test_runs_outside_the_bounds_fail_at_load_time(self) -> None:
        self.assert_load_fails(self.payload_with(runs=99), "between 100 and 20000")
        self.assert_load_fails(self.payload_with(runs=20_001), "between 100 and 20000")

    def test_a_horizon_outside_the_bounds_fails_at_load_time(self) -> None:
        self.assert_load_fails(self.payload_with(horizon_days=0), "between 1 and 2520")

    def test_a_non_integer_count_is_rejected(self) -> None:
        self.assert_load_fails(self.payload_with(runs=200.5), "must be a JSON integer")

    def test_unknown_monte_carlo_keys_are_rejected(self) -> None:
        self.assert_load_fails(
            self.payload_with(walkers=3), "unknown monte_carlo key\\(s\\): walkers"
        )


class WriteArtifactsTests(unittest.TestCase):
    def setUp(self) -> None:
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.output = Path(holder.name) / "results"

    def test_artifacts_are_written_and_returned(self) -> None:
        results_path, report_path = write_run_artifacts(self.output, "{}\n", "# Report\n")
        self.assertEqual(results_path.read_text(encoding="utf-8"), "{}\n")
        self.assertEqual(report_path.read_text(encoding="utf-8"), "# Report\n")

    def test_existing_artifacts_are_not_overwritten_without_force(self) -> None:
        write_run_artifacts(self.output, "first\n", "first\n")
        with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
            write_run_artifacts(self.output, "second\n", "second\n")
        self.assertEqual((self.output / "results.json").read_text(encoding="utf-8"), "first\n")

    def test_force_replaces_existing_artifacts(self) -> None:
        write_run_artifacts(self.output, "first\n", "first\n")
        write_run_artifacts(self.output, "second\n", "second\n", force=True)
        self.assertEqual((self.output / "results.json").read_text(encoding="utf-8"), "second\n")

    def test_no_staging_files_are_left_behind(self) -> None:
        write_run_artifacts(self.output, "{}\n", "# Report\n")
        leftovers = [path.name for path in self.output.iterdir()]
        self.assertEqual(sorted(leftovers), ["report.md", "results.json"])


class AtomicPairPublishTests(unittest.TestCase):
    """The pair publishes completely or not at all — never a mix (#24)."""

    def setUp(self) -> None:
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.output = Path(holder.name) / "results"

    @contextlib.contextmanager
    def failing_report_promotion(self) -> Iterator[None]:
        """Fail the atomic replace that would publish ``report.md``."""

        real_replace = Path.replace

        def fail_second_promotion(source: Path, target: str | Path) -> Path:
            if source.name.endswith(".stage.tmp") and Path(target).name == "report.md":
                raise OSError("synthetic second-artifact failure")
            return real_replace(source, target)

        with patch.object(Path, "replace", new=fail_second_promotion):
            yield

    def read_pair(self) -> dict[str, bytes]:
        return {path.name: path.read_bytes() for path in sorted(self.output.iterdir())}

    def test_a_failed_second_promotion_publishes_nothing_into_a_fresh_directory(self) -> None:
        with (
            self.failing_report_promotion(),
            self.assertRaisesRegex(OSError, "second-artifact failure"),
        ):
            write_run_artifacts(self.output, "orphan\n", "# never published\n")
        self.assertEqual(sorted(path.name for path in self.output.iterdir()), [])

    def test_a_failed_second_promotion_restores_the_forced_over_pair(self) -> None:
        write_run_artifacts(self.output, "old results\n", "old report\n")
        original = self.read_pair()
        with (
            self.failing_report_promotion(),
            self.assertRaisesRegex(OSError, "second-artifact failure"),
        ):
            write_run_artifacts(self.output, "new results\n", "new report\n", force=True)
        self.assertEqual(self.read_pair(), original)

    def test_a_failed_second_staging_leaves_the_original_pair_untouched(self) -> None:
        write_run_artifacts(self.output, "old results\n", "old report\n")
        original = self.read_pair()
        real_stage = quantrisk.io._stage_text

        def fail_report_staging(path: Path, content: str) -> Path:
            if path.name == "report.md":
                raise OSError("synthetic staging failure")
            return real_stage(path, content)

        with (
            patch("quantrisk.io._stage_text", new=fail_report_staging),
            self.assertRaisesRegex(OSError, "staging failure"),
        ):
            write_run_artifacts(self.output, "new results\n", "new report\n", force=True)
        self.assertEqual(self.read_pair(), original)

    def test_a_forced_success_leaves_only_the_new_pair(self) -> None:
        write_run_artifacts(self.output, "old results\n", "old report\n")
        write_run_artifacts(self.output, "new results\n", "new report\n", force=True)
        self.assertEqual(
            self.read_pair(),
            {"report.md": b"new report\n", "results.json": b"new results\n"},
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
