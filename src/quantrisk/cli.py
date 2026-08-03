"""Command-line interface: one spec file in, results and a report out."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, NoReturn

from quantrisk.io import RunSpec, load_spec, write_run_artifacts
from quantrisk.report import evaluate_spec, render_report, results_payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantrisk",
        description="Compute portfolio risk evidence from a JSON spec file.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate", help="parse a spec and report its contents without computing"
    )
    validate.add_argument("--spec", type=Path, required=True)

    run = subparsers.add_parser("run", help="evaluate a spec and write results.json and report.md")
    run.add_argument("--spec", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--force", action="store_true")
    return parser


def _fail(message: str) -> NoReturn:
    raise SystemExit(f"error: {message}")


def _describe(spec: RunSpec) -> dict[str, Any]:
    """What the spec contains, without computing any metric."""

    monte_carlo: dict[str, Any] | None = None
    if spec.monte_carlo is not None:
        monte_carlo = {
            "mode": spec.monte_carlo.mode,
            "runs": spec.monte_carlo.runs,
            "horizon_days": spec.monte_carlo.horizon_days,
            "seed": spec.monte_carlo.seed,
        }
    grid = spec.asset_series[0].dates
    return {
        "status": "valid",
        "prices_csv": spec.prices_csv_name,
        "trading_days": len(grid),
        "first_date": grid[0],
        "last_date": grid[-1],
        "weights": dict(zip(spec.portfolio.names, spec.portfolio.weights, strict=True)),
        "benchmark": spec.benchmark,
        "risk_free_rate_annual": spec.risk_free_rate_annual,
        "stress_windows": [window.name for window in spec.stress_windows],
        "stress_shocks": list(spec.stress_shocks),
        "monte_carlo": monte_carlo,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        spec = load_spec(args.spec)
        if args.command == "validate":
            print(json.dumps(_describe(spec), indent=2, sort_keys=True))
            return 0

        evaluation = evaluate_spec(spec)
        results_content = (
            json.dumps(results_payload(evaluation), allow_nan=False, indent=2, sort_keys=True)
            + "\n"
        )
        report_content = render_report(evaluation, source_name=args.spec.name)
        results_path, report_path = write_run_artifacts(
            args.output, results_content, report_content, force=args.force
        )
        print(f"results: {results_path}")
        print(f"report: {report_path}")
        return 0
    except (OSError, ValueError) as exc:
        _fail(str(exc))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
