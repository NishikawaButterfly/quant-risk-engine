"""Bounded spec ingestion and failure-safe artifact publication.

A run spec is a single JSON document that describes everything one run
needs: the prices CSV, the portfolio weights by ticker, and optionally
a risk-free rate, a benchmark ticker, a stress block, and a Monte
Carlo block with its mandatory seed. Loading is bounded and strict —
both files have size caps, duplicate JSON keys, unknown keys, and
wrong types are rejected with the offending field named, and a spec
either loads completely or not at all; no partial objects escape.

The prices CSV path must be relative and resolve inside the spec's own
directory. A spec that could read files elsewhere on the machine is a
confused deputy waiting to happen, and a spec whose data does not
travel with it cannot be reproduced by whoever receives it.

Writing stages every artifact next to its target and publishes with
atomic replaces, and it refuses to overwrite existing results unless
explicitly forced.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quantrisk.montecarlo import (
    MAX_HORIZON_DAYS,
    MAX_RUNS,
    MIN_HORIZON_DAYS,
    MIN_RUNS,
    SimulationMode,
)
from quantrisk.portfolio import Portfolio
from quantrisk.series import PriceSeries, align
from quantrisk.stress import StressWindow, historical_stress, shock_stress

SPEC_SCHEMA_VERSION = 1
_MAX_SPEC_BYTES = 1_000_000
_MAX_PRICES_BYTES = 5_000_000

#: A prices CSV needs the date column plus at least the two tickers a
#: portfolio must hold.
_MIN_CSV_COLUMNS = 3

_ROOT_KEYS = {
    "schema_version",
    "prices_csv",
    "portfolio",
    "risk_free_rate_annual",
    "benchmark",
    "stress",
    "monte_carlo",
}
_STRESS_KEYS = {"windows", "shocks"}
_WINDOW_KEYS = {"start", "end"}
_MONTE_CARLO_KEYS = {"mode", "runs", "horizon_days", "seed"}
_SIMULATION_MODES = ("bootstrap", "parametric_normal")


class SpecFileError(ValueError):
    """Raised when a spec or its prices CSV is unsafe, malformed, or unsupported."""


@dataclass(frozen=True, slots=True)
class MonteCarloSpec:
    """The Monte Carlo request of a spec: which mode, how often, which seed."""

    mode: SimulationMode
    runs: int
    horizon_days: int
    seed: int


@dataclass(frozen=True, slots=True)
class RunSpec:
    """One fully parsed run spec with its price data already aligned.

    ``asset_series`` are the portfolio's assets in portfolio order and
    ``benchmark_series`` (when a benchmark is configured) shares their
    date grid; both come out of :func:`~quantrisk.series.align`, so the
    intersection policy has already been applied. The stress and Monte
    Carlo blocks are empty (or ``None``) when the document does not
    carry them.
    """

    prices_csv_name: str
    portfolio: Portfolio
    asset_series: tuple[PriceSeries, ...]
    benchmark: str | None
    benchmark_series: PriceSeries | None
    risk_free_rate_annual: float
    stress_windows: tuple[StressWindow, ...]
    stress_shocks: dict[str, dict[str, float]]
    monte_carlo: MonteCarloSpec | None


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SpecFileError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_bounded_text(path: Path, maximum_bytes: int) -> str:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise SpecFileError(f"cannot access {path}: {exc}") from exc
    if size > maximum_bytes:
        raise SpecFileError(f"{path.name} exceeds the {maximum_bytes}-byte limit")
    try:
        return path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise SpecFileError(f"cannot read {path} as UTF-8: {exc}") from exc


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SpecFileError(f"{path} must be a JSON object")
    return value


def _reject_unknown_keys(mapping: dict[str, Any], path: str, allowed: set[str]) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise SpecFileError(f"unknown {path} key(s): {', '.join(unknown)}")


def _required_string(mapping: dict[str, Any], path: str, key: str) -> str:
    if key not in mapping:
        raise SpecFileError(f"{path}.{key} is required")
    value = mapping[key]
    if not isinstance(value, str) or not value.strip():
        raise SpecFileError(f"{path}.{key} must be a non-blank string")
    return value


def _number(mapping: dict[str, Any], path: str, key: str) -> float:
    if key not in mapping:
        raise SpecFileError(f"{path}.{key} is required")
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SpecFileError(f"{path}.{key} must be a JSON number")
    return float(value)


def _integer(mapping: dict[str, Any], path: str, key: str) -> int:
    if key not in mapping:
        raise SpecFileError(f"{path}.{key} is required")
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpecFileError(f"{path}.{key} must be a JSON integer")
    return value


@contextmanager
def _model_errors(path: str) -> Iterator[None]:
    """Re-raise a model's own ``ValueError`` with the spec path prefixed."""

    try:
        yield
    except SpecFileError:
        raise
    except ValueError as exc:
        raise SpecFileError(f"{path}: {exc}") from exc


def _resolve_prices_path(spec_path: Path, relative: str) -> Path:
    """Apply the path rule: relative, and inside the spec's directory."""

    candidate = Path(relative)
    if candidate.is_absolute():
        raise SpecFileError(f"prices_csv must be a relative path, got {relative!r}")
    directory = spec_path.parent.resolve()
    resolved = (directory / candidate).resolve()
    if directory != resolved and directory not in resolved.parents:
        raise SpecFileError(
            f"prices_csv {relative!r} escapes the spec's directory; the price data "
            "must live beside the spec so a run can be reproduced from the pair"
        )
    return resolved


def _parse_prices_csv(path: Path) -> tuple[PriceSeries, ...]:
    """Every column of the CSV as a validated price series, in column order."""

    lines = _read_bounded_text(path, _MAX_PRICES_BYTES).splitlines()
    if not lines:
        raise SpecFileError(f"prices CSV {path.name} is empty")
    header = lines[0].split(",")
    if header[0] != "Date" or len(header) < _MIN_CSV_COLUMNS:
        raise SpecFileError(
            f"prices CSV {path.name} header must be 'Date' followed by at least "
            f"two tickers, got {lines[0]!r}"
        )
    tickers = header[1:]
    if len(set(tickers)) != len(tickers):
        raise SpecFileError(f"prices CSV {path.name} has duplicate tickers: {tickers}")
    dates: list[str] = []
    columns: list[list[float]] = [[] for _ in tickers]
    for line_number, line in enumerate(lines[1:], start=2):
        cells = line.split(",")
        if len(cells) != len(header):
            raise SpecFileError(
                f"prices CSV {path.name} line {line_number} has {len(cells)} cells; "
                f"expected {len(header)}"
            )
        dates.append(cells[0])
        for ticker, column, cell in zip(tickers, columns, cells[1:], strict=True):
            try:
                column.append(float(cell))
            except ValueError as exc:
                raise SpecFileError(
                    f"prices CSV {path.name} line {line_number}: {ticker} price "
                    f"{cell!r} is not a number"
                ) from exc
    series = []
    for ticker, column in zip(tickers, columns, strict=True):
        with _model_errors(f"prices CSV {path.name}"):
            series.append(PriceSeries(name=ticker, dates=tuple(dates), prices=tuple(column)))
    return tuple(series)


def _parse_portfolio(data: dict[str, Any], available: tuple[str, ...]) -> Portfolio:
    if not data:
        raise SpecFileError("portfolio must map at least two tickers to weights")
    missing = sorted(set(data) - set(available))
    if missing:
        raise SpecFileError(
            f"portfolio ticker(s) {missing} are not columns of the prices CSV "
            f"(available: {list(available)})"
        )
    weights = [_number(data, "portfolio", ticker) for ticker in data]
    with _model_errors("portfolio"):
        return Portfolio(names=tuple(data), weights=tuple(weights))


def _parse_benchmark(
    root: dict[str, Any], available: tuple[str, ...], portfolio: Portfolio
) -> str | None:
    if "benchmark" not in root:
        return None
    ticker = _required_string(root, "spec", "benchmark")
    if ticker not in available:
        raise SpecFileError(
            f"benchmark {ticker!r} is not a column of the prices CSV "
            f"(available: {list(available)})"
        )
    if ticker in portfolio.names:
        raise SpecFileError(
            f"benchmark {ticker!r} also carries a portfolio weight; a portfolio "
            "measured against itself would flatter every statistic, so the "
            "benchmark must stay outside the portfolio"
        )
    return ticker


def _parse_windows(data: dict[str, Any]) -> tuple[StressWindow, ...]:
    if not data:
        raise SpecFileError("stress.windows must contain at least one named window")
    windows = []
    for name, entry in data.items():
        path = f"stress.windows[{name!r}]"
        entry_map = _mapping(entry, path)
        _reject_unknown_keys(entry_map, path, _WINDOW_KEYS)
        with _model_errors(path):
            windows.append(
                StressWindow(
                    name=name,
                    start=_required_string(entry_map, path, "start"),
                    end=_required_string(entry_map, path, "end"),
                )
            )
    return tuple(windows)


def _parse_shocks(data: dict[str, Any]) -> dict[str, dict[str, float]]:
    if not data:
        raise SpecFileError("stress.shocks must contain at least one named shock vector")
    shocks: dict[str, dict[str, float]] = {}
    for name, entry in data.items():
        path = f"stress.shocks[{name!r}]"
        entry_map = _mapping(entry, path)
        if not entry_map:
            raise SpecFileError(f"{path} must map at least one asset to a shock")
        shocks[name] = {asset: _number(entry_map, path, asset) for asset in entry_map}
    return shocks


def _parse_monte_carlo(data: dict[str, Any]) -> MonteCarloSpec:
    _reject_unknown_keys(data, "monte_carlo", _MONTE_CARLO_KEYS)
    mode = data.get("mode")
    if mode not in _SIMULATION_MODES:
        raise SpecFileError(f"monte_carlo.mode must be one of: {', '.join(_SIMULATION_MODES)}")
    if "seed" not in data:
        raise SpecFileError(
            "monte_carlo.seed is required; an unseeded simulation cannot be reproduced"
        )
    runs = _integer(data, "monte_carlo", "runs")
    if not MIN_RUNS <= runs <= MAX_RUNS:
        raise SpecFileError(f"monte_carlo.runs must be between {MIN_RUNS} and {MAX_RUNS}")
    horizon_days = _integer(data, "monte_carlo", "horizon_days")
    if not MIN_HORIZON_DAYS <= horizon_days <= MAX_HORIZON_DAYS:
        raise SpecFileError(
            f"monte_carlo.horizon_days must be between {MIN_HORIZON_DAYS} and {MAX_HORIZON_DAYS}"
        )
    return MonteCarloSpec(
        mode=mode,
        runs=runs,
        horizon_days=horizon_days,
        seed=_integer(data, "monte_carlo", "seed"),
    )


def _check_stress_against_data(
    portfolio: Portfolio,
    asset_series: tuple[PriceSeries, ...],
    windows: tuple[StressWindow, ...],
    shocks: dict[str, dict[str, float]],
) -> None:
    """Surface every stress error at load time, with the field named.

    The stress engine itself owns the rules — window inside the grid,
    enough observations, complete shock vectors above -1 — so the load
    runs each scenario through it once and discards the results rather
    than re-stating the rules here. The arithmetic involved is a few
    hundred multiplications; what matters is that ``validate`` and
    ``run`` cannot disagree about what a valid spec is.
    """

    for window in windows:
        with _model_errors(f"stress.windows[{window.name!r}]"):
            historical_stress(portfolio, asset_series, [window])
    for name, vector in shocks.items():
        with _model_errors(f"stress.shocks[{name!r}]"):
            shock_stress(portfolio, {name: vector})


def load_spec(path: str | Path) -> RunSpec:
    """Load one JSON run spec and its price data, without side effects.

    Both reads are bounded, duplicate keys are rejected, and every
    field error names the offending field. The prices CSV path must be
    relative and stay inside the spec's directory. Cross-references are
    checked here — portfolio tickers and the benchmark must be CSV
    columns, the benchmark must not carry a weight, and every stress
    scenario must be valid against the loaded data — so a spec that
    loads is a spec that runs.
    """

    spec_path = Path(path)
    try:
        payload = json.loads(
            _read_bounded_text(spec_path, _MAX_SPEC_BYTES),
            object_pairs_hook=_unique_json_object,
        )
    except json.JSONDecodeError as exc:
        raise SpecFileError(f"invalid spec JSON: {exc}") from exc
    root = _mapping(payload, "spec")
    _reject_unknown_keys(root, "spec", _ROOT_KEYS)
    if root.get("schema_version") != SPEC_SCHEMA_VERSION:
        raise SpecFileError(
            f"schema_version must be {SPEC_SCHEMA_VERSION}; "
            f"received {root.get('schema_version')!r}"
        )
    prices_path = _resolve_prices_path(spec_path, _required_string(root, "spec", "prices_csv"))
    all_series = _parse_prices_csv(prices_path)
    available = tuple(item.name for item in all_series)
    if "portfolio" not in root:
        raise SpecFileError("portfolio is required")
    portfolio = _parse_portfolio(_mapping(root["portfolio"], "portfolio"), available)
    benchmark = _parse_benchmark(root, available, portfolio)

    risk_free_rate_annual = 0.0
    if "risk_free_rate_annual" in root:
        risk_free_rate_annual = _number(root, "spec", "risk_free_rate_annual")

    used_names = [*portfolio.names, *([benchmark] if benchmark is not None else [])]
    by_name = {item.name: item for item in all_series}
    with _model_errors("prices_csv"):
        aligned = align([by_name[name] for name in used_names])
    asset_series = aligned[: len(portfolio.names)]
    benchmark_series = aligned[-1] if benchmark is not None else None

    windows: tuple[StressWindow, ...] = ()
    shocks: dict[str, dict[str, float]] = {}
    if "stress" in root:
        stress = _mapping(root["stress"], "stress")
        _reject_unknown_keys(stress, "stress", _STRESS_KEYS)
        if not stress:
            raise SpecFileError("stress must contain windows and/or shocks")
        if "windows" in stress:
            windows = _parse_windows(_mapping(stress["windows"], "stress.windows"))
        if "shocks" in stress:
            shocks = _parse_shocks(_mapping(stress["shocks"], "stress.shocks"))
        _check_stress_against_data(portfolio, asset_series, windows, shocks)

    monte_carlo = (
        _parse_monte_carlo(_mapping(root["monte_carlo"], "monte_carlo"))
        if "monte_carlo" in root
        else None
    )
    return RunSpec(
        prices_csv_name=prices_path.name,
        portfolio=portfolio,
        asset_series=asset_series,
        benchmark=benchmark,
        benchmark_series=benchmark_series,
        risk_free_rate_annual=risk_free_rate_annual,
        stress_windows=windows,
        stress_shocks=shocks,
        monte_carlo=monte_carlo,
    )


def _stage_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".stage.tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        with suppress(OSError):
            temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def write_run_artifacts(
    output_directory: str | Path,
    results_json: str,
    report_markdown: str,
    *,
    force: bool = False,
) -> tuple[Path, Path]:
    """Publish ``results.json`` and ``report.md``, refusing silent overwrite.

    Both artifacts are staged next to their targets first, so a failure
    while staging publishes nothing, and each publication is an atomic
    replace. Existing results are only replaced when ``force`` is set.
    """

    output = Path(output_directory)
    results_path = output / "results.json"
    report_path = output / "report.md"
    existing = [path for path in (results_path, report_path) if path.exists()]
    if existing and not force:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"refusing to overwrite {names}; pass --force to replace them")

    staged: list[tuple[Path, Path]] = []
    try:
        for target, content in ((results_path, results_json), (report_path, report_markdown)):
            staged.append((target, _stage_text(target, content)))
        for target, temporary in staged:
            temporary.replace(target)
    finally:
        for _, temporary in staged:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
    return results_path, report_path
