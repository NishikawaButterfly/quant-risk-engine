"""Bounded spec ingestion and failure-safe artifact publication.

A run spec is a single JSON document that describes everything one run
needs: the prices CSV, the portfolio weights by ticker, and optionally
a risk-free rate, a benchmark ticker, a stress block, and a Monte
Carlo block with its mandatory seed. Loading is bounded and strict —
both files have size caps, duplicate JSON keys, unknown keys, and
wrong types are rejected with the offending field named, and a spec
either loads completely or not at all; no partial objects escape.
Numbers are strict too: JSON ``true``/``false`` never count as
numbers, and the non-standard ``NaN``/``Infinity``/``-Infinity``
tokens (which Python's :mod:`json` would otherwise happily parse into
floats) are refused at parse, so no non-finite value can enter through
a spec.

The prices CSV path must be relative and resolve inside the spec's own
directory. A spec that could read files elsewhere on the machine is a
confused deputy waiting to happen, and a spec whose data does not
travel with it cannot be reproduced by whoever receives it.

Both input files are hashed with SHA-256 at read time, over the raw
bytes exactly as they came off the disk — before any decoding or
parsing — and the digests travel on the :class:`RunSpec`. The results
payload publishes them in its provenance block, so an artifact states
verifiably which input bytes produced it.

Writing publishes the artifact pair atomically as a pair: both files
are staged next to their targets, any existing pair is set aside, and
only then are both promoted with atomic replaces. If anything fails
before both promotions complete, the staged files are removed and the
set-aside pair is restored, so the output directory holds either the
complete new pair or exactly what it held before — never one new file
beside one old one. Overwriting existing results still requires an
explicit ``force``.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quantrisk._validation import require_finite_number
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
    carry them. ``spec_sha256`` and ``prices_csv_sha256`` are SHA-256
    hex digests of the two input files' raw bytes as read, taken
    before decoding or parsing, so they identify the exact file
    contents rather than any re-serialization.
    """

    prices_csv_name: str
    spec_sha256: str
    prices_csv_sha256: str
    portfolio: Portfolio
    asset_series: tuple[PriceSeries, ...]
    benchmark: str | None
    benchmark_series: PriceSeries | None
    risk_free_rate_annual: float
    stress_windows: tuple[StressWindow, ...]
    stress_shocks: dict[str, dict[str, float]]
    monte_carlo: MonteCarloSpec | None


def _reject_nonfinite_token(token: str) -> Any:
    """Refuse the non-standard JSON tokens ``NaN`` and ``±Infinity``.

    Python's :func:`json.loads` parses them into floats by default;
    a spec carrying one describes no finite number, so the document is
    rejected at parse rather than validated field by field later.
    """

    raise SpecFileError(f"spec JSON contains {token}; every number in a spec must be finite")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SpecFileError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_bounded_text(path: Path, maximum_bytes: int) -> tuple[str, str]:
    """The file's decoded text and the SHA-256 hex digest of its raw bytes.

    The digest is taken over the bytes exactly as read, before the
    UTF-8 decode (so a byte-order mark counts): it identifies the file
    content on disk, which is what a verifier will hash independently.
    """

    try:
        size = path.stat().st_size
    except OSError as exc:
        raise SpecFileError(f"cannot access {path}: {exc}") from exc
    if size > maximum_bytes:
        raise SpecFileError(f"{path.name} exceeds the {maximum_bytes}-byte limit")
    try:
        raw = path.read_bytes()
        return raw.decode("utf-8-sig"), hashlib.sha256(raw).hexdigest()
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
    # Defense in depth: the parse hook already refuses the tokens that
    # produce non-finite floats, so this shared check should never
    # fire from load_spec; it stays so the field rule holds on its own.
    try:
        return require_finite_number(value, f"{path}.{key}")
    except ValueError as exc:
        raise SpecFileError(str(exc)) from exc


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


def _parse_prices_csv(path: Path, text: str) -> tuple[PriceSeries, ...]:
    """Every column of the CSV as a validated price series, in column order."""

    lines = text.splitlines()
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
    field error names the offending field. Each file's raw bytes are
    hashed with SHA-256 as they are read, and the digests travel on
    the returned spec so the results can publish them as provenance.
    The prices CSV path must be relative and stay inside the spec's
    directory. Cross-references are
    checked here — portfolio tickers and the benchmark must be CSV
    columns, the benchmark must not carry a weight, and every stress
    scenario must be valid against the loaded data. Whether the loaded
    spec can then be *evaluated* — constant series, too few paired
    returns, non-finite figures — is the job of
    :func:`quantrisk.feasibility.check_feasibility`, which the CLI
    calls right after this function on both its paths.
    """

    spec_path = Path(path)
    spec_text, spec_sha256 = _read_bounded_text(spec_path, _MAX_SPEC_BYTES)
    try:
        payload = json.loads(
            spec_text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_nonfinite_token,
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
    prices_text, prices_csv_sha256 = _read_bounded_text(prices_path, _MAX_PRICES_BYTES)
    all_series = _parse_prices_csv(prices_path, prices_text)
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
        spec_sha256=spec_sha256,
        prices_csv_sha256=prices_csv_sha256,
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


def _reserve_backup_path(path: Path) -> Path:
    """A collision-safe sibling name for setting an existing target aside."""

    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".backup.tmp",
    )
    os.close(descriptor)
    backup_path = Path(temporary_name)
    backup_path.unlink()
    return backup_path


def _undo_publication(published: set[Path], backups: dict[Path, Path]) -> list[OSError]:
    """Remove promoted files and move backups back, collecting what failed."""

    errors: list[OSError] = []
    for target in published.difference(backups):
        try:
            target.unlink(missing_ok=True)
        except OSError as exc:
            errors.append(exc)
    for target, backup in reversed(tuple(backups.items())):
        try:
            backup.replace(target)
        except OSError as exc:
            errors.append(exc)
    return errors


def _publish_pair(artifacts: tuple[tuple[Path, str], ...]) -> None:
    """Stage every artifact, publish them together, restore the prior pair on failure.

    Three phases: stage all contents beside their targets, move every
    existing target aside to a backup name, then promote every staged
    file with an atomic :meth:`Path.replace`. A failure in any phase
    unwinds this invocation completely — promoted files are removed,
    backups are moved back — so the directory ends holding either the
    whole new pair or exactly the files it held before. Each individual
    step is an ``os.replace`` within one directory, so it never spans
    filesystems; on Windows it would fail if another process held a
    published file open, in which case the rollback runs and the raised
    error names any file it could not restore.
    """

    staged: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    published: set[Path] = set()
    try:
        for target, content in artifacts:
            staged[target] = _stage_text(target, content)

        for target, _ in artifacts:
            if target.exists():
                backup = _reserve_backup_path(target)
                target.replace(backup)
                backups[target] = backup

        for target, _ in artifacts:
            staged[target].replace(target)
            published.add(target)
    except Exception as publication_error:
        rollback_errors = _undo_publication(published, backups)
        if rollback_errors:
            raise OSError(
                "artifact publication failed and rollback was incomplete; the "
                "previous artifacts survive as hidden .backup.tmp files beside "
                f"their targets: {'; '.join(str(error) for error in rollback_errors)}"
            ) from publication_error
        raise
    else:
        for backup in backups.values():
            with suppress(OSError):
                backup.unlink(missing_ok=True)
    finally:
        for temporary in staged.values():
            with suppress(OSError):
                temporary.unlink(missing_ok=True)


def write_run_artifacts(
    output_directory: str | Path,
    results_json: str,
    report_markdown: str,
    *,
    force: bool = False,
) -> tuple[Path, Path]:
    """Publish ``results.json`` and ``report.md`` as one atomic pair.

    The two artifacts are one result, so they publish together or not
    at all: on any failure the output directory holds either the
    complete new pair or exactly what it held before — a forced
    overwrite that fails restores the previous pair byte for byte.
    Existing results are only replaced when ``force`` is set. A process
    kill in mid-publish can leave hidden ``.*.tmp`` staging or backup
    files beside the targets; they are safe to delete.
    """

    output = Path(output_directory)
    results_path = output / "results.json"
    report_path = output / "report.md"
    existing = [path for path in (results_path, report_path) if path.exists()]
    if existing and not force:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"refusing to overwrite {names}; pass --force to replace them")

    _publish_pair(((results_path, results_json), (report_path, report_markdown)))
    return results_path, report_path
