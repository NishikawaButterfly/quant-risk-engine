"""The reporting layer: evaluate a spec once, render every artifact from it.

``evaluate_spec`` runs the loaded spec through the existing entry
points — the metrics, portfolio, benchmark, stress, and Monte Carlo
modules — and returns one frozen evaluation. ``results_payload`` and
``render_report`` both read that evaluation and never recompute a
number, so the JSON results and the Markdown report cannot disagree.
Neither function touches a file, prints, or looks at the clock: the
same spec always renders the same bytes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from quantrisk.benchmark import BenchmarkComparison, compare_to_benchmark
from quantrisk.io import SPEC_SCHEMA_VERSION, RunSpec
from quantrisk.metrics import (
    TRADING_DAYS_PER_YEAR,
    Drawdown,
    annualized_volatility,
    historical_cvar,
    historical_var,
    max_drawdown,
    sharpe_ratio,
)
from quantrisk.montecarlo import MonteCarloResult, run_bootstrap, run_parametric_normal
from quantrisk.series import PriceSeries
from quantrisk.stress import (
    ShockStressResult,
    WindowStressResult,
    historical_stress,
    shock_stress,
)

#: Every VaR and CVaR in a report is historical at this confidence.
VAR_CONFIDENCE = 0.95


@dataclass(frozen=True, slots=True)
class AssetMetrics:
    """One return series' metrics under the engine's stated conventions."""

    name: str
    total_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: Drawdown
    var_95: float
    cvar_95: float


@dataclass(frozen=True, slots=True)
class PortfolioMetrics:
    """The portfolio's own metrics plus its risk decomposition.

    ``condition_number`` is the covariance matrix's 2-norm condition
    number (:data:`math.inf` for an exactly singular matrix) and
    ``conditioning_warning`` its disclosure past the warn limit — see
    :meth:`quantrisk.portfolio.Portfolio.covariance_diagnostics`.
    """

    total_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: Drawdown
    var_95: float
    cvar_95: float
    diversification_benefit: float
    risk_contributions: dict[str, float]
    condition_number: float
    conditioning_warning: str | None


@dataclass(frozen=True, slots=True)
class SpecEvaluation:
    """Everything one spec evaluates to, in one frozen object."""

    spec: RunSpec
    assets: tuple[AssetMetrics, ...]
    portfolio: PortfolioMetrics
    benchmark: AssetMetrics | None
    benchmark_comparison: BenchmarkComparison | None
    stress_windows: tuple[WindowStressResult, ...]
    stress_shocks: tuple[ShockStressResult, ...]
    monte_carlo: MonteCarloResult | None


def _series_metrics(name: str, series: PriceSeries, risk_free_rate_annual: float) -> AssetMetrics:
    returns = series.simple_returns()
    return AssetMetrics(
        name=name,
        total_return=series.prices[-1] / series.prices[0] - 1.0,
        annualized_volatility=annualized_volatility(returns),
        sharpe_ratio=sharpe_ratio(returns, risk_free_rate_annual),
        max_drawdown=max_drawdown(series),
        var_95=historical_var(returns, VAR_CONFIDENCE),
        cvar_95=historical_cvar(returns, VAR_CONFIDENCE),
    )


def _portfolio_value_path(spec: RunSpec) -> PriceSeries:
    """The portfolio's daily-rebalanced value path, compounded from 1.0."""

    values = [1.0]
    for value in spec.portfolio.return_series(spec.asset_series):
        values.append(values[-1] * (1.0 + value))
    return PriceSeries(
        name="portfolio",
        dates=spec.asset_series[0].dates,
        prices=tuple(values),
    )


def _portfolio_metrics(spec: RunSpec) -> PortfolioMetrics:
    returns = spec.portfolio.return_series(spec.asset_series)
    path = _portfolio_value_path(spec)
    diagnostics = spec.portfolio.covariance_diagnostics(spec.asset_series)
    return PortfolioMetrics(
        total_return=path.prices[-1] - 1.0,
        annualized_volatility=spec.portfolio.annualized_volatility(spec.asset_series),
        sharpe_ratio=sharpe_ratio(returns, spec.risk_free_rate_annual),
        max_drawdown=max_drawdown(path),
        var_95=historical_var(returns, VAR_CONFIDENCE),
        cvar_95=historical_cvar(returns, VAR_CONFIDENCE),
        diversification_benefit=spec.portfolio.diversification_benefit(spec.asset_series),
        risk_contributions=spec.portfolio.risk_contributions(spec.asset_series),
        condition_number=diagnostics.condition_number,
        conditioning_warning=diagnostics.conditioning_warning,
    )


def _monte_carlo(spec: RunSpec) -> MonteCarloResult | None:
    block = spec.monte_carlo
    if block is None:
        return None
    run = run_bootstrap if block.mode == "bootstrap" else run_parametric_normal
    return run(
        spec.portfolio,
        spec.asset_series,
        horizon_days=block.horizon_days,
        runs=block.runs,
        seed=block.seed,
    )


def evaluate_spec(spec: RunSpec) -> SpecEvaluation:
    """Evaluate everything the spec carries through the existing layers.

    The per-asset and portfolio metrics always run; the benchmark
    comparison, the stress scenarios, and the Monte Carlo simulation
    run only when the spec configures them. Every number comes from the
    same entry points a direct library call would use.
    """

    rate = spec.risk_free_rate_annual
    benchmark_metrics: AssetMetrics | None = None
    comparison: BenchmarkComparison | None = None
    if spec.benchmark is not None and spec.benchmark_series is not None:
        benchmark_metrics = _series_metrics(spec.benchmark, spec.benchmark_series, rate)
        # The spec's series come out of align() at load time, so the
        # comparison's own grid check passes by construction; the
        # comparison still re-derives both return series itself.
        comparison = compare_to_benchmark(
            spec.portfolio,
            spec.asset_series,
            spec.benchmark_series,
            risk_free_rate_daily=rate / TRADING_DAYS_PER_YEAR,
        )
    return SpecEvaluation(
        spec=spec,
        assets=tuple(_series_metrics(series.name, series, rate) for series in spec.asset_series),
        portfolio=_portfolio_metrics(spec),
        benchmark=benchmark_metrics,
        benchmark_comparison=comparison,
        stress_windows=(
            historical_stress(spec.portfolio, spec.asset_series, spec.stress_windows)
            if spec.stress_windows
            else ()
        ),
        stress_shocks=(
            shock_stress(spec.portfolio, spec.stress_shocks) if spec.stress_shocks else ()
        ),
        monte_carlo=_monte_carlo(spec),
    )


def _metrics_dict(metrics: AssetMetrics) -> dict[str, Any]:
    payload = asdict(metrics)
    del payload["name"]
    return payload


def results_payload(evaluation: SpecEvaluation) -> dict[str, Any]:
    """Every computed number as one JSON-ready dictionary.

    The Monte Carlo section keeps the seed, the run count, and the full
    sorted terminal values, so any summary figure can be recomputed and
    checked. Nothing here reads the clock: the payload of a given spec
    is always the same.
    """

    spec = evaluation.spec
    payload: dict[str, Any] = {
        "schema_version": SPEC_SCHEMA_VERSION,
        "prices_csv": spec.prices_csv_name,
        "conventions": {
            "annualization_days": TRADING_DAYS_PER_YEAR,
            "sample_denominator": "n - 1",
            "returns": "daily simple returns as plain fractions",
            "portfolio_returns": "weighted daily returns (daily rebalancing)",
            "var_confidence": VAR_CONFIDENCE,
            "percentiles": "linear interpolation between closest ranks",
        },
        "portfolio": {
            "weights": dict(zip(spec.portfolio.names, spec.portfolio.weights, strict=True)),
            "risk_free_rate_annual": spec.risk_free_rate_annual,
            "metrics": {
                "total_return": evaluation.portfolio.total_return,
                "annualized_volatility": evaluation.portfolio.annualized_volatility,
                "sharpe_ratio": evaluation.portfolio.sharpe_ratio,
                "max_drawdown": asdict(evaluation.portfolio.max_drawdown),
                "var_95": evaluation.portfolio.var_95,
                "cvar_95": evaluation.portfolio.cvar_95,
                "diversification_benefit": evaluation.portfolio.diversification_benefit,
                "risk_contributions": evaluation.portfolio.risk_contributions,
                # JSON has no infinity; an exactly singular covariance
                # reports null with its warning still stating the fact.
                "condition_number": (
                    evaluation.portfolio.condition_number
                    if math.isfinite(evaluation.portfolio.condition_number)
                    else None
                ),
                "conditioning_warning": evaluation.portfolio.conditioning_warning,
            },
        },
        "assets": {metrics.name: _metrics_dict(metrics) for metrics in evaluation.assets},
        "benchmark": None,
        "stress": {
            "windows": [
                {
                    "name": result.window.name,
                    "start": result.window.start,
                    "end": result.window.end,
                    "observations": result.observations,
                    "total_return": result.total_return,
                    "annualized_volatility": result.annualized_volatility,
                    "max_drawdown": asdict(result.max_drawdown),
                }
                for result in evaluation.stress_windows
            ],
            "shocks": [
                {
                    "name": result.name,
                    "asset_shocks": dict(
                        zip(spec.portfolio.names, result.asset_shocks, strict=True)
                    ),
                    "portfolio_return": result.portfolio_return,
                }
                for result in evaluation.stress_shocks
            ],
        },
        "monte_carlo": None,
    }
    if evaluation.benchmark is not None and evaluation.benchmark_comparison is not None:
        comparison = evaluation.benchmark_comparison
        payload["benchmark"] = {
            "ticker": evaluation.benchmark.name,
            "metrics": _metrics_dict(evaluation.benchmark),
            "comparison": asdict(comparison),
        }
    if evaluation.monte_carlo is not None:
        simulation = evaluation.monte_carlo
        payload["monte_carlo"] = {
            "mode": simulation.mode,
            "seed": simulation.seed,
            "runs": simulation.runs,
            "horizon_days": simulation.horizon_days,
            "terminal_mean": simulation.terminal_mean,
            "terminal_stddev": simulation.terminal_stddev,
            "terminal_percentiles": asdict(simulation.terminal_percentiles),
            "probability_below_initial": simulation.probability_below_initial,
            "terminal_values": list(simulation.terminal_values),
        }
    return payload


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _number(value: float) -> str:
    return f"{value:.2f}"


def _factor(value: float) -> str:
    return f"{value:.4f}"


def _optional_number(value: float | None) -> str:
    return "—" if value is None else _number(value)


def _drawdown_cell(drawdown: Drawdown) -> str:
    return f"{_percent(drawdown.depth)} ({drawdown.peak_date} to {drawdown.trough_date})"


def _spoken_list(items: Sequence[str]) -> str:
    *head, tail = items
    if not head:
        return tail
    if len(head) == 1:
        return f"{head[0]} and {tail}"
    return ", ".join(head) + ", and " + tail


def _table(header: tuple[str, ...], rows: Sequence[tuple[str, ...]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join([":---"] + ["---:"] * (len(header) - 1)) + "|")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _holdings_lines(spec: RunSpec) -> list[str]:
    grid = spec.asset_series[0].dates
    holdings = _spoken_list(
        [
            f"{name} at {_percent(weight)}"
            for name, weight in zip(spec.portfolio.names, spec.portfolio.weights, strict=True)
        ]
    )
    against = (
        f", measured against {spec.benchmark} as the benchmark"
        if spec.benchmark is not None
        else ""
    )
    first = (
        f"The portfolio holds {holdings} over the {len(grid)} shared trading days "
        f"from {grid[0]} to {grid[-1]} in `{spec.prices_csv_name}`{against}. The "
        f"annual risk-free rate is {_percent(spec.risk_free_rate_annual)}."
    )
    second = (
        "Returns are daily simple returns; annualization uses 252 trading days, "
        "volatility scaling with the square root of time; all second moments are "
        "sample moments (denominator n - 1). The portfolio's daily return is the "
        "weighted sum of that day's asset returns — daily rebalancing, so "
        "buy-and-hold drift is not modelled. VaR and CVaR are historical at 95% "
        "confidence, reported as positive loss fractions, with percentiles "
        "interpolated linearly between closest ranks."
    )
    return [first, "", second]


def _metrics_row(metrics: AssetMetrics) -> tuple[str, ...]:
    return (
        metrics.name,
        _percent(metrics.total_return),
        _percent(metrics.annualized_volatility),
        _number(metrics.sharpe_ratio),
        _percent(metrics.max_drawdown.depth),
        _percent(metrics.var_95),
        _percent(metrics.cvar_95),
    )


_METRICS_HEADER = (
    "Asset",
    "Total return",
    "Volatility",
    "Sharpe",
    "Max DD",
    "VaR 95%",
    "CVaR 95%",
)


def _portfolio_lines(evaluation: SpecEvaluation) -> list[str]:
    spec = evaluation.spec
    metrics = evaluation.portfolio
    rows = [
        ("Total return", _percent(metrics.total_return)),
        ("Annualized volatility", _percent(metrics.annualized_volatility)),
        ("Sharpe ratio", _number(metrics.sharpe_ratio)),
        ("Max drawdown", _drawdown_cell(metrics.max_drawdown)),
        ("Historical VaR 95%", _percent(metrics.var_95)),
        ("Historical CVaR 95%", _percent(metrics.cvar_95)),
        ("Diversification benefit", _percent(metrics.diversification_benefit)),
    ]
    lines = _table(("Metric", "Value"), rows)
    lines.append("")
    lines.append(
        "Each asset's risk contribution is its weight times its marginal "
        "covariance with the portfolio, as a fraction of total portfolio "
        "variance; the contributions sum to 100% by construction."
    )
    lines.append("")
    contribution_rows = [
        (name, _percent(weight), _percent(metrics.risk_contributions[name]))
        for name, weight in zip(spec.portfolio.names, spec.portfolio.weights, strict=True)
    ]
    lines.extend(_table(("Asset", "Weight", "Risk contribution"), contribution_rows))
    return lines


def _benchmark_lines(evaluation: SpecEvaluation) -> list[str]:
    benchmark = evaluation.benchmark
    comparison = evaluation.benchmark_comparison
    if benchmark is None or comparison is None:
        return ["The spec names no benchmark."]
    lines = [
        (
            f"The portfolio is compared against {benchmark.name}, which returned "
            f"{_percent(benchmark.total_return)} over the same days at "
            f"{_percent(benchmark.annualized_volatility)} annualized volatility. "
            "Alpha is the arithmetic CAPM residual over excess returns; its annual "
            "figure is the daily one times 252, not compounded."
        ),
        "",
    ]
    rows = [
        ("Beta", _number(comparison.beta)),
        ("Alpha (daily)", _percent(comparison.alpha_daily)),
        ("Alpha (annualized)", _percent(comparison.alpha_annualized)),
        ("Tracking error", _percent(comparison.tracking_error)),
        ("Information ratio", _optional_number(comparison.information_ratio)),
        (
            "Up capture",
            "—" if comparison.up_capture is None else _number(comparison.up_capture),
        ),
        (
            "Down capture",
            "—" if comparison.down_capture is None else _number(comparison.down_capture),
        ),
    ]
    lines.extend(_table(("Statistic", "Value"), rows))
    return lines


def _stress_lines(evaluation: SpecEvaluation) -> list[str]:
    if not evaluation.stress_windows and not evaluation.stress_shocks:
        return ["The spec requests no stress scenarios."]
    lines: list[str] = []
    if evaluation.stress_windows:
        lines.append(
            "Each historical window replays the portfolio's own daily return "
            "series through the named dates; nothing is invented, and the replay "
            "keeps the daily-rebalancing convention."
        )
        lines.append("")
        window_rows: list[tuple[str, ...]] = [
            (
                result.window.name,
                result.window.start,
                result.window.end,
                str(result.observations),
                _percent(result.total_return),
                _percent(result.annualized_volatility),
                _percent(result.max_drawdown.depth),
            )
            for result in evaluation.stress_windows
        ]
        lines.extend(
            _table(
                ("Window", "Start", "End", "Days", "Total return", "Volatility", "Max DD"),
                window_rows,
            )
        )
    if evaluation.stress_shocks:
        if lines:
            lines.append("")
        names = evaluation.spec.portfolio.names
        lines.append(
            "Each shock is a hypothetical one-day simple-return vector; the "
            "portfolio figure is the weighted sum — exact for one day of plain "
            "positions, but linear and correlation-free: the co-movements are the "
            "author's assumption, and no probability attaches to any scenario."
        )
        lines.append("")
        shock_rows: list[tuple[str, ...]] = [
            (
                result.name,
                *[_percent(shock) for shock in result.asset_shocks],
                _percent(result.portfolio_return),
            )
            for result in evaluation.stress_shocks
        ]
        lines.extend(_table(("Shock", *names, "Portfolio"), shock_rows))
    return lines


def _monte_carlo_lines(evaluation: SpecEvaluation) -> list[str]:
    simulation = evaluation.monte_carlo
    if simulation is None:
        return ["The spec requests no Monte Carlo simulation."]
    percentiles = simulation.terminal_percentiles
    mode = (
        "bootstrap (i.i.d. resampling of the portfolio's daily returns)"
        if simulation.mode == "bootstrap"
        else "parametric normal (draws fitted to the portfolio's daily returns)"
    )
    lines = [
        (
            f"The simulation ran {simulation.runs} {mode} runs with seed "
            f"{simulation.seed} over a horizon of {simulation.horizon_days} trading "
            "days, compounding daily returns from an initial value of 1.0. The "
            "same seed reproduces every figure exactly."
        ),
        "",
    ]
    rows = [
        (f"P{level}", _factor(getattr(percentiles, f"p{level}"))) for level in (5, 25, 50, 75, 95)
    ]
    lines.extend(_table(("Percentile", "Terminal value"), rows))
    lines.append("")
    lines.append(
        f"The mean terminal value is {_factor(simulation.terminal_mean)} with a "
        f"standard deviation of {_factor(simulation.terminal_stddev)}, and "
        f"{_percent(simulation.probability_below_initial)} of runs finished below "
        "the initial value."
    )
    return lines


def render_report(evaluation: SpecEvaluation, source_name: str) -> str:
    """Render the committee report as Markdown, deterministically.

    ``source_name`` is the spec's file name, cited so a reader knows
    which document reproduces the numbers. No line of the report ever
    contains a timestamp.
    """

    lines: list[str] = ["# Portfolio risk report"]
    lines.append("")
    lines.append(
        f"Prepared by quantrisk from `{source_name}`. Every figure below is "
        "recomputable from that file and its price data alone; the Monte Carlo "
        "section states the seed that reproduces it."
    )
    lines.append("")
    lines.append("## Holdings and conventions")
    lines.append("")
    lines.extend(_holdings_lines(evaluation.spec))
    lines.append("")
    lines.append("## Per-asset metrics")
    lines.append("")
    lines.extend(_table(_METRICS_HEADER, [_metrics_row(metrics) for metrics in evaluation.assets]))
    lines.append("")
    lines.append("## Portfolio risk")
    lines.append("")
    lines.extend(_portfolio_lines(evaluation))
    lines.append("")
    lines.append("## Benchmark comparison")
    lines.append("")
    lines.extend(_benchmark_lines(evaluation))
    lines.append("")
    lines.append("## Stress scenarios")
    lines.append("")
    lines.extend(_stress_lines(evaluation))
    lines.append("")
    lines.append("## Monte Carlo")
    lines.append("")
    lines.extend(_monte_carlo_lines(evaluation))
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- This is not a prediction tool. Every figure describes the supplied "
        "price history under the stated conventions; nothing here forecasts "
        "prices, returns, or risk."
    )
    lines.append(
        "- The bootstrap simulation resamples daily returns independently, which "
        "destroys autocorrelation and volatility clustering; when turbulent days "
        "cluster, multi-day drawdown risk is understated."
    )
    lines.append(
        "- Parametric figures assume normal tails. Daily equity returns are "
        "fatter-tailed than a normal distribution, so at high confidences the "
        "parametric VaR and the parametric_normal mode understate tail losses."
    )
    lines.append(
        "- The sample data shipped with this repository is fictional, generated "
        "by a seeded script; results over it demonstrate the machinery, not any "
        "market."
    )
    if evaluation.portfolio.conditioning_warning is not None:
        lines.append(
            f"- Numerical conditioning: {evaluation.portfolio.conditioning_warning}. "
            "Near-collinear holdings make weight-space figures (risk "
            "contributions above all) sensitive to tiny changes in the input "
            "prices; treat them as indicative, not precise."
        )
    lines.append("- Nothing in this report is investment advice.")
    lines.append("")
    return "\n".join(lines)
