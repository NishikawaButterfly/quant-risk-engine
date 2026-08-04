# The spec file and the CLI

The engine computes risk evidence from series handed to it as Python
objects. The CLI wraps that in one JSON document — the run spec — so a
whole portfolio analysis can be described, versioned, and rerun from a
single file and the price data beside it. `quantrisk run` evaluates a
spec and writes two artifacts; `quantrisk validate` proves the same
spec would evaluate cleanly, without writing anything.

## The spec document

A spec is a single JSON object. `prices_csv` and `portfolio` are
required; every other block is optional, and the run evaluates exactly
what is present. The shipped example,
[sample-data/portfolio-spec.json](../sample-data/portfolio-spec.json),
weights the fictional tickers AAA, BBB, and CCC at 50/30/20 over the
synthetic 2025 price year, measures them against DDD, replays one
stress window, applies two shock vectors, and runs a 1000-run seeded
bootstrap simulation.

| Field | Meaning |
|:------|:--------|
| `schema_version` | Always `1`. |
| `prices_csv` | Relative path to a CSV of daily closes: a `Date` column of ISO dates, then one column per ticker. |
| `portfolio` | Ticker-to-weight map; weights are fractions summing to one. Every ticker must be a CSV column. |
| `risk_free_rate_annual` | Optional annual risk-free rate as a fraction; defaults to zero. Enters the Sharpe ratio and the benchmark alpha. |
| `benchmark` | Optional ticker to compare against. Must be a CSV column and must not carry a portfolio weight. |
| `stress` | Optional; `windows` (name to `start`/`end` ISO dates) and/or `shocks` (name to ticker-to-shock map, every held ticker required, each shock a simple return above −1). |
| `monte_carlo` | Optional; `mode` (`bootstrap` or `parametric_normal`), `runs`, `horizon_days`, and the mandatory `seed`. |

The prices path must be relative and must resolve inside the spec's
own directory: a spec that could read files elsewhere on the machine
is a hazard, and a spec whose data does not travel with it cannot be
reproduced by whoever receives it. Rates, weights, and shocks are
plain fractions, so `0.02` is two percent. The seed is required in the
file for the same reason it is required in the code: an unseeded
simulation cannot be re-run to the same numbers, so none of its
figures can be checked.

Loading is deliberately strict. Both files have size caps, duplicate
JSON keys are rejected, and an unknown key or a wrong type fails the
whole load with the offending field named — for example
`monte_carlo.runs must be a JSON integer` or
`stress.windows['early']: ... lies outside the aligned grid`. Every
cross-reference is checked at load time too — tickers against CSV
columns, the benchmark against the portfolio, every stress scenario
against the loaded data. A spec either loads completely or not at all.

## Validation implies runnability

A spec accepted by `validate` will run: `quantrisk run` on the same
spec and price data cannot fail evaluation. Both commands prove it the
same way — after loading, they hand the spec to one shared feasibility
check that evaluates it through the run's own code paths, serializes
the payload under the run's own no-non-finite-numbers rule, renders
the report, and discards everything. A spec that is well formed but
unevaluable — a constant series with no Sharpe ratio, a benchmark grid
too short for three paired returns, a short position driving the
compounded value path to zero, a price ratio overflowing the float
range — now fails `validate` with exactly the message `run` would have
produced, for example `spec cannot be evaluated: Sharpe ratio is
undefined for constant returns`.

The Monte Carlo block is the one part not always re-executed: for a
bootstrap request a cheap arithmetic bound usually proves the
simulation cannot produce a non-finite figure (every draw is a
historical return, so the terminals are bounded by
`(1 + max return) ^ horizon`), and only when that bound is
inconclusive — or for the parametric mode, whose normal draws are
unbounded — does `validate` run the seeded simulation itself, which
its input caps keep to seconds at worst. Simulated bankruptcies do not
threaten the guarantee either way: a path absorbed at zero is a
counted result, not an error.

The guarantee's limits, stated plainly: it covers evaluation and
rendering, not publication. `quantrisk run` can still fail *writing*
its artifacts — an output directory that already holds results and no
`--force`, a full disk, a permissions error — and it can fail if the
spec or CSV files change between the two invocations. Validation
proves the spec evaluable, not the filesystem writable. The test suite
holds the property itself under a seeded sweep of adversarial specs:
every generated spec is either rejected by `validate` or run to
completion.

## Commands

```bash
quantrisk validate --spec sample-data/portfolio-spec.json
quantrisk run --spec sample-data/portfolio-spec.json --output results
```

`validate` parses the spec and its CSV, proves the run would complete
(see "Validation implies runnability" above), and prints a JSON
summary of what the run would cover — the weights, the date grid, the
benchmark, the stress scenario names, the Monte Carlo request —
without publishing a single number.

`run` evaluates everything through the same entry points a direct
library call would use and writes two artifacts into the output
directory:

- `results.json` — every computed number: per-asset and portfolio
  metrics with risk contributions, the benchmark comparison, the
  stress results, and the Monte Carlo summary with its seed, run
  count, the full sorted terminal values, and the `bankruptcies`
  count of runs absorbed at zero (see the bankruptcy section of
  [methodology.md](methodology.md)), so any summary figure can be
  recomputed and checked. The portfolio metrics include the
  covariance matrix's 2-norm `condition_number` (`null` when it is
  infinite) and a `conditioning_warning` that is `null` unless the
  matrix is ill-conditioned — see the numerical-conditioning section
  of [methodology.md](methodology.md).
- `report.md` — a short committee-style report: the holdings and
  conventions, the per-asset table, portfolio risk with
  contributions, the benchmark table, the stress tables, the Monte
  Carlo percentiles with the seed stated, and the caveats. When the
  covariance is ill-conditioned, the conditioning warning appears
  under Caveats.

Both artifacts render from one evaluation, so they cannot disagree,
and neither embeds a timestamp: the same spec always produces
byte-identical output, and the test suite asserts that equality.

The pair is published atomically as a pair. Both files are staged next
to their targets, any existing pair is set aside, and only then are
both promoted with atomic replaces; if anything fails before both
promotions complete, the staged files are removed and the set-aside
pair is restored. On any failure the output directory therefore holds
either the complete new pair or exactly what it held before — never a
new `results.json` beside an old `report.md`, and never one artifact
without the other. An output directory that already holds results is
never overwritten unless `--force` is passed, and a forced run that
fails restores the previous pair byte for byte. Two edges, stated
plainly: a process kill in mid-publish can leave hidden `.*.tmp`
staging or backup files beside the targets, which are safe to delete;
and if the restore itself also fails (a second, independent
filesystem error), the command says so and the previous artifacts
survive as hidden `.*.backup.tmp` files.

## Conventions the numbers follow

Everything is inherited from the library and stated in
[methodology.md](methodology.md): daily simple returns, 252-day
annualization, sample moments with denominator n − 1, historical VaR
and CVaR at 95% as positive loss fractions, the portfolio's daily
return as the weighted (daily-rebalanced) sum, and percentiles by
linear interpolation between closest ranks. The CLI adds no arithmetic
of its own.
