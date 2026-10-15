# First-call latency: where the time went and what was changed

Measured on 2026-10-06, starting from `3c7559d`, on one
machine: Julia 1.13.1, a 13-core `znver3` under WSL2, `JULIA_NUM_THREADS=7`
unless a row says otherwise. Every number is a fresh process started with
`julia --startup-file=no`. "Baseline" is a pristine checkout of `3c7559d`;
"changed" is the tree with the two changes described below.

The machine's speed drifted by about 10% in both directions during the
evening, so the before-and-after tables come from runs that alternate
baseline, changed, changed, baseline. Earlier, unpaired runs are quoted only
where they show something the paired runs do not, and they give the 110 s and
155 s in the next paragraph.

## The question

The first `estimate` of a session took about 110 s on the README example, and
about 155 s with Julia's stock thread setting. A repeat run took 8 s, so
nearly all of it was compilation. The plan was to make the package's
precompile workload run a full default estimation, so that this code is
compiled when the package is installed.

## Where the time went

`--trace-compile --trace-compile-timing` on the first call, summed over
threads:

| | README example | built-in `lotka_volterra()` |
|---|---|---|
| Code for HomotopyContinuation's compiled systems | 146 s, 21 systems | 186 s, 21 systems |
| Code for its mixed systems (polyhedral start) | 20 s, 2 systems | 12 s, 2 systems |
| Generated functions (ModelingToolkit, Symbolics) | 8 s | 9 s |
| Everything else | 67 s | 68 s |
| Total | 242 s | 275 s |

Only the last row does not depend on the model, so only it can be compiled in
advance. That was confirmed directly: estimating the proposed workload model
first and the README example second in one session left the README example at
63 s (from 110 s), `lotka_volterra()` at 86 s (from 121 s) and `simple()` (now
`harmonic_oscillator()`) at 71 s (from 91 s). A workload alone could not halve
the wait.

## Why 21 systems

The package tells HomotopyContinuation to compile every system
(`hc_compile_mode = :all`). HomotopyContinuation keys its compiled code on
the content of the system, coefficients included.

`solve_with_hc_parameterized` passes the data as parameters, so one template
serves every time point. Column scaling did not: `scale_hc_system`
substituted `scale * variable` into the expressions, and the scales are
floating-point numbers computed from the data
(`max(largest derivative of that order, 1)`). Each of the nine interpolators
produces slightly different derivatives, so each got a system with different
coefficients, for each template. Most of the 21 systems were copies that
differed only in those numbers: with the scales passed as parameters the
count is 7.

The same thing happened for every new data set. In one session, on the README
model:

| Estimate | Baseline | Changed |
|---|---|---|
| First data set | 111 s | 36 s |
| Same data again | 8.0 s | 8.1 s |
| Same model, other parameter values | 33 s | 9.8 s |
| Same model, 1% noise, seed 1 | 34 s | 6.6 s |
| Same model, 1% noise, seed 2 | 32 s | 8.5 s |
| Seed 2 again | 9.5 s | 8.8 s |

About 25 s of each new data set was HomotopyContinuation compiling systems it
would never see again. The estimates are the same in both columns.

## The two changes

**Scales as parameters.** `scale_hc_system` now replaces each scalable
unknown `v` by `s * v` with `s` a new parameter of the system, listed after
the data, and each solve passes `vcat(data, scales)`. The scaled system no
longer depends on the data, so HomotopyContinuation compiles it once for the
model. Only derivatives of order one and above get a scale parameter: the
rule in `compute_column_scales` never scales anything else. The scale values,
the equations and the coordinates the paths are tracked in are what they
were.

**A default estimation during precompilation.** `src/precompile_workload.jl`
builds a two-state, four-parameter model through the keyword constructor,
simulates exact data and calls `estimate` with no options. The model is the
size of the README example because the refinement's ODE solve is compiled for
the number of unknowns. It is not the README model, so the README example
measures what a user's own model of that size gains.

## Results

Alternating runs, two per cell, seconds:

| | Baseline | Changed |
|---|---|---|
| README example, first `estimate` | 105, 103 | 32, 32 |
| README example, second | 7.5, 7.4 | 7.4, 7.5 |
| `lotka_volterra()`, first | 112, 111 | 51, 49 |
| `lotka_volterra()`, second | 8.4, 9.1 | 9.2, 8.5 |
| `simple()`, first | 78, 85 | 34, 33 |
| `simple()`, second | 2.2, 3.0 | 3.2, 2.6 |
| README example, first, stock threads | 146 | 45 |
| `lotka_volterra()`, first, stock threads | 163 | 65 |
| `using ODEParameterEstimation` (median of 5) | 11.9 | 12.3 |

Repeat runs were also timed on their own, six per session in four alternating
sessions per model: 8.06 s against 8.23 s on the README example and 12.00 s
against 11.73 s on `lotka_volterra()` (with 101 points). The differences are
inside the spread between sessions.

Two larger built-in models, 101 points, seeded, the same alternation, with
four repeat runs per session:

| | Baseline | Changed |
|---|---|---|
| `daisy_mamil3()`, first | 140, 139 | 64, 65 |
| `daisy_mamil3()`, repeat run (median) | 9.2, 9.2 | 9.2, 9.1 |
| `fitzhugh_nagumo()`, first | 133, 124 | 59, 62 |
| `fitzhugh_nagumo()`, repeat run (median) | 19.1, 18.4 | 18.1, 18.6 |

A seeded run returns the same values in every process, with one thread and
with seven. On the README example and on `simple()` the changed tree returns
exactly the baseline's values. On `lotka_volterra()` they differ by 7e-14
relative. For scale, `hc_compile_mode = :mixed` differs from `:all` by 4e-14
on the README example.

After the change the trace of a first call shows 7 compiled systems in place
of 21, and its sum over threads fell from 242 s to 65 s. That sum counts the
time a thread spends waiting for another thread's compilation, so the
wall-clock times above are the better measure.

## Costs

| | No workload | Baseline | Changed |
|---|---|---|---|
| Precompiling the package | 23 s | 272 to 284 s | 347 to 371 s |
| Compiled cache (`.so`) | 10 MB | 229 MB | 312 MB |
| `using` | | 11.9 s | 12.3 s |

The workload runs on one thread, which is what Julia gives a precompile
worker. "No workload" is the changed tree with the `precompile_workload`
preference set to `false`: nearly all of the precompile time and of the cache
is the workload, and the old one-state workload already cost over four
minutes and 219 MB.

The old workload also left its model in the loaded package:
`_LAST_ESTIMATION_REUSE` held `precompile_simple` and a cached system. The
new one calls `_reset_session_state!` when it finishes, and
`test/test_precompile_workload.jl` checks a freshly started process.

## Checks on the changed tree

| Gate | Result | Time, before | Time, after |
|---|---|---|---|
| `test/current.jl unit` | 564 of 564 | 29 s | 29 s |
| `test/current.jl all` | 2,454 of 2,454 | 17 min 25 s | 13 min 28 s |
| `test/current.jl benchmark` | 10 of 10 | 11 min 17 s | 5 min 50 s |
| `test/registered.jl all default` | 2,454 of 2,454 | 16 min 53 s | 15 min 5 s |
| `docs/make.jl` | builds | | |

The times are the test summaries of the same gates run earlier the same day
on `b9ae402`, which had 2,420 tests and differs from `3c7559d` only in
docstrings. The benchmark gate estimates several noisy data sets in one
process, which is the case the scaling change speeds up.

Tests added: the scaled system does not depend on the scale values
(`test/column_scaling.jl`); a second and third data set compile no new system
(`test/fast_core.jl`); the workload is a quiet, accurate default run that
leaves the caller's random stream alone, and a freshly started process
carries no state from it (`test/test_precompile_workload.jl`).

## What is still compiled per model

For a model the size of the workload's: the seven systems, the two polyhedral
start systems, about 7 s of generated functions, and about 12 s of other
code, 8 s of it in this package, where argument types carry the model (the
4 s `_maybe_synthesize_aggregate_candidates` call takes a `NamedTuple` with
the SI template in it).

A model with a different number of unknowns also recompiles the refinement's
dual-number ODE solve, 13 s on `lotka_volterra()` (five unknowns against the
workload's six). ForwardDiff's chunk size is the number of unknowns
(`polish_residual.jl`). A second workload model would cover one more size.

## Not done

- **`hc_compile_mode = :none`.** On the README example with the baseline
  code: first call 96 to 108 s, repeat run 10 to 14 s against 7.5 to 8 s, a
  new data set 8 to 12 s. With a workload its best case was 37 s. It removes
  the per-data-set cost as the scaling change does, and slows every repeat
  run by a third or more. `:mixed` gave 130 s and 11.3 s.
- **A fixed ForwardDiff chunk size**, to make the refinement's compiled code
  independent of the number of unknowns. The ODE solver's step control sees
  the dual parts, so results would change with the chunking.
- **Solving the generic start on the scaled system**, which would save two
  more compiled systems per model. The start solve is deliberately unscaled.

## Reproducing

The timing script, in outline:

```julia
load = @elapsed using ODEParameterEstimation, ModelingToolkit
# build the problem, sample data
first = @elapsed results = estimate(problem)
second = @elapsed estimate(problem)
seeded = estimate(problem; seed = 1)   # values compared across builds
```

run in a fresh process per number, in an environment that has the checkout
developed and ModelingToolkit added. Precompile time is
`@elapsed Base.compilecache(Base.identify_package("ODEParameterEstimation"))`.
The scripts and raw output are not in the repository.
