# Internal GP and symbolic construction backends

The functionality the package used from GaussianProcesses.jl and SIAN-Julia
now lives in two internal modules, so neither package is a dependency. The
public estimator options, GP candidate and interpolation wrapper are unchanged.

## Architecture and provenance

- [GPBackend](../../src/internal/gp/README.md) owns the dense zero-mean SE fitting
  path, adapted from validated GP revision `3e896e9`. It retains analytic
  gradients, log standard deviations, LBFGS/backtracking and the original
  Cholesky/noise policy. It has no PDMats dependency or external-type methods.
- [SIANBackend](../../src/internal/sian/README.md) owns the exact jet construction
  helpers from SIAN revision `2f78ca8`. Equation selection, structural fixing
  and multiplicity remain in ODEPE; structural classification remains in
  StructuralIdentifiability.jl.
- Each module has its upstream MIT notice and an explicit replacement boundary.
  Neither module imports the parent package. AGP/AGPUQ policies remain separate.
- Standard tests use frozen upstream fixtures. The optional upstream oracle
  environment in [test/reference](../../test/reference/README.md) is excluded from
  normal installation, test dependency resolution and CI.

## Numerical findings

The first direct GP extraction reproduced the noisy fits but changed the clean
fit. The cause was the likelihood constant: computing `log(2*pi)` differs by
one ulp from the correctly rounded upstream `log2π` constant. Matching the
constant restored identical optimized likelihoods and derivatives. Matching
the upstream matrix-vector reduction also removed the remaining prediction
roundoff difference.

On the four frozen synthetic datasets, the final module matched upstream
likelihoods, predictions and derivatives through order six exactly on both
Julia 1.13.1 and Julia 1.12.7 with Optim 2.3.2. Each comparison uses that
runtime's native resolved BLAS stack. This is evidence for these inputs and
stacks, not a guarantee of bitwise agreement across Julia, BLAS or optimizer
versions. The older `agp_gpr` log-variance initialization remains a separate
policy and is unchanged.

## Validation record

The baseline source is `81bcf14e4c9e16fb60391a7427586974498450bd`; the after
measurements use the internalization worktree based on that commit. Exact
fixture provenance is retained in `test/fixtures/internal_backends/backends.toml`.

The Julia 1.13.1 comparison stack uses Optim 2.3.2, Nemo 0.56.2,
StructuralIdentifiability 0.5.34, ModelingToolkit 11.45.3,
OrdinaryDiffEq 7.8.1, SciMLBase 3.57.0 and OrderedCollections 2.0.1.
Fresh registered environments resolve without GP, SIAN or dependency checkout
overrides. PDMats can still occur transitively through other packages.

| Gate | Result |
|---|---|
| Exact SIAN polynomial and rational fixtures | Match upstream, including ordering, denominators, sampled values and Jacobians |
| Four paired optimized GP fits | Identical likelihoods, predictions and derivatives through order six |
| Focused internal-backend contracts | 118/118 |
| Local unit gate | 419/419 |
| Local full FAST gate | 2,176/2,176 |
| Local benchmark | 10/10 |
| Fresh registered modern full suite, Julia 1.13.1 | 2,176/2,176 |
| Fresh registered full suite, Julia 1.12.7 | 2,176/2,176 |
| Fresh registered modern benchmark | 10/10 |
| Direct-optimization UQ canary after test-budget correction | 20/20 |

The first registered full run reported 2,172 passes, one failure and three
follow-on errors in the direct-optimization UQ canary. Its 30-second deadline
could expire during compilation before the first optimization step. An
isolated ordinary run passed all 20 canary assertions; a forced first-callback
timeout reproduced the exact failed score (`14.883738311421581`) and all four
failures. This canary now retains its 200-iteration budget with no wall-clock
deadline. Its accuracy, covariance and stationarity assertions are unchanged;
`test_polish_maxtime.jl` separately checks deadline enforcement. No production
UQ acceptance rule or optimization timeout changed.

The first Julia 1.12 full run passed all 2,058 pre-existing assertions but
failed five new fixture assertions. Two came from allowing `Dict` iteration
to choose the fixture's output order. The fixture now passes explicit state
and output vectors; upstream, internal and frozen symbolic results agree
exactly on both runtimes. Three came from comparing the nearly noiseless GP's
fourth through sixth derivatives across native BLAS versions (0.3.29 versus
0.3.30). The [Julia 1.12 upstream comparison](../../test/reference/results/fits_julia112.toml)
matches the internal backend exactly, while the largest cross-stack relative
jet difference is `5.75e-4`. Only this clean fit's orders 4–6 now use `1e-3`
relative tolerance; noisy fits, lower orders and fixed-parameter derivative
contracts retain their tighter tolerances. Production code did not change.
The corrected Julia 1.12 full rerun passed all 2,176 assertions; the corrected
backend file also passed 118/118 again on Julia 1.13. Test output now flushes
after each file so assertion failures appear before the next long test starts.

### Frozen ODE recovery

The [recorded inputs and paired outputs](../../test/reference/results/) retain all
eight runs. The protocol uses the package model constructors, 81 observations,
noise `1e-6`, adaptive shooting and trajectory polish. Every pair matches
exactly in rank-one parameter error, best-branch parameter error, trajectory
error and returned candidate count.

| Model | GP-only rank-one relative parameter error, before = after | Default-pool error, before = after |
|---|---:|---:|
| Lotka–Volterra | 0.00122837434 | 0.00122837415 |
| Van der Pol | 5.00220575e-7 | 5.00198182e-7 |
| FitzHugh–Nagumo | 0.000215290229 | 0.000215290243 |
| Biohydrogenation | 4.50123635 | 4.48171242 |

Biohydrogenation remains a poor parameter-recovery case despite trajectory
errors near `1.2e-10`. Preserving that result establishes compatibility, not
estimator quality. The runs shared a machine with other validation jobs, so
their elapsed times are not evidence of a performance change.

### Initial backend cost

The [warmed comparison](../../test/reference/results/performance.toml) uses six
measurements per arm, alternating order with one BLAS thread. Both
implementations run in the same process and dependency environment.

| Workload | Upstream median | Internal median | Internal/upstream allocated bytes |
|---|---:|---:|---:|
| SIAN polynomial fixture | 0.94 ms | 0.99 ms | 1.00× |
| SIAN rational fixture | 1.18 ms | 1.12 ms | 1.00× |
| GP smooth clean, 41 points | 2.67 ms | 3.12 ms | 8.08× |
| GP smooth noisy, 41 points | 1.98 ms | 2.24 ms | 5.81× |
| GP mixed scale, 61 points | 2.65 ms | 2.89 ms | 5.54× |
| GP transient, 41 points | 1.21 ms | 1.49 ms | 4.88× |
| GP smooth noisy, 201 points | 30.18 ms | 37.87 ms | 7.10× |

The symbolic extraction has comparable cost. The initial compact GP
implementation reconstructed dense arrays per likelihood evaluation, whereas
upstream reused more storage. That cost about 9–25% more time on this panel
and substantially more allocated bytes. The follow-up below addresses the
repeated allocations; large-grid performance has not been established.

### October 5: GP buffer reuse

`fit_se` now owns a workspace for covariance, factorization, inverse-score,
solve and gradient storage. Every trial refills its inputs, including after a
failed Cholesky factorization. The final posterior retains its arrays; other
fits have independent storage and predictions do not mutate it. The
non-mutating `evaluate_se` convenience function still returns an independent
state. Likelihood arithmetic, accumulation order and the optimizer are unchanged.

The [paired follow-up](../../test/reference/results/performance_buffers.toml)
compares the buffered implementation with both upstream and the original
internal module from `00387ca95690197c605ee0eb7e4ba0c06f5bee7d` in the same
process. Each pair uses six warmed measurements in alternating order with
one BLAS thread, Julia 1.13.1, Optim 2.3.2 and native OpenBLAS_jll 0.3.30.

| GP workload | Original internal median | Buffered median | Allocated bytes, original → buffered |
|---|---:|---:|---:|
| Smooth clean, 41 points | 2.84 ms | 2.54 ms | 3,805,008 → 70,168 |
| Smooth noisy, 41 points | 2.24 ms | 1.89 ms | 2,835,328 → 65,368 |
| Mixed scale, 61 points | 2.90 ms | 2.63 ms | 3,998,880 → 108,696 |
| Transient, 41 points | 1.48 ms | 1.36 ms | 1,530,912 → 57,320 |
| Smooth noisy, 201 points | 34.94 ms | 32.03 ms | 50,562,688 → 993,864 |

The 201-point workload allocates 98% fewer bytes and takes about 8% less time
than the original internal version. In the separate upstream/current series,
the buffered implementation takes about 9% more time than upstream and
allocates 86% fewer bytes. These small warmed panels do not establish
large-grid or end-to-end speedups.

All four frozen optimized GP cases were rerun against upstream on both native
Julia 1.12.7 and Julia 1.13.1 stacks. Likelihoods, predictions and derivatives
through order six match exactly within each stack. The complete parsed rerun
records equal the existing `test/fixtures/internal_backends/fits.toml` and
`test/reference/results/fits_julia112.toml` records, so no duplicate fixtures
were added. The new standard contracts check workspace reset after failed
factorization, posterior independence and a warmed allocation budget that
detects accidental dense scratch allocation.

The local Julia 1.13.1 full FAST gate passed **2,188/2,188**, including the
12 new workspace assertions, and the recovery benchmark passed **10/10**.
Both used `test/current.jl` with `allow_reresolve=false`. A separate warmed
128-point likelihood-and-gradient evaluation on Julia 1.12.7 allocated
**zero bytes**, within the new 4 KiB regression budget. Full-fit allocations
in the table also include workspace construction and optimizer storage.

### Removing the old global packages

The normal package and test dependency graphs no longer require `SIAN` or
`GaussianProcesses`. Removal was checked in a copy of a Julia 1.13 global
environment that had both installed. Its old manifest still listed them under the ODEPE path
entry after `Pkg.rm`; resolving refreshed that entry and removed both packages
from the complete dependency graph:

```julia
using Pkg
Pkg.rm(["SIAN", "GaussianProcesses"])
Pkg.resolve()
```

The optional upstream comparison environment in `test/reference` still needs
those packages.

### Reproduction

Run the supported local gates from the global Julia environment:

```sh
julia --startup-file=no test/current.jl unit
julia --startup-file=no test/current.jl all
julia --startup-file=no test/current.jl benchmark
julia --startup-file=no test/registered.jl all modern
julia --startup-file=no test/registered.jl benchmark modern
```

Run `test/registered.jl all` with Julia 1.12 as well. The reference environment
and frozen-source setup are documented in [test/reference](../../test/reference/README.md).
Nightly CI is advisory; no nightly result is claimed here.

## October 5 release-candidate portability follow-up

The buffered commit `ef35226` passed all three required Julia 1.13 CI jobs,
including registered recovery, but Julia 1.12 passed 2,187/2,188 assertions.
Its sole failure was the optimized clean GP sixth derivative against a frozen
Julia 1.13 fit. The observed derivative had relative error 0.001207 against
the frozen fit and 0.000664 against the known generating curve.

Additional upstream/internal pairs on native Julia 1.12.7, with coverage and
with Haswell/Sandybridge BLAS kernels, agree exactly within each run. Their
clean sixth-derivative errors against the curve range from 0.000299 to
0.001021. The compact [portability record](../../test/reference/results/gp_portability.toml)
includes objectives, runtime details, raw-output hashes and the CI observation.
This isolates sensitivity in the optimized, nearly noiseless fit; it provides
no evidence of changed internal-backend arithmetic.

The candidate checks clean optimized derivatives 4–6 against the analytic
curve with relative bounds 1e-3, 1e-3 and 2e-3. Fixed-parameter contracts,
noisy cases, and lower optimized derivatives retain their strict comparisons.
The test runner prints Julia, CPU, BLAS, thread and coverage metadata. CI also
preserves the resolved Project/Manifest and validation metadata as artifacts,
including on test failure. Production fitting code is unchanged.
