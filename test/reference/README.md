# Optional upstream comparisons

This directory is a separately activated environment. It is never activated by
`Pkg.test`, CI, or ordinary installation. Runtime and standard tests use neither
GaussianProcesses nor SIAN; the standard tests consume frozen TOML fixtures.

The reference ODEPE source is `81bcf14e4c9e16fb60391a7427586974498450bd`,
SIAN is `2f78ca8a0cc93f99eb2f800f08c1dbd20f8b28d9`, and GP is
`3e896e9dbd0c41341c723ab16dcf0c261fc7b95a` from the validated local checkout.
The equivalent GP source tree can also be assembled using the preparation
script preserved at ODEPE revision `81bcf14` under
`.github/scripts/prepare-gp-checkout.sh`; that script verifies its tree hash.
Use separate source checkouts and `Pkg.develop` them in this environment; use
the recorded versions rather than resolving arbitrary current upstream source.
`Project.toml` lists imports but deliberately is not an exact environment lock.
The recorded fixtures were captured on Julia 1.13.1, Nemo 0.56.2, Optim 2.3.2,
and StructuralIdentifiability 0.5.34. All Julia commands use `--startup-file=no`.

- `capture_backends.jl OUTPUT.toml` captures exact SIAN equations/samples and
  GP covariance, objective, gradient, and derivatives at fixed parameters.
- `compare_fits.jl CURRENT_CHECKOUT OUTPUT.toml` loads only the current private
  GP module beside upstream GP, compares four identical frozen observation
  draws, and records objectives, parameters, derivatives 0–6 and errors.
- `recovery.jl INPUT.toml OUTPUT.toml` creates frozen ODE data if INPUT does not
  exist, then tests LV, VDP, FHN and biohydrogenation with the GP-only and default
  pools. Run it with the baseline and current package in separate environments,
  using the same input file. Every completed case is written immediately.
- `compare_performance.jl CURRENT_CHECKOUT OUTPUT.toml [BASELINE_MODULE]` warms both backends,
  alternates six measurements per arm and fixes BLAS to one thread. It records
  backend time and allocations separately from the end-to-end recovery runs.
  An optional standalone `GPBackend.jl` adds a `baseline_comparison` to each
  GP case: its `reference_*` measurements are the older internal implementation,
  while `internal_*` measurements are the current implementation.

These scripts are reference tools, not substitutes for the package full suite
and recovery benchmark. Raw reproduction output goes outside the working tree;
promote compact, reviewed fixtures and provenance explicitly.

`results/performance_buffers.toml` records the October 5 buffer-reuse follow-up
on Julia 1.13.1 with Optim 2.3.2 and native OpenBLAS_jll 0.3.30. The baseline
module is `src/internal/gp/GPBackend.jl` from ODEPE revision
`00387ca95690197c605ee0eb7e4ba0c06f5bee7d`. For example, extract it with
`git show 00387ca:src/internal/gp/GPBackend.jl > /tmp/GPBackend_before.jl` and pass
that file as the optional third argument. The top-level case measurements
still compare the pinned upstream GP to the current implementation; each
baseline comparison is a separate interleaved measurement series.

## Julia and BLAS portability

Resolve each reference environment with the Julia executable being tested;
do not reuse a different Julia minor's manifest without resolving it. The
standard-library and BLAS versions must describe the running executable.

`results/fits_julia112.toml` records the paired comparison with Julia 1.12.7,
native OpenBLAS_jll 0.3.29, Optim 2.3.2 and the same pinned upstream GP source
and frozen input observations. It matches upstream exactly through derivative
order six. The Julia 1.13.1 record is `../fixtures/internal_backends/fits.toml`
(native OpenBLAS_jll 0.3.30). The clean fit's high derivatives vary slightly
between these stacks; the ordinary tests allow that measured variation while
retaining tight fixed-parameter derivative contracts.

`results/gp_portability.toml` records the follow-up on ODEPE `ef35226`: native
Julia 1.12.7 with and without coverage, and one-thread Haswell/Sandybridge BLAS
kernels on the same host. Upstream and internal fits match exactly in every
paired run. The clean optimized sixth derivative varies with BLAS, including
on hosted CI; this is optimizer sensitivity on nearly noiseless observations.
For that clean case, standard tests now compare orders 4–6 to the derivatives
of the generating curve `sin(1.4x) + 0.2cos(3.1x)`, with relative tolerances
`1e-3`, `1e-3`, and `2e-3`. Noisy fits, orders 0–3, and all fixed-parameter
contracts retain their tighter frozen comparisons.

To reproduce, run `compare_fits.jl` in the pinned Julia 1.12 reference
environment normally and with `--code-coverage=user`. On a host supporting
the corresponding instructions, also run with
`OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=1` and
`OPENBLAS_CORETYPE=Sandybridge OPENBLAS_NUM_THREADS=1`. The record separates
paired local evidence from the hosted CI observation, which was not paired
with upstream on that runner.

## Recorded recovery comparison

`results/recovery_inputs.toml` freezes the exact observations used on both
checkouts. `recovery_before.toml` is the baseline above;
`recovery_after.toml` is the internalization worktree based on that revision.
Both ran on Julia 1.13.1, with Optim 2.3.2, Nemo 0.56.2,
StructuralIdentifiability 0.5.34, ModelingToolkit 11.45.3,
OrdinaryDiffEq 7.8.1, SciMLBase 3.57.0 and OrderedCollections 2.0.1.

All eight cases match exactly in returned candidate count, rank-one and
best-branch relative parameter errors, and rank-one trajectory fit error.
Biohydrogenation has poor parameter recovery in both versions despite small
trajectory error. These are package-model compatibility checks. Recorded
elapsed times include compilation
and concurrent validation load and must not be used as a performance comparison.
