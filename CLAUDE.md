# ODEParameterEstimation.jl Guidelines

## Active integration notes

- [`docs/internal/review_map.md`](docs/internal/review_map.md) — canonical multi-agent review
  coordination map. Read this before starting broad code review, refactors, or
  review-lane assignment.
- [`MULTIPLICITY_INTEGRATION.md`](docs/internal/MULTIPLICITY_INTEGRATION.md) — algebraic
  multiplicity (M) auto-detection: production behavior, the registered Groebner
  baseline and historical PR #218 fix, and PEB integration context. Read
  before working on result.csv truncation or multiplicity.

## Open investigations (read before starting reconditioning / numerical-stability work)

- **UQ regime + production contract.** Start with the audited PEB LV/VDP/FHN
  canaries
  [`docs/internal/2026-08-14_peb_audited_uq_canaries.md`](docs/internal/2026-08-14_peb_audited_uq_canaries.md),
  then read the current routing/API contract
  [`docs/internal/2026-08-14_estimator_aware_uq.md`](docs/internal/2026-08-14_estimator_aware_uq.md)
  and empirical recap
  [`docs/internal/2026-08-14_uq_session_recap.md`](docs/internal/2026-08-14_uq_session_recap.md)
  and the GP-jet bias decision note
  [`docs/internal/2026-08-14_gp_jet_bias_decision_note.md`](docs/internal/2026-08-14_gp_jet_bias_decision_note.md)
  and the staged research protocol
  [`docs/internal/2026-08-15_estimation_uq_research_program.md`](docs/internal/2026-08-15_estimation_uq_research_program.md)
  and its completed LV Stage 1--3 decision record
  [`docs/internal/2026-08-16_lv_multipoint_bias_results.md`](docs/internal/2026-08-16_lv_multipoint_bias_results.md)
  and the model-assisted correction prototype/discovery record
  [`docs/internal/2026-08-16_model_assisted_correction_discovery.md`](docs/internal/2026-08-16_model_assisted_correction_discovery.md)
  and its repeated-noise/polish decision record
  [`docs/internal/2026-08-16_model_assisted_repeated_noise.md`](docs/internal/2026-08-16_model_assisted_repeated_noise.md)
  and the clean-revision audited hard-model campaign
  [`docs/internal/2026-08-16_audited_repeated_uq_campaign.md`](docs/internal/2026-08-16_audited_repeated_uq_campaign.md)
  before touching uncertainty quantification. Production UQ targets exactly the
  returned rank-one estimator and supports retained SP/MP algebraic roots,
  trajectory polish/direct score equations, and branch propagation. The N=60
  low-noise calibration result applies to the single-point estimator; nonlinear
  MP/polish coverage remains to be established. The older package-constructor
  pilot is routing stress, not audited model evidence. `:degenerate` is only the
  legacy summary flag: inspect `uq_reliability`'s separate numerical and
  interval-width axes, and never read a single-run status as a calibration
  certificate.

- **Variable (column) scaling of the polynomial system.** ODEPE now has
  power-of-2 problem rescaling (`auto_rescale`) and column scaling for
  parameterized HC systems (`use_column_scaling`), both enabled by default.
  Start with [`src/core/problem_rescaling.jl`](src/core/problem_rescaling.jl),
  [`src/core/homotopy_continuation.jl`](src/core/homotopy_continuation.jl),
  and their tests in `test/test_rescaling.jl` and `test/column_scaling.jl`.
  HC.jl also performs Skeel row scaling automatically. The diagnostic numbers
  and proposed implementation levels in
  [`docs/internal/2026-05-01_variable_scaling_investigation.md`](docs/internal/2026-05-01_variable_scaling_investigation.md)
  predate these implementations; use them as historical evidence when
  investigating remaining conditioning problems.

- **Compile time on the first call and on new data.** HomotopyContinuation
  compiles every distinct system it is given, so writing data-derived numbers
  into the polynomial system costs a compilation for every data set. Column
  scales are passed to it as parameters for that reason. The package also
  runs a default estimation while it is precompiled
  ([`src/precompile_workload.jl`](src/precompile_workload.jl)). Read
  [`docs/internal/2026-10-06_first_call_latency.md`](docs/internal/2026-10-06_first_call_latency.md)
  before changing column scaling, `hc_compile_mode` or that workload.

## User documentation

- The manual is `docs/src`, built with Documenter and published at
  <https://orebas.github.io/ODEParameterEstimation.jl/dev/>. The README is its
  front page. Engineering notes go in `docs/internal`, never in `docs/src`.
- Build it with `julia --startup-file=no --project=docs docs/make.jl`. Code in
  `@example` blocks runs during the build, so every estimation costs build
  time. Keep to one or two per page, and never put `progress = true` in an
  executed block.
- Write for someone who has never seen the package: plain words, short
  sentences, code first and explanation after. No dates, benchmark names or
  internal terms. Say what the package does without hedging, and do not quote
  digits that change from run to run.
- Say a thing once. A selling point belongs in the README and gets explained
  on one page of the manual; do not restate it on the others.
- A change a user can see needs its page and docstring changed with it.
  `?EstimationOptions` must name every option, and a test checks that it does.

## Build/Test Commands
- **Always use `--startup-file=no`** when invoking Julia (Revise.jl caused exit segfaults on Julia 1.12).
- Start local tests from the global Julia environment (plain `julia`, not `julia --project`). `Pkg.test` creates the isolated test environment and installs the dependencies declared in `test/Project.toml`.
- **Full FAST gate** (required for estimation-touching changes):
  `julia --startup-file=no -e 'using Pkg; Pkg.test("ODEParameterEstimation"; allow_reresolve=false)'`
- **Quiet unit contracts** (does not replace the full gate):
  `julia --startup-file=no -e 'using Pkg; Pkg.test("ODEParameterEstimation"; allow_reresolve=false, test_args=["unit"])'`
- **Benchmark smoke** (seeded, noisy, full-scale recovery guard; run before handing a build to the cluster):
  `julia --startup-file=no -e 'using Pkg; Pkg.test("ODEParameterEstimation"; allow_reresolve=false, test_args=["benchmark"])'`
- **Example models** (estimates every quick model in `ODEParameterEstimation.Examples` with default options; `examples_slow` does the slow ones and runs weekly on CI):
  `julia --startup-file=no test/current.jl examples`
- `test/current.jl` wraps these commands, records the active environment, and verifies that it points at this checkout. `allow_reresolve=false` preserves dependency versions and local development paths; a test dependency conflict must fail visibly.
- The full gate includes feature regressions, a build-and-simulate test of every example model, and a run of each script in `examples/`. Expected answers for the example models are in `test/example_expectations.jl`. Direct `include("test/...")` commands require their imports to be direct dependencies of the active environment; they are not a substitute for checking `Pkg.test`.
- Precompiling the package takes about six minutes after any change to `src`, because it runs a default estimation. `docs/src/contributing.md` says how to switch that off while iterating. Switch it back on before timing anything or running the final gates.
- For dependency/registration checks, run `julia --startup-file=no test/registered.jl` to resolve and test a fresh temporary environment using registered dependencies. Do not infer reproducibility from a global environment containing local development overrides. The current baseline and remaining release work are in [`docs/internal/2026-09-10_production_readiness.md`](docs/internal/2026-09-10_production_readiness.md).

## Code Style Guidelines
- Imports: Group related packages, with ModelingToolkit, OrdinaryDiffEq first
- Types: Use concrete types for function arguments, especially core types
- Functions: Document with docstrings using the triple quote format with Arguments/Returns sections
- Naming: Use snake_case for functions/variables, PascalCase for types
- Error handling: Use informative error messages with try/catch for numerical operations
- Parameters: Use OrderedDict for parameters and states to maintain consistent ordering
- ODE convention: Use t as the independent variable, D for differentiation
- Documentation: Document complex algorithms with explanatory inline comments

## Type Stability Guidelines
- Avoid `Any` type in struct fields and function signatures
- Ensure functions return consistent types
- Use concrete parameter types instead of generic ones
- Add explicit return type annotations to complex functions
- Prefer using Union types over Any when multiple specific types are possible
- Use @code_warntype to check for type instabilities in critical functions

## Constants and Configuration
- Default ODE solver: `package_wide_default_ode_solver = AutoVern9(Rodas5P())`
- Algorithm thresholds are defined in core_types.jl

## Naming Conventions
- Error thresholds: Use descriptive names with consistent notation (e.g., `XXX_THRESHOLD`)
- Function parameters: Use consistent names across similar functions:
  - `abstol`/`reltol` for tolerances (not atol/rtol)
  - `interp_func` for interpolation functions
  - Put `problem` or `model` as first parameter when applicable
- File organization: Keep related functionality in the same file or module
