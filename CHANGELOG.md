# Changelog

## Unreleased

### Documentation checked against the changes above — 2026-10-15

- `print_options`, and so the way `EstimationOptions` values print, covers
  every option. It left out about a third of them, so a changed
  `branch_top_k`, for example, did not show.
- The docstrings of the uncertainty types and functions, and of
  `ResultProvenance`, say in plain words what their fields mean.
- The manual and docstrings now say that `sample_problem_data` makes its
  `ode_solver` the problem's solver; that `estimate` leaves out the states the
  package adds for inputs such as `sin(c * t)`, which
  `analyze_parameter_estimation_problem` keeps; that the error statistics
  leave out the quantities the data cannot determine; that the fit error is
  measured on rescaled data; and that bounds cannot yet be used with inputs
  such as `sin(c * t)`.

### Options for old or research-only behaviour removed — 2026-10-15

- Removed `rank_strategy` and `cluster_method`. Results are ranked by fit
  error and clustered as before by default; the alternatives were kept only
  to reproduce older runs.
- Removed `t0_state_completion`. Its `:seed_for_polish` value filled in
  states with random starting values when recovering them failed; it was
  never tested. The default behaviour is unchanged.
- Removed `dump_raw_candidates_path`, `dump_polished_path` and
  `si_placeholder_fail_categories`, which were tools for studying the method.
- Removed `heartbeat`. With `nooutput = false` the stage lines are printed as
  before; `progress = true` prints them on their own.
- Removed `uneven_sampling`. A non-empty `uneven_sampling_times` now selects
  uneven sampling by itself.
- Fixed: with `uneven_sampling_times`, the model was simulated only over
  `time_interval`, so later sample times failed, and the initial conditions
  were applied at the start of `time_interval`. It is now simulated from the
  first sample time to the last, with the initial conditions at the first.

### A plain `Dict` of data is accepted again — 2026-10-15

- The nine-field `ParameterEstimationProblem` constructor accepts the data as
  a plain `Dict` keyed by `"t"` and the measured right-hand sides, as its
  docstring says. Since the `ObservationData` change of 2026-09-10 it raised a
  `MethodError` unless the data was already an `OrderedDict` of the exact
  type.

### Options that did nothing removed — 2026-10-14

- Removed the options `debug_solver`, `debug_cas_diagnostics` and
  `debug_dimensional_analysis`, which nothing read, and `gp_s3_refinement`,
  which was deprecated.
- `noise_model` accepts `:additive`, `:relative` and `:none`. Four synonyms
  for the first two are no longer accepted.
- `sample_problem_data` uses `abstol` and `reltol` when it simulates the
  model. It used 1e-14, the default of both, whatever they were set to.
- The docstring of `analyze_parameter_estimation_problem` says which fields of
  its `analysis` value are supported.
- `?EstimationOptions` marks each option that is experimental. They are
  mostly the tuning of equation construction, homotopy tracking, clustering,
  ranking and refinement, the sensitivity seeds, and everything to do with
  uncertainty, along with a few others such as `uneven_sampling_times` and
  `profile_phases`. These may change or be removed in a minor release; the
  others change only in a major release. `use_si_template`, which has no
  effect, is among them.

### Fewer exported names, and clearer ones — 2026-10-14

- `using ODEParameterEstimation` now brings in 28 names instead of 190:
  - setting up: `ParameterEstimationProblem`, `create_ordered_ode_system`,
    `sample_problem_data`, `ObservationData`, `ObservationSeries`,
    `observation_times`
  - estimating: `estimate`, `analyze_parameter_estimation_problem`,
    `EstimationOptions`
  - option values: `SolverHC`, `SolverTrustRegion`,
    `SolverLevenbergMarquardt`, `FlowStandard`, `FlowDirectOpt`,
    `PolishLeastSquares`, `PolishBFGS`, `PolishLBFGS`, `InterpolatorAAA`,
    `InterpolatorGP`, `InterpolatorGPUncertainty`, `InterpolatorCustom`
  - results and errors: `ParameterEstimationResult`, `UncertaintyReport`,
    `UncertaintyUnavailable`, `print_uncertainty_results`,
    `UnsupportedModelClassError`, `UnsupportedDerivativeOrderError`,
    `SamplingFailureError`
- Marked `public` but not exported, so written with the module name:
  `Examples`, the option types (`InterpolatorMethod` and the others),
  `OrderedODESystem`, `ResultProvenance`, `AbstractUQOutcome`,
  `UncertaintyError`, `uq_reliability`, `UQReliabilityAssessment`,
  `merge_options`, `validate_options` and `print_options`.
- Everything else is internal. It can still be reached as
  `ODEParameterEstimation.name` or imported by name, but it may change in any
  release. This covers the individual curve-fit and solver functions, the
  stages of the estimation, the uncertainty internals, the result bookkeeping
  helpers, the noise helpers, the constants, `PolishFastLMBoundedLog`,
  `PolishNewtonTrust` (the old default for refinement), and the curve fits
  other than the four exported. The default list of curve fits is
  unchanged.
- Renamed, so that each says what it does:

  | Old | New |
  |---|---|
  | `SolverFastNLOpt` | `SolverLevenbergMarquardt` |
  | `SolverRobust` | `SolverTrustRegion` |
  | `PolishLSOBoundedLog` | `PolishLeastSquares` |
  | `InterpolatorAAAD` | `InterpolatorAAA` |
  | `InterpolatorAGPRobust` | `InterpolatorGP` |
  | `InterpolatorAGPUQ` | `InterpolatorGPUncertainty` |
  | `UQUnavailable` | `UncertaintyUnavailable` |
  | `UQComputationError` | `UncertaintyError` |

- Removed `InterpolatorS3SE`, `InterpolatorS3RQ`, `InterpolatorS3SEpRQ` and
  `InterpolatorS3SExRQ`. They were marked deprecated and did what the
  `InterpolatorS3Adapt` values do.

### Equation building trimmed — 2026-10-14

- `construct_equation_system_from_si_template` now requires `si_template`,
  `precomputed_interpolants` and `time_index_set`, and no longer takes
  `interpolator` or `diagnostics`. Every caller already passed a template and
  interpolants, so the code that built them was never used.
  `instantiate_si_template_equations` no longer takes `diagnostics`; nothing
  passed it through from the options, so its output never appeared.
- Removed the debugging output switched on by the environment variable
  `BIOH_MWE_CAPTURE`.

### `diagnose` moved to the `research` branch — 2026-10-14

- Removed `diagnose` and everything only it used: `diagnose_derivative_accuracy`,
  `diagnose_polynomial_system`, `diagnose_sensitivity`, the error budget
  (`compute_error_budget`, `compute_multipoint_error_budget`), the HTML
  diagnostic report and its plots, `physicalize_uncertainty_report`, and the
  report types `DiagnosticReport`, `ComprehensiveDiagnosticReport`,
  `DerivativeAccuracyReport`, `PolynomialFeasibilityReport`,
  `SensitivityReport`, `MultipointDiagnosticAnalysis`, `ErrorBudgetEntry`,
  `ErrorBudgetReport`, `EstimationResultsReport`, `BacksolveUQReport`,
  `ParameterSpreadEntry` and `CrossSolutionSpread`. `diagnose` compares the
  interpolated derivatives with exact ones computed from the true parameter
  values, so it works only on simulated problems; it is a tool for developing
  the method and stays on the `research` branch. The uncertainty report
  (`compute_uncertainty = true`) is unchanged.
- Removed the `point_hint` option. Estimation never read it.
- Everything removed is in the history at commit `35d9a73`.

### One Levenberg–Marquardt solver — 2026-10-14

- Removed `SolverNLOpt` and `solve_with_nlopt`. They did what
  `SolverFastNLOpt` does (Levenberg–Marquardt from one starting point) without
  its Jacobian or retry. The local solvers left are `SolverFastNLOpt` and
  `SolverRobust`; both are now tested by estimating a model.
- `SolverFastNLOpt` returns a point only when the solver converged. It used to
  return a point that had not converged whenever the residual had gone down.
- `solve_with_robust` returns the same shape of result when it finds nothing
  as when it finds something.

### Small corrections found by new tests — 2026-10-14

- `polish_maxtime` of zero or less is now rejected. It used to draw a warning
  that promised the default of 3600 s, which was never applied, and the two
  refinement paths then read it differently. Use `Inf` for no limit.
- `validate_options` returns `false` when `opt_lb` and `opt_ub` have
  different lengths. It used to throw a `DimensionMismatch`.
- With `uneven_sampling = true`, `uneven_sampling_times` must have `datasize`
  entries in increasing order, and `validate_options` says so before any
  sampling starts.
- Files written by `save_system` are named without colons, so they can be
  stored on Windows, and a second file in the same millisecond no longer
  overwrites the first.

### Unused code removed — 2026-10-14

- Removed functions that nothing in the package calls. Some were research
  tools, which remain on the `research` branch: the alternative ways of
  choosing time points for the multipoint system
  (`select_time_point_pairs` and its `_gp_quality`, `_sensitivity` and
  `_homotopy_probed` variants, `select_time_points_by_conditioning`), the
  sensitivity-aware equation stripping behind the `strip_strategy` keyword of
  `build_multipoint_template`, `diagnose_model` with
  `compute_cross_solution_spread`, and `agp_gpr` (`agp_gpr_robust` and
  `agp_gpr_uq` remain).
- Removed the Fourier interpolator (`InterpolatorFourierAdaptive`,
  `fourier_adaptive`, `FourierApprox`).
- Removed the Matérn-5/2 kernel and its four interpolators
  (`InterpolatorAGPRobustMatern52`, `InterpolatorS3Matern52`,
  `InterpolatorS3AdaptMatern52`, `InterpolatorS3BICMatern52`). A Matérn-5/2
  fit is only four times differentiable at the data points, where the
  estimator takes higher derivatives; the plain GP one returned NaN for the
  first derivative there.
- Removed exported helpers that nothing used: `calculate_observable_derivatives`,
  `joint_derivative_covariance_cross_time`, `print_stats_table`,
  `sigma_d_diagonal`, `s3_symbol`, `is_gp_interpolator`,
  `is_matern_interpolator`, `configure_logging`, `log_matrix`,
  `log_equations` and `log_dict`, and about twenty internal ones.
- The package no longer sets the global logger. It only ever did so while it
  was being precompiled, so a session never saw it, and the `ODEPE_DEBUG`
  variable that controlled it had no effect.
- Everything removed is in the history at commit `784d865`.

### The final refinement works again — 2026-10-13

- Since the precompile step was added on 2026-10-06, the final refinement of
  a default `estimate` did nothing: it ran, failed inside, and returned the
  answer it was given. Results were the unrefined ones, typically good to
  five or six digits where the refined ones are good to twelve or more. It is
  fixed. On the built-in `brusselator` the largest error goes from 1e-5 to
  2e-15, and on `hiv` from 3e-5 to 1e-13.
- The cause: the refinement differentiates an ODE solve with ForwardDiff, and
  a stiff solver differentiates inside it with ForwardDiff too. ForwardDiff
  tells the two layers apart by a number each gets when it is first compiled.
  Precompiling saved ours, and the solver's was assigned afresh in each
  session, so their order was reversed. These derivatives now carry a tag
  whose place in the order is fixed (`src/core/nested_ad.jl`).
- An ODE solve that throws inside the refinement is now logged once, at
  warning level, instead of being discarded without a trace.
- A test estimates a model in a newly started Julia process and checks that
  the refinement was applied.

### The example models move to a submodule — 2026-10-13

- The ready-made models are now in `ODEParameterEstimation.Examples`, and the
  package no longer exports them. Write `using ODEParameterEstimation.Examples`
  to get them. This takes 31 names out of the package's exports.
- There are 40 models in six groups: first steps, models where something
  cannot be determined, models with several answers, larger models from the
  literature, control systems, and a model that has to be rewritten.
  `example_groups()`, `example_models()` and `example_problem(name)` list and
  build them. Every model has a docstring, true values and a time interval,
  and the manual's example models page shows them all.
- `sample_problem_data(problem; ...)` uses an example model's own time
  interval when you give none. A problem you build yourself carries no
  interval, so it behaves as before.
- Renamed models:

  | Before | Now |
  |---|---|
  | `simple` | `harmonic_oscillator` |
  | `simple_linear_combination` | `oscillator_mixed_outputs` |
  | `onevar_exp` | `exponential_decay` |
  | `onesp_cubed` | `decay_cubed_output` |
  | `trivial_unident` | `sum_of_rates` |
  | `global_unident_test` | `partly_observed_chain` |
  | `sum_test` | `hidden_scale` |
  | `latent_subpopulation_branch` | `hidden_subpopulations` |
  | `latent_subpopulation_observed_control` | `hidden_subpopulations_measured` |
  | `receptor_subtype_binding_branch` | `receptor_subtypes` |
  | `receptor_subtype_binding_observed_control` | `receptor_subtypes_measured` |
  | `crauste_corrected` | `crauste` |
  | `tank_level_poly` | `tank_level_rewritten` |

- Each control system keeps one model, the one with a sinusoidal input, under
  the plain name: `dc_motor` is what `dc_motor_sinusoidal` was, and likewise
  `quadrotor_altitude`, `forced_lotka_volterra`, `magnetic_levitation`,
  `aircraft_pitch`, `bicycle_model`, `boost_converter` and `bilinear_system`.
  The constant-input models that had those names are gone, and so are the
  `_identifiable` ones.
- Changed models: `dc_motor` treats the torque constant as known, so its two
  remaining parameters can be found. `tank_level` and `tank_level_rewritten`
  are now the same tank written two ways. `sum_of_rates` has the true values
  its source meant, 0.4 and 0.6. `fitzhugh_nagumo` is watched over [0, 10].
- Removed: earlier versions of models (`crauste_revised`, `hiv_old_wrong`,
  `biohydrogenation_debug`, the old `crauste`), the `_m1` benchmark variants,
  near-copies (`harmonic`, `lv_periodic`), `thermal_system`, `flexible_arm`,
  and the cart-pole, ball-and-beam, swing-equation, CSTR and two-tank models,
  which were either refused or stand-ins for the real system.
- Removed `available_models`, `available_model_categories`,
  `run_parameter_estimation_examples` and the model lists such as
  `ALL_MODELS`.
- The scripts move from `src/examples` to a top-level `examples` folder, and a
  test runs each one. The control-system walkthroughs that relied on
  unsupported cases, the adiabatic reactor experiment and the biohydrogenation
  data script are removed.
- Tests build and simulate every model on each run. Three new test groups
  estimate them with default options: `examples_ci` does five models on
  every push, and `examples` (the quick ones) and `examples_slow` (the ones
  that take minutes each) run weekly. Three models are not
  estimated, because the package does not solve them in reasonable time:
  `crauste`, `sirsforced` and `receptor_subtypes`. Their docstrings say so.
- Everything removed here is in the history at commit `fadccc6`.

### One polynomial system for a model, whatever the data — 2026-10-06

- Column scales now reach HomotopyContinuation as parameters of the polynomial
  system. They were numbers written into its coefficients, so every
  interpolator and every data set produced a system HomotopyContinuation had
  not seen, and it compiled each one: 21 systems in a default run of a
  two-state model, and most of them again for each new data set. It now
  compiles 7, once for the model.
- A second data set for a model takes as long as a repeat run. On the README
  example it took about 33 s and now takes about 9 s. The first call of a
  session is about 20 s shorter, and repeat runs are as fast as before.
- The scale values and the equations are unchanged, and only unknowns that can
  be scaled get a scale parameter. A seeded run of the README example or of
  `simple()` returns exactly the values it returned before. On the built-in
  `lotka_volterra()` the values differ in the 14th digit.

### A default estimation runs while the package is precompiled — 2026-10-06

- Precompiling the package now estimates a small model with default options,
  so that most of what `estimate` needs is compiled before the first call.
  With the change above, the first `estimate` on the README example takes
  about 32 s where it took about 105 s, and about 45 s where it took about
  150 s on Julia's default single thread. A model of another size gains less:
  the built-in `lotka_volterra()` goes from about 112 s to about 50 s.
- Precompiling the package takes about a minute and a half longer, and its
  cache file grows from 229 MB to 312 MB.
- If that estimation fails, precompilation warns and carries on, and
  `ODEParameterEstimation._WORKLOAD_STATUS[]` is `:failed`.
- The earlier precompile step left the model it had estimated in the loaded
  package. Nothing is carried over now, and a test checks a freshly loaded
  package.

### Small corrections — 2026-10-06

- `?EstimationOptions` said that `opt_maxiters` limits the terminal fallback.
  It limits `FlowDirectOpt` only. The fallback is limited by `polish_maxiters`
  and `polish_maxtime`.
- The direct optimizer is LBFGS. Two docstrings and a message said BFGS.

### Citation files — 2026-10-06

- Add `CITATION.cff`, which GitHub reads for its "Cite this repository"
  button, and `CITATION.bib`, with BibTeX entries for the two papers and for
  the software. The paper to cite is the 2026 arXiv paper.
- Add `.zenodo.json`, which describes the software to Zenodo when a release is
  archived there.

## 1.0.0 — registration candidate, 2026-10-05

This is the intended first registered release. Earlier source-install users
should review the development-snapshot changes below. Registration and tagging
are separate from preparing this candidate.

### Release preparation

- Check high derivatives of the optimized noiseless GP against its analytic
  generating curve, with measured BLAS portability budgets. Retain strict
  fixed-parameter upstream contracts and noisy-fit comparisons.
- Record numerical-runtime metadata and preserve resolved CI manifests.
- Build a split reference for the exported API with Documenter 1.19.
- Prepare the `ODEParameterEstimation.jl` repository URL, user documentation,
  and release workflows for Julia 1.12 and 1.13.

### Uncertainty is marked experimental — 2026-10-06

- The manual, the option documentation and the docstrings of `estimate` and
  `analyze_parameter_estimation_problem` now say that uncertainty estimation
  is experimental, and the tutorial asks for reports of how it did. Its
  behavior is unchanged.

### Estimates are refined by default — 2026-10-06

- `polish_solutions` now defaults to `true`: each solution of the equations is
  fitted to the data by least squares before it is returned. On Lotka–Volterra
  with 101 points and 2% noise, the worst parameter error fell from 30% to 2%.
  On exact data the estimates of six models, `hiv` among them, stayed exact.
  `polish_solutions = false` returns the solutions of the equations as they
  are.
- With `compute_uncertainty = true`, the standard errors now describe the
  refined estimate, since that is the one returned.

### One way to choose interpolators — 2026-10-06

- Remove the `interpolator` and `custom_interpolator` options. Beside the
  default `interpolators` list they had no effect: a run that set
  `interpolator = InterpolatorAAAD` used all nine default interpolators. Name
  the interpolators to use in `interpolators`, and give the functions for its
  `InterpolatorCustom` entries in `custom_interpolators`.
- Setting a removed option is an error, and `estimate` points to the list form.
  An empty `interpolators` list is refused, since there is no longer a single
  interpolator to fall back on.
- Tests and examples that set only the removed option ran the default list, and
  still do. The few that emptied the list to reach the single interpolator now
  name it in the list.

### A short path from a model and data to estimates — 2026-10-06

- Add `ParameterEstimationProblem(system, measured_quantities; data, true_values, name, solver)`.
  It takes a ModelingToolkit `System` as it is, completed or compiled or
  neither, and data under the names of the measured quantities: a
  `NamedTuple`, a dictionary, or a table with named columns. True values are
  optional. Numbers the system carries, as in `@parameters a = 0.4`, are used
  when present. The nine-argument constructor is unchanged.
- Add `estimate(problem; options...)`, which returns the solutions found, best
  fit first. A mistyped option name is reported by name.
- Results print as a short table that marks parameters the data cannot
  determine. `result[a]`, `result[:a]` and `result["a"]` look up one value.
- `sample_problem_data(problem; datasize = 51, noise_level = 0.01)` takes its
  options by keyword, as `estimate` does. It reports which true values are
  missing instead of simulating with `NaN`.
- For a model with a `sin(c*t)`, `cos(c*t)` or `exp(c*t)` input, `estimate`
  returns the model's own states only, without the helper states the input is
  rewritten with.
- A model that is not rational is refused before any work is done, in a
  message that quotes the term as it was written. The messages of
  `UnsupportedModelClassError`, `UnsupportedDerivativeOrderError` and
  `SamplingFailureError` are rewritten for users, and the three are documented.
- `create_ordered_ode_system` names the system after its `name` argument. It
  was always named `model`.
- Rewrite the docstrings a user meets first: `EstimationOptions`, which now
  describes every option, `ParameterEstimationProblem`,
  `ParameterEstimationResult`, `ObservationData` and `ObservationSeries`.
  Document `analyze_parameter_estimation_problem` and the package itself.

### Two fixes found while writing the manual — 2026-10-06

- A `sin(c*t)` or `cos(c*t)` input gave a wrong coefficient when the data
  covered a stretch on which the input stayed small. Rescaling halved the
  helper state for the input and doubled its coefficient, while the state's
  values were filled in unscaled. On `forced_decay` over `[-0.5, 0.5]` the
  input's coefficient came back as 4 instead of 2. Helper states are no longer
  rescaled.
- State names that are not ASCII, such as `θ(t)`, raised `StringIndexError` in
  models with a quantity that cannot be determined, because names were cut by
  byte offset.

### A manual — 2026-10-06

- Replace the README and the Documenter site with documentation written for
  someone meeting the package for the first time: a getting-started page,
  tutorials on your own data, noisy data, models with more than one answer and
  uncertainty, guides to supported models, options, results, the method and
  troubleshooting, and a reference split into the main API and everything
  else. Every example runs when the site is built.
- Publish the site from a new `Documentation` workflow.
- Remove `examples/estimation_options_example.jl`, which no longer ran, rewrite
  `src/examples/first_example.jl` for the short path, and move the
  biohydrogenation benchmark script out of the repository root.

### Quiet by default — 2026-10-06

- A run with default options now prints nothing, passes only errors to the
  logger, and writes no files. The defaults changed to `nooutput = true`,
  `diagnostics = false` and `save_system = false`. Estimates are unaffected:
  with a seed, all combinations of these flags gave identical candidate pools
  on `simple` and Lotka–Volterra. Set `nooutput = false` or
  `diagnostics = true` for the earlier output.
- Add `progress = true`, which prints one timestamped line as each phase starts
  and finishes, and nothing else.
- The candidate-synthesis log (`synthesis_log.csv`) is written only with
  `diagnostics = true`. It had been written on every run.
- Options are checked before output is silenced, so configuration warnings and
  errors are still shown.
- Counting the solutions of a candidate system no longer draws
  HomotopyContinuation's progress meter, which has no switch and showed on
  stderr for models slow enough to trigger it.

### Reproducible runs — 2026-10-05

- Add `EstimationOptions(seed = ...)`. With an integer, `sample_problem_data`
  and the estimation entry points each run on their own random stream and then
  restore Julia's default RNG. Repeated runs give identical results, and the
  caller's random stream is untouched. The default, `nothing`, keeps the
  earlier behavior of drawing from the caller's default RNG.
- Without a seed, the candidate pool for `simple` and Lotka–Volterra differed
  on every run and the selected `simple` estimate alternated between two
  candidates. With a seed, fresh processes with one and four threads returned
  identical pools.

### Levenberg–Marquardt name clash — 2026-10-05

- `LevenbergMarquardt` is exported by both NonlinearSolve and LeastSquaresOptim,
  so the unqualified name had been undefined in the package since
  LeastSquaresOptim was added on 2026-05-06. Qualify it in `solve_with_robust`,
  where `:algorithm => :levenberg` returned no solution, and in
  `solve_multipoint_overdetermined`, whose refinement step was silently skipped.
- Remove `PolishLevenberg` and `PolishGaussNewton`. NonlinearSolve algorithms
  are not optimizers for the scalar `Optimization.solve` polish, so the first
  raised `UndefVarError` and the second failed every polish. Use
  `PolishLSOBoundedLog` (the default) or `PolishFastLMBoundedLog` for
  Levenberg–Marquardt polishing.
- Add contracts: every remaining polish method yields a runnable optimizer,
  and no package method references a name that two imports both export.

### Dependency cleanup — 2026-10-05

- Remove eight declared dependencies that the package did not use. `Plots`,
  `Zygote`, `BlackBoxOptim` and `MultivariatePolynomials` were never imported.
  `DynamicPolynomials` and `PolynomialRoots` were imported without a call
  site. `Enzyme` and `SciMLSensitivity` served only the backend removed below.
- Remove `opt_ad_backend = :enzyme`. Enzyme cannot compile the trajectory
  loss, which rebuilds the ODE problem through symbolic indexing. In testing,
  `:enzyme` polishes raised after a long compilation and kept the unpolished
  candidate. `validate_options` now rejects the value before any estimation
  work; `:forward` (the default) and `:finite` are unchanged.
- A fresh registered resolve on Julia 1.13.1 selects 308 packages instead of
  434, adds none, and changes one indirect version (UnsafeAtomics 0.3.2 to
  0.3.3). GPUCompiler, where the advisory nightly job fails, is no longer in
  the graph. Median warm load time fell from 16.3 s to 12.5 s on the
  development machine.

### GP workspace reuse — 2026-10-05

- Reuse covariance, factorization and gradient scratch buffers within each
  internal GP fit, preserving fitting arithmetic and independent posteriors.
- Reduce allocated bytes by 98% on the measured 201-point fit (50.6 MB to
  0.99 MB); median fitting time fell from 34.9 ms to 32.0 ms in the paired run.
  Frozen fits still match upstream through derivative order six on Julia
  1.12 and 1.13. Add workspace recovery, ownership and allocation contracts.

### Internal dependency backends — 2026-10-04

- Replace the used SIAN-Julia helpers and GaussianProcesses.jl dense SE fit
  with private modules under `src/internal`, retaining upstream MIT notices
  and documented replacement boundaries.
- Preserve the GP interpolator option, fitting policy and higher derivatives;
  retain AGP/AGPUQ as separate implementations. Remove GP, SIAN and the direct
  PDMats dependency, including the GP-specific global method bridge.
- Replace CI's dependency fork checkouts with registered default and modern
  dependency profiles. Add frozen upstream contracts and optional paired
  recovery/performance tools outside the standard test dependency graph.
- Remove the wall-clock cap from the direct-optimizer UQ convergence canary;
  retain its iteration limit and numerical assertions. Production timeouts and
  their dedicated tests are unchanged.
- Record equivalence results, performance costs and remaining release work in
  [the internalization record](docs/internal/2026-10-04_internal_backends.md).

### Julia 1.13 stabilization

- Normalize dictionary inputs in all `ParameterEstimationResult` constructors
  explicitly for OrderedCollections 2, preserving already typed ordered maps.
- Match states and parameters by symbolic key when clustering results.
- Repair exported derivative utilities and clear equation denominators on both
  sides; restore their tests to the unit and full gates.
- Use `Symbolics.diff2term` at symbolic conversion sites and bound ForwardDiff
  chunk size in the noise-frontier rank probe to reduce compilation cost.
- Bridge the autonomous-model SI dispatch issue only on affected versions that
  lack the upstream method; patched SI checkouts receive no redundant bridge.
- Run isolated test files through `Pkg.test`, preserve active development
  versions, declare test imports, and contain diagnostic sidecars. CI initially
  checked registered dependencies and reproducible modern GP/SIAN/SI patches;
  the October internalization supersedes that patched setup.
- Update the quickstart, result contract, and review map. Verified environments,
  gate results, and remaining release work are recorded in
  [production readiness](docs/internal/2026-09-10_production_readiness.md).

### Changes from earlier development snapshots

- **`EstimationOptions`: 13 fields removed** (2026-08-12/13 dead-options
  cleanup): `rtol`, `output_precision`, `imag_threshold`, `branch_resid_factor`,
  `branch_min_size`, `max_deriv_level`, `trap_debug`, `display_system`,
  `use_monodromy`, `si_infolevel`, `log_dir`, `polish_only`, `ideal`. None had a
  live consumer (audit-verified; PEB sets none). Setting one now errors loudly
  at construction. Four previously-dead fields were WIRED instead (defaults
  preserve old behavior exactly): `clustering_threshold`, `si_probability`,
  `point_hint`, `save_filepath`. New fields: `hc_threading`, `hc_compile_mode`,
  `heartbeat`.
- **`ResultProvenance`: `residual_fix_set` removed;
  `template_status_before/after_residual_fix` collapsed to `template_status`**
  (2026-08-12; the pair was identical by construction and the set always empty —
  no residual-repair mechanism ever existed). Metadata writers should use the
  new exported `provenance_metadata_dict`.
- `solve_with_hc` lost its dead `use_monodromy`/`display_system` kwargs.

### Added

- `provenance_metadata_dict(::ResultProvenance)` — exported single source for
  the `odepe_metadata.json` provenance block (consumed by PEB + experiments
  templates).
- Default-on `[HB]` phase heartbeats (`EstimationOptions.heartbeat`).
- Scoped `RunContext` (auto-M hand-off, timing, per-analysis HC solver config).
- `_rethrow_if_interrupt` cancellation discipline across all production catches.

### Versioning note

Oren selected **1.0.0** for the first registered release. The changes above are
relative to earlier unregistered development snapshots; no previous registered
release is being declared compatible. Subsequent registered releases should
follow semantic versioning for the documented public API.
