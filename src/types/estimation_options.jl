# estimation_options.jl
# Comprehensive options struct for ODEParameterEstimation.jl

# Enums for type-safe option selection
"""
	SystemSolverMethod

How the polynomial systems are solved, chosen with the `system_solver` option.

- `SolverHC` (default): homotopy continuation, which finds every solution.
- `SolverTrustRegion`: a trust-region method started from several points.
- `SolverLevenbergMarquardt`: Levenberg–Marquardt from one point.

The last two are local solvers and can miss solutions.
"""
@enum SystemSolverMethod begin
	SolverHC                  # solve_with_hc
	SolverLevenbergMarquardt  # solve_with_fast_nlopt
	SolverTrustRegion         # solve_with_robust
end

"""
	InterpolatorMethod

The curve fits that estimate the derivatives of the data, listed in the
`interpolators` option.

- `InterpolatorAAA`: a rational approximation. Suits exact or nearly exact
  data.
- `InterpolatorGP`: a Gaussian process. Suits noisy data.
- `InterpolatorGPUncertainty`: a Gaussian process that also estimates the noise
  in the data. Needed for `compute_uncertainty`.
- `InterpolatorCustom`: a function of your own, from `custom_interpolators`.

The default list runs nine curve fits. The values not named here are internal
and may change in any release.
"""
@enum InterpolatorMethod begin
	InterpolatorAAA           # aaad - Basic AAA rational interpolation
	InterpolatorAAADGPR        # aaad_gpr_pivot - GPR-based AAA (default)
	InterpolatorAAADOld        # aaad_old_reliable - Conservative AAA
	InterpolatorFHD            # Floater-Hormann interpolation
	InterpolatorGP      # agp_gpr_robust - Robust GP that handles smooth/noiseless data
	InterpolatorAGPRobustRQ    # agp_gpr_robust with Rational Quadratic kernel
	InterpolatorAGPRobustSEpRQ # agp_gpr_robust with SE + RQ sum kernel
	InterpolatorAGPRobustSExRQ # agp_gpr_robust with SE * RQ product kernel
	InterpolatorS2AAAMLE           # S2 composite: AAA(raw data) → MLE (no GP)
	InterpolatorS3AdaptSE          # S3 adaptive-tol: GP(SE) → AAA(adaptive) → MLE
	InterpolatorS3AdaptRQ          # S3 adaptive-tol: GP(RQ) → AAA(adaptive) → MLE
	InterpolatorS3AdaptSEpRQ       # S3 adaptive-tol: GP(SE+RQ) → AAA(adaptive) → MLE
	InterpolatorS3AdaptSExRQ       # S3 adaptive-tol: GP(SE×RQ) → AAA(adaptive) → MLE
	InterpolatorS3BICSE            # S3 BIC: GP(SE) → AAA(BIC-selected) → MLE
	InterpolatorS3BICRQ            # S3 BIC: GP(RQ) → AAA(BIC-selected) → MLE
	InterpolatorS3BICSEpRQ         # S3 BIC: GP(SE+RQ) → AAA(BIC-selected) → MLE
	InterpolatorS3BICSExRQ         # S3 BIC: GP(SE×RQ) → AAA(BIC-selected) → MLE
	InterpolatorGPUncertainty          # agp_gpr_uq - GP with full UQ (for calibrated uncertainty)
	InterpolatorChebyshevAICc  # Chebyshev polynomial with AICc degree selection (spectral)
	InterpolatorChebyshevBIC   # Chebyshev polynomial with BIC degree selection (spectral)
	InterpolatorCustom         # User-provided custom interpolator
end

"""
	PolishMethod

The optimizer that refines each solution against the data, chosen with the
`polish_method` option.

- `PolishLeastSquares` (default): Levenberg–Marquardt on the residuals.
- `PolishBFGS`, `PolishLBFGS`: minimize the sum of squares directly, with the
  Optim.jl method of that name.
"""
@enum PolishMethod begin
	PolishNewtonTrust      # NewtonTrustRegion from Optim.jl (legacy default)
	PolishBFGS             # BFGS from Optim.jl
	PolishLBFGS            # LBFGS from Optim.jl
	# Residual-mode polishers operating in per-variable transformed coordinates
	# (`:auto` policy: log for positive bounds, shifted-log for signed bounds, linear
	# for unbounded). Switch back to `PolishNewtonTrust` to restore the legacy scalar
	# polish path; the dispatch is fully reversible at runtime.
	PolishLeastSquares        # LeastSquaresOptim.LevenbergMarquardt() with bounds
	PolishFastLMBoundedLog     # FastLevenbergMarquardt.lmsolve!() with bounds
end

"""
	is_residual_polish_method(method::PolishMethod) -> Bool

True when `method` uses the residual-vector polish path (`_polish_single_residual`)
rather than the scalar-loss `Optimization.solve` path.
"""
function is_residual_polish_method(method::PolishMethod)
	return method === PolishLeastSquares || method === PolishFastLMBoundedLog
end

"""
	EstimationFlow

How the estimate is made, chosen with the `flow` option.

- `FlowStandard` (default): the algebraic method.
- `FlowDirectOpt`: local optimization alone, from a random start.
"""
@enum EstimationFlow begin
	FlowStandard     # optimized_multishot_parameter_estimation
	FlowDirectOpt    # direct_optimization_parameter_estimation
end

const SI_PLACEHOLDER_CATEGORY_ALIASES = Dict{Symbol, Symbol}(
	:dd_derivative_unmapped => :observable_derivative_overflow,
	:nonobservable_derivative => :support_jet,
	:state_or_input_jet => :support_jet,
	:unknown_variable => :true_unknown_variable,
)

const SI_PLACEHOLDER_CANONICAL_CATEGORIES = Set((
	:dd_observable_index_oob,
	:observable_derivative_overflow,
	:measured_rhs_jet,
	:state_jet,
	:parameter_or_ic_symbol,
	:transformed_analytic_support,
	:support_jet,
	:no_dd_derivative,
	:sian_auxiliary,
	:true_unknown_variable,
	:late_map_miss,
))

canonicalize_si_placeholder_category(category::Symbol) = get(SI_PLACEHOLDER_CATEGORY_ALIASES, category, category)

"""
	EstimationOptions(; options...)

Everything that can be adjusted about an estimation. Every option has a
default, and the defaults are meant to be used: `estimate(problem)` is the
normal call.

Options are given by keyword, either directly or through this struct:

```julia
estimate(problem; seed = 1, progress = true)

options = EstimationOptions(seed = 1, progress = true)
estimate(problem, options)
```

The options most people reach for come first. The rest adjust one stage of the
method each, and are grouped by stage.

Options marked *experimental* are for studying the method. They may change or
be removed in a minor release. The others change only in a major release.

# Everyday options

- `interpolators`: the curve fits used to estimate derivatives of the data, as
  a vector such as `[InterpolatorAAA]`. Each is run and their candidate
  solutions are pooled. The default runs nine: Gaussian processes, rational
  approximations and Chebyshev series. A shorter list is faster.
  `InterpolatorAAA`, a rational approximation, suits exact or nearly exact
  data; `InterpolatorGP`, a Gaussian process, suits noisy data;
  `InterpolatorGPUncertainty` is needed for `compute_uncertainty`; and
  `InterpolatorCustom` uses a function of your own.
- `polish_solutions` (default `true`): refine each solution by least squares
  against the data. `false` returns the algebraic estimates as they are.
- `opt_lb`, `opt_ub` (default `nothing`): lower and upper bounds, as vectors
  with the states first and then the parameters. Refinement stays inside them.
  They cannot yet be used with a model that has an input such as `sin(c * t)`.
- `seed` (default `nothing`): with an integer, repeated runs give identical
  results and Julia's default random number generator is left as it was. With
  `nothing`, random draws come from the default generator, so results vary a
  little from run to run unless you call `Random.seed!` first.
- `progress` (default `false`): print one timestamped line as each stage starts
  and finishes.
- `shooting_points` (default `12`): how many time points the equations are
  solved at. Fewer is faster.
- `abstol`, `reltol` (default `1e-14`): tolerances for solving the model's
  ODEs, both when simulating data and during refinement and the other steps
  that integrate the model.

# Simulated data

These are read by [`sample_problem_data`](@ref), which also uses `abstol`,
`reltol` and `seed`.

- `datasize` (default `21`): the number of time points.
- `time_interval` (default `[-0.5, 0.5]`): the first and last time.
- `noise_level` (default `0.0`): the size of the Gaussian noise added.
- `noise_model` (default `:additive`): with `:additive`, the noise has standard
  deviation `noise_level` times the mean absolute value of the series. With
  `:relative`, each value is multiplied by `1 + noise_level * randn()`. `:none`
  adds no noise.
- `uneven_sampling_times` (default empty, experimental): if not empty, sample
  at these times, of which there must be `datasize`, instead of evenly over
  `time_interval`. The initial conditions are the values at the first of them.
- `ode_solver` (default `AutoVern9(Rodas5P())`): the ODE solver for the
  simulation. The problem `sample_problem_data` returns keeps it as its
  solver for the estimation.

# Output

- `nooutput` (default `true`): print nothing and pass only errors to the
  logger. `false` prints the stages and a summary of the results.
- `diagnostics` (default `false`): print detailed diagnostics and write the
  candidate-synthesis log under `artifacts/diagnostics/`.
- `save_system` (default `false`), `save_filepath` (default `""`): save the
  polynomial systems to files, under `save_filepath` or else `saved_systems/`.
- `profile_phases` (default `false`, experimental): print the time and memory
  each stage took.
- `hc_show_progress` (default `false`): show HomotopyContinuation's progress
  bar.

# Estimating derivatives

- `custom_interpolators` (default `Function[]`): your own functions
  `(times, values) -> curve`, one for each `InterpolatorCustom` entry in
  `interpolators`, in order.
- `auto_filter_interpolators` (default `true`): leave out the rational
  interpolators when the data look noisy, because they amplify noise.
- `s3_adapt_k` (default `10.0`, experimental): noise multiplier for the
  tolerance of the two default curve fits that pass a Gaussian process through a
  rational approximation. Higher keeps fewer support points.
- `gp_derivative_lengthscale_factor` (default `1.0`, experimental): multiplies
  the fitted length scale of `InterpolatorGPUncertainty`. Values below one
  smooth less.

# Building and solving the equations

- `flow` (default `FlowStandard`): `FlowStandard` is the algebraic method.
  `FlowDirectOpt` fits by local optimization alone.
- `system_solver` (default `SolverHC`): how the polynomial systems are solved.
  `SolverHC` is homotopy continuation, which finds every solution. The others
  are local solvers and can miss solutions: `SolverTrustRegion` is a
  trust-region method started from several points, and
  `SolverLevenbergMarquardt` is Levenberg–Marquardt from one point.
- `use_si_template` (default `true`, experimental): has no effect. The
  equations are always built with StructuralIdentifiability.jl, and `false` is
  refused.
- `si_probability` (default `0.99`): the probability with which the
  identifiability analysis is correct.
- `si_fix_strategy` (default `:local_basis`): how values are chosen for
  quantities that cannot be determined: `:local_basis` or
  `:identifiable_functions`.
- `system_construction_policy` (default `:noise_frontier`, experimental): which
  equations are used. `:noise_frontier` prefers systems that need lower
  derivatives of the data and have fewer solutions. `:legacy` is the earlier
  choice.
- `construction_candidate_limit` (default `64`), `construction_beam_width`
  (default `16`), `construction_compute_mixed_volume` (default `true`),
  experimental: limits on the search that `:noise_frontier` does.
- `shooting_warp` (default `true`), `shooting_warp_beta` (default `3.0`),
  experimental: place more of the `shooting_points` near the start of the data,
  and how strongly.
- `use_multipoint` (default `true`): also solve systems that combine equations
  from several time points.
- `multipoint_n_points` (default `2`), `multipoint_max_pairs` (default `20`),
  `multipoint_pair_strategy` (default `:spread`), experimental: how many time
  points each of those systems combines, how many combinations are tried, and
  how they are chosen, `:spread` or `:boundary_order`.
- `auto_handle_transcendentals` (default `true`): rewrite `sin(c*t)`,
  `cos(c*t)` and `exp(c*t)` in polynomial form.
- `auto_rescale` (default `true`): rescale states, parameters and data by
  powers of two to bring them near one, and convert the estimates back.
- `use_parameter_homotopy` (default `true`): solve at one time point and track
  the solutions to the others, which is faster than solving each afresh.
- `homotopy_tracking_mode` (default `:generic_start`, experimental): how that
  tracking is done: `:generic_start`, `:gamma_straight`,
  `:gamma_straight_fallback` or `:parameter`.
- `gamma_max_seeds` (default `5`), `gamma_seed` (default `0`), experimental:
  the number of
  random restarts for the tracking, and their seed. `0` uses a fixed seed for
  each problem, a positive value is used as given, and a negative value draws a
  new one every run, which `seed` cannot make repeatable.
- `use_column_scaling` (default `true`, experimental): scale the unknowns of
  each polynomial system by the size of the data derivatives.
- `hc_real_tol` (default `1e-9`, experimental): how small the imaginary part of
  a solution must be for it to count as real.
- `hc_threading` (default `true`): let HomotopyContinuation use threads.
- `hc_compile_mode` (default `:all`, experimental): whether
  HomotopyContinuation compiles the systems: `:all`, `:none` or `:mixed`.

# Choosing the solutions to return

- `algebraic_multiplicity` (default `nothing`): the number of solutions the
  equations have. It is worked out automatically unless you set it. At most
  this many results are returned.
- `compute_algebraic_multiplicity` (default `true`, experimental): `false`
  skips working it out.
- `branch_top_k` (default `20`): the most results returned when that number is
  not known. `0` returns all of them.
- `branch_detection` (default `true`, experimental): rank without using the true
  values. `false` ranks by distance to them, which is only meaningful in a
  benchmark.
- `clustering_threshold` (default `1e-5`, experimental): the relative distance
  below which two candidates are the same solution.
- `rough_cluster_eps` (default `1.0`), `subspace_cluster_eps` (default `0.05`),
  experimental: how near-duplicate results are merged. Candidates that are far
  apart are separated first, then close ones within each group are merged, at
  these two distances.
- `branch_cluster_eps` (default `0.001`), `branch_err_factor` (default `100.0`),
  experimental: before refinement, drop candidates whose fit error is more than
  `branch_err_factor` times the best, and merge the rest at this distance.
- `branch_diversity_selection` (default `true`), `branch_diversity_eps` (default
  `0.01`), experimental: when several results are returned, prefer ones that
  differ by at least this relative distance.
- `branch_completion` (default `true`): when the equations have several
  solutions, take the best one found and solve for the others from it.
- `branch_completion_max_anchors` (default `1`),
  `branch_completion_residual_tol` (default `1e-6`), experimental: how many of
  the best solutions to start from, and the residual a new solution must
  reach.
- `synthesize_aggregate_candidates` (default `true`): add candidates made by
  combining the others: the median, mean and trimmed mean of each value.

# Refining solutions

- `polish_solver_solutions` (default `true`): refine each solution of the
  polynomial equations against those equations.
- `polish_solver_jacobian` (default `:forwarddiff`), `polish_solver_chunk_size`
  (default `1`), experimental: how the Jacobian for that is computed,
  `:forwarddiff`, `:symbolic` or `:finitediff`, and ForwardDiff's chunk size,
  where `0` chooses automatically.
- `polish_method` (default `PolishLeastSquares`): the optimizer for
  `polish_solutions`. `PolishLeastSquares` is Levenberg–Marquardt.
  `PolishBFGS` and `PolishLBFGS` minimize the sum of squares directly.
- `polish_maxiters` (default `100`), `polish_maxtime` (default `3600.0`
  seconds), `polish_max_concurrency` (default `Threads.nthreads()`): limits on
  the iterations and time spent on each solution, and on how many are refined
  at once.
- `polish_divergence_factor` (default `10.0`), `polish_stagnation_window`
  (default `50`), experimental: stop when the loss has grown by this factor, or
  has not improved for this many iterations.
- `polish_ode_maxiters` (default `5000`, experimental): the step limit for the
  ODE solver during refinement.
- `polish_coordinate_policy` (default `:auto`, experimental): the coordinates
  refinement works in. `:auto` takes logarithms of quantities whose bounds are
  positive. The others are `:linear`, `:log_only` and `:shifted_log_only`.
- `polish_regularization_lambda` (default `0.0`, experimental): the strength of
  an L2 penalty in those coordinates.
- `polish_softwall_lambda` (default `1e-2`), `polish_softwall_epsilon` (default
  `0.10`), experimental: a penalty that keeps values off their bounds: its
  strength, and the width of the band beside each bound where it acts, as a
  fraction of the range.
- `polish_lso_delta` (default `10.0`), `polish_lso_x_tol`, `polish_lso_f_tol`,
  `polish_lso_g_tol` (default `-1.0`), experimental: the trust-region radius
  and tolerances of `PolishLeastSquares`. A negative tolerance means `abstol` or
  `reltol`.
- `opt_maxiters` (default `10000`): the iteration limit for the local
  optimization of `FlowDirectOpt`.
- `opt_ad_backend` (default `:forward`, experimental): how the optimizers get
  derivatives: `:forward` for ForwardDiff or `:finite` for finite differences.
- `terminal_fallback` (default `:direct_opt`): what to do when the algebraic
  method finds nothing. `:direct_opt` fits by local optimization, within
  `polish_maxiters` and `polish_maxtime`. `:none` returns no results.
- `backsolve_recovery` (default `:algebraic_resolve`): what to do when a
  solution blows up as it is integrated back to the first time point.
  `:algebraic_resolve` keeps its parameters and solves for the states at that
  time again. `:none` does nothing.
- `use_sensitivity_seeds` (default `false`), `sensitivity_seed_probe_scale`
  (default `1.0`), `sensitivity_seed_eigenvalue_threshold` (default `0.01`),
  `sensitivity_seed_mahalanobis_threshold` (default `-1.0`),
  `sensitivity_sigma_d_kappa` (default `2.0`), experimental: add starting
  points for refinement along the directions in which the data constrain a
  solution least.

# Uncertainty

All of these are experimental.

- `compute_uncertainty` (default `false`): also estimate standard errors for
  the best solution. Needs `InterpolatorGPUncertainty` in `interpolators`.
- `uq_failure_policy` (default `:return_failed`): when standard errors cannot be
  computed, `:return_failed` returns an [`UncertaintyUnavailable`](@ref) that
  says why, and `:throw` throws an `ODEParameterEstimation.UncertaintyError`.
- `uq_noise_source` (default `:learned_gp_homoscedastic`): where the noise
  level comes from: the Gaussian process fit, or with `:smoother_residual_edf`
  the residuals of that fit.
"""
Base.@kwdef struct EstimationOptions
	# Solver and Algorithm Selection
	system_solver::SystemSolverMethod = SolverHC
	ode_solver::Any = AutoVern9(Rodas5P())  # Any type due to ODE solver type complexity

	# The curve fits to run. Each is run and their candidates are pooled.
	interpolators::Vector{InterpolatorMethod} = InterpolatorMethod[
		InterpolatorGP,        # SE kernel — robust GP baseline
		InterpolatorAGPRobustRQ,      # RQ kernel — Rational Quadratic (heavier-tail length-scale mixture)
		InterpolatorS3AdaptSE,        # GP→AAA(adaptive)→MLE
		InterpolatorS3AdaptRQ,        # GP→AAA(adaptive)→MLE with RQ kernel
		InterpolatorChebyshevBIC,     # Spectral, BIC degree selection
		InterpolatorChebyshevAICc,    # Spectral, AICc degree selection
		InterpolatorAAADGPR,          # GPR-pivoted AAA — proven for ill-conditioned systems
		InterpolatorAAA,             # Pure AAA rational interpolation
		InterpolatorS2AAAMLE,         # AAA→MLE (no GP step)
	]
	custom_interpolators::Vector{Function} = Function[]

	# Numerical Tolerances
	abstol::Float64 = 1e-14
	reltol::Float64 = 1e-14

	# Solution Filtering and Validation
	clustering_threshold::Float64 = 1e-5

	# Branch detection. When enabled, candidates are reduced in two stages:
	#   pre-polish: drop candidates with err > branch_err_factor × min(err);
	#               L-inf normalized clustering in identifiable-only variable space
	#               at branch_cluster_eps; one rep per cluster polished.
	#   post-polish: re-cluster polished reps in id-only space; each cluster
	#                surfaces as one "algebraic branch", branch_size attached.
	branch_detection::Bool = true
	branch_cluster_eps::Float64 = 0.001            # Target precision in MAD-normalized identifiable space. Larger values (0.05) merge distinct nearby basins on low-noise data; the extra rows this keeps are capped by branch_top_k.
	branch_err_factor::Float64 = 100.0
	branch_top_k::Int = 20                         # Maximum cluster reps to return at the output stage, sorted by fit error before slicing. Returning more than 20 was not found to recover more correct answers. Set to 0 to return all reps. Acts as a safety cap when `algebraic_multiplicity` is set: actual output is `min(algebraic_multiplicity, branch_top_k, length(cluster_reps))`.

	# Algebraic solution count, including multiplicities, on the chosen
	# representative slice. When set, the output is truncated to this many rows
	# (capped above by `branch_top_k` as a safety net). When `nothing`, output is
	# the full top-K candidate list (length up to `branch_top_k`).
	#
	# Computed automatically when this field is left `nothing`: ODEPE
	# detects M during the SI template build (si_equation_builder.jl —
	# `Groebner.quotient_basis` on the SIAN ideal; gauge/positive-dimensional
	# systems via projection to finite-valued coordinates) and hands it off
	# through the scoped RunContext (`_run_ctx_take_auto_m!`, consumed once in
	# `analyze_parameter_estimation_problem`). A caller-supplied value always
	# skips the automatic computation. Detection failure leaves the field
	# `nothing`: no truncation, no M≥2 full-space clustering protection.
	#
	# Type: positive Int, or `nothing` (compute automatically unless disabled).
	algebraic_multiplicity::Union{Int, Nothing} = nothing
	# Compute M only when it was not supplied. Disable for exploratory runs that
	# leave M unknown; ordinary branch_top_k filtering then remains in effect.
	compute_algebraic_multiplicity::Bool = true

	# Results are ranked by fit error (`err`, SSE).
	#
	# Output-stage clustering is two-stage: rough basin separation at a coarse
	# threshold, then within-basin MAD-normalized L∞ merging at
	# `subspace_cluster_eps`. This collapses near-duplicate rows along
	# practical-non-identifiability axes while preserving algebraically distinct
	# basins. Full-space dedup at `clustering_threshold` applies instead when
	# algebraic siblings may be present (successful branch completion, or a
	# detected/declared algebraic_multiplicity ≥ 2). See
	# `cluster_solutions_identifiable_subspace` and `_cluster_output_candidates`.
	# Rough-cluster threshold for stage-1 (basin identification). 1.0 = parameters
	# differ by more than the mean magnitude (i.e. different basins). Tight
	# enough that distinct algebraic solutions stay separate.
	rough_cluster_eps::Float64 = 1.0
	# Stage-2 within-basin MAD-normalized L∞ threshold. 0.05 ≈ "5% of MAD-scaled
	# spread" — coarser than the 1e-5 bit-identical dedup but tight enough to
	# preserve genuine sub-basin structure.
	subspace_cluster_eps::Float64 = 0.05

	# Output-stage M-selection diversity. When algebraic_multiplicity asks for
	# multiple returned rows, prefer candidates separated by at least this
	# relative solution-distance threshold before filling remaining slots from
	# the original rank order. This is best-effort: it cannot invent a missing
	# branch, and it deliberately falls back to ranked near-duplicates if too few
	# separated candidates exist.
	branch_diversity_selection::Bool = true
	branch_diversity_eps::Float64 = 0.01

	# Algebraic branch completion. When a system has
	# algebraic_multiplicity > 1, ODEPE uses the normal candidate pool only to
	# choose an anchor, then solves the SI-template equations with exact jets
	# implied by that anchor and replaces the returned pool with the verified
	# algebraic sibling set.
	branch_completion::Bool = true
	branch_completion_max_anchors::Int = 1
	branch_completion_residual_tol::Float64 = 1e-6

	# Multi-shot Parameters
	shooting_points::Int = 12
	shooting_warp::Bool = true                   # true = exponential warp, false = equidistant
	shooting_warp_beta::Float64 = 3.0            # warp strength (0≈uniform, 3=default)

	# Derivative and Reconstruction Parameters

	# Optimization Parameters
	polish_solutions::Bool = true
	polish_solver_solutions::Bool = true
	polish_solver_jacobian::Symbol = :forwarddiff
	polish_solver_chunk_size::Int = 1
	# Default: bounded LeastSquaresOptim Levenberg–Marquardt in per-variable
	# transformed coordinates.
	polish_method::PolishMethod = PolishLeastSquares
	polish_maxiters::Int = 100
	opt_maxiters::Int = 10000
	opt_lb::Union{Nothing, Vector{Float64}} = nothing
	opt_ub::Union{Nothing, Vector{Float64}} = nothing
	opt_ad_backend::Symbol = :forward
	polish_maxtime::Float64 = 3600.0         # Per-solution wall-clock timeout (seconds). With bounded concurrency most polishes finish in well under 100 s, but stiff or multi-basin models can need much longer to escape singular limits.
	polish_max_concurrency::Int = Threads.nthreads()  # Maximum number of polish tasks running in parallel. Defaults to Threads.nthreads() so each polish gets a fair CPU share. Prevents thread contention from slowing each polish ~N/T× when many candidates are submitted at once (each polish assembles ForwardDiff Jacobians through ODE integration; when they compete for cores, every polish can run into polish_maxtime).
	polish_divergence_factor::Float64 = 10.0 # Stop if loss > initial_loss * this
	polish_stagnation_window::Int = 50       # Stop if no improvement in N iters
	polish_ode_maxiters::Int = 5000          # ODE solver maxiters inside polish loss (DiffEq default: 100000)

	# Per-variable polish coordinate transform policy. `:auto` selects per variable:
	#  - `:log` for `lb > 0 && isfinite(ub)`
	#  - `:shifted_log` for finite signed bounds
	#  - `:linear` for unbounded variables
	# Other accepted values: `:linear`, `:log_only`, `:shifted_log_only`.
	polish_coordinate_policy::Symbol = :auto

	# Optional log-space L2 regularization for residual-mode polish. Augments the
	# residual with `√λ · x_internal`, pulling toward `x = 1` (or `x = -shift+1`)
	# in scaled coordinates. Default off; nonzero values can rescue ill-conditioned
	# `crauste`/`seir`/`hiv`-style cases at the cost of bias on well-posed ones.
	# No robust auto-selection across models exists yet.
	polish_regularization_lambda::Float64 = 0.0

	# Soft-wall penalty near bounds for residual-mode polish. When
	# `polish_softwall_lambda > 0`, augments the residual with one row per parameter:
	# zero in the interior of the bound interval, growing quadratically as the
	# parameter approaches either bound. Targets parameters that run onto a bound
	# during refinement (e.g. biohydrogenation's k10), without biasing interior
	# solutions. λ acts almost as an on/off switch: any value past about 1e-4 has
	# the same effect, and ε sets the width of the penalty zone. With ε = 0.10 and
	# bounds [1e-5, 10], the penalty applies only outside roughly [3e-5, 5].
	# Set both to 0.0 to disable. See polish_residual.jl for the exact form.
	polish_softwall_lambda::Float64 = 1e-2
	polish_softwall_epsilon::Float64 = 0.10

	# LSO (LeastSquaresOptim.LevenbergMarquardt) trust-region & tolerance knobs.
	# Negative tolerances mean "use `reltol`/`abstol`". `polish_lso_delta` is the
	# initial trust-region radius (LeastSquaresOptim's own default is 1.0).
	polish_lso_delta::Float64 = 10.0
	polish_lso_x_tol::Float64 = -1.0
	polish_lso_f_tol::Float64 = -1.0
	polish_lso_g_tol::Float64 = -1.0

	terminal_fallback::Symbol = :direct_opt   # :none | :direct_opt
	backsolve_recovery::Symbol = :algebraic_resolve  # :none | :algebraic_resolve

	# Sensitivity-based seed generation (σ_d-aware probing). Off by default.
	# When enabled, computes per-candidate Σ_x = S · diag(σ_d²) · S' and emits
	# additional polish seeds along significant eigenvectors (sloppy directions),
	# plus pairwise Mahalanobis-gated blends between candidates.
	use_sensitivity_seeds::Bool = false
	# Multiplier on √λ for eigenvector probes. 1.0 emits ±1σ along sloppy directions;
	# larger values explore farther at the cost of more polish budget.
	sensitivity_seed_probe_scale::Float64 = 1.0
	# Eigenvalue significance: only emit probes for eigenvalues with λ_i / λ_max above
	# this fraction. Tighter = fewer probes.
	sensitivity_seed_eigenvalue_threshold::Float64 = 0.01
	# Mahalanobis distance² threshold for cross-candidate blending. Negative means
	# "use chi² 95% quantile for n_unknowns degrees of freedom."
	sensitivity_seed_mahalanobis_threshold::Float64 = -1.0
	# Order-decay floor parameter for σ_d construction: σ_d_floor(order) = noise_level · κ^order.
	sensitivity_sigma_d_kappa::Float64 = 2.0

	# Data Sampling Parameters
	datasize::Int = 21
	time_interval::Vector{Float64} = [-0.5, 0.5]
	noise_level::Float64 = 0.0
	noise_model::Symbol = :additive
	uneven_sampling_times::Vector{Float64} = Float64[]

	# Reproducibility
	seed::Union{Nothing, Int} = nothing  # Integer ⇒ sampling noise and estimation are reproducible and leave the caller's RNG untouched; nothing ⇒ draw from the caller's default RNG

	# Debug and Output Flags
	nooutput::Bool = true
	diagnostics::Bool = false
	profile_phases::Bool = false  # Print per-phase timing/allocation breakdown
	progress::Bool = false        # The same phase markers on their own, for a quiet run

	# Feature Flags
	flow::EstimationFlow = FlowStandard
	use_si_template::Bool = true
	system_construction_policy::Symbol = :noise_frontier
	construction_candidate_limit::Int = 64
	construction_beam_width::Int = 16
	construction_compute_mixed_volume::Bool = true
	save_system::Bool = false
	compute_uncertainty::Bool = false  # Experimental GP/IFT sidecar
	uq_failure_policy::Symbol = :return_failed  # :return_failed | :throw
	uq_noise_source::Symbol = :learned_gp_homoscedastic
	gp_derivative_lengthscale_factor::Float64 = 1.0
	auto_handle_transcendentals::Bool = true  # Automatically detect and handle sin/cos/exp in equations
	auto_rescale::Bool = true  # Power-of-2 rescale states/params/observables/data to O(1) before estimation, un-rescale results after (see core/problem_rescaling.jl). Near-identity (mild powers of 2) on already-O(1) models; rescues ill-scaled ones (e.g. raw hiv 43.7→1.2e-3). Set false to disable.
	auto_filter_interpolators::Bool = true  # Filter AAA-family interpolators (S2/AAAD/AAADOld) by data-driven σ̂ before the SP loop
	synthesize_aggregate_candidates::Bool = true  # Inject per-component median/mean/trim25 aggregates of SP, MP, SP∪MP candidates as extra polish seeds before clustering. Logged to artifacts/diagnostics/<model>/synthesis_log.csv when diagnostics = true.
	s3_adapt_k::Float64 = 10.0     # Noise multiplier for S3 adaptive tolerance (higher = fewer support points)

	# HomotopyContinuation Specific
	use_parameter_homotopy::Bool = true  # Use parameter homotopy for multi-shot (track solutions between points)
	use_column_scaling::Bool = true  # Data-driven per-order column (variable) rescaling for the parameterized HC
	# solve. Rescales each unknown x = s.*x̂ using per-derivative-order observable-derivative
	# magnitudes (order-0 vars left at 1.0), solves the rescaled system (Newton polytopes / mixed volume unchanged),
	# and unscales solutions by s. Benign (~identity) when observable-derivative magnitudes are ~O(1). Tames the
	# ~1e7 jet-coordinate dynamic range that defeats unscaled polyhedral tracking on stiff/transient systems.
	hc_real_tol::Float64 = 1e-9
	# Per-analysis HC solver config, carried to _hc_solve via the scoped
	# RunContext (run_context.jl). The module Refs HC_SOLVE_THREADING /
	# HC_COMPILE_MODE remain the manual/global escape hatch when no context
	# is bound (bare solver calls outside analyze_*).
	hc_threading::Bool = true
	hc_compile_mode::Symbol = :all  # :all | :none | :mixed (HC compile policy)
	hc_show_progress::Bool = false
	# Multi-shot parameter-transition tracking mode (points i>1). :gamma_straight tracks
	# F(·;p_start)→F(·;p_target) via HC.jl's fixed-system StraightLineHomotopy WITH the γ-trick
	# (H = γ·t·F(·;p_start) + (1−t)·F(·;p_target)) — a discriminant-avoiding path that lands EXACTLY at
	# the real target; robust where the old straight parameter path collapses (on the receptor model
	# it lands 10 of 10 paths where that path lands 4 of 16). :parameter = HC.jl ParameterHomotopy (old
	# straight real parameter path, no γ, kept for comparison). :gamma_straight_fallback = :parameter first,
	# then γ-straight instead of a fresh solve on path loss (strictly dominates :parameter).
	# :generic_start (the default) seeds from a
	# generic COMPLEX parameter point p0 (off the discriminant ⇒ full generic root count N, well-conditioned)
	# then fans out by tracking p0→each real point — robust to a deficient point-1 solve, and makes the count
	# target the true N (fixing the initial_solution_count anchor). Falls back to per-point fresh+γ chain.
	homotopy_tracking_mode::Symbol = :generic_start  # The generic solve runs once and serves every interpolator (single-point and multipoint)
	gamma_max_seeds::Int = 5  # γ-straight: try up to this many random γ seeds, keep the most-complete result
	gamma_seed::Int = 0       # γ RNG seed: 0 ⇒ deterministic per-problem auto-seed (reproducible); >0 ⇒ that exact seed; <0 ⇒ entropy

	# Multi-point template (combines polynomial systems from N time points)
	use_multipoint::Bool = true   # Enable multi-point polynomial template system
	multipoint_n_points::Int = 2  # Number of time points per evaluation (2 recommended)
	multipoint_max_pairs::Int = 20  # Maximum number of time point pairs to solve
	multipoint_pair_strategy::Symbol = :spread  # :spread | :boundary_order

	# StructuralIdentifiability Parameters
	si_probability::Float64 = 0.99
	si_fix_strategy::Symbol = :local_basis

	# File I/O
	save_filepath::String = ""
end

# Single source of truth for option-bridge fallbacks: internal solver paths that
# receive a plain options Dict must default missing keys to THESE struct
# defaults, never to hardcoded literals, which silently drift: `use_column_scaling`
# once fell back to false and `homotopy_tracking_mode` to :gamma_straight while
# the struct defaults are true / :generic_start. Constructed once at load; read-only.
const _OPT_STRUCT_DEFAULTS = EstimationOptions()

"""
	get_solver_function(method::SystemSolverMethod) -> Function

Convert SystemSolverMethod enum to actual solver function.
"""
function get_solver_function(method::SystemSolverMethod)
	if method == SolverHC
		return solve_with_hc
	elseif method == SolverLevenbergMarquardt
		return solve_with_fast_nlopt
	elseif method == SolverTrustRegion
		return solve_with_robust
	else
		error("Unknown solver method: $method")
	end
end

"""
	get_interpolator_function(method::InterpolatorMethod, custom::Union{Nothing, Function}=nothing; s3_adapt_k=10.0) -> Function

Convert InterpolatorMethod enum to actual interpolator function.
The `s3_adapt_k` keyword controls the noise multiplier for S3 adaptive-tolerance methods.
"""
function get_interpolator_function(method::InterpolatorMethod, custom::Union{Nothing, Function} = nothing;
                                   s3_adapt_k::Float64 = 10.0)
	if method == InterpolatorCustom
		if isnothing(custom)
			error("InterpolatorCustom is listed without a function for it in custom_interpolators")
		end
		return custom
	end
	# If a custom override was provided (e.g., from GP caching layer), use it
	if custom !== nothing
		return custom
	end
	if method == InterpolatorAAA
		return aaad
	elseif method == InterpolatorAAADGPR
		return aaad_gpr_pivot
	elseif method == InterpolatorAAADOld
		return aaad_old_reliable
	elseif method == InterpolatorFHD
		return fhdn(5)  # Default to degree 5 FHD
	elseif method == InterpolatorGPUncertainty
		return agp_gpr_uq
	elseif method == InterpolatorGP
		return agp_gpr_robust
	elseif method == InterpolatorAGPRobustRQ
		return (xs, ys) -> agp_gpr_robust(xs, ys; kernel_type=:rq)
	elseif method == InterpolatorAGPRobustSEpRQ
		return (xs, ys) -> agp_gpr_robust(xs, ys; kernel_type=:se_plus_rq)
	elseif method == InterpolatorAGPRobustSExRQ
		return (xs, ys) -> agp_gpr_robust(xs, ys; kernel_type=:se_times_rq)
	elseif method == InterpolatorS2AAAMLE
		return s2_aaa_mle_interpolator
	# --- S3 Adaptive tolerance ---
	elseif method == InterpolatorS3AdaptSE
		k = s3_adapt_k
		return (xs, ys) -> s3_adapt_se_interpolator(xs, ys; k = k)
	elseif method == InterpolatorS3AdaptRQ
		k = s3_adapt_k
		return (xs, ys) -> s3_adapt_rq_interpolator(xs, ys; k = k)
	elseif method == InterpolatorS3AdaptSEpRQ
		k = s3_adapt_k
		return (xs, ys) -> s3_adapt_se_plus_rq_interpolator(xs, ys; k = k)
	elseif method == InterpolatorS3AdaptSExRQ
		k = s3_adapt_k
		return (xs, ys) -> s3_adapt_se_times_rq_interpolator(xs, ys; k = k)
	# --- S3 BIC model selection ---
	elseif method == InterpolatorS3BICSE
		return s3_bic_se_interpolator
	elseif method == InterpolatorS3BICRQ
		return s3_bic_rq_interpolator
	elseif method == InterpolatorS3BICSEpRQ
		return s3_bic_se_plus_rq_interpolator
	elseif method == InterpolatorS3BICSExRQ
		return s3_bic_se_times_rq_interpolator
	elseif method == InterpolatorChebyshevAICc
		return chebyshev_aicc
	elseif method == InterpolatorChebyshevBIC
		return chebyshev_bic
	else
		error("Unknown interpolator method: $method")
	end
end

"""
	interpolator_method_to_symbol(method::InterpolatorMethod) -> Symbol

Convert an InterpolatorMethod enum value to a Symbol for tagging results.
"""
function interpolator_method_to_symbol(method::InterpolatorMethod)
	method == InterpolatorAAA && return :aaad
	method == InterpolatorAAADGPR && return :aaad_gpr
	method == InterpolatorAAADOld && return :aaad_old
	method == InterpolatorFHD && return :fhd
	method == InterpolatorGPUncertainty && return :agp_uq
	method == InterpolatorGP && return :agp_robust
	method == InterpolatorAGPRobustRQ && return :agp_robust_rq
	method == InterpolatorAGPRobustSEpRQ && return :agp_robust_se_plus_rq
	method == InterpolatorAGPRobustSExRQ && return :agp_robust_se_times_rq
	method == InterpolatorS2AAAMLE && return :s2_aaa_mle
	method == InterpolatorS3AdaptSE && return :s3_adapt_se
	method == InterpolatorS3AdaptRQ && return :s3_adapt_rq
	method == InterpolatorS3AdaptSEpRQ && return :s3_adapt_se_plus_rq
	method == InterpolatorS3AdaptSExRQ && return :s3_adapt_se_times_rq
	method == InterpolatorS3BICSE && return :s3_bic_se
	method == InterpolatorS3BICRQ && return :s3_bic_rq
	method == InterpolatorS3BICSEpRQ && return :s3_bic_se_plus_rq
	method == InterpolatorS3BICSExRQ && return :s3_bic_se_times_rq
	method == InterpolatorChebyshevAICc && return :chebyshev_aicc
	method == InterpolatorChebyshevBIC && return :chebyshev_bic
	method == InterpolatorCustom && return :custom
	return :unknown
end


"""
	resolve_interpolator_list(opts::EstimationOptions) -> Vector{Tuple{InterpolatorMethod, Union{Nothing, Function}}}

The interpolators to run, as `(method, custom_func_or_nothing)` tuples, in the
order of `opts.interpolators`. Each `InterpolatorCustom` entry takes the next
function in `opts.custom_interpolators`.
"""
function resolve_interpolator_list(opts::EstimationOptions)
	result = Vector{Tuple{InterpolatorMethod, Union{Nothing, Function}}}()
	custom_idx = 0
	for method in opts.interpolators
		if method == InterpolatorCustom
			custom_idx += 1
			func = custom_idx <= length(opts.custom_interpolators) ? opts.custom_interpolators[custom_idx] : nothing
			push!(result, (method, func))
		else
			push!(result, (method, nothing))
		end
	end

	# Optional derivative-oriented undersmoothing for the exact AGPUQ route.
	# The default leaves the public function object and estimator behavior alone.
	if opts.gp_derivative_lengthscale_factor != 1.0
		factor = opts.gp_derivative_lengthscale_factor
		for i in eachindex(result)
			method, custom = result[i]
			if method == InterpolatorGPUncertainty && isnothing(custom)
				result[i] = (
					method,
					(xs, ys) -> agp_gpr_uq(
						xs, ys; lengthscale_factor = factor,
					),
				)
			end
		end
	end

	# Inject GP caching for shared kernels
	_inject_gp_caching!(result; s3_adapt_k = opts.s3_adapt_k)

	return result
end

"""
	_apply_noise_filter(list, σ̂) -> Vector

Remove AAA-family interpolators that produce catastrophic derivative errors
above their respective noise thresholds. Logs a `@warn` for each method
dropped. Returns the filtered list (possibly empty — caller is responsible
for fallback).

Thresholds (calibrated empirically on a forced Lotka–Volterra model):
- `InterpolatorS2AAAMLE`: drop when σ̂ > 1e-4
- `InterpolatorAAA`, `InterpolatorAAADOld`: drop when σ̂ > 1e-5
"""
function _apply_noise_filter(list::AbstractVector, σ̂::Float64)
	THRESHOLD_S2     = 1e-4
	THRESHOLD_AAAD   = 1e-5
	out = empty(list)
	for entry in list
		method = entry[1]
		if method == InterpolatorS2AAAMLE && σ̂ > THRESHOLD_S2
			@warn "[INTERP-GATE] Skipping S2AAAMLE: σ̂=$σ̂ > threshold $THRESHOLD_S2"
			continue
		end
		if (method == InterpolatorAAA || method == InterpolatorAAADOld) && σ̂ > THRESHOLD_AAAD
			@warn "[INTERP-GATE] Skipping $method: σ̂=$σ̂ > threshold $THRESHOLD_AAAD"
			continue
		end
		push!(out, entry)
	end
	return out
end

"""
	_gp_kernel_of(method::InterpolatorMethod) -> Union{Nothing, Symbol}

Map any GP-based interpolator method to its kernel type, or `nothing` if not GP-based.
"""
function _gp_kernel_of(method::InterpolatorMethod)::Union{Nothing, Symbol}
	method == InterpolatorGP && return :se
	method == InterpolatorAGPRobustRQ && return :rq
	method == InterpolatorAGPRobustSEpRQ && return :se_plus_rq
	method == InterpolatorAGPRobustSExRQ && return :se_times_rq
	# S3 Adaptive
	method == InterpolatorS3AdaptSE && return :se
	method == InterpolatorS3AdaptRQ && return :rq
	method == InterpolatorS3AdaptSEpRQ && return :se_plus_rq
	method == InterpolatorS3AdaptSExRQ && return :se_times_rq
	# S3 BIC
	method == InterpolatorS3BICSE && return :se
	method == InterpolatorS3BICRQ && return :rq
	method == InterpolatorS3BICSEpRQ && return :se_plus_rq
	method == InterpolatorS3BICSExRQ && return :se_times_rq
	return nothing
end

"""
	_is_s3_method(method::InterpolatorMethod) -> Bool

Returns true if the interpolator method is an S3 composite (GP→AAA→MLE).
"""
function _is_s3_method(method::InterpolatorMethod)::Bool
	return method in (InterpolatorS3AdaptSE, InterpolatorS3AdaptRQ, InterpolatorS3AdaptSEpRQ,
	                  InterpolatorS3AdaptSExRQ,
	                  InterpolatorS3BICSE, InterpolatorS3BICRQ, InterpolatorS3BICSEpRQ,
	                  InterpolatorS3BICSExRQ)
end


"""
	_is_s3_bic_method(method::InterpolatorMethod) -> Bool

Returns true for S3 BIC model-selection methods.
"""
function _is_s3_bic_method(method::InterpolatorMethod)::Bool
	return method in (InterpolatorS3BICSE, InterpolatorS3BICRQ, InterpolatorS3BICSEpRQ,
	                  InterpolatorS3BICSExRQ)
end

"""
	_make_cached_gp_func(kernel_type, cache) -> Function

Create a closure that fits a GP and stores it in the shared cache, keyed by `hash(ys)`.
"""
function _make_cached_gp_func(kernel_type::Symbol, cache::Dict{UInt64, Any})
	return function(xs, ys)
		key = hash(ys)
		gp = agp_gpr_robust(Vector{Float64}(xs), Vector{Float64}(ys); kernel_type = kernel_type)
		cache[key] = gp
		return gp
	end
end


"""
	_make_cached_s3_adapt_func(kernel_type, cache; k=10.0) -> Function

Create a closure that reuses a cached GP for S3 adaptive-tolerance refinement.
"""
function _make_cached_s3_adapt_func(kernel_type::Symbol, cache::Dict{UInt64, Any}; k::Float64 = 10.0)
	return function(xs, ys)
		key = hash(ys)
		t_vec = Vector{Float64}(xs)
		y_vec = Vector{Float64}(ys)
		gp = get(cache, key, nothing)
		if gp === nothing
			gp = agp_gpr_robust(t_vec, y_vec; kernel_type = kernel_type)
		end
		return s3_refine_gp_adaptive(gp, t_vec, y_vec; k = k)
	end
end

"""
	_make_cached_s3_bic_func(kernel_type, cache) -> Function

Create a closure that reuses a cached GP for S3 BIC model-selection refinement.
"""
function _make_cached_s3_bic_func(kernel_type::Symbol, cache::Dict{UInt64, Any})
	return function(xs, ys)
		key = hash(ys)
		t_vec = Vector{Float64}(xs)
		y_vec = Vector{Float64}(ys)
		gp = get(cache, key, nothing)
		if gp === nothing
			gp = agp_gpr_robust(t_vec, y_vec; kernel_type = kernel_type)
		end
		return s3_refine_gp_bic(gp, t_vec, y_vec)
	end
end

"""
	_inject_gp_caching!(interpolator_list; s3_adapt_k=10.0)

For interpolator methods that share a GP kernel (e.g., AGPRobust + S3AdaptSE both use SE),
inject closure-wrapped functions that share a `Dict` cache so the GP is fitted only once.
"""
function _inject_gp_caching!(interpolator_list::Vector{Tuple{InterpolatorMethod, Union{Nothing, Function}}};
                              s3_adapt_k::Float64 = 10.0)
	# Count how many methods share each GP kernel
	kernel_counts = Dict{Symbol, Int}()
	for (method, _) in interpolator_list
		kt = _gp_kernel_of(method)
		kt !== nothing && (kernel_counts[kt] = get(kernel_counts, kt, 0) + 1)
	end
	shared_kernels = Set(k for (k, c) in kernel_counts if c > 1)
	isempty(shared_kernels) && return

	# Create per-kernel caches
	caches = Dict(k => Dict{UInt64, Any}() for k in shared_kernels)

	# Replace functions for methods that share a kernel
	for i in eachindex(interpolator_list)
		method, custom = interpolator_list[i]
		custom !== nothing && continue  # don't override user-provided functions
		kt = _gp_kernel_of(method)
		kt === nothing && continue
		kt ∉ shared_kernels && continue
		cache = caches[kt]
		if _is_s3_bic_method(method)
			interpolator_list[i] = (method, _make_cached_s3_bic_func(kt, cache))
		elseif _is_s3_method(method)  # covers adapt + old S3
			interpolator_list[i] = (method, _make_cached_s3_adapt_func(kt, cache; k = s3_adapt_k))
		else
			interpolator_list[i] = (method, _make_cached_gp_func(kt, cache))
		end
	end
end

"""
	compute_shooting_indices(n_points, n_total; warp=true, beta=3.0) -> Vector{Int}

Compute shooting point indices across a time vector of length `n_total`.

When `warp=true` and `beta > 0`, uses an exponential warp to cluster more points
near the start of the interval (where transient dynamics are typically richest).
When `warp=false` or `beta ≈ 0`, returns equidistant indices.

Returns a sorted vector of unique indices in `[1, n_total]`.
"""
function compute_shooting_indices(n_points::Int, n_total::Int; warp::Bool = true, beta::Float64 = 3.0)
	n_total <= 0 && return Int[]
	n_total == 1 && return [1]
	n_total == 2 && return [1, 2]
	n_points <= 0 && return [max(1, n_total ÷ 2)]
	n_points == 1 && return [1]
	n_points >= n_total && return collect(1:n_total)

	if !warp || abs(beta) < 1e-10
		# Equidistant
		indices = round.(Int, range(1, n_total, length = n_points))
	else
		u = range(0.0, 1.0, length = n_points)
		frac = (exp.(beta .* u) .- 1) ./ (exp(beta) - 1)
		indices = round.(Int, 1 .+ frac .* (n_total - 1))
		indices = clamp.(indices, 1, n_total)
	end
	return unique(indices)
end

"""
	get_polish_optimizer(method::PolishMethod)

Convert PolishMethod enum to actual optimizer object/type.

For scalar `Optimization.solve`-based methods (legacy default `PolishNewtonTrust`,
plus `PolishBFGS`/`PolishLBFGS`) returns a zero-arg constructor — the call site
does `optimizer_type()` to instantiate.

For residual-mode methods (`PolishLeastSquares`, `PolishFastLMBoundedLog`) returns
a tagged tuple `(kind::Symbol, factory)` consumed by `_polish_single_residual`.
The kinds are `:lso_direct` (LeastSquaresOptim) and `:fastlm_direct` (FastLevenbergMarquardt).
"""
function get_polish_optimizer(method::PolishMethod)
	if method == PolishNewtonTrust
		return NewtonTrustRegion
	elseif method == PolishBFGS
		return BFGS
	elseif method == PolishLBFGS
		return LBFGS
	elseif method == PolishLeastSquares
		return (:lso_direct, () -> LeastSquaresOptim.LevenbergMarquardt())
	elseif method == PolishFastLMBoundedLog
		return (:fastlm_direct, () -> nothing)
	else
		error("Unknown polish method: $method")
	end
end

"""
	get_ad_backend(backend::Symbol)

Convert AD backend symbol to an Optimization.jl AD type.

# Supported backends
- `:forward` → `AutoForwardDiff()` (default, works with most problems)
- `:finite` → `AutoFiniteDiff()` (fallback, no AD required)

Note: `:zygote` and `:enzyme` were removed. Zygote segfaults Julia 1.12's JIT compiler,
and Enzyme cannot compile the trajectory loss, which rebuilds the ODE problem through
symbolic indexing. ForwardDiff is the recommended backend (the only AD backend that
works through adaptive ODE solvers).
"""
function get_ad_backend(backend::Symbol)
	backend === :forward && return Optimization.AutoForwardDiff()
	backend === :finite && return Optimization.AutoFiniteDiff()
	error("Unknown AD backend :$backend. Supported backends: :forward, :finite")
end

"""
	merge_options(base::EstimationOptions; kwargs...) -> EstimationOptions

Create a new EstimationOptions struct by merging keyword arguments with an existing options struct.
This is useful for temporarily overriding specific options.

# Examples
```julia
base_opts = EstimationOptions()
new_opts = ODEParameterEstimation.merge_options(base_opts; abstol = 1e-10, diagnostics = true)
```
"""
function merge_options(base::EstimationOptions; kwargs...)
	# Get all field names and their values from base
	fields = fieldnames(EstimationOptions)
	values = Dict(f => getfield(base, f) for f in fields)

	# Override with any provided kwargs
	for (k, v) in kwargs
		if k in fields
			values[k] = v
		else
			error("Unknown option: $k")
		end
	end

	# Create new struct
	return EstimationOptions(; values...)
end

"""
	_options_from_keywords(keywords) -> EstimationOptions

`EstimationOptions(; keywords...)`, with a mistyped name reported by name rather
than as a `MethodError` that lists every field.
"""
function _options_from_keywords(keywords)
	unknown = setdiff(keys(keywords), fieldnames(EstimationOptions))
	if !isempty(unknown)
		names = join(("`$name`" for name in unknown), ", ", " and ")
		verb = length(unknown) == 1 ? "is not an estimation option" : "are not estimation options"
		# `interpolator` and `custom_interpolator` were options before 1.0.
		hint = any(in(unknown), (:interpolator, :custom_interpolator)) ?
			" Interpolators are given as a list, as in `interpolators = [InterpolatorAAA]`." : ""
		throw(ArgumentError("$names $verb.$hint `?EstimationOptions` lists the options."))
	end
	return EstimationOptions(; keywords...)
end

"""
	validate_options(opts::EstimationOptions) -> Bool

Check the options. Each problem is logged: as an error when the options cannot
work, as a warning when they work but are unwise.

# Returns
- `true` if the options can be used
- `false` if any of them cannot
"""
function validate_options(opts::EstimationOptions)
	valid = true

	# Check algorithm selections that otherwise fail only after expensive setup.
	if !(opts.opt_ad_backend in (:forward, :finite))
		@error "opt_ad_backend must be :forward or :finite (got $(opts.opt_ad_backend))"
		valid = false
	end

	if !(opts.hc_compile_mode in (:all, :none, :mixed))
		@error "hc_compile_mode must be :all, :none, or :mixed (got $(opts.hc_compile_mode))"
		valid = false
	end

	if isempty(opts.interpolators)
		@error "interpolators is empty. List at least one, such as [InterpolatorAAA]."
		valid = false
	else
		required_custom = count(==(InterpolatorCustom), opts.interpolators)
		if length(opts.custom_interpolators) < required_custom
			@error "interpolators contains $required_custom InterpolatorCustom entr$(required_custom == 1 ? "y" : "ies"), but only $(length(opts.custom_interpolators)) custom_interpolators function(s) were provided"
			valid = false
		end
	end

	# Check tolerances
	if opts.abstol <= 0 || opts.reltol <= 0
		@error "Tolerances must be positive"
		valid = false
	end

	if opts.abstol < 1e-16 || opts.reltol < 1e-16
		@warn "Extremely small tolerances may cause numerical issues"
	end

	# Check thresholds
	if opts.clustering_threshold < 0
		@error "clustering_threshold must be non-negative"
		valid = false
	end

	if opts.shooting_points < 0
		@error "shooting_points must be non-negative"
		valid = false
	end

	# Check optimization bounds
	if !isnothing(opts.opt_lb) && !isnothing(opts.opt_ub)
		if length(opts.opt_lb) != length(opts.opt_ub)
			@error "Optimization bounds must have the same length"
			valid = false
		elseif any(opts.opt_lb .> opts.opt_ub)
			@error "Lower bounds must not exceed upper bounds"
			valid = false
		end
	end

	# Check polish safeguard parameters
	if !(opts.polish_maxtime > 0)
		@error "polish_maxtime must be positive (got $(opts.polish_maxtime)); use Inf for no limit"
		valid = false
	end
	if opts.polish_stagnation_window < 5
		@warn "polish_stagnation_window < 5 is too aggressive; may stop prematurely"
	end
	if opts.polish_max_concurrency < 1
		@warn "polish_max_concurrency must be ≥ 1 (got $(opts.polish_max_concurrency)); will treat as 1"
	end

	# Polish coordinate transform policy
	if !(opts.polish_coordinate_policy in (:auto, :linear, :log_only, :shifted_log_only))
		@error "polish_coordinate_policy must be one of :auto, :linear, :log_only, :shifted_log_only (got $(opts.polish_coordinate_policy))"
		valid = false
	end

	# Polish regularization
	if opts.polish_regularization_lambda < 0
		@error "polish_regularization_lambda must be non-negative (got $(opts.polish_regularization_lambda))"
		valid = false
	end

	# Polish soft-wall
	if opts.polish_softwall_lambda < 0
		@error "polish_softwall_lambda must be non-negative (got $(opts.polish_softwall_lambda))"
		valid = false
	end
	if !(0.0 <= opts.polish_softwall_epsilon < 0.5)
		@error "polish_softwall_epsilon must be in [0, 0.5) (got $(opts.polish_softwall_epsilon))"
		valid = false
	end

	# LSO trust-region delta must be positive
	if opts.polish_lso_delta <= 0
		@error "polish_lso_delta must be positive (got $(opts.polish_lso_delta))"
		valid = false
	end

	# Sensitivity-seed parameters
	if opts.sensitivity_seed_probe_scale <= 0
		@error "sensitivity_seed_probe_scale must be positive (got $(opts.sensitivity_seed_probe_scale))"
		valid = false
	end
	if opts.sensitivity_seed_eigenvalue_threshold < 0 || opts.sensitivity_seed_eigenvalue_threshold > 1
		@error "sensitivity_seed_eigenvalue_threshold must be in [0, 1] (got $(opts.sensitivity_seed_eigenvalue_threshold))"
		valid = false
	end
	if opts.sensitivity_sigma_d_kappa < 1.0
		@warn "sensitivity_sigma_d_kappa < 1.0 implies higher derivatives are MORE certain than lower (got $(opts.sensitivity_sigma_d_kappa))"
	end
	if !(opts.terminal_fallback in (:none, :direct_opt))
		@error "terminal_fallback must be :none or :direct_opt"
		valid = false
	end
	if !(opts.homotopy_tracking_mode in (:parameter, :gamma_straight, :gamma_straight_fallback, :generic_start))
		@error "homotopy_tracking_mode must be :parameter, :gamma_straight, :gamma_straight_fallback, or :generic_start (got $(opts.homotopy_tracking_mode))"
		valid = false
	end
	if opts.gamma_max_seeds < 1
		@error "gamma_max_seeds must be ≥ 1 (got $(opts.gamma_max_seeds))"
		valid = false
	end
	if !isnothing(opts.algebraic_multiplicity) && opts.algebraic_multiplicity <= 0
		@error "algebraic_multiplicity must be a positive integer or `nothing` (got $(opts.algebraic_multiplicity))"
		valid = false
	end
	if opts.rough_cluster_eps <= 0
		@error "rough_cluster_eps must be positive (got $(opts.rough_cluster_eps))"
		valid = false
	end
	if opts.subspace_cluster_eps <= 0
		@error "subspace_cluster_eps must be positive (got $(opts.subspace_cluster_eps))"
		valid = false
	end
	if opts.branch_diversity_eps < 0
		@error "branch_diversity_eps must be non-negative (got $(opts.branch_diversity_eps))"
		valid = false
	end
	if opts.branch_completion_max_anchors <= 0
		@error "branch_completion_max_anchors must be positive (got $(opts.branch_completion_max_anchors))"
		valid = false
	end
	if opts.branch_completion_residual_tol < 0
		@error "branch_completion_residual_tol must be non-negative (got $(opts.branch_completion_residual_tol))"
		valid = false
	end
	if !(opts.backsolve_recovery in (:none, :algebraic_resolve))
		@error "backsolve_recovery must be :none or :algebraic_resolve"
		valid = false
	end

	# Check data parameters
	if opts.datasize < 3
		@warn "Very small datasize (<3) may lead to underdetermined systems"
	end

	if length(opts.time_interval) != 2 || opts.time_interval[1] >= opts.time_interval[2]
		@error "time_interval must be [t_start, t_end] with t_start < t_end"
		valid = false
	end

	if opts.noise_level < 0
		@error "noise_level must be non-negative"
		valid = false
	end

	times = opts.uneven_sampling_times
	if !isempty(times)
		if length(times) != opts.datasize
			@error "uneven_sampling_times must have datasize = $(opts.datasize) entries (got $(length(times)))"
			valid = false
		elseif !issorted(times; lt = <=)
			@error "uneven_sampling_times must be increasing"
			valid = false
		end
	end
	if !(opts.noise_model in (:additive, :relative, :none))
		@error "noise_model must be :additive, :relative or :none"
		valid = false
	end

	# Check SI parameters
	if !(opts.si_fix_strategy in (:local_basis, :identifiable_functions))
		@error "si_fix_strategy must be :local_basis or :identifiable_functions"
		valid = false
	end
	if opts.si_probability <= 0 || opts.si_probability > 1
		@error "si_probability must be in (0, 1]"
		valid = false
	end

	if opts.nooutput && opts.diagnostics
		@info "diagnostics=true with nooutput=true prints diagnostic lines but not progress or the result summary; set nooutput=false as well for the full output"
	end

	if opts.terminal_fallback != :none && opts.flow == FlowDirectOpt
		@warn "terminal_fallback is ignored when flow=FlowDirectOpt (FlowDirectOpt already is the terminal optimizer)"
	end

	if opts.backsolve_recovery != :none && opts.flow != FlowStandard
		@info "backsolve_recovery is ignored outside FlowStandard"
	end

	if !(opts.polish_solver_jacobian in (:symbolic, :forwarddiff, :finitediff))
		@error "polish_solver_jacobian must be :symbolic, :forwarddiff, or :finitediff"
		valid = false
	end
	if opts.polish_solver_chunk_size < 0
		@error "polish_solver_chunk_size must be nonnegative"
		valid = false
	end

	if !isnothing(opts.seed) && opts.gamma_seed < 0
		@warn "seed is set, but gamma_seed < 0 draws fresh entropy for the homotopy start, so results will not be reproducible"
	end

	if !(opts.system_construction_policy in (:legacy, :noise_frontier))
		@error "system_construction_policy must be :legacy or :noise_frontier (got $(opts.system_construction_policy))"
		valid = false
	end
	if opts.construction_candidate_limit <= 0
		@error "construction_candidate_limit must be positive (got $(opts.construction_candidate_limit))"
		valid = false
	end
	if opts.construction_beam_width <= 0
		@error "construction_beam_width must be positive (got $(opts.construction_beam_width))"
		valid = false
	end
	if !(opts.multipoint_pair_strategy in (:spread, :boundary_order))
		@error "multipoint_pair_strategy must be :spread or :boundary_order"
		valid = false
	end

	if !(opts.uq_failure_policy in (:return_failed, :throw))
		@error "uq_failure_policy must be :return_failed or :throw"
		valid = false
	end
	if !(opts.uq_noise_source in (:learned_gp_homoscedastic, :smoother_residual_edf))
		@error "uq_noise_source must be :learned_gp_homoscedastic or :smoother_residual_edf"
		valid = false
	end
	if !(isfinite(opts.gp_derivative_lengthscale_factor) &&
			opts.gp_derivative_lengthscale_factor > 0)
		@error "gp_derivative_lengthscale_factor must be finite and positive"
		valid = false
	end
	if opts.compute_uncertainty
		configured = resolve_interpolator_list(opts)
		if !any(first(pair) == InterpolatorGPUncertainty for pair in configured)
			@warn "compute_uncertainty=true but no exact UQ-capable interpolator is configured. Add InterpolatorGPUncertainty explicitly; the estimator pool will not be changed automatically."
		end
	end


	return valid
end

"""
	print_options(io::IO, opts::EstimationOptions; compact=false)

Print every option and its value, grouped by topic. With `compact = true`, print
only the options that differ from their defaults, and the defaults beside them;
this is how `EstimationOptions` values show themselves.
"""
function print_options(io::IO, opts::EstimationOptions; compact = false)
	defaults = EstimationOptions()

	println(io, "EstimationOptions:")

	categories = [
		("Solver and Algorithm", [:system_solver, :ode_solver, :interpolators]),
		("Tolerances", [:abstol, :reltol]),
		("Solution Validation", [:clustering_threshold]),
		("Multi-shot", [:shooting_points, :shooting_warp, :shooting_warp_beta]),
		("Optimization", [:polish_solutions, :polish_solver_solutions, :polish_solver_jacobian, :polish_solver_chunk_size, :polish_method, :polish_maxiters, :opt_maxiters,
			:opt_lb, :opt_ub, :opt_ad_backend, :polish_maxtime, :polish_divergence_factor, :polish_stagnation_window, :polish_ode_maxiters]),
		("Rescue Policy", [:terminal_fallback, :backsolve_recovery]),
		("Data Sampling", [:datasize, :time_interval, :noise_level, :noise_model,
			:uneven_sampling_times]),
		("Reproducibility", [:seed]),
		("Debug Flags", [:nooutput, :diagnostics, :progress, :profile_phases]),
		("Feature Flags", [:flow, :use_si_template, :save_system,
			:compute_uncertainty, :uq_failure_policy, :uq_noise_source,
			:gp_derivative_lengthscale_factor,
			:auto_handle_transcendentals]),
		("System Construction", [:system_construction_policy, :construction_candidate_limit,
			:construction_beam_width, :construction_compute_mixed_volume]),
		("HomotopyContinuation", [:use_parameter_homotopy, :hc_real_tol, :hc_show_progress, :homotopy_tracking_mode, :gamma_max_seeds, :gamma_seed, :use_multipoint, :multipoint_n_points, :multipoint_max_pairs, :multipoint_pair_strategy]),
		("StructuralIdentifiability", [:si_probability, :si_fix_strategy]),
		("File I/O", [:save_filepath]),
	]
	grouped = Set(Iterators.flatten(last.(categories)))
	push!(categories, ("Other", [f for f in fieldnames(EstimationOptions) if !(f in grouped)]))

	for (category, fields) in categories
		printed_header = false
		for field in fields
			val = getfield(opts, field)
			default_val = getfield(defaults, field)

			if !compact || val != default_val
				if !printed_header
					println(io, "\n  $category:")
					printed_header = true
				end

				# Special formatting for functions
				if isa(val, Function)
					val_str = string(val)
				else
					val_str = repr(val)
				end

				if compact && val != default_val
					println(io, "    $field: $val_str (default: $(repr(default_val)))")
				else
					println(io, "    $field: $val_str")
				end
			end
		end
	end
end

# Define show method for pretty printing
Base.show(io::IO, opts::EstimationOptions) = print_options(io, opts; compact = true)
