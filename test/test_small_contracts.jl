# Small, fast contracts: every option check, the interpolator names, how
# results are ranked, the phase table, the point picker and error messages.
# None of these runs an estimation. Added 2026-10-14, when a coverage review
# found them untested.

using ODEParameterEstimation
using OrderedCollections
using Test

const _SMALL = ODEParameterEstimation

# `validate_options` result and log records for one set of options.
function _checked(; options...)
	valid = Ref(true)
	logs, _ = Test.collect_test_logs(; min_level = Base.CoreLogging.Info) do
		valid[] = validate_options(EstimationOptions(; options...))
	end
	return valid[], logs
end

_has_record(logs, level, pattern) =
	any(record -> record.level == level && occursin(pattern, string(record.message)), logs)

@testset "Every rejected option says which one it is" begin
	cases = [
		(hc_compile_mode = :fast,) => "hc_compile_mode",
		(abstol = 0.0,) => "Tolerances must be positive",
		(reltol = -1.0,) => "Tolerances must be positive",
		(clustering_threshold = -1.0,) => "clustering_threshold",
		(shooting_points = -1,) => "shooting_points",
		(opt_lb = [0.0, 0.0], opt_ub = [1.0, 1.0, 1.0]) => "bounds must have the same length",
		(opt_lb = [2.0], opt_ub = [1.0]) => "Lower bounds must not exceed upper bounds",
		(polish_maxtime = 0.0,) => "polish_maxtime",
		(polish_coordinate_policy = :sideways,) => "polish_coordinate_policy",
		(polish_regularization_lambda = -1.0,) => "polish_regularization_lambda",
		(polish_softwall_lambda = -1.0,) => "polish_softwall_lambda",
		(polish_softwall_epsilon = 0.5,) => "polish_softwall_epsilon",
		(polish_lso_delta = 0.0,) => "polish_lso_delta",
		(sensitivity_seed_probe_scale = 0.0,) => "sensitivity_seed_probe_scale",
		(sensitivity_seed_eigenvalue_threshold = 2.0,) => "sensitivity_seed_eigenvalue_threshold",
		(terminal_fallback = :retry,) => "terminal_fallback",
		(rank_strategy = :random,) => "rank_strategy",
		(cluster_method = :kmeans,) => "cluster_method",
		(homotopy_tracking_mode = :fast,) => "homotopy_tracking_mode",
		(gamma_max_seeds = 0,) => "gamma_max_seeds",
		(algebraic_multiplicity = 0,) => "algebraic_multiplicity",
		(rough_cluster_eps = 0.0,) => "rough_cluster_eps",
		(subspace_cluster_eps = 0.0,) => "subspace_cluster_eps",
		(branch_diversity_eps = -1.0,) => "branch_diversity_eps",
		(branch_completion_max_anchors = 0,) => "branch_completion_max_anchors",
		(branch_completion_residual_tol = -1.0,) => "branch_completion_residual_tol",
		(backsolve_recovery = :retry,) => "backsolve_recovery",
		(t0_state_completion = :guess,) => "t0_state_completion",
		(time_interval = [1.0, 0.0],) => "time_interval",
		(time_interval = [0.0, 1.0, 2.0],) => "time_interval",
		(noise_level = -0.1,) => "noise_level",
		(noise_model = :pink,) => "noise_model",
		(si_probability = 0.0,) => "si_probability",
		(si_probability = 1.5,) => "si_probability",
		(system_construction_policy = :other,) => "system_construction_policy",
		(construction_candidate_limit = 0,) => "construction_candidate_limit",
		(construction_beam_width = 0,) => "construction_beam_width",
		(uq_failure_policy = :ignore,) => "uq_failure_policy",
		(uq_noise_source = :guess,) => "uq_noise_source",
		(si_placeholder_fail_categories = [:not_a_category],) => "si_placeholder_fail_categories",
		(uneven_sampling = true,) => "uneven_sampling_times",
		(uneven_sampling = true, datasize = 3, uneven_sampling_times = [0.0, 1.0]) => "uneven_sampling_times",
		(uneven_sampling = true, datasize = 3, uneven_sampling_times = [0.0, 2.0, 1.0]) => "uneven_sampling_times",
	]
	for (options, message) in cases
		valid, logs = _checked(; options...)
		@test !valid
		@test _has_record(logs, Base.CoreLogging.Error, message)
	end
	# The default options pass, quietly.
	valid, logs = _checked()
	@test valid
	@test isempty(logs)
	@test first(_checked(uneven_sampling = true, datasize = 3, uneven_sampling_times = [0.0, 0.5, 2.0]))
end

@testset "Options that work but are unwise draw a warning" begin
	warnings = [
		(abstol = 1e-17,) => "Extremely small tolerances",
		(polish_stagnation_window = 2,) => "polish_stagnation_window",
		(polish_max_concurrency = 0,) => "polish_max_concurrency",
		(sensitivity_sigma_d_kappa = 0.5,) => "sensitivity_sigma_d_kappa",
		(datasize = 2,) => "Very small datasize",
		(flow = FlowDirectOpt,) => "terminal_fallback is ignored",
		(t0_state_completion = :seed_for_polish, polish_solutions = false) => "requires polish_solutions",
		(compute_uncertainty = true,) => "InterpolatorAGPUQ",
	]
	for (options, message) in warnings
		valid, logs = _checked(; options...)
		@test valid
		@test _has_record(logs, Base.CoreLogging.Warn, message)
	end
	notes = [
		(diagnostics = true,) => "diagnostics=true with nooutput=true",
		(flow = FlowDirectOpt, terminal_fallback = :none) => "backsolve_recovery is ignored",
		(flow = FlowDirectOpt, terminal_fallback = :none, backsolve_recovery = :none, t0_state_completion = :seed_for_polish) =>
			"t0_state_completion is ignored",
	]
	for (options, message) in notes
		valid, logs = _checked(; options...)
		@test valid
		@test _has_record(logs, Base.CoreLogging.Info, message)
	end
end

@testset "Every interpolator and solver has a function and a name" begin
	names = Symbol[]
	for method in instances(InterpolatorMethod)
		method == InterpolatorCustom && continue
		@test get_interpolator_function(method) isa Function
		push!(names, interpolator_method_to_symbol(method))
	end
	@test allunique(names)
	@test !(:unknown in names)
	@test interpolator_method_to_symbol(InterpolatorCustom) == :custom
	@test_throws "InterpolatorCustom is listed without a function" get_interpolator_function(InterpolatorCustom)
	mine(xs, ys) = x -> 0.0
	@test get_interpolator_function(InterpolatorCustom, mine) === mine
	# A supplied function takes the place of the built-in one.
	@test get_interpolator_function(InterpolatorAAAD, mine) === mine
	for method in instances(SystemSolverMethod)
		@test get_solver_function(method) isa Function
	end
end

@testset "Results are ranked as rank_strategy says" begin
	result(err, parameters; source = 1) =
		(err = err, parameters = OrderedDict(parameters), provenance = (polish_source_hc_idx = source,))
	lb, ub = [1e-5], [10.0]

	# Two parameters pinned at the bounds, one comfortably inside them.
	pinned = result(1e-12, [:a => 9.99, :b => 1.0e-5])
	typical = result(1e-8, [:a => 0.7, :b => 1.3])
	untagged = result(1e-10, [:a => 0.8, :b => 1.1]; source = -1)
	@test _SMALL.saturation_count(pinned, lb, ub) == 2
	@test _SMALL.saturation_count(typical, lb, ub) == 0
	@test _SMALL.saturation_count(typical, nothing, ub) == 0
	@test _SMALL.saturation_count(typical, [-1.0], ub) == 0

	ranked(strategy; options...) = _SMALL.rank_cluster_representatives(
		[pinned, typical, untagged], EstimationOptions(; rank_strategy = strategy, options...))
	@test ranked(:err_only) == [pinned, untagged, typical]
	@test ranked(:sat_err; opt_lb = lb, opt_ub = ub) == [untagged, typical, pinned]
	@test ranked(:sat_neg1_err; opt_lb = lb, opt_ub = ub) == [typical, untagged, pinned]
	@test ranked(:lognorm_err) == [untagged, typical, pinned]
	@test ranked(:lognorm_neg1_err) == [untagged, typical, pinned]
	# With equal parameter sizes, a row from the algebraic solve beats an untagged one.
	twin = result(1e-10, [:a => 0.8, :b => 1.1]; source = 3)
	@test _SMALL.rank_cluster_representatives([untagged, twin], EstimationOptions(rank_strategy = :lognorm_neg1_err)) == [twin, untagged]
	@test _SMALL.lognorm_score(result(0.0, [:a => -1.0])) == Inf
end

@testset "The phase table" begin
	@test _SMALL._format_bytes(1023) == "1023 B"
	@test _SMALL._format_bytes(1024) == "1.0 KiB"
	@test _SMALL._format_bytes(1024^2) == "1.0 MiB"
	@test _SMALL._format_bytes(3 * 1024^3) == "3.00 GiB"

	printed(stats) = mktemp() do path, io
		redirect_stdout(() -> _SMALL._print_phase_profile(stats), io)
		close(io)
		read(path, String)
	end
	with_memory = OrderedDict{String, NamedTuple}(
		"Setup" => (time = 1.5, bytes = 2048, gctime = 0.1, rss_mb = 500.0, rss_delta_mb = 20.0),
		"Solving" => (time = 3.0, bytes = 5 * 1024^2, gctime = 0.0, rss_mb = 620.0, rss_delta_mb = 120.0))
	table = printed(with_memory)
	@test occursin("1. Setup", table) && occursin("2. Solving", table)
	@test occursin("TOTAL", table) && occursin("RSS (MB)", table)
	@test occursin("4.50", table)   # total seconds
	older = OrderedDict{String, NamedTuple}("Setup" => (time = 1.0, bytes = 10, gctime = 0.0))
	table = printed(older)
	@test occursin("TOTAL", table) && !occursin("RSS", table)
	@test printed(OrderedDict{String, NamedTuple}()) == ""
end

@testset "Error messages" begin
	x = "D(h) ~ k * sqrt(h)"
	sqrt_error = _SMALL.classify_unsupported_substitution_error(x, "sqrt(x) + 1")
	@test sqrt_error isa UnsupportedModelClassError
	@test sqrt_error.category == :sqrt_nonlinearity
	@test sqrt_error.expressions == [x]
	@test _SMALL.classify_unsupported_substitution_error(x, "sin(x)").category == :state_trigonometric
	@test _SMALL.classify_unsupported_substitution_error(x, "cos(x)").category == :state_trigonometric
	@test _SMALL.classify_unsupported_substitution_error(x, "exp(x)").category == :unsupported_transcendental
	@test isnothing(_SMALL.classify_unsupported_substitution_error(x, "x^2 + 1"))

	order = _SMALL.UnsupportedDerivativeOrderError(25, 20, :taylordiff, nothing)
	@test occursin("needs derivative 25", sprint(showerror, order))
	@test !occursin("It came up", sprint(showerror, order))
	in_context = _SMALL.UnsupportedDerivativeOrderError(25, 20, :taylordiff, "while instantiating the template")
	@test occursin("It came up while instantiating the template.", sprint(showerror, in_context))

	sampling = _SMALL.SamplingFailureError("blowup", :Unstable, 41, OrderedDict("y1" => 12))
	message = sprint(showerror, sampling)
	@test occursin("Could not simulate data for blowup", message)
	@test occursin("41 time points", message)

	@test uq_reliability(nothing).availability == :not_requested
end

@testset "A GP interpolator reports its variance" begin
	xs = collect(range(-0.5, 0.5; length = 15))
	fit = agp_gpr_robust(xs, sin.(xs))
	mean, variance = mean_and_var(fit, 0.1)
	@test mean ≈ fit(0.1)
	@test mean ≈ sin(0.1) atol = 1e-3
	@test variance >= 0
end
