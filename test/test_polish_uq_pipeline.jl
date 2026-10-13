using ODEParameterEstimation
using ODEParameterEstimation.Examples
using Test
using Logging
using LinearAlgebra
using Random

function _pup_quiet(f)
	redirect_stdout(devnull) do
		redirect_stderr(devnull) do
			with_logger(NullLogger()) do
				return f()
			end
		end
	end
end

function _pup_physical_vector(result, labels::Vector{String})
	parameter_values = Dict(
		replace(string(key), r"\(.*\)" => "") => Float64(value)
		for (key, value) in result.parameters
	)
	state_values = Dict(
		replace(string(key), r"\(.*\)" => "") => Float64(value)
		for (key, value) in result.states
	)
	return Float64[
		haskey(parameter_values, label) ? parameter_values[label] : state_values[label]
		for label in labels
	]
end

@testset "trajectory-polish estimator-aware UQ pipeline" begin
	Random.seed!(7731)
	sample_options = EstimationOptions(
		datasize = 61, time_interval = [0.0, 1.0], noise_level = 1e-4,
		nooutput = true,
	)
	problem = sample_problem_data(simple(), sample_options)
	options = EstimationOptions(
		datasize = 61,
		time_interval = [0.0, 1.0],
		noise_level = 1e-4,
		nooutput = true,
		diagnostics = false,
		compute_uncertainty = true,
		interpolators = InterpolatorMethod[InterpolatorAGPUQ],
		auto_filter_interpolators = false,
		shooting_points = 1,
		use_parameter_homotopy = false,
		use_multipoint = false,
		synthesize_aggregate_candidates = false,
		polish_solutions = true,
		polish_solver_solutions = true,
		polish_maxtime = 30.0,
		polish_maxiters = 100,
		polish_max_concurrency = 1,
		branch_completion = false,
		save_system = false,
		# Keep the retained influence and public report in the same coordinates
		# for the direct-dispatch and perturb/refit comparisons below. Automatic
		# rescaling itself is covered independently in test_rescaling.jl.
		auto_rescale = false,
	)

	observed, _ = _pup_quiet() do
		ODEParameterEstimation._with_run_context(
			selection_recipe = ODEParameterEstimation.FixedSinglePointRecipe(
				20; interpolator_source = :agp_uq,
			),
		) do
			pipeline = analyze_parameter_estimation_problem(deepcopy(problem), options)
			_, analysis, report = pipeline
			selected = first(analysis.returned_results)
			identity = selected.provenance.estimator_identity
			artifact = ODEParameterEstimation._run_ctx_artifact(identity.candidate_id)
			influence = ODEParameterEstimation._run_ctx_uq_influence(identity.candidate_id)
			(; pipeline, selected, identity, artifact, influence)
		end
	end

	_, analysis, report = observed.pipeline
	@test report isa UncertaintyReport
	@test observed.identity.estimator_kind == :trajectory_polish
	@test report.target.identity.candidate_id == observed.identity.candidate_id
	@test report.target.artifact_match == :exact
	@test observed.artifact isa ODEParameterEstimation.PolishUQArtifact
	@test !isnothing(observed.influence)
	@test observed.influence.coordinate_labels == report.param_labels
	@test all(isfinite, report.param_covariance)
	@test isfinite(report.linearization_diagnostics.jacobian_condition_equilibrated)
	@test isfinite(report.linearization_diagnostics.linear_solve_backward_error)

	# Exercise the actual direct-optimizer production path, including creation
	# and lookup of its retained score-equation artifact.
	Random.seed!(9191)
	direct_options = EstimationOptions(
		flow = FlowDirectOpt,
		datasize = 61,
		time_interval = [0.0, 1.0],
		noise_level = 1e-4,
		nooutput = true,
		diagnostics = false,
		compute_uncertainty = true,
		interpolators = InterpolatorMethod[InterpolatorAGPUQ],
		auto_filter_interpolators = false,
		opt_lb = fill(1e-4, 4),
		opt_ub = fill(5.0, 4),
		opt_maxiters = 200,
		# The first callback includes cold compilation, which can exhaust a
		# 30-second budget before even one optimizer step on a busy CI host.
		# This checks convergence/UQ; test_polish_maxtime.jl covers deadlines.
		polish_maxtime = Inf,
		save_system = false,
		auto_rescale = false,
	)
	_, direct_analysis, direct_report = _pup_quiet() do
		analyze_parameter_estimation_problem(deepcopy(problem), direct_options)
	end
	@test direct_report isa UncertaintyReport
	@test first(direct_analysis.returned_results).provenance.estimator_identity.estimator_kind ==
		:direct_optimization
	@test direct_report.target.identity.estimator_kind == :direct_optimization
	@test direct_report.target.artifact_match == :exact
	@test all(isfinite, direct_report.param_covariance)
	@test isfinite(direct_report.linearization_diagnostics.jacobian_condition_equilibrated)
	@test maximum(abs.(direct_report.estimate_values .- direct_report.param_true_values)) < 1e-2

	# Perturb one retained raw observation and refit from the selected optimum.
	# This is an actual tiny-ODE optimizer check of the reported influence, not a
	# quadratic surrogate: it covers observation ordering, sign, and the physical
	# parameter/state coordinate map end to end.
	data_column = findfirst(==("y1(t_index=17)"), observed.influence.observation_labels)
	@test !isnothing(data_column)
	if !isnothing(data_column)
		perturbed = deepcopy(problem)
		measurement_key = first(problem.measured_quantities).rhs
		step = 1e-4 * max(abs(perturbed.data_sample[measurement_key][17]), 1.0)
		perturbed.data_sample[measurement_key][17] += step
		perturbed_context = _pup_quiet() do
			ODEParameterEstimation._build_polish_context(perturbed; opts = options)
		end
		seed = vcat(
			[observed.selected.states[state] for state in perturbed_context.unknown_syms],
			[observed.selected.parameters[param] for param in perturbed_context.param_syms],
		)
		refit, _ = _pup_quiet() do
			ODEParameterEstimation._polish_single_from_context(
				perturbed_context, seed;
				polish_method = options.polish_method,
				maxiters = 100,
				maxtime = 30.0,
			)
		end
		observed_delta = _pup_physical_vector(refit, report.param_labels) .-
			report.estimate_values
		predicted_delta = observed.influence.influence[:, data_column] .* step
		relative_discrepancy = norm(observed_delta - predicted_delta) /
			max(norm(predicted_delta), 1e-10)
		@test dot(observed_delta, predicted_delta) > 0
		@test relative_discrepancy < 0.25
	end
end
