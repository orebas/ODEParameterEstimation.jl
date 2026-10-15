using ODEParameterEstimation
using ODEParameterEstimation: InterpolatorMethod
using ODEParameterEstimation.Examples
using Test
using Logging
using LinearAlgebra
using Random

function _bup_quiet(f)
	redirect_stdout(devnull) do
		redirect_stderr(devnull) do
			with_logger(NullLogger()) do
				return f()
			end
		end
	end
end

@testset "branch-completion estimator-aware UQ pipeline" begin
	Random.seed!(8144)
	sample_options = EstimationOptions(
		datasize = 41, time_interval = [0.0, 1.0], noise_level = 1e-4,
		nooutput = true,
	)
	problem = sample_problem_data(harmonic_oscillator(), sample_options)
	options = EstimationOptions(
		datasize = 41,
		time_interval = [0.0, 1.0],
		noise_level = 1e-4,
		nooutput = true,
		diagnostics = false,
		compute_uncertainty = true,
		interpolators = InterpolatorMethod[InterpolatorGPUncertainty],
		auto_filter_interpolators = false,
		shooting_points = 1,
		use_parameter_homotopy = false,
		use_multipoint = false,
		synthesize_aggregate_candidates = false,
		polish_solutions = false,
		branch_completion = false,
		save_system = false,
		auto_rescale = false,
	)

	observed, _ = _bup_quiet() do
		ODEParameterEstimation._with_run_context(
			selection_recipe = ODEParameterEstimation.FixedSinglePointRecipe(
				17; interpolator_source = :agp_uq,
			),
		) do
			pipeline = analyze_parameter_estimation_problem(deepcopy(problem), options)
			_, parent_analysis, parent_report = pipeline
			parent = first(parent_analysis.returned_results)

			ident = ODEParameterEstimation.setup_identifiability(
				problem; max_num_points = 1, nooutput = true,
			)
			si_template, _ = ODEParameterEstimation.prepare_si_template_with_structural_fix(
				problem.model,
				problem.measured_quantities,
				problem.data_sample,
				ident.good_DD,
				false;
				states = ident.states,
				params = ident.params,
				infolevel = 0,
				si_probability = options.si_probability,
				placeholder_fail_categories = options.si_placeholder_fail_categories,
			)
			interpolants = ODEParameterEstimation.create_interpolants(
				problem.measured_quantities,
				problem.data_sample,
				ident.t_vector,
				ODEParameterEstimation.agp_gpr_uq,
			)
			setup = (
				states = ident.states,
				params = ident.params,
				t_vector = ident.t_vector,
				interpolants = interpolants,
				good_num_points = ident.good_num_points,
				good_deriv_level = ident.good_deriv_level,
				good_udict = ident.good_udict,
				good_varlist = ident.good_varlist,
				good_DD = ident.good_DD,
				time_index_set = [17],
				all_unidentifiable = ident.all_unidentifiable,
				numerical_advisory = ident.numerical_advisory,
				si_template = si_template,
			)
			completion = ODEParameterEstimation.complete_branches_from_anchor_report(
				problem, parent, setup, options,
			)
			child = first(completion.results)
			child_analysis = (returned_results = Any[child],)
			child_report = ODEParameterEstimation._compute_uq_result(
				problem, child_analysis, options,
			)
			child_id = child.provenance.estimator_identity
			child_influence = ODEParameterEstimation._run_ctx_uq_influence(
				child_id.candidate_id,
			)
			(; pipeline, completion, child, child_report, child_influence)
		end
	end

	_, _, parent_report = observed.pipeline
	@test observed.completion.status == :completed
	@test !isempty(observed.completion.results)
	@test observed.child.provenance.estimator_identity.estimator_kind == :branch_completed
	@test observed.child_report isa UncertaintyReport
	@test observed.child_report.target.artifact_match == :exact
	@test observed.child_report.target.identity.candidate_id ==
		observed.child.provenance.estimator_identity.candidate_id
	@test length(observed.child_report.target.lineage) == 2
	@test observed.child_report.target.lineage[2].estimator_kind ==
		:single_point_algebraic
	@test all(isfinite, observed.child_report.param_covariance)
	@test !isnothing(observed.child_influence)
	@test observed.child_influence.coordinate_labels == observed.child_report.param_labels
	@test observed.child_report.linearization_diagnostics.gradient_norm <= 1e-3
	@test !observed.child_report.linearization_diagnostics.degraded

	# `simple` is globally identifiable, so completing its sole algebraic branch
	# is a useful identity-like composition check: parent -> exact anchor jets ->
	# retained child root should preserve the physical estimate and covariance.
	@test observed.child_report.param_labels == parent_report.param_labels
	@test observed.child_report.estimate_values ≈ parent_report.estimate_values rtol = 5e-4
	@test observed.child_report.param_covariance ≈ parent_report.param_covariance rtol = 5e-2
end
