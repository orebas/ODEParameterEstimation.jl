using ODEParameterEstimation
using ODEParameterEstimation: InterpolatorMethod
# The uncertainty report (compute_uncertainty=true) is evaluated at the
# estimate and the interpolants' derivatives, not at the true values, so it
# runs on real data where pep.p_true and pep.ic are unknown.

using Test
using Logging
using Random
using ModelingToolkit
using OrderedCollections
using LinearAlgebra

function _ec_quiet(f)
	redirect_stdout(devnull) do
		redirect_stderr(devnull) do
			with_logger(NullLogger()) do
				return f()
			end
		end
	end
end

Random.seed!(2468)

@testset "estimate-conditioned UQ (Stream B phase 1)" begin
	pep_ec = ODEParameterEstimation.Examples.harmonic_oscillator()   # ẋ1 = -a·x2, ẋ2 = b·x1; y = (x1, x2)
	sample_opts = EstimationOptions(
		datasize = 121, time_interval = [0.0, 1.0], noise_level = 0.0, nooutput = true)
	pep_data = ODEParameterEstimation.sample_problem_data(pep_ec, sample_opts)

	@testset "full pipeline: compute_uncertainty runs estimate-conditioned" begin
		uq_opts = EstimationOptions(
			datasize = 121, time_interval = [0.0, 1.0], noise_level = 0.0,
			nooutput = true, diagnostics = false, compute_uncertainty = true,
			interpolators = InterpolatorMethod[InterpolatorGPUncertainty],
			auto_filter_interpolators = false, shooting_points = 1,
			use_parameter_homotopy = false, use_multipoint = false,
			synthesize_aggregate_candidates = false, polish_solutions = false,
			branch_completion = false, save_system = false)
		raw_ec, analysis_ec, uq_ec = _ec_quiet() do
			ODEParameterEstimation.analyze_parameter_estimation_problem(pep_data, uq_opts)
		end
		@test uq_ec isa ODEParameterEstimation.UncertaintyReport
		@test uq_ec.status in (:ok, :wide_ci)
		@test !isempty(uq_ec.param_std)
		@test all(isfinite, uq_ec.param_std)
		@test uq_ec.target.identity.candidate_id ==
			analysis_ec.returned_results[1].provenance.estimator_identity.candidate_id
		@test uq_ec.estimate_values ≈ vcat(
			collect(values(analysis_ec.returned_results[1].parameters)),
			collect(values(analysis_ec.returned_results[1].states)))

		fixed_row = 37
		fixed_value, _ = _ec_quiet() do
			ODEParameterEstimation._with_run_context(
				selection_recipe = ODEParameterEstimation.FixedSinglePointRecipe(
					fixed_row; interpolator_source = :agp_uq,
				),
			) do
				ODEParameterEstimation._analyze_parameter_estimation_problem_impl(
					deepcopy(pep_data), uq_opts,
				)
			end
		end
		_, fixed_analysis, fixed_uq = fixed_value
		@test fixed_uq isa ODEParameterEstimation.UncertaintyReport
		@test fixed_uq.target.identity.estimator_kind == :single_point_algebraic
		@test fixed_uq.target.identity.time_indices == [fixed_row]
		@test fixed_uq.target.identity.time_values == [pep_data.data_sample["t"][fixed_row]]
		@test fixed_analysis.returned_results[1].provenance.estimator_identity.time_indices ==
			[fixed_row]
	end
end
