using ODEParameterEstimation
# Stream B phase 1 (2026-08-13): estimate-conditioned sensitivity S.
# The production UQ path (compute_uncertainty=true) must evaluate the IFT
# sensitivity at (θ̂, x̂, GP interpolant jets) — NOT at ground truth — so it
# can run on real data where pep.p_true/pep.ic are unavailable. Oracle mode
# stays as the calibration/testing diagnostic.

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
	setup_ec = _ec_quiet() do
		ODEParameterEstimation.setup_parameter_estimation(
			pep_data; interpolator = ODEParameterEstimation.agp_gpr_uq, nooutput = true)
	end
	t_vec = pep_data.data_sample["t"]
	t_eval_ec = t_vec[setup_ec.time_index_set[1]]

	# Synthetic "estimate" that equals truth: parameters = p_true, states = t0 ICs
	# (ParameterEstimationResult.states are t0 initial conditions by convention;
	# the resolver forward-solves them to t_eval).
	truth_est = ODEParameterEstimation.ParameterEstimationResult(
		copy(pep_data.p_true), copy(pep_data.ic), t_eval_ec, 0.0, :Success,
		sample_opts.datasize, nothing, nothing, Set{Num}(), nothing)

	@testset "estimate Taylor ≈ oracle Taylor when estimate = truth" begin
		st_est = ODEParameterEstimation.compute_estimate_taylor_coefficients(
			pep_data, truth_est, t_eval_ec, 6)
		st_orc = ODEParameterEstimation.compute_oracle_taylor_coefficients(
			pep_data, t_eval_ec, 6)
		for s in ModelingToolkit.unknowns(pep_data.model.system)
			@test st_est[s] ≈ st_orc[s] rtol = 1e-6 atol = 1e-9
		end
	end

	sens_est = _ec_quiet() do
		ODEParameterEstimation.diagnose_sensitivity(
			pep_data; setup_data = setup_ec, t_eval = t_eval_ec, estimate_result = truth_est)
	end
	sens_orc = _ec_quiet() do
		ODEParameterEstimation.diagnose_sensitivity(
			pep_data; setup_data = setup_ec, t_eval = t_eval_ec)
	end

	@testset "mode recorded + S well-formed" begin
		@test sens_est.value_source == :estimate
		@test sens_orc.value_source == :oracle
		@test !isempty(sens_est.data_sensitivity_matrix)
		@test all(isfinite, sens_est.data_sensitivity_matrix)
		@test isfinite(sens_est.jacobian_cond)
		@test sens_est.data_sensitivity_data_labels == sens_orc.data_sensitivity_data_labels
		@test sens_est.data_sensitivity_unknown_labels == sens_orc.data_sensitivity_unknown_labels
	end

	@testset "S(estimate=truth, GP jets) ≈ S(oracle) on clean data" begin
		S_est = sens_est.data_sensitivity_matrix
		S_orc = sens_orc.data_sensitivity_matrix
		@test size(S_est) == size(S_orc)
		# Same linearization point up to GP-jet vs oracle-jet differences on
		# noiseless data — entries should agree to a few percent of the largest.
		scale = max(maximum(abs.(S_orc)), 1e-12)
		@test maximum(abs.(S_est .- S_orc)) / scale < 0.05
	end

	@testset "runs with truth stripped (real-data capability)" begin
		nan_p = OrderedDict{Num, Float64}(k => NaN for k in keys(pep_data.p_true))
		nan_ic = OrderedDict{Num, Float64}(k => NaN for k in keys(pep_data.ic))
		pep_nan = ODEParameterEstimation.ParameterEstimationProblem(
			pep_data.name, pep_data.model, pep_data.measured_quantities,
			pep_data.data_sample, pep_data.recommended_time_interval,
			pep_data.solver, nan_p, nan_ic, pep_data.unident_count)
		sens_nan = _ec_quiet() do
			ODEParameterEstimation.diagnose_sensitivity(
				pep_nan; setup_data = setup_ec, t_eval = t_eval_ec, estimate_result = truth_est)
		end
		@test sens_nan.value_source == :estimate
		@test !isempty(sens_nan.data_sensitivity_matrix)
		@test all(isfinite, sens_nan.data_sensitivity_matrix)
		# Truth values must be irrelevant to the estimate-conditioned S
		@test isapprox(sens_nan.data_sensitivity_matrix, sens_est.data_sensitivity_matrix;
			rtol = 1e-10, atol = 1e-12)
	end

	@testset "kwarg contracts" begin
		@test_throws ArgumentError ODEParameterEstimation.diagnose_sensitivity(
			pep_data; setup_data = setup_ec, t_eval = t_eval_ec, value_source = :estimate)
		@test_throws ArgumentError ODEParameterEstimation.diagnose_sensitivity(
			pep_data; setup_data = setup_ec, t_eval = t_eval_ec, value_source = :bogus,
			estimate_result = truth_est)
	end

	@testset "full pipeline: compute_uncertainty runs estimate-conditioned" begin
		uq_opts = EstimationOptions(
			datasize = 121, time_interval = [0.0, 1.0], noise_level = 0.0,
			nooutput = true, diagnostics = false, compute_uncertainty = true,
			interpolators = InterpolatorMethod[InterpolatorAGPUQ],
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
