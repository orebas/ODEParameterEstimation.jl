using ODEParameterEstimation
using ODEParameterEstimation: UQTargetSnapshot
using Symbolics
# The uncertainty report for a multipoint estimate, built from the exact
# template, time points and interpolants that produced it.

using Test
using Logging
using LinearAlgebra
using ForwardDiff
using OrderedCollections
using ModelingToolkit

function _mps_quiet(f)
	redirect_stdout(devnull) do
		redirect_stderr(devnull) do
			with_logger(NullLogger()) do
				return f()
			end
		end
	end
end

# Newton on the square system F(x, d) = 0 for fixed d, from x0.
# The compiled template function returns BigFloat (BigFloat coefficients from
# the SIAN/rational pipeline), so every iterate is narrowed back to Float64.
function _mps_newton(fn, x0::AbstractVector{<:Real}, d::AbstractVector{<:Real};
	iters::Int = 60, tol::Float64 = 1e-14)
	x = Vector{Float64}(x0)
	dv = Vector{Float64}(d)
	for _ in 1:iters
		F = Float64.(fn(vcat(x, dv)))
		norm(F, Inf) < tol && break
		J = Float64.(ForwardDiff.jacobian(xx -> fn(vcat(xx, dv)), x))
		step = try
			J \ F
		catch
			return x, false
		end
		all(isfinite, step) || return x, false
		x -= step
	end
	return x, norm(Float64.(fn(vcat(x, dv))), Inf) < 1e-9
end

@testset "multipoint uncertainty report" begin
	opts_mps = EstimationOptions(
		datasize = 61, time_interval = [-0.5, 0.5], noise_level = 0.0,
		nooutput = true, diagnostics = false, interpolators = [InterpolatorAAA])
	pep_mps = ODEParameterEstimation.sample_problem_data(ODEParameterEstimation.Examples.harmonic_oscillator(), opts_mps)

	mpt, setup_mps = _mps_quiet() do
		ident = ODEParameterEstimation.setup_identifiability(pep_mps; max_num_points = 1, nooutput = true)
		si_template, _ = ODEParameterEstimation.prepare_si_template_with_structural_fix(
			pep_mps.model, pep_mps.measured_quantities, pep_mps.data_sample,
			ident.good_DD, false;
			states = ident.states, params = ident.params, infolevel = 0,
			placeholder_fail_categories = opts_mps.si_placeholder_fail_categories)
		interp_func = ODEParameterEstimation.get_interpolator_function(only(opts_mps.interpolators))
		interpolants = ODEParameterEstimation.create_interpolants(
			pep_mps.measured_quantities, pep_mps.data_sample, ident.t_vector, interp_func)
		setup = (good_deriv_level = ident.good_deriv_level, good_udict = ident.good_udict,
			good_varlist = ident.good_varlist, good_DD = ident.good_DD,
			interpolants = interpolants)
		template = ODEParameterEstimation.build_multipoint_template(
			pep_mps, setup, si_template; n_points = 2, diagnostics = false)
		(template, setup)
	end

	# Truth as the estimate: on noiseless data it is, numerically, the
	# algebraic root.
	combo = [15, 45]
	t_combo1 = pep_mps.data_sample["t"][combo[1]]
	est_mps = ODEParameterEstimation.ParameterEstimationResult(
		copy(pep_mps.p_true), copy(pep_mps.ic), t_combo1, 0.0, :Truth,
		opts_mps.datasize, nothing, nothing, Set{Num}(), nothing)

	# The true values of the template's unknowns at the combo times, for Newton
	# to start from.
	t_combo = Float64[pep_mps.data_sample["t"][i] for i in combo]
	max_order = maximum(mpt.solve_vars) do v
		_, clean = ODEParameterEstimation._multipoint_solve_var_point(string(v))
		parsed = ODEParameterEstimation.parse_derivative_variable_name(clean)
		isnothing(parsed) ? 0 : parsed[2]
	end
	state_taylor = [ODEParameterEstimation.compute_oracle_taylor_coefficients(pep_mps, t, max_order + 2) for t in t_combo]
	x_true = map(mpt.solve_vars) do v
		point, clean = ODEParameterEstimation._multipoint_solve_var_point(string(v))
		handle = point == 1 ? v : Symbolics.variable(Symbol(clean))
		ODEParameterEstimation._lookup_true_value(pep_mps, handle;
			state_taylor = state_taylor[point], t_eval = t_combo[point])
	end
	@test all(isfinite, x_true)

	@testset "_multipoint_solve_var_point parses point suffixes" begin
		@test ODEParameterEstimation._multipoint_solve_var_point("w_2") == (1, "w_2")
		@test ODEParameterEstimation._multipoint_solve_var_point("w_2_pt3") == (3, "w_2")
		@test ODEParameterEstimation._multipoint_solve_var_point("k1_0") == (1, "k1_0")
	end

	@testset "exact retained multipoint artifact produces estimator-matched UQ" begin
		agp_interpolants = _mps_quiet() do
			ODEParameterEstimation.create_interpolants(
				pep_mps.measured_quantities, pep_mps.data_sample,
				Float64.(pep_mps.data_sample["t"]), ODEParameterEstimation.agp_gpr_uq)
		end
		evaluation = ODEParameterEstimation.evaluate_multipoint_template(
			mpt, combo, agp_interpolants, pep_mps.data_sample)
		fn = ODEParameterEstimation._compile_system_function(
			mpt.stripped_equations, vcat(mpt.solve_vars, mpt.data_vars))
		root, root_ok = _mps_newton(fn, x_true, evaluation.data_values)
		@test root_ok

		(report, retained_influence), _ = ODEParameterEstimation._with_run_context() do
			ODEParameterEstimation._run_ctx_set_capture_uq!(true)
			identity = ODEParameterEstimation.set_result_estimator_identity!(est_mps;
				estimator_kind = :multipoint_algebraic,
				data_scope = :point_set,
				time_indices = combo,
				time_values = evaluation.t_values,
				interpolator_source = :agp_uq)
			artifact = ODEParameterEstimation.MultipointUQArtifact(
				evaluation, root, agp_interpolants, :agp_uq)
			ODEParameterEstimation._run_ctx_register_artifact!(identity.candidate_id, artifact)
			target = UQTargetSnapshot(
				identity = identity,
				lineage = ODEParameterEstimation._run_ctx_lineage(identity),
				artifact_match = :exact)
			report = ODEParameterEstimation._compute_estimator_aware_uq(
				pep_mps, est_mps, target,
				EstimationOptions(compute_uncertainty = true,
					uq_noise_source = :learned_gp_homoscedastic,
					nooutput = true))
			(
				report,
				ODEParameterEstimation._run_ctx_uq_influence(identity.candidate_id),
			)
		end

		@test report isa UncertaintyReport
		@test report.target.identity.estimator_kind == :multipoint_algebraic
		@test report.target.identity.time_indices == combo
		@test report.target.identity.time_values == evaluation.t_values
		@test report.target.artifact_match == :exact
		@test report.t_eval == evaluation.t_values[1]
		@test size(report.data_covariance) == (length(report.data_labels), length(report.data_labels))
		@test length(report.data_labels) == length(mpt.data_vars) + 2
		@test count(label -> startswith(label, "raw::"), report.data_labels) == 2
		@test size(report.param_covariance, 1) == length(report.param_labels)
		@test length(report.estimate_values) == length(report.param_labels)
		@test all(isfinite, report.estimate_values)
		@test report.linearization_diagnostics.reason == :ok
		@test !isnothing(retained_influence)
		@test retained_influence.coordinate_labels == report.param_labels
		@test size(retained_influence.influence, 1) == length(report.param_labels)
		@test size(retained_influence.observation_covariance, 1) ==
			length(retained_influence.observation_labels)
		# This stripped system happens to use only point-2 GP jets, while production
		# reports its eliminated directly observed states from point 1. Their
		# covariance must still share the same raw-observation block; treating the
		# supplemental state rows as independent would zero every entry below.
		cross_entries = Float64[]
		for (i, meta) in enumerate(mpt.data_var_meta)
			meta.kind == :observable_jet || continue
			obs_name = replace(string(pep_mps.measured_quantities[meta.obs_idx].lhs), r"\(.*\)" => "")
			j = findfirst(==("raw::$obs_name(t_index=$(combo[1]))"), report.data_labels)
			isnothing(j) || push!(cross_entries, report.data_covariance[i, j])
		end
		@test !isempty(cross_entries)
		@test maximum(abs, cross_entries) > 0

		# The retained local estimator values must reproduce the exact raw values
		# used by `process_estimation_results`, not the GP posterior mean at point 1.
		local_snapshot = report.local_coordinate_report
		@test !isnothing(local_snapshot)
		for state in pep_mps.model.original_states
			state_name = replace(string(state), "(t)" => "")
			row = findfirst(==(state_name), local_snapshot.param_labels)
			obs_idx = findfirst(mq -> isequal(Symbolics.diff2term(mq.rhs), state),
			                    pep_mps.measured_quantities)
			raw_series = ODEParameterEstimation._uq_measurement_series(
				pep_mps, pep_mps.measured_quantities[obs_idx])
			@test local_snapshot.coordinate_values[row] == raw_series[combo[1]]
		end
	end
end
