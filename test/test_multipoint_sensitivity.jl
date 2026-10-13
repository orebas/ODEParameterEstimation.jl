using ODEParameterEstimation
using Symbolics
# Multipoint UQ v1 step 3 (2026-08-14): estimate-conditioned IFT sensitivity
# over a MultiPointTemplate, with the root-residual and conditioning gates.
#
# The load-bearing test is the finite-difference validation (design-consult
# menu item 7): perturb one data coordinate, re-solve the SAME local root
# branch by Newton, and compare the displacement against S's column. That
# checks the whole chain at once — variable ordering, the solve/data
# partition, the Jacobian, and the sign convention.

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

@testset "multipoint estimate-conditioned sensitivity (v1 step 3)" begin
	opts_mps = EstimationOptions(
		datasize = 61, time_interval = [-0.5, 0.5], noise_level = 0.0,
		nooutput = true, diagnostics = false, interpolators = [InterpolatorAAAD])
	pep_mps = ODEParameterEstimation.sample_problem_data(ODEParameterEstimation.Examples.simple(), opts_mps)

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

	# Truth-as-estimate: on noiseless data the estimate IS (numerically) the
	# algebraic root, so the residual gate must pass and FD must match S.
	combo = [15, 45]
	t_combo1 = pep_mps.data_sample["t"][combo[1]]
	est_mps = ODEParameterEstimation.ParameterEstimationResult(
		copy(pep_mps.p_true), copy(pep_mps.ic), t_combo1, 0.0, :Truth,
		opts_mps.datasize, nothing, nothing, Set{Num}(), nothing)

	S, d_labels, d_roles, x_labels, x_roles, info = _mps_quiet() do
		ODEParameterEstimation._compute_multipoint_data_sensitivity(
			mpt, combo, est_mps, pep_mps, setup_mps)
	end

	@testset "shape, labels, roles" begin
		@test info.reason == :ok
		@test size(S) == (length(mpt.solve_vars), length(mpt.data_vars))
		@test all(isfinite, S)
		# Narrowed at the boundary: the template's BigFloat coefficients must
		# not leak an extended-precision S into the Float64 report field.
		@test eltype(S) == Float64
		@test eltype(info.x_hat) == Float64 && eltype(info.d_hat) == Float64
		@test x_labels == String[string(v) for v in mpt.solve_vars]
		@test d_labels == String[string(v) for v in mpt.data_vars]
		@test length(info.x_hat) == length(mpt.solve_vars)
		@test length(info.d_hat) == length(mpt.data_vars)
		@test all(isfinite, info.x_hat) && all(isfinite, info.d_hat)
		# Roles resolve through the _ptK-stripped name (suffixed vars included)
		@test all(haskey(x_roles, l) for l in x_labels)
		@test any(r -> r == :parameter, values(x_roles))
	end

	@testset "gates: near-root, well-conditioned, not degraded" begin
		@test info.root_residual_rel < 1e-3
		@test !info.residual_degraded
		@test isfinite(info.jx_cond)
		@test !info.ift_degraded
		@test !info.degraded
	end

	@testset "finite-difference validation of S columns" begin
		combined_vars = vcat(mpt.solve_vars, mpt.data_vars)
		fn = ODEParameterEstimation._compile_system_function(mpt.stripped_equations, combined_vars)
		x0, ok0 = _mps_newton(fn, Vector{Float64}(info.x_hat), Vector{Float64}(info.d_hat))
		@test ok0   # the estimate point converges to a genuine root

		checked = 0
		for j in 1:length(info.d_hat)
			h = 1e-6 * max(abs(info.d_hat[j]), 1.0)
			d_pert = copy(Vector{Float64}(info.d_hat))
			d_pert[j] += h
			xp, okp = _mps_newton(fn, x0, d_pert)
			okp || continue
			fd = (xp .- x0) ./ h
			scale = max(maximum(abs, @view S[:, j]), 1e-8)
			@test maximum(abs, fd .- S[:, j]) / scale < 1e-3
			checked += 1
		end
		@test checked >= max(1, length(info.d_hat) - 1)
	end

	@testset "guards: bad combo length, non-finite estimate" begin
		@test_throws ArgumentError ODEParameterEstimation._compute_multipoint_data_sensitivity(
			mpt, [15], est_mps, pep_mps, setup_mps)

		nan_est = ODEParameterEstimation.ParameterEstimationResult(
			OrderedDict{Num, Float64}(k => NaN for k in keys(pep_mps.p_true)),
			OrderedDict{Num, Float64}(k => NaN for k in keys(pep_mps.ic)),
			t_combo1, 0.0, :Bad, opts_mps.datasize, nothing, nothing, Set{Num}(), nothing)
		# An unusable estimate must DEGRADE (status + empty S), never throw.
		S_bad, _, _, _, _, info_bad = _mps_quiet() do
			ODEParameterEstimation._compute_multipoint_data_sensitivity(
				mpt, combo, nan_est, pep_mps, setup_mps)
		end
		@test isempty(S_bad)
		@test info_bad.degraded
		@test info_bad.reason == :estimate_taylor_failed
	end

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
		root, root_ok = _mps_newton(fn, info.x_hat, evaluation.data_values)
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
