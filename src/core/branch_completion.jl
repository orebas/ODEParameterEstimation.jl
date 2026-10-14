# Experimental branch completion (`complete_branches_from_anchor_report`).
#
# The normal pipeline still does all interpolation, shooting, synthesis, and
# polish work needed to choose a strong anchor candidate. Branch completion then
# uses that anchor only to build exact observable jets, instantiates the existing
# SI template with those jets, solves the trimmed algebraic system, verifies the
# solutions against any rank-trimmed dropped equations, and returns the verified
# algebraic sibling set.

const _LAST_BRANCH_COMPLETION_DEBUG = Ref{Any}(nothing)

function _branch_completion_multiplicity(opts::EstimationOptions, setup_data)
	!isnothing(opts.algebraic_multiplicity) && return opts.algebraic_multiplicity
	if hasproperty(setup_data, :si_template) &&
	   hasproperty(setup_data.si_template, :rank_trimming_metadata) &&
	   hasproperty(setup_data.si_template.rank_trimming_metadata, :algebraic_multiplicity)
		return setup_data.si_template.rank_trimming_metadata.algebraic_multiplicity
	end
	return nothing
end

function _branch_completion_max_required_deriv(si_template)
	if hasproperty(si_template, :deriv_dict) && !isempty(si_template.deriv_dict)
		return maximum(values(si_template.deriv_dict))
	end
	return 0
end

function _branch_completion_dropped_equations(si_template)
	(!hasproperty(si_template, :rank_trimming_metadata)) && return Any[]
	metadata = si_template.rank_trimming_metadata
	full_equations = hasproperty(metadata, :full_equations) ? metadata.full_equations :
		(hasproperty(si_template, :all_equations) ? si_template.all_equations : si_template.equations)
	dropped_indices = hasproperty(metadata, :dropped_equation_indices) ? metadata.dropped_equation_indices : Int[]
	return [full_equations[i] for i in dropped_indices if 1 <= i <= length(full_equations)]
end

function _branch_completion_setup_data(setup_data)
	# Completed branches are generated at the first data time because final
	# ParameterEstimationResult rows are reported/backsolved at that time.
	return (; setup_data..., time_index_set = [1])
end

function _branch_completion_solution_data(solutions, hc_vars, trivial_dict, trimmed_vars, final_vars, good_udict)
	n = length(solutions)
	return (
		solns = solutions,
		hcvarlist = hc_vars,
		trivial_dict = trivial_dict,
		trimmed_varlist = trimmed_vars,
		forward_subst_dict = [OrderedDict{Num, Any}()],
		reverse_subst_dict = [OrderedDict{Num, Num}()],
		final_varlist = final_vars,
		good_udict = good_udict,
		solution_time_indices = fill(1, n),
		solution_interpolator_sources = fill(:branch_completion, n),
		solution_source_types = fill(:branch_completed, n),
		solution_mp_time_indices = fill(nothing, n),
		solution_mp_combo_indices = fill(nothing, n),
	)
end

function _branch_completion_verify_dropped(
	PEP::ParameterEstimationProblem,
	candidate::ParameterEstimationResult,
	dropped_instantiation,
	opts::EstimationOptions,
	max_required_deriv::Int,
)
	trivial_residuals = dropped_instantiation.trivial_residuals
	if !isempty(trivial_residuals)
		finite_residuals = filter(isfinite, trivial_residuals)
		max_trivial = isempty(finite_residuals) ? Inf : maximum(finite_residuals)
		max_trivial <= opts.branch_completion_residual_tol || return false
	end

	dropped_equations = dropped_instantiation.equations
	isempty(dropped_equations) && return true

	candidate_pep = _consensus_candidate_pep(PEP, candidate)
	t0 = Float64(first(PEP.data_sample["t"]))
	state_taylor = compute_oracle_taylor_coefficients(
		candidate_pep,
		t0,
		max_required_deriv;
		solver = PEP.solver,
		abstol = opts.abstol,
		reltol = opts.reltol,
	)
	obs_taylor = compute_observable_taylor_coefficients(candidate_pep, state_taylor, t0, max_required_deriv)
	vals = _build_true_value_vector(
		candidate_pep,
		dropped_instantiation.vars;
		state_taylor = state_taylor,
		obs_taylor = obs_taylor,
		t_eval = t0,
	)
	residual = _compute_residual(dropped_equations, dropped_instantiation.vars, vals)
	return isfinite(residual) && residual <= opts.branch_completion_residual_tol
end

function _branch_completion_process_roots(
	PEP::ParameterEstimationProblem,
	solutions,
	hc_vars,
	trivial_dict,
	trimmed_vars,
	instantiated_vars,
	setup_data,
	opts::EstimationOptions,
)
	isempty(solutions) && return ParameterEstimationResult[]
	setup_for_completion = _branch_completion_setup_data(setup_data)
	solution_data = _branch_completion_solution_data(
		solutions,
		hc_vars,
		trivial_dict,
		trimmed_vars,
		instantiated_vars,
		setup_for_completion.good_udict,
	)
	opts_no_polish = merge_options(opts; polish_solutions = false)
	return process_estimation_results(PEP, solution_data, setup_for_completion; opts = opts_no_polish)
end

function _branch_completion_report(;
	results = ParameterEstimationResult[],
	status::Symbol,
	n_hc_roots::Int = 0,
	n_processed::Int = 0,
	n_bounds_passed::Int = 0,
	n_dropped_passed::Int = 0,
	n_returned::Int = length(results),
)
	return (
		results = results,
		status = status,
		n_hc_roots = n_hc_roots,
		n_processed = n_processed,
		n_bounds_passed = n_bounds_passed,
		n_dropped_passed = n_dropped_passed,
		n_returned = n_returned,
	)
end

function _branch_completion_note!(results, status::Symbol)
	note = Symbol("branch_completion_", String(status))
	for result in results
		note_provenance!(result.provenance, note)
	end
	return results
end

function _branch_completion_anchor_values(
	PEP::ParameterEstimationProblem,
	anchor_pep::ParameterEstimationProblem,
	vars,
	t0::Float64,
	max_required_deriv::Int,
	opts::EstimationOptions,
)
	state_taylor = compute_oracle_taylor_coefficients(
		anchor_pep,
		t0,
		max_required_deriv;
		solver = PEP.solver,
		abstol = opts.abstol,
		reltol = opts.reltol,
	)
	obs_taylor = compute_observable_taylor_coefficients(anchor_pep, state_taylor, t0, max_required_deriv)
	return _build_true_value_vector(
		anchor_pep,
		vars;
		state_taylor = state_taylor,
		obs_taylor = obs_taylor,
		t_eval = t0,
	)
end

function _branch_completion_hc_debug(equations, vars)
	try
		hc_system, _hc_variables = convert_to_hc_format(equations, vars)
		raw = _hc_solve(hc_system, show_progress = false)
		real_solutions = HomotopyContinuation.solutions(raw, only_nonsingular = false, only_real = true, real_tol = 1e-9)
		all_solutions = HomotopyContinuation.solutions(raw, only_nonsingular = false)
		solution_imag_summary = [
			(
				max_abs_imag = maximum(abs.(imag.(s))),
				max_abs_real = maximum(abs.(real.(s))),
				relative_imag = maximum(abs.(imag.(s))) / max(maximum(abs.(real.(s))), 1.0),
			)
			for s in all_solutions
		]
		return (
			status = :ok,
			raw_result_type = string(typeof(raw)),
			n_real_solutions = length(real_solutions),
			n_all_solutions = length(all_solutions),
			all_solutions = all_solutions,
			solution_imag_summary = solution_imag_summary,
			result_summary = sprint(show, raw),
		)
	catch err
		_rethrow_if_interrupt(err)
		return (
			status = :error,
			error_type = string(typeof(err)),
			error_message = sprint(showerror, err),
		)
	end
end

function _record_branch_completion_debug!(;
	status::Symbol,
	PEP::ParameterEstimationProblem,
	anchor::ParameterEstimationResult,
	anchor_pep::ParameterEstimationProblem,
	instantiated,
	t0::Float64,
	max_required_deriv::Int,
	opts::EstimationOptions,
	hc_debug = nothing,
)
	anchor_values = try
		_branch_completion_anchor_values(PEP, anchor_pep, instantiated.vars, t0, max_required_deriv, opts)
	catch err
		_rethrow_if_interrupt(err)
		err
	end
	anchor_residual = anchor_values isa Exception ? NaN : _compute_residual(instantiated.equations, instantiated.vars, anchor_values)
	hc_anchor_residual = try
		if anchor_values isa Exception
			NaN
		else
			hc_system, _ = convert_to_hc_format(instantiated.equations, instantiated.vars)
			norm(HomotopyContinuation.evaluate(hc_system, ComplexF64.(anchor_values)))
		end
	catch err
		_rethrow_if_interrupt(err)
		NaN
	end
	_LAST_BRANCH_COMPLETION_DEBUG[] = (
		status = status,
		model = PEP.name,
		n_equations = length(instantiated.equations),
		n_vars = length(instantiated.vars),
		vars = instantiated.vars,
		equations = instantiated.equations,
		anchor_values = anchor_values,
		anchor_residual = anchor_residual,
		hc_anchor_residual = hc_anchor_residual,
		anchor_err = anchor.err,
		anchor_parameters = copy(anchor.parameters),
		anchor_states = copy(anchor.states),
		hc_debug = hc_debug,
	)
	return _LAST_BRANCH_COMPLETION_DEBUG[]
end

function complete_branches_from_anchor_report(
	PEP::ParameterEstimationProblem,
	anchor::ParameterEstimationResult,
	setup_data,
	opts::EstimationOptions,
)
	(hasproperty(setup_data, :si_template) && !isnothing(setup_data.si_template)) ||
		return _branch_completion_report(status = :no_si_template)
	opts.system_solver == SolverHC || return _branch_completion_report(status = :non_hc_solver)

	si_template = setup_data.si_template
	max_required_deriv = _branch_completion_max_required_deriv(si_template)
	t0 = Float64(first(PEP.data_sample["t"]))
	anchor_pep = _consensus_candidate_pep(PEP, anchor)

	perfect_interpolants = try
		build_perfect_interpolants(
			anchor_pep,
			t0,
			max_required_deriv;
			solver = PEP.solver,
			abstol = opts.abstol,
			reltol = opts.reltol,
		)
	catch err
		_rethrow_if_interrupt(err)
		opts.diagnostics && @warn "[Branch completion] failed to build exact anchor interpolants" exception = (err, catch_backtrace())
		return _branch_completion_report(status = :anchor_interpolants_failed)
	end

	instantiated = try
		instantiate_si_template_equations(
			si_template.equations,
			PEP.measured_quantities,
			PEP.data_sample,
			si_template.deriv_dict,
			si_template.template_DD;
			interpolants = perfect_interpolants,
			time_index = 1,
			prune_overdetermined = true,
		)
	catch err
		_rethrow_if_interrupt(err)
		opts.diagnostics && @warn "[Branch completion] failed to instantiate SI template" exception = (err, catch_backtrace())
		return _branch_completion_report(status = :template_instantiation_failed)
	end

	if isempty(instantiated.equations) || isempty(instantiated.vars)
		return _branch_completion_report(status = :empty_instantiated_template)
	end
	if length(instantiated.equations) != length(instantiated.vars)
		opts.diagnostics && @warn "[Branch completion] skipping non-square instantiated template" equations = length(instantiated.equations) vars = length(instantiated.vars)
		return _branch_completion_report(status = :non_square_template)
	end

	solver_options = Dict(:show_progress => opts.hc_show_progress, :real_tol => opts.hc_real_tol)
	solutions, hc_vars, trivial_dict, trimmed_vars = solve_with_hc(instantiated.equations, instantiated.vars; options = solver_options)
	if isempty(solutions)
		hc_debug = _branch_completion_hc_debug(instantiated.equations, instantiated.vars)
		debug = _record_branch_completion_debug!(
			status = :no_hc_roots,
			PEP = PEP,
			anchor = anchor,
			anchor_pep = anchor_pep,
			instantiated = instantiated,
			t0 = t0,
			max_required_deriv = max_required_deriv,
			opts = opts,
			hc_debug = hc_debug,
		)
		@warn "[Branch completion] HC found no roots for anchor-completed system" anchor_residual = debug.anchor_residual n_equations = debug.n_equations n_vars = debug.n_vars hc_debug = hc_debug
		return _branch_completion_report(status = :no_hc_roots)
	end

	completed = _branch_completion_process_roots(
		PEP,
		solutions,
		hc_vars,
		trivial_dict,
		trimmed_vars,
		instantiated.vars,
		setup_data,
		opts,
	)
	if isempty(completed)
		return _branch_completion_report(status = :no_processed_rows, n_hc_roots = length(solutions))
	end
	anchor_identity = ensure_result_estimator_identity!(anchor)
	for (idx, candidate) in enumerate(completed)
		identity = set_result_estimator_identity!(candidate;
			estimator_kind = :branch_completed,
			data_scope = :derived,
			time_indices = Int[1],
			time_values = Float64[t0],
			interpolator_source = :branch_completion,
			parent_candidate_ids = Int[anchor_identity.candidate_id],
		)
		if opts.compute_uncertainty && idx <= length(solutions)
			artifact = _run_ctx_try_uq_capture("branch-completion artifact") do
				payload = (
					pep = PEP,
					anchor = anchor,
					anchor_pep = anchor_pep,
					si_template = si_template,
					perfect_interpolants = perfect_interpolants,
					instantiated = instantiated,
					solve_vars = copy(hc_vars),
					root = Float64.(real.(solutions[idx])),
					max_required_deriv = max_required_deriv,
					t0 = t0,
				)
				BranchCompletionUQArtifact(anchor_identity.candidate_id, payload)
			end
			!isnothing(artifact) && _run_ctx_register_artifact!(identity.candidate_id, artifact)
		end
	end

	dropped_equations = _branch_completion_dropped_equations(si_template)
	dropped_instantiation = if isempty(dropped_equations)
		(equations = Any[], vars = Any[], trivial_residuals = Float64[])
	else
		try
			instantiate_si_template_equations(
				dropped_equations,
				PEP.measured_quantities,
				PEP.data_sample,
				si_template.deriv_dict,
				si_template.template_DD;
				interpolants = perfect_interpolants,
				time_index = 1,
				prune_overdetermined = false,
			)
		catch err
			_rethrow_if_interrupt(err)
			opts.diagnostics && @warn "[Branch completion] failed to instantiate dropped equations" exception = (err, catch_backtrace())
			return _branch_completion_report(
				status = :dropped_instantiation_failed,
				n_hc_roots = length(solutions),
				n_processed = length(completed),
			)
		end
	end

	n_bounds_passed = 0
	n_dropped_passed = 0
	verified = ParameterEstimationResult[]
	for candidate in completed
		n_bounds_passed += 1
		_branch_completion_verify_dropped(PEP, candidate, dropped_instantiation, opts, max_required_deriv) || continue
		n_dropped_passed += 1
		note_provenance!(candidate.provenance, :branch_completion)
		push!(verified, candidate)
	end

	if isempty(verified)
		return _branch_completion_report(
			status = :all_failed_dropped_equations,
			n_hc_roots = length(solutions),
			n_processed = length(completed),
			n_bounds_passed = n_bounds_passed,
			n_dropped_passed = n_dropped_passed,
		)
	end

	ranked = ranked_full_space_representatives(verified, opts)
	status = isempty(ranked) ? :no_ranked_rows : :completed
	return _branch_completion_report(
		results = ranked,
		status = status,
		n_hc_roots = length(solutions),
		n_processed = length(completed),
		n_bounds_passed = n_bounds_passed,
		n_dropped_passed = n_dropped_passed,
		n_returned = length(ranked),
	)
end


function maybe_replace_with_branch_completion(
	PEP::ParameterEstimationProblem,
	solved_res,
	setup_data,
	opts::EstimationOptions,
)
	opts.branch_completion || return solved_res
	isempty(solved_res) && return solved_res
	M = _branch_completion_multiplicity(opts, setup_data)
	(isnothing(M) || M <= 1) && return solved_res

	ranked = ranked_candidate_representatives(solved_res, opts)
	isempty(ranked) && return solved_res
	max_anchors = min(opts.branch_completion_max_anchors, length(ranked))

	reports = []
	for anchor in ranked[1:max_anchors]
		report = complete_branches_from_anchor_report(PEP, anchor, setup_data, opts)
		push!(reports, report)
		completed = report.results
		isempty(completed) && continue
		if opts.branch_diversity_selection
			completed = select_branch_diverse_reps(completed, M, opts)
		end
		if opts.branch_top_k > 0
			completed = completed[1:min(length(completed), min(M, opts.branch_top_k))]
		else
			completed = completed[1:min(length(completed), M)]
		end
		if !isempty(completed)
			status = length(completed) < M ? :returned_short_set : :replaced
			_branch_completion_note!(completed, status)
			if status === :returned_short_set
				@warn "[Branch completion] returning fewer completed rows than requested" requested = M returned = length(completed) report_status = report.status n_hc_roots = report.n_hc_roots n_processed = report.n_processed n_bounds_passed = report.n_bounds_passed n_dropped_passed = report.n_dropped_passed
			end
			return completed
		end
	end

	statuses = [report.status for report in reports]
	_branch_completion_note!(solved_res, :kept_original_pool)
	for status in statuses
		_branch_completion_note!(solved_res, status)
	end
	@warn "[Branch completion] keeping original candidate pool; completion produced no replacement rows" requested = M anchors_tried = length(reports) statuses = statuses original_rows = length(solved_res)
	return solved_res
end
