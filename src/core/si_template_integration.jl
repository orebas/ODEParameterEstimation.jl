"""
SI.jl Template Integration for ODEParameterEstimation

This module provides the glue between SI.jl's polynomial template
and ODEPE's multipoint system construction.
"""

function _numeric_residual_value(expr)
	try
		return abs(Float64(Symbolics.value(Symbolics.simplify(expr))))
	catch err
		_rethrow_if_interrupt(err)
		return NaN
	end
end

# Timing SINKS live on the scoped RunContext (run_context.jl) as of 2026-07-24.
# The two context-label STACKS below remain module-global (worst case under
# concurrent runs: mislabeled timing rows) — follow-up candidate.
const _RESOLVE_TIMING_CONTEXT_STACK = Symbol[]
const _DETAILED_TIMING_CONTEXT_STACK = Symbol[]

# Label stacks live on the scoped RunContext when one is bound (per-run
# isolation); the module stacks below remain the fallback for bare calls
# outside analyze_* (8 of the 12 push sites can run context-free).
_resolve_label_stack() = (c = _run_ctx(); c === nothing ? _RESOLVE_TIMING_CONTEXT_STACK : c.resolve_timing_labels)
_detailed_label_stack() = (c = _run_ctx(); c === nothing ? _DETAILED_TIMING_CONTEXT_STACK : c.detailed_timing_labels)

function _current_resolve_timing_context()
	stack = _resolve_label_stack()
	return isempty(stack) ? :unspecified : last(stack)
end

function _with_resolve_timing_context(f::Function, context::Symbol)
	stack = _resolve_label_stack()
	push!(stack, context)
	try
		return f()
	finally
		pop!(stack)
	end
end

function _record_resolve_timing!(record::NamedTuple)
	sink = _run_ctx_resolve_sink()
	isnothing(sink) && return nothing
	push!(sink, record)
	return nothing
end

function _timed_resolve_stage!(f::Function, stages::OrderedDict{Symbol, Float64}, stage::Symbol)
	t0 = time()
	try
		return f()
	finally
		stages[stage] = get(stages, stage, 0.0) + (time() - t0)
	end
end

function _current_detailed_timing_context()
	stack = _detailed_label_stack()
	return isempty(stack) ? :unspecified : last(stack)
end

function _with_detailed_timing_context(f::Function, context::Symbol)
	stack = _detailed_label_stack()
	push!(stack, context)
	try
		return f()
	finally
		pop!(stack)
	end
end

function _record_detailed_timing!(record::NamedTuple)
	sink = _run_ctx_detailed_sink()
	isnothing(sink) && return nothing
	push!(sink, record)
	return nothing
end

function _timed_detail_stage!(f::Function, stages::OrderedDict{Symbol, Float64}, stage::Symbol)
	t0 = time()
	try
		return f()
	finally
		stages[stage] = get(stages, stage, 0.0) + (time() - t0)
	end
end

"""
	instantiate_si_template_equations(
		template_equations, measured_quantities, data_sample, derivative_dict, template_DD;
		interpolants, time_index, prune_overdetermined=true
	)

Substitute derivative data and analytic transcendental-input values into an
already-built SI template. This is the shared instantiation path used by the
normal estimator and by branch completion, so they see identical
polynomial systems for the same template and interpolants.
"""
function instantiate_si_template_equations(
	template_equations,
	measured_quantities_in,
	data_sample,
	derivative_dict,
	template_DD;
	interpolants,
	time_index::Int,
	prune_overdetermined::Bool = true,
	substitute_trfn::Bool = true,
)
	# Create a set of all variables present in the template equations
	vars_in_template = OrderedSet{Num}()
	for eq in template_equations
		union!(vars_in_template, Num.(Symbolics.get_variables(eq)))
	end

	# Interpolate data for the required derivatives at the specified time point
	interpolated_values_dict = Dict()
	t_point = data_sample["t"][time_index]

	# SIAN's derivative dictionary describes its entire jet ring, including jets
	# absent from the retained equations. Evaluating that universe can request
	# unsupported derivatives (Fujita: order 21 although the template uses <= 7).
	# Determine the required order from the actual template support instead.
	max_required_deriv = maximum((level - 1
		for (level, lhs_vars) in enumerate(template_DD.obs_lhs)
		for lhs_var in lhs_vars if Num(lhs_var) in vars_in_template); init = 0)

	# Populate only observable jets present in these equations.
	for (obs_idx, obs_eqn) in enumerate(measured_quantities_in)
		obs_rhs = Symbolics.diff2term(obs_eqn.rhs)
		# Skip _trfn_ observables — no interpolant exists (skipped in create_interpolants),
		# and values are always set analytically by the _trfn_ substitution block below.
		if _is_trfn_observable(Symbolics.wrap(obs_rhs))
			continue
		end
		obs_interp = interpolants[obs_rhs]

		for i in 0:max_required_deriv
			# Find the corresponding lhs variable in the DD structure
			if i + 1 <= length(template_DD.obs_lhs) && obs_idx <= length(template_DD.obs_lhs[i+1])
				lhs_var = template_DD.obs_lhs[i+1][obs_idx]
				Num(lhs_var) in vars_in_template || continue
				val = try
					_estimation_derivative(obs_interp, i, t_point)
				catch err
					if err isa UnsupportedDerivativeOrderError
						context = "while instantiating $(lhs_var) for observable $(obs_rhs) at time $(t_point)"
						throw(UnsupportedDerivativeOrderError(err.requested_order, err.supported_order, err.backend, context))
					end
					rethrow()
				end
				if isnan(val)
					# The NaN propagates into the instantiated template, where the
					# downstream nonfinite guards reject this shooting point —
					# visible failure, never silent data. (One structured warn;
					# the old triple-warn + nearby-point probe was debug leftover
					# that fired 951× across PEB logs.)
					@warn "NaN derivative from interpolator; this shooting point's instantiation will be rejected downstream" observable = obs_rhs deriv_order = i time_point = t_point
				end
				interpolated_values_dict[lhs_var] = val
			else
				# This may not be an error if a high derivative of one observable is needed,
				# but not for this specific one.
			end
		end
	end

	# Also substitute _trfn_ transcendental input variables with their known analytical values.
	# These represent sin(c*t), cos(c*t), and their derivatives at the shooting point — they
	# are known functions of time, not unknowns.  After substitution, equations that only
	# contained data_vars + _trfn_ vars will become trivially satisfied (0 ≈ 0) and get
	# removed by the zero-variable filter below.
	n_trfn_substituted = 0
	if substitute_trfn
		for v in vars_in_template
			var_name = string(v)
			trfn_val = evaluate_trfn_template_variable(var_name, t_point)
			if isnothing(trfn_val)
				trfn_val = evaluate_obs_trfn_template_variable(var_name, t_point)
			end
			if !isnothing(trfn_val)
				interpolated_values_dict[v] = trfn_val
				n_trfn_substituted += 1
			end
		end
	end
	if n_trfn_substituted > 0
		@info "[TEMPLATE] Substituted $n_trfn_substituted _trfn_/_obs_trfn_ variable(s) at t=$t_point"
	end

	# Substitute interpolated values into the template equations (broadcast over the vector)
	substituted_equations = Symbolics.substitute.(template_equations, Ref(interpolated_values_dict))

	# Extract variables from each equation *after* substitution,
	# and filter out trivially-satisfied equations (0 remaining variables).
	# These arise from oscillator coupling constraints when _trfn_ observable
	# derivatives are substituted — the equations become "0 ≈ 0".
	final_vars = OrderedSet()
	kept_equations = eltype(substituted_equations)[]
	kept_source_indices = Int[]
	trivial_residuals = Float64[]
	n_trivial = 0
	for (eq_idx, eq) in enumerate(substituted_equations)
		vars_in_eq = Symbolics.get_variables(eq)
		if isempty(vars_in_eq)
			# Equation has no remaining variables — trivially satisfied after substitution
			n_trivial += 1
			push!(trivial_residuals, _numeric_residual_value(eq))
		else
			push!(kept_equations, eq)
			push!(kept_source_indices, eq_idx)
			union!(final_vars, vars_in_eq)
		end
	end

	if n_trivial > 0
		@info "[TEMPLATE] Removed $n_trivial trivially-satisfied equations after data substitution"
	end

	# Log the final variables list
	@info "[DEBUG-SI-VARS] Final variables list ($(length(final_vars))): $(collect(final_vars))"

	# Always log this critical count info
	@info "[DEBUG-EQ-VAR-COUNT] After template instantiation: $(length(kept_equations)) equations, $(length(final_vars)) variables"

	# Handle overdetermined systems: when _trfn_ substitution + data substitution
	# leaves more equations than variables, remove redundant equations.
	# An equation is "removable" if every variable in it also appears in at least one
	# other kept equation — removing it won't lose any variable.
	# Among removable equations, prefer removing from the end (highest derivative order,
	# most numerically sensitive).
	if prune_overdetermined
		while length(kept_equations) > length(final_vars)
			n_excess = length(kept_equations) - length(final_vars)
			# Build per-variable occurrence count across all kept equations
			var_eq_count = Dict{Any, Int}()
			for eq in kept_equations
				for v in Symbolics.get_variables(eq)
					var_eq_count[v] = get(var_eq_count, v, 0) + 1
				end
			end
			# Find removable equations (from the end, for numerical stability)
			removed = false
			for idx in length(kept_equations):-1:1
				eq_vars = Symbolics.get_variables(kept_equations[idx])
				# Safe to remove if every variable appears in at least 2 equations (this one + another)
				if all(v -> get(var_eq_count, v, 0) >= 2, eq_vars)
					@info "[TEMPLATE] Overdetermined ($n_excess excess): removing equation $idx (all $(length(eq_vars)) vars appear elsewhere)"
					deleteat!(kept_equations, idx)
					deleteat!(kept_source_indices, idx)
					removed = true
					break
				end
			end
			if !removed
				@warn "[TEMPLATE] Cannot safely remove any equation without losing a variable. Keeping overdetermined system."
				break
			end
			# Recompute final_vars after removal
			final_vars = OrderedSet()
			for eq in kept_equations
				union!(final_vars, Symbolics.get_variables(eq))
			end
		end
	end

	return (
		equations = kept_equations,
		vars = collect(final_vars),
		source_indices = kept_source_indices,
		trivial_residuals = trivial_residuals,
		substituted_values = interpolated_values_dict,
		t_point = t_point,
	)
end

"""
	construct_equation_system_from_si_template(
		model, measured_quantities, data_sample, deriv_level, unident_dict, varlist, DD;
		si_template, precomputed_interpolants, time_index_set,
	)

Build the polynomial system at one data point from an equation template.

The derivatives of each observable at `data_sample["t"][time_index_set[1]]`
are read from `precomputed_interpolants` and put into `si_template.equations`.
Equations left with no unknowns are dropped. If more equations than unknowns
remain, equations are dropped from the end as long as every unknown still
appears in one of the rest.

Returns the equations and their unknowns. `DD` is used only when `si_template`
has no `template_DD`; `model`, `deriv_level`, `unident_dict` and `varlist` are
not used.
"""
function construct_equation_system_from_si_template(
	model::ModelingToolkit.AbstractSystem,
	measured_quantities,
	data_sample,
	deriv_level,
	unident_dict,
	varlist,
	DD;
	si_template,
	precomputed_interpolants,
	time_index_set,
)
	template_DD = hasproperty(si_template, :template_DD) ? si_template.template_DD : DD
	instantiated = instantiate_si_template_equations(
		si_template.equations,
		measured_quantities,
		data_sample,
		si_template.deriv_dict,
		template_DD;
		interpolants = precomputed_interpolants,
		time_index = time_index_set[1],
		prune_overdetermined = true,
	)
	return instantiated.equations, instantiated.vars
end

"""
	_cascade_resolve(equations, template_vars, resolve_stages; diagnostics = false)

The fallback of `resolve_states_with_fixed_params` when homotopy continuation
finds no solution or the system is not square. Equations in one unknown are
solved for it and substituted into the rest, repeatedly. If equations remain,
homotopy continuation solves them when there are at least as many as their
unknowns, and each of its solutions is merged with the substituted values.

Returns `solutions`, `state_vars` (the order of each solution),
`missing_vars_per_solution`, `cascading_status` (`:full`, `:merged`,
`:merge_failed_partial`, `:partial` or `:failed`), `status` and `notes`.
"""
function _cascade_resolve(equations, template_vars, resolve_stages; diagnostics::Bool = false)
	solutions = Vector{Float64}[]
	state_vars = template_vars
	missing_vars_per_solution = Vector{Vector{Any}}()
	resolve_notes = Symbol[]
	resolve_status = :no_solution
	cascading_status = :attempted
	cascade_t0 = time()

	cascade_subst = Dict{Any, Any}()
	cascade_pass = 0
	reduced_eqs = copy(equations)
	changed = true

	while changed
		changed = false
		cascade_pass += 1
		new_eqs = eltype(reduced_eqs)[]
		for eq in reduced_eqs
			eq_vars = Symbolics.get_variables(eq)
			if isempty(eq_vars)
				continue
			elseif length(eq_vars) == 1
				v = only(eq_vars)
				haskey(cascade_subst, v) && continue
				try
					solved = Symbolics.symbolic_linear_solve(eq, v)
					cascade_subst[v] = solved
					changed = true
					if diagnostics
						@info "[RESOLVE] Cascade (pass $cascade_pass): solved $v = $solved"
					end
				catch err
					_rethrow_if_interrupt(err)
					push!(new_eqs, eq)
				end
			else
				push!(new_eqs, eq)
			end
		end
		reduced_eqs = changed ? Symbolics.substitute.(new_eqs, Ref(cascade_subst)) : new_eqs
	end

	if diagnostics
		@info "[RESOLVE] Cascading completed in $cascade_pass passes, solved $(length(cascade_subst)) variables"
	end

	if !isempty(cascade_subst)
		# Resolve dependency chains (A→B→C→number)
		resolved = Dict{Any, Any}(k => v for (k, v) in cascade_subst)
		for _pass in 1:10
			all_numeric = true
			for (v, expr) in resolved
				new_expr = Symbolics.substitute(expr, resolved)
				resolved[v] = new_expr
				if !isempty(Symbolics.get_variables(new_expr))
					all_numeric = false
				end
			end
			all_numeric && break
		end

		final_vals = Dict{Any, Float64}()
		failed_vals = Any[]
		for (v, expr) in resolved
			try
				final_vals[v] = Float64(Symbolics.value(expr))
			catch e
				_rethrow_if_interrupt(e)
				@warn "[RESOLVE] Failed to convert $v = $expr to Float64: $e"
				push!(failed_vals, v)
			end
		end
		if !isempty(failed_vals)
			@warn "[RESOLVE] $(length(failed_vals)) variables failed Float64 conversion and were left unresolved: $(failed_vals)"
			push!(resolve_notes, :cascade_float64_conversion_failed)
		end

		# Check if any equations remain unsolved
		remaining_vars = OrderedSet()
		remaining_eqs = eltype(reduced_eqs)[]
		for eq in reduced_eqs
			eq_vars = Symbolics.get_variables(eq)
			if !isempty(eq_vars)
				push!(remaining_eqs, eq)
				union!(remaining_vars, eq_vars)
			end
		end

		if isempty(remaining_eqs)
			@info "[RESOLVE] All variables solved by cascading (no HC.jl needed)"
			solutions = [collect(values(final_vals))]
			state_vars = collect(keys(final_vals))
			missing_vars_per_solution = [collect(failed_vals)]
			cascading_status = :full
			resolve_status = isempty(failed_vals) ? :cascade_full_solution : :cascade_full_with_unresolved
		else
			remaining_state_vars = collect(remaining_vars)
			n_rem_eqs = length(remaining_eqs)
			n_rem_vars = length(remaining_state_vars)

			hc_solutions = Vector{Float64}[]
			if n_rem_eqs >= n_rem_vars && n_rem_vars > 0
				# Square or overdetermined — HC.jl can handle this
				@info "[RESOLVE] Trying HC.jl on remaining $n_rem_eqs eqs, $n_rem_vars vars"
				hc_solutions, _, _, _ = _timed_resolve_stage!(resolve_stages, :cascade_hc_solve) do
					solve_with_hc(remaining_eqs, remaining_state_vars)
				end
			elseif n_rem_vars > 0
				# Underdetermined — HC.jl would throw FiniteException
				@warn "[RESOLVE] Remaining system is underdetermined ($n_rem_eqs eqs, $n_rem_vars vars) — skipping HC.jl"
				push!(resolve_notes, :remaining_system_underdetermined)
			end

			# Assert HC solution length matches variable count
			for sol in hc_solutions
				@assert length(sol) == n_rem_vars "[RESOLVE] HC.jl solution length $(length(sol)) != remaining variable count $n_rem_vars"
			end

			if !isempty(hc_solutions)
				# Merge cascaded + HC solutions
				all_vars = copy(remaining_state_vars)
				cascade_keys = collect(keys(cascade_subst))
				append!(all_vars, cascade_keys)
				merged_solutions = Vector{Vector{Float64}}()
				merged_missing = Vector{Vector{Any}}()

				for sol in hc_solutions
					sol_dict = Dict{Any, Any}(remaining_state_vars[j] => sol[j] for j in eachindex(remaining_state_vars))
					merged = copy(sol)
					solution_failed_vars = Any[]
					for cvar in cascade_keys
						expr = cascade_subst[cvar]
						val = Symbolics.substitute(expr, merge(sol_dict, cascade_subst))
						try
							push!(merged, Float64(Symbolics.value(val)))
						catch e
							_rethrow_if_interrupt(e)
							@warn "[RESOLVE] Failed to convert merged value for $cvar: $e"
							push!(solution_failed_vars, cvar)
						end
					end
					if isempty(solution_failed_vars)
						push!(merged_solutions, merged)
						push!(merged_missing, Any[])
					end
				end
				if isempty(merged_solutions)
					@warn "[RESOLVE] No fully mergeable HC+cascade solutions; returning partial cascade-only solution"
					state_vars = collect(keys(final_vals))
					solutions = [collect(values(final_vals))]
					unresolved = setdiff(collect(template_vars), state_vars)
					append!(unresolved, failed_vals)
					missing_vars_per_solution = [collect(OrderedSet(unresolved))]
					cascading_status = :merge_failed_partial
					resolve_status = :cascade_merge_failed_partial
					push!(resolve_notes, :cascade_merge_failed_partial)
				else
					solutions = merged_solutions
					missing_vars_per_solution = merged_missing
					state_vars = all_vars
					cascading_status = :merged
					resolve_status = :cascade_hc_merged
				end
			else
				# HC.jl failed or was skipped — return cascade-only partial solution.
				@info "[RESOLVE] Returning partial solution from cascading ($(length(final_vals)) of $(length(template_vars)) vars solved)"
				solutions = [collect(values(final_vals))]
				state_vars = collect(keys(final_vals))
				unresolved = setdiff(collect(template_vars), state_vars)
				append!(unresolved, failed_vals)
				missing_vars_per_solution = [collect(OrderedSet(unresolved))]
				cascading_status = :partial
				resolve_status = :cascade_partial_solution
			end
		end
	else
		cascading_status = :failed
		push!(resolve_notes, :cascade_found_no_substitutions)
	end
	resolve_stages[:cascading] = get(resolve_stages, :cascading, 0.0) + (time() - cascade_t0)

	return (
		solutions = solutions,
		state_vars = state_vars,
		missing_vars_per_solution = missing_vars_per_solution,
		cascading_status = cascading_status,
		status = resolve_status,
		notes = resolve_notes,
	)
end

"""
	resolve_states_with_fixed_params(
		model, measured_quantities, data_sample,
		deriv_level, unident_dict, varlist, DD,
		known_param_dict, interpolants;
		si_template, time_index, diagnostics
	)

Re-solve for unknown state initial conditions at a specific time point,
with parameter values fixed from a previous shooting-point solution.

This is the algebraic fallback for when backward ODE integration (backsolving)
fails due to stiff eigenvalues amplifying numerical errors.

**Approach**: Re-run SIAN with all parameters pre-fixed at the model level,
so SIAN generates a purpose-built polynomial system where the only unknowns
are state derivative variables. This produces a square system (N eqs = N vars),
unlike post-hoc parameter substitution which creates an overdetermined system
with numerically inconsistent redundant equations.

Returns a named tuple:
- `solutions::Vector{Vector{Float64}}` — each inner vector has values
  in the same order as `state_vars`
- `state_vars::Vector` — the Symbolics variables for the returned solution vectors
- `missing_vars_per_solution::Vector{Vector}` — explicit unresolved variables for each returned solution
- `used_cascading::Bool` — whether the cascading-substitution fallback was used
- `status::Symbol` — high-level branch outcome for the rescue attempt
- `notes::Vector{Symbol}` — branch notes describing why paths were skipped or partial
- `hc_status::Symbol` — `:success`, `:no_solutions`, `:skipped_nonsquare`, or `:not_attempted`
- `cascading_status::Symbol` — `:not_used`, `:full`, `:partial`, `:merged`, `:merge_failed_partial`, or `:failed`
"""
function resolve_states_with_fixed_params(
	model::ModelingToolkit.AbstractSystem,
	measured_quantities,
	data_sample,
	deriv_level,
	unident_dict,
	varlist,
	DD,
	known_param_dict::OrderedDict,
	interpolants;
	si_template = nothing,  # only its analysis settings are reused, not its model-specific analysis
	time_index::Int = 1,
	diagnostics::Bool = false,
	placeholder_fail_categories = Symbol[],
	compute_multiplicity::Bool = true,
	si_fix_strategy::Symbol = isnothing(si_template) || !hasproperty(si_template, :structural_analysis) ?
		_OPT_STRUCT_DEFAULTS.si_fix_strategy : si_template.structural_analysis.strategy,
	si_probability::Float64 = isnothing(si_template) || !hasproperty(si_template, :structural_analysis) ?
		_OPT_STRUCT_DEFAULTS.si_probability : si_template.structural_analysis.probability,
)
	resolve_t0 = time()
	resolve_stages = OrderedDict{Symbol, Float64}()
	resolve_template_eq_count = 0
	resolve_instantiated_eq_count = 0
	resolve_state_var_count = 0

	function record_resolve_result(result)
		_record_resolve_timing!((
			context = _current_resolve_timing_context(),
			total_seconds = time() - resolve_t0,
			stage_seconds = copy(resolve_stages),
			fixed_parameter_count = length(known_param_dict),
			time_index = time_index,
			template_equations = resolve_template_eq_count,
			instantiated_equations = resolve_instantiated_eq_count,
			state_vars = resolve_state_var_count,
			solution_count = hasproperty(result, :solutions) ? length(result.solutions) : 0,
			status = hasproperty(result, :status) ? result.status : :unknown,
			hc_status = hasproperty(result, :hc_status) ? result.hc_status : :unknown,
			cascading_status = hasproperty(result, :cascading_status) ? result.cascading_status : :unknown,
		))
		return result
	end

	try
	@info "[RESOLVE] Re-running SIAN with $(length(known_param_dict)) fixed parameters"

	empty_result(;
		status::Symbol = :no_solution,
		notes::Vector{Symbol} = Symbol[],
		hc_status::Symbol = :not_attempted,
		cascading_status::Symbol = :not_used,
	) = (
		solutions = Vector{Vector{Float64}}(),
		state_vars = Any[],
		missing_vars_per_solution = Vector{Vector{Any}}(),
		used_cascading = false,
		status = status,
		notes = notes,
		hc_status = hc_status,
		cascading_status = cascading_status,
	)

	# Step 1: Build OrderedODESystem for the original model
	ordered_model = _timed_resolve_stage!(resolve_stages, :ordered_model) do
		model_states = ModelingToolkit.unknowns(model)
		model_params = ModelingToolkit.parameters(model)
		if isa(model, ODEParameterEstimation.OrderedODESystem)
			model
		else
			ODEParameterEstimation.OrderedODESystem(model, model_states, model_params)
		end
	end

	# Step 2: Apply all parameters as pre-fixed → model with 0 parameters, values baked in
	fixed_model, fixed_mq = _timed_resolve_stage!(resolve_stages, :fixed_model) do
		apply_prefixed_params_to_model(ordered_model, measured_quantities, known_param_dict)
	end

	if diagnostics
		fixed_sys = isa(fixed_model, ODEParameterEstimation.OrderedODESystem) ? fixed_model.system : fixed_model
		n_remaining = length(ModelingToolkit.parameters(fixed_sys))
		@info "[RESOLVE] Fixed model: $(n_remaining) remaining parameters (should be 0)"
	end

	# Step 3: Re-run SIAN on the parameter-free model → template with only state unknowns
	new_template_eqs, new_deriv_dict, new_unident, new_id_funcs, new_si_variable_role_summary, new_si_template_metadata = _timed_resolve_stage!(resolve_stages, :sian_rerun) do
		get_si_equation_system(
			fixed_model, fixed_mq, data_sample;
			DD = DD,
			si_fix_strategy = si_fix_strategy,
			p = si_probability,
			infolevel = diagnostics ? 1 : 0,
			placeholder_fail_categories = placeholder_fail_categories,
			compute_multiplicity = compute_multiplicity,
		)
	end
	resolve_template_eq_count = length(new_template_eqs)
	new_template_DD = _timed_resolve_stage!(resolve_stages, :ensure_template_dd) do
		ensure_si_template_dd_support(fixed_model, fixed_mq, DD, new_deriv_dict)
	end

	if isempty(new_template_eqs)
		@warn "[RESOLVE] SIAN re-run produced no template equations"
		return record_resolve_result(empty_result(status = :empty_template, notes = [:empty_template]))
	end

	new_si_template = (
		equations = new_template_eqs,
		all_equations = hasproperty(new_si_template_metadata, :full_equations) ? new_si_template_metadata.full_equations : new_template_eqs,
		deriv_dict = new_deriv_dict,
		template_DD = new_template_DD,
		unidentifiable = new_unident,
		identifiable_funcs = new_id_funcs,
		structural_analysis = new_si_template_metadata.structural_analysis,
		si_variable_role_summary = new_si_variable_role_summary,
		rank_trimming_metadata = new_si_template_metadata,
	)

	@info "[RESOLVE] SIAN re-run produced $(length(new_template_eqs)) template eqs, $(length(new_deriv_dict)) deriv vars"

	# Step 4: Instantiate at t=0 using the new template (parameters already baked in)
	equations, template_vars = _timed_resolve_stage!(resolve_stages, :instantiate) do
		construct_equation_system_from_si_template(
			model,
			measured_quantities,
			data_sample,
			deriv_level,
			OrderedDict(),  # empty unident_dict — all params fixed, nothing unidentifiable
			varlist,
			DD;
			time_index_set = [time_index],
			precomputed_interpolants = interpolants,
			si_template = new_si_template,
		)
	end

	if isempty(equations)
		@warn "[RESOLVE] No equations after template instantiation at time_index=$time_index"
		return record_resolve_result(empty_result(status = :no_equations_after_instantiation, notes = [:no_equations_after_instantiation]))
	end

	n_eqs = length(equations)
	n_vars = length(template_vars)
	resolve_instantiated_eq_count = n_eqs
	resolve_state_var_count = n_vars
	@info "[RESOLVE] Instantiated system: $n_eqs eqs, $n_vars state vars" * (n_eqs == n_vars ? " (square)" : " (NOT square)")

	if diagnostics
		@info "[RESOLVE] Variables: $(template_vars)"
		for (i, eq) in enumerate(equations)
			eq_vars = Symbolics.get_variables(eq)
			@info "[RESOLVE] Eq$i ($(length(eq_vars)) vars): $eq"
		end
	end

	# Step 5: Try HC.jl first — square systems are HC.jl's sweet spot
	state_vars = template_vars
	solutions = Vector{Float64}[]
	missing_vars_per_solution = Vector{Vector{Any}}()
	used_cascading = false
	resolve_notes = Symbol[]
	hc_status = :not_attempted
	cascading_status = :not_used
	resolve_status = :not_attempted

	if n_eqs == n_vars && n_vars > 0
		@info "[RESOLVE] Solving square system with HC.jl"
		hc_status = :attempted
		solutions, _hc_varlist, _, _ = _timed_resolve_stage!(resolve_stages, :hc_solve) do
			solve_with_hc(equations, state_vars)
		end
		if isempty(solutions)
			@warn "[RESOLVE] HC.jl found no solutions for square system"
			hc_status = :no_solutions
			push!(resolve_notes, :hc_no_solutions)
		else
			# Assert that HC returned solutions in the expected variable order
			for sol in solutions
				@assert length(sol) == length(state_vars) "[RESOLVE] HC.jl solution length $(length(sol)) != variable count $(length(state_vars))"
			end
			@info "[RESOLVE] HC.jl found $(length(solutions)) solution(s)"
			missing_vars_per_solution = [Any[] for _ in 1:length(solutions)]
			hc_status = :success
			resolve_status = :hc_success
		end
	elseif n_eqs != n_vars
		@warn "[RESOLVE] Non-square system ($n_eqs eqs, $n_vars vars) — skipping HC.jl"
		hc_status = :skipped_nonsquare
		push!(resolve_notes, :hc_skipped_nonsquare)
	end

	# Fallback: cascading substitution if HC.jl fails or system isn't square
	if isempty(solutions)
		@info "[RESOLVE] Attempting cascading substitution fallback"
		used_cascading = true
		push!(resolve_notes, :cascading_attempted)
		cascade = _cascade_resolve(equations, template_vars, resolve_stages; diagnostics = diagnostics)
		solutions = cascade.solutions
		state_vars = cascade.state_vars
		missing_vars_per_solution = cascade.missing_vars_per_solution
		cascading_status = cascade.cascading_status
		resolve_status = cascade.status
		append!(resolve_notes, cascade.notes)
	end

	if isempty(solutions)
		@warn "[RESOLVE] No solutions found (neither HC.jl nor cascading)"
		resolve_status = :no_solution
		cascading_status == :attempted && (cascading_status = :failed)
	else
		@info "[RESOLVE] Final: $(length(solutions)) solution(s) with $(length(state_vars)) variables"
	end

	return record_resolve_result((
		solutions = solutions,
		state_vars = state_vars,
		missing_vars_per_solution = missing_vars_per_solution,
		used_cascading = used_cascading,
		status = resolve_status,
		notes = unique(resolve_notes),
		hc_status = hc_status,
		cascading_status = cascading_status,
	))
	catch err
		_record_resolve_timing!((
			context = _current_resolve_timing_context(),
			total_seconds = time() - resolve_t0,
			stage_seconds = copy(resolve_stages),
			fixed_parameter_count = length(known_param_dict),
			time_index = time_index,
			template_equations = resolve_template_eq_count,
			instantiated_equations = resolve_instantiated_eq_count,
			state_vars = resolve_state_var_count,
			solution_count = 0,
			status = :threw,
			hc_status = :unknown,
			cascading_status = :unknown,
			error = sprint(showerror, err),
		))
		rethrow()
	end
end

# Export the template-based constructor
export construct_equation_system_from_si_template
