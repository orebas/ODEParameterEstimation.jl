# REFACTORING NOTE:
# Several functions have been moved out of this file to improve organization:
# - Parameter estimation helper functions -> moved to parameter_estimation_helpers.jl

"""
	populate_derivatives(model, measured_quantities_in, max_deriv_level, unident_dict; include_cleared=true)

Populate a DerivativeData object by taking derivatives of state variable and measured quantity equations.
Derivatives retain their Differential form so substitutions preserve symbolic identity.
Rank and support calculations can skip the cleared tables, then request them later
with `ensure_cleared_derivatives!` if polynomial equations are needed.

# Arguments
- `model::ModelingToolkit.System`: The ODE system
- `measured_quantities_in`: Input measured quantities
- `max_deriv_level`: Maximum derivative level
- `unident_dict`: Dictionary of unidentifiable variables
- `include_cleared`: Also construct derivatives of the cleared base equations

# Returns
- DerivativeData object
"""
function populate_derivatives(model::ModelingToolkit.AbstractSystem, measured_quantities_in, max_deriv_level, unident_dict;
	include_cleared::Bool = true)
	(t, model_eq, model_states, model_ps) = unpack_ODE(model)
	measured_quantities = deepcopy(measured_quantities_in)

	DD = DerivativeData(
		Vector{Vector{Num}}(), Vector{Vector{Num}}(),
		Vector{Vector{Num}}(), Vector{Vector{Num}}(),
		Vector{Vector{Num}}(), Vector{Vector{Num}}(),
		Vector{Vector{Num}}(), Vector{Vector{Num}}(),
		Set{Num}(),
	)

	#First, we fully substitute values we have chosen for an unidentifiable variables.
	unident_subst!(model_eq, measured_quantities, unident_dict)

	DD.states_lhs = [[eq.lhs for eq in model_eq], expand_derivatives.(D.([eq.lhs for eq in model_eq]))]
	DD.states_rhs = [[eq.rhs for eq in model_eq], expand_derivatives.(D.([eq.rhs for eq in model_eq]))]
	DD.obs_lhs = [[eq.lhs for eq in measured_quantities], expand_derivatives.(D.([eq.lhs for eq in measured_quantities]))]
	DD.obs_rhs = [[eq.rhs for eq in measured_quantities], expand_derivatives.(D.([eq.rhs for eq in measured_quantities]))]

	extra_levels = 0
	for i in 1:(max_deriv_level-2)
		new_states_lhs = expand_derivatives.(D.(DD.states_lhs[end]))
		temp = DD.states_rhs[end]
		temp2 = D.(temp)
		temp4 = Num[]
		for j in 1:length(temp2)
			push!(temp4, expand_derivatives(temp2[j]))
		end

		# Stop if symbolic coefficients have overflowed Float64
		if any(expr -> (let s = string(expr); occursin("Inf", s) || occursin("NaN", s) end), temp4)
			@warn "populate_derivatives: coefficient overflow at derivative level $(length(DD.states_rhs) + 1), capping at $(length(DD.states_rhs))"
			break
		end

		push!(DD.states_lhs, new_states_lhs)
		push!(DD.states_rhs, temp4)
		extra_levels += 1
	end

	# Build observable derivatives (must match state derivative levels for substitution to work)
	# Note: obs derivatives must not exceed state derivatives, otherwise substitution
	# will fail with missing keys
	for i in 1:extra_levels
		push!(DD.obs_lhs, expand_derivatives.(D.(DD.obs_lhs[end])))
		push!(DD.obs_rhs, expand_derivatives.(D.(DD.obs_rhs[end])))
	end
	include_cleared && ensure_cleared_derivatives!(DD)

	# NOTE: We intentionally do NOT apply diff2term here.
	# diff2term converts Differential(t)(x(t)) to xˍt(t), but it creates NEW symbol objects
	# each time, causing substitution failures (Symbolics.substitute uses object identity).
	# By keeping everything in Differential form, structural equality ensures matching.
	return DD
end

"""
	ensure_cleared_derivatives!(DD, max_derivative_order=length(DD.obs_lhs)-1)

Construct missing cleared derivative tables through the requested order. Clear each
base equation once, then differentiate that polynomial equation, matching the eager
construction. This is not clearing each differentiated rational equation separately.

# Arguments
- `DD::DerivativeData`: Rational derivative tables, with any existing cleared prefix
- `max_derivative_order::Int`: Zero-based order within the available rational tables

# Returns
The same `DerivativeData`, retaining already constructed cleared derivatives.
"""
function ensure_cleared_derivatives!(DD::DerivativeData,
	max_derivative_order::Int = length(DD.obs_lhs) - 1)::DerivativeData
	available_order = minimum(length, (DD.states_lhs, DD.states_rhs, DD.obs_lhs, DD.obs_rhs)) - 1
	0 <= max_derivative_order <= available_order || throw(ArgumentError(
		"Cleared derivative order must be in 0:$available_order; got $max_derivative_order"))
	for (raw_lhs, raw_rhs, cleared_lhs, cleared_rhs) in (
		(DD.states_lhs, DD.states_rhs, DD.states_lhs_cleared, DD.states_rhs_cleared),
		(DD.obs_lhs, DD.obs_rhs, DD.obs_lhs_cleared, DD.obs_rhs_cleared),
	)
		if isempty(cleared_lhs)
			base = clear_denoms.([lhs ~ rhs for (lhs, rhs) in zip(first(raw_lhs), first(raw_rhs))])
			push!(cleared_lhs, [eq.lhs for eq in base])
			push!(cleared_rhs, [eq.rhs for eq in base])
		end
		while length(cleared_lhs) <= max_derivative_order
			next_lhs = expand_derivatives.(D.(last(cleared_lhs)))
			next_rhs = expand_derivatives.(D.(last(cleared_rhs)))
			push!(cleared_lhs, next_lhs)
			push!(cleared_rhs, next_rhs)
		end
	end
	return DD
end


"""
	convert_to_real_or_complex_array(values) -> Union{Array{Float64,1}, Array{ComplexF64,1}}

Converts input values to either real or complex array based on the values' properties.

# Arguments
- `values`: Input values to convert

# Returns
- `Array{Float64,1}` if all values are real (within numerical precision)
- `Array{ComplexF64,1}` if any values have non-negligible imaginary components
"""
function convert_to_real_or_complex_array(values)::Union{Array{Float64, 1}, Array{ComplexF64, 1}}
	# First convert to complex to handle mixed inputs
	newvalues = Base.convert(Array{ComplexF64, 1}, values)

	# If all values are real (within tolerance), convert to real array
	if isreal(newvalues)
		return Base.convert(Array{Float64, 1}, newvalues)
	else
		return newvalues
	end
end


"""
	create_interpolants(
		measured_quantities::Vector{ModelingToolkit.Equation},
		data_sample::OrderedDict,
		t_vector::Vector{Float64},
		interp_func::Function
	) -> Dict{Any, AbstractInterpolator}

Creates interpolant functions for measured quantities using the provided interpolation function.

# Arguments
- `measured_quantities::Vector{ModelingToolkit.Equation}`: Equations for measured quantities
- `data_sample::OrderedDict`: Sample data for measured quantities
- `t_vector::Vector{Float64}`: Time points at which measurements were taken
- `interp_func::Function`: Function to use for interpolation (e.g., aaad, aaad_gpr_pivot)

# Returns
- Dictionary mapping each quantity to its corresponding interpolant function
"""
function create_interpolants(
	measured_quantities::Vector{ModelingToolkit.Equation},
	data_sample::AbstractDict,
	t_vector::Vector{Float64},
	interp_func::Function,
)::Dict{Num, AbstractInterpolator}
	interpolants = Dict{Num, AbstractInterpolator}()

	for j in measured_quantities
		r = j.rhs
		# Skip _trfn_ auxiliary observables — their interpolated values are always
		# overwritten by analytical evaluation in the SI template, so fitting an
		# interpolant is pure waste.
		if _is_trfn_observable(Symbolics.wrap(r))
			continue
		end
		# Look up data in data_sample - use rhs if available, otherwise use lhs
		key = haskey(data_sample, r) ? r : Symbolics.wrap(j.lhs)
		y_vector = data_sample[key]

		# Create interpolant and store in dictionary
		series_times = data_sample isa ObservationData ? observation_times(data_sample, key) : t_vector
		interpolants[r] = interp_func(series_times, y_vector)
	end

	return interpolants
end


function determine_optimal_points_count(model, measured_quantities, max_num_points, t_vector, nooutput)
	(t, eqns, states, params) = unpack_ODE(model)
	time_interval = extrema(t_vector)
	large_num_points = min(length(params), max_num_points, length(t_vector))
	good_num_points = large_num_points
	@debug "Large num points: $large_num_points"
	@debug "Good num points: $good_num_points"

	time_index_set, solns, good_udict, forward_subst_dict, trivial_dict, final_varlist, trimmed_varlist =
		[[] for _ in 1:7]
	good_DD = nothing

	@debug "Starting parameter estimation..."
	if good_num_points > 1
		(target_deriv_level, target_udict, target_varlist, target_DD) = multipoint_local_identifiability_analysis(model, measured_quantities, large_num_points)

		while (good_num_points > 1)
			good_num_points = good_num_points - 1
			(test_deriv_level, test_udict, test_varlist, test_DD) = multipoint_local_identifiability_analysis(model, measured_quantities, good_num_points)
			if !(test_deriv_level == target_deriv_level)
				good_num_points = good_num_points + 1
				break
			end
		end
	end

	(good_deriv_level, good_udict, good_varlist, good_DD) = multipoint_local_identifiability_analysis(model, measured_quantities, good_num_points)

	@debug "Final analysis with $(good_num_points) points"
	@debug "Final unidentifiable dict: $(good_udict)"
	@debug "Final varlist: $(good_varlist)"

	return good_num_points, good_deriv_level, good_udict, good_varlist, good_DD

end


function _state_base_name_set(states)
	state_base_names = Set{String}()
	if !isnothing(states)
		for s in states
			# chopsuffix, not a byte range: names such as θ(t) are not ASCII.
			push!(state_base_names, chopsuffix(string(s), "(t)"))
		end
	end
	return state_base_names
end

function _model_symbol_from_name(name::AbstractString, states = nothing, params = nothing)
	if !isnothing(params)
		for p in params
			if string(p) == name
				return p
			end
		end
	end
	if !isnothing(states)
		for s in states
			if chopsuffix(string(s), "(t)") == name
				return s
			end
		end
	end
	return Symbolics.variable(Symbol(name))
end

function _symbolic_identifiable_functions(identifiable_funcs)
	nemo_to_mtk_map = Dict()
	return [nemo_to_symbolics(f, nemo_to_mtk_map) for f in identifiable_funcs]
end

function _candidate_fix_variables(unidentifiable_params, already_fixed::Set, states)
	state_base_names = _state_base_name_set(states)
	param_like = Any[]
	state_like = Any[]
	already_fixed_names = Set(string.(collect(already_fixed)))
	for p in unidentifiable_params
		pstr = string(p)
		pstr in already_fixed_names && continue
		if pstr in state_base_names
			push!(state_like, p)
		else
			push!(param_like, p)
		end
	end
	return isempty(param_like) ? state_like : param_like
end

function _rank_based_fix_candidates(candidate_vars, symbolic_identifiable_funcs, diagnostics)
	isempty(candidate_vars) && return Any[]
	param_syms = [Symbolics.variable(Symbol(string(p))) for p in candidate_vars]
	candidate_names = Set(string.(candidate_vars))
	funcs_filtered = [
		f for f in symbolic_identifiable_funcs
		if any(string(v) in candidate_names for v in Symbolics.get_variables(f))
	]

	if isempty(funcs_filtered)
		diagnostics && println("[STRUCTURAL-FIX] No identifiable functions for candidates; selecting first candidate")
		return Any[candidate_vars[1]]
	end

	J_sym = Symbolics.jacobian(funcs_filtered, param_syms)
	all_vars = OrderedSet{Any}()
	for f in funcs_filtered
		union!(all_vars, Symbolics.get_variables(f))
	end
	# Seeded (postcampaign review P1): this Jacobian's pivoted-QR column order
	# decides WHICH unidentifiable parameter gets fixed to 1.0 — a structural
	# decision that must not vary run-to-run with the global RNG.
	fix_probe_rng = MersenneTwister(0x9f4d0329)
	val_dict = Dict{Num, Float64}()
	for v in all_vars
		val_dict[v] = 0.5 + rand(fix_probe_rng)
	end
	J_num = Array{Float64}(undef, length(funcs_filtered), length(param_syms))
	for i in 1:size(J_sym, 1)
		for j in 1:size(J_sym, 2)
			entry = Symbolics.substitute(J_sym[i, j], val_dict)
			J_num[i, j] = Float64(Symbolics.value(entry))
		end
	end
	F = LinearAlgebra.qr(J_num, LinearAlgebra.ColumnNorm())
	R = Array(F.R)
	tol = 1e-10 * maximum(size(J_num)) * (isempty(R) ? 0.0 : maximum(abs, diag(R)))
	rnk = sum(abs.(diag(R)) .> tol)
	num_to_fix = max(length(candidate_vars) - rnk, 0)
	if num_to_fix == 0
		diagnostics && println("[STRUCTURAL-FIX] Jacobian rank indicates no remaining structural fix variables are required")
		return Any[]
	end

	cols_ordered = collect(F.p)
	pivot_cols = rnk > 0 ? Set(cols_ordered[1:rnk]) : Set{Int}()
	dof_cols = [j for j in 1:length(param_syms) if !(j in pivot_cols)]
	selected = Any[]
	for j in dof_cols
		push!(selected, candidate_vars[j])
		length(selected) >= num_to_fix && break
	end
	while length(selected) < num_to_fix && length(selected) < length(candidate_vars)
		for candidate in candidate_vars
			candidate in selected && continue
			push!(selected, candidate)
			length(selected) >= num_to_fix && break
		end
	end
	if diagnostics
		println("[STRUCTURAL-FIX] Jacobian rank: $rnk, DOF cols: $dof_cols")
		println("[STRUCTURAL-FIX] Selected structural fix candidates: $selected")
	end
	return selected
end

function derive_structural_fix_set(si_template, diagnostics; states = nothing, params = nothing)
	unidentifiable_params = si_template.unidentifiable
	isempty(unidentifiable_params) && return (
		pre_fixed = OrderedDict{Num, Float64}(),
		reported = OrderedDict{Num, Float64}(),
		structural_unidentifiable = Set{Num}(),
	)

	analysis = hasproperty(si_template, :structural_analysis) ? si_template.structural_analysis : nothing
	selected = if !isnothing(analysis) && analysis.strategy == :local_basis
		# SI pivots the state columns before the parameter columns, preferring
		# parameter fixes where possible. Nonpivots supply exactly the continuous
		# freedoms, including any independent initial-state freedoms.
		analysis.coordinate_basis
	else
		candidate_vars = _candidate_fix_variables(unidentifiable_params, Set(), states)
		symbolic_identifiable_funcs = _symbolic_identifiable_functions(si_template.identifiable_funcs)
		_rank_based_fix_candidates(candidate_vars, symbolic_identifiable_funcs, diagnostics)
	end

	pre_fixed = OrderedDict{Num, Float64}()
	reported = OrderedDict{Num, Float64}()
	for candidate in selected
		candidate_name = string(candidate)
		pre_fixed[Symbolics.variable(Symbol(candidate_name))] = 1.0
		reported[_model_symbol_from_name(candidate_name, states, params)] = 1.0
	end

	structural_unidentifiable = Set{Num}()
	for candidate in unidentifiable_params
		push!(structural_unidentifiable, _model_symbol_from_name(string(candidate), states, params))
	end

	if diagnostics
		println("[STRUCTURAL-FIX] Structural unidentifiable set from SI: $structural_unidentifiable")
		println("[STRUCTURAL-FIX] Representative structural fix set: $reported")
		println("[STRUCTURAL-FIX] Practical/numerical identifiability status: not_assessed")
	end

	return (
		pre_fixed = pre_fixed,
		reported = reported,
		structural_unidentifiable = structural_unidentifiable,
	)
end

function analyze_si_template_structure(si_template)
	template_equations = si_template.equations
	template_DD = hasproperty(si_template, :template_DD) ? si_template.template_DD : nothing
	data_vars = isnothing(template_DD) ? Any[] : extract_data_variables_from_DD(template_DD)
	data_vars_set = Set(data_vars)
	vars_in_system = OrderedSet{Any}()
	for eq in template_equations
		union!(vars_in_system, Symbolics.get_variables(eq))
	end
	unknown_vars = OrderedSet{Any}()
	for v in vars_in_system
		if !(v in data_vars_set)
			push!(unknown_vars, v)
		end
	end
	trfn_var_info, real_solve_vars, trfn_only_eq_indices = classify_trfn_in_template(
		collect(unknown_vars), data_vars_set, template_equations
	)
	n_equations = length(template_equations)
	n_variables = length(unknown_vars)
	n_data_vars = length(data_vars)
	n_trfn_vars = length(trfn_var_info)
	n_trfn_only_eqs = length(trfn_only_eq_indices)
	n_effective_eqs = n_equations - n_trfn_only_eqs
	n_effective_vars = n_variables - n_trfn_vars
	status = if n_effective_eqs == n_effective_vars
		:determined
	elseif n_effective_eqs > n_effective_vars && n_effective_eqs <= n_effective_vars + 2
		:slightly_overdetermined
	elseif n_effective_eqs < n_effective_vars
		:residual_underdetermined
	else
		:severely_overdetermined
	end
	rank_trim_meta = hasproperty(si_template, :rank_trimming_metadata) ? si_template.rank_trimming_metadata : (
		selected_equation_indices = Int[],
		dropped_equation_indices = Int[],
		original_equation_count = n_equations,
	)
	return (
		status = status,
		n_equations = n_equations,
		n_variables = n_variables,
		n_data_vars = n_data_vars,
		n_effective_eqs = n_effective_eqs,
		n_effective_vars = n_effective_vars,
		n_trfn_vars = n_trfn_vars,
		n_trfn_only_eqs = n_trfn_only_eqs,
		trfn_var_info = trfn_var_info,
		real_solve_vars = real_solve_vars,
		trfn_only_eq_indices = trfn_only_eq_indices,
		dropped_equation_indices = rank_trim_meta.dropped_equation_indices,
		selected_equation_indices = rank_trim_meta.selected_equation_indices,
		original_equation_count = rank_trim_meta.original_equation_count,
	)
end

struct SITemplateShapeError <: Exception
	status::Symbol
	n_equations::Int
	n_variables::Int
	n_data_vars::Int
	n_effective_eqs::Int
	n_effective_vars::Int
	structural_fix_set::OrderedDict{Num, Float64}
	dropped_equation_indices::Vector{Int}
	suspicious_si_roles::Dict{Symbol, Int}
end

function Base.showerror(io::IO, err::SITemplateShapeError)
	print(
		io,
		"SI template remained $(err.status) after structural fixing; ",
		"effective system has $(err.n_effective_eqs) equations and $(err.n_effective_vars) unknowns ",
		"(raw: $(err.n_equations) equations, $(err.n_variables) unknowns, $(err.n_data_vars) data variables).",
	)
	!isempty(err.structural_fix_set) && print(io, " Structural fix set: $(err.structural_fix_set).")
	!isempty(err.dropped_equation_indices) && print(io, " Rank-trimmed equations dropped: $(err.dropped_equation_indices).")
	!isempty(err.suspicious_si_roles) && print(io, " Suspicious SI roles: $(err.suspicious_si_roles).")
end

function throw_on_nonsquare_si_template(structure, structural_fix_set, si_variable_role_summary)
	structure.status == :determined && return nothing
	throw(SITemplateShapeError(
		structure.status,
		structure.n_equations,
		structure.n_variables,
		structure.n_data_vars,
		structure.n_effective_eqs,
		structure.n_effective_vars,
		deepcopy(structural_fix_set),
		copy(structure.dropped_equation_indices),
		Dict{Symbol, Int}(si_variable_role_summary.suspicious_categories),
	))
end

function build_si_template_for_fixed_params(
	ordered_model,
	measured_quantities,
	data_sample,
	base_DD;
	infolevel = 0,
	si_probability = 0.99,
	si_fix_strategy::Symbol = _OPT_STRUCT_DEFAULTS.si_fix_strategy,
	structural_analysis::Union{Nothing, SIStructuralAnalysis} = nothing,
	pre_fixed_params = OrderedDict{Num, Float64}(),
	compute_multiplicity = true,
)
	template_equations, derivative_dict, unidentifiable, identifiable_funcs, si_variable_role_summary, si_template_metadata = get_si_equation_system(
		ordered_model,
		measured_quantities,
		data_sample;
		DD = base_DD,
		infolevel = infolevel,
		p = si_probability,
		si_fix_strategy = si_fix_strategy,
		structural_analysis = structural_analysis,
		pre_fixed_params = pre_fixed_params,
		compute_multiplicity = compute_multiplicity,
	)
	template_DD = ensure_si_template_dd_support(ordered_model, measured_quantities, base_DD, derivative_dict)
	return (
		equations = template_equations,
		all_equations = hasproperty(si_template_metadata, :full_equations) ? si_template_metadata.full_equations : template_equations,
		deriv_dict = derivative_dict,
		template_DD = template_DD,
		unidentifiable = unidentifiable,
		identifiable_funcs = identifiable_funcs,
		structural_analysis = si_template_metadata.structural_analysis,
		si_variable_role_summary = si_variable_role_summary,
		rank_trimming_metadata = si_template_metadata,
	)
end


function prepare_si_template_with_structural_fix(
	ordered_model,
	measured_quantities,
	data_sample,
	base_DD,
	diagnostics;
	states = nothing,
	params = nothing,
	infolevel = diagnostics ? 1 : 0,
	si_probability = 0.99,
	si_fix_strategy::Symbol = _OPT_STRUCT_DEFAULTS.si_fix_strategy,
	compute_multiplicity::Bool = true,
)
	initial_template = build_si_template_for_fixed_params(
		ordered_model,
		measured_quantities,
		data_sample,
		base_DD;
		infolevel = infolevel,
		si_probability = si_probability,
		si_fix_strategy = si_fix_strategy,
		pre_fixed_params = OrderedDict{Num, Float64}(),
		# Detection pass: discovers what is unidentifiable on the raw model, so a
		# partially identifiable system is positive-dimensional here by design.
		# M is computed only on the final fixed template below.
		compute_multiplicity = false,
	)

	structural_fix_info = derive_structural_fix_set(initial_template, diagnostics; states = states, params = params)
	structural_fix_set = structural_fix_info.pre_fixed
	structural_fix_report = structural_fix_info.reported
	structural_unidentifiable = structural_fix_info.structural_unidentifiable
	final_template = build_si_template_for_fixed_params(
		ordered_model,
		measured_quantities,
		data_sample,
		base_DD;
		infolevel = infolevel,
		si_probability = si_probability,
		si_fix_strategy = si_fix_strategy,
		structural_analysis = initial_template.structural_analysis,
		pre_fixed_params = OrderedDict{Num, Float64}(k => v for (k, v) in structural_fix_set),
		compute_multiplicity = compute_multiplicity,
	)
	structure = analyze_si_template_structure(final_template)

	if diagnostics
		@info "[TEMPLATE-STRUCTURE] System status: $(structure.n_equations) equations, $(structure.n_variables) unknowns (+ $(structure.n_data_vars) data variables)"
		if structure.n_trfn_vars > 0
			@info "[TEMPLATE-STRUCTURE] _trfn_ vars: $(structure.n_trfn_vars) known inputs, $(structure.n_trfn_only_eqs) trivial equations"
			@info "[TEMPLATE-STRUCTURE] Effective system: $(structure.n_effective_eqs) equations, $(structure.n_effective_vars) real unknowns"
		end
	end

	final_template = (
		equations = final_template.equations,
		all_equations = hasproperty(final_template, :all_equations) ? final_template.all_equations : final_template.equations,
		deriv_dict = final_template.deriv_dict,
		template_DD = final_template.template_DD,
		unidentifiable = final_template.unidentifiable,
		identifiable_funcs = final_template.identifiable_funcs,
		structural_analysis = final_template.structural_analysis,
		si_variable_role_summary = final_template.si_variable_role_summary,
		rank_trimming_metadata = final_template.rank_trimming_metadata,
		structural_unidentifiable = structural_unidentifiable,
		structural_fix_set = structural_fix_report,
		template_status = structure.status,
		practical_identifiability_status = :not_assessed,
	)
	throw_on_nonsquare_si_template(structure, structural_fix_report, final_template.si_variable_role_summary)
	return final_template, structure
end


"""
	compute_default_bounds(PEP::ParameterEstimationProblem; multiplier = DEFAULT_BOUND_MULTIPLIER)

Compute default optimization bounds based on the scale of observed data.
Returns `(lb, ub)` vectors of length `n_states + n_params`, scaled by
`multiplier * max(1, max_data_value)`.

`multiplier` decouples the two consumers: blown-backsolve DETECTION keeps the wide
`DEFAULT_BOUND_MULTIPLIER` (1e9), while the no-user-bounds POLISH search box passes the
narrower `DEFAULT_POLISH_BOUND_MULTIPLIER` (1e6) so the optimizer doesn't roam a
1e9-wide box. The box is symmetric, so `_choose_polish_transforms(:auto)` selects
`:linear` for it (sign unknown, negative not ≥1 OOM smaller than positive).
"""
function compute_default_bounds(PEP::ParameterEstimationProblem; multiplier::Real = DEFAULT_BOUND_MULTIPLIER)
	data_vals = Float64[]
	for (k, v) in PEP.data_sample
		k == "t" && continue
		append!(data_vals, abs.(Float64.(v)))
	end
	data_scale = isempty(data_vals) ? 1.0 : max(1.0, maximum(data_vals))
	n_states = length(ModelingToolkit.unknowns(PEP.model.system))
	n_params = length(ModelingToolkit.parameters(PEP.model.system))
	p_size = n_states + n_params
	bound = multiplier * data_scale
	lb = fill(-bound, p_size)
	ub = fill(bound, p_size)
	return lb, ub
end

function process_raw_solution(raw_sol, model::OrderedODESystem, data_sample, ode_solver; abstol = 1e-12, reltol = 1e-12)
	process_t0 = time()
	process_stages = OrderedDict{Symbol, Float64}()
	# Create ordered collections for states and parameters
	ordered_states = OrderedDict()
	ordered_params = OrderedDict()

	# Get current ordering from ModelingToolkit
	current_states = ModelingToolkit.unknowns(model.system)
	current_params = ModelingToolkit.parameters(model.system)


	# Reorder states according to original ordering
	for (i, state) in enumerate(model.original_states)
		idx = findfirst(s -> isequal(s, state), current_states)
		if isnothing(idx)
			@warn "State $state not found in current states, using original index $i"
			idx = i
		end
		ordered_states[state] = raw_sol[idx]
	end

	# Reorder parameters according to original ordering. raw_sol's parameter block
	# is in CURRENT (MTK) order — the same convention as the states block above and
	# as the producers (unpack_ODE-based assembly). Resolve each original parameter
	# to its CURRENT index, exactly like the states loop.
	param_offset = length(current_states)
	for (i, param) in enumerate(model.original_parameters)
		idx = findfirst(p -> isequal(p, param), current_params)
		if isnothing(idx)
			@warn "Parameter $param not found in current parameters, using original index $i"
			idx = i
		end
		ordered_params[param] = raw_sol[param_offset+idx]
	end

	ic = collect(values(ordered_states))
	ps = collect(values(ordered_params))


	# Solve ODE problem
	tspan = (_initial_time(data_sample), last(_observation_union(data_sample)))

	prob = _timed_detail_stage!(process_stages, :ode_problem_build) do
		ODEProblem(complete(model.system), merge(ordered_states, ordered_params), tspan)
	end
	# Catch exceptions thrown from inside the integrator (e.g. SingularException from
	# implicit-Newton step on rank-deficient candidate Jacobians at high noise on
	# stiff systems). Existing retcode-based blowup handling below treats `nothing`
	# the same as `retcode != Success`, rejecting the candidate via err = 1e+15.
	ode_threw = false
	ode_solution = _timed_detail_stage!(process_stages, :ode_solve) do
		try
			ModelingToolkit.solve(prob, ode_solver, saveat = _observation_union(data_sample), abstol = abstol, reltol = reltol)
		catch e
			_rethrow_if_interrupt(e)
			ode_threw = true
			@warn "ODE integration of HC candidate threw $(typeof(e)); rejecting candidate" exception = e
			nothing
		end
	end

	# Calculate error
	err = 0
	_timed_detail_stage!(process_stages, :error_eval) do
		if ode_solution !== nothing && ode_solution.retcode == ReturnCode.Success
			err = 0.0
			for (key, sample) in data_sample
				if isequal(key, "t")
					continue
				end
				# SSE — the SINGLE canonical fit-error (see `_trajectory_sse`): sum of
				# squared residuals over every observable and time point. The same unit
				# the optimizer minimizes and every other candidate's `.err` is reported
				# in, so algebraic / polished / synthesized candidates rank in one unit.
				# (Was `norm(resid)/N` averaged over observables — a different unit that
				# made small-residual candidates incomparable across sources.)
				err += sum(abs2, (ode_solution(observation_times(data_sample, key))[key]) .- sample)
			end
		else
			err = 1e+15
		end
	end


	# (The former post-err second parameter pass was folded into the single pass
	# above — postcampaign review P0#1: two passes with different ordering
	# conventions meant the ODE solve/err and the returned dict could describe
	# different parameter vectors whenever MTK order != declaration order.)

	_record_detailed_timing!((
		category = :process_raw_solution,
		context = _current_detailed_timing_context(),
		total_seconds = time() - process_t0,
		stage_seconds = copy(process_stages),
		state_count = length(current_states),
		parameter_count = length(current_params),
		data_point_count = length(data_sample["t"]),
		data_series_count = max(length(data_sample) - 1, 0),
		ode_success = ode_solution !== nothing && ode_solution.retcode == ReturnCode.Success,
		ode_threw = ode_threw,
		err = Float64(err),
	))

	return ordered_states, ordered_params, ode_solution, err
end

"""
	equilibrate_jacobian(jac::Matrix{Float64}) -> Matrix{Float64}

Apply full matrix equilibration (row and column scaling) to improve numerical stability.

When Jacobian matrices have values spanning many orders of magnitude (e.g., 10^0 to 10^36),
SVD-based nullspace computation can fail due to numerical precision issues. This function
normalizes the matrix so all rows and columns have similar magnitudes.

# Arguments
- `jac::Matrix{Float64}`: The input Jacobian matrix

# Returns
- Equilibrated copy of the Jacobian matrix with normalized rows and columns
"""
function equilibrate_jacobian(jac::Matrix{Float64})
	scaled_jac = copy(jac)

	# Step 1: Row scaling - normalize each row by its L2 norm
	row_norms = [norm(scaled_jac[i, :]) for i in 1:size(scaled_jac, 1)]
	for i in 1:size(scaled_jac, 1)
		if row_norms[i] > 1e-10
			scaled_jac[i, :] ./= row_norms[i]
		end
	end

	# Step 2: Column scaling - normalize each column by its L2 norm
	col_norms = [norm(scaled_jac[:, j]) for j in 1:size(scaled_jac, 2)]
	for j in 1:size(scaled_jac, 2)
		if col_norms[j] > 1e-10
			scaled_jac[:, j] ./= col_norms[j]
		end
	end

	return scaled_jac
end


"""
	multipoint_numerical_jacobian(
		model::ModelingToolkit.System,
		measured_quantities::Vector{ModelingToolkit.Equation},
		max_deriv_level::Int,
		max_num_points::Int,
		unident_dict::Dict,
		varlist::Vector{Num},
		param_dict,
		ic_dict_vector,
		values_dict,
		DD::Union{DerivativeData,Symbol} = :nothing
	) -> Tuple{Matrix{Float64}, DerivativeData}

Computes the numerical Jacobian at multiple points.
The multiple points have different values for states, but the same parameters.

# Arguments
- `model::ModelingToolkit.System`: The ODE system model
- `measured_quantities::Vector{ModelingToolkit.Equation}`: Input measured quantities
- `max_deriv_level::Int`: Maximum derivative level to compute
- `max_num_points::Int`: Maximum number of points to use
- `unident_dict::Dict`: Dictionary of unidentifiable variables
- `varlist::Vector{Num}`: List of variables
- `param_dict::OrderedDict{Num,Float64}`: Dictionary of parameters
- `ic_dict_vector::Vector{OrderedDict{Num,Float64}}`: Vector of initial condition dictionaries
- `values_dict::OrderedDict{Num,Float64}`: Dictionary of values; used for dictionary structure
- `DD::Union{DerivativeData,Symbol}`: DerivativeData object (optional, default: :nothing)

# Returns
- Tuple containing the Jacobian matrix and DerivativeData object
"""
function multipoint_numerical_jacobian(
	model::ModelingToolkit.AbstractSystem,
	measured_quantities::Vector{Equation},
	max_deriv_level::Int,
	max_num_points::Int,
	unident_dict::Dict,
	varlist::Vector{Num},
	param_dict,
	ic_dict_vector,
	values_dict,
	DD::Union{DerivativeData, Symbol} = :nothing,
)::Tuple{Matrix{Float64}, DerivativeData}
	(t, model_eq, model_states, model_ps) = unpack_ODE(model)
	measured_quantities_local = deepcopy(measured_quantities)

	states_count = length(model_states)
	ps_count = length(model_ps)
	D = Differential(t)
	subst_dict = Dict()

	num_real_params = length(keys(param_dict))
	num_real_states = length(keys(ic_dict_vector[1]))

	if (DD == :nothing)
		DD = populate_derivatives(model, measured_quantities_local, max_deriv_level, unident_dict;
			include_cleared = false)
	end

	function f(param_and_ic_values_vec)
		T = eltype(param_and_ic_values_vec)  # Float64 or ForwardDiff.Dual
		obs_deriv_vals = T[]
		for k in eachindex(ic_dict_vector)
			evaluated_subst_dict = OrderedDict{Num, Any}(values_dict)
			thekeys = collect(keys(evaluated_subst_dict))
			for i in 1:num_real_params
				evaluated_subst_dict[thekeys[i]] = param_and_ic_values_vec[i]
			end
			for i in 1:num_real_states
				evaluated_subst_dict[thekeys[i+num_real_params]] =
					param_and_ic_values_vec[(k-1)*num_real_states+num_real_params+i]
			end

		for i in eachindex(DD.states_rhs)
				for j in eachindex(DD.states_rhs[i])
					substituted_val = Symbolics.substitute(DD.states_rhs[i][j], evaluated_subst_dict)
					# If the substituted value is a constant, unwrap it from the symbolic type.
					if !Symbolics.iscall(substituted_val)
						evaluated_subst_dict[DD.states_lhs[i][j]] = Symbolics.value(substituted_val)
					else
						evaluated_subst_dict[DD.states_lhs[i][j]] = substituted_val
					end
				end
			end
			for i in eachindex(DD.obs_rhs), j in eachindex(DD.obs_rhs[i])
				substituted = Symbolics.substitute(DD.obs_rhs[i][j], evaluated_subst_dict)
				val = Symbolics.value(substituted)
				# Convert to the appropriate numeric type for ForwardDiff compatibility
				if val isa Number
					push!(obs_deriv_vals, T(val))
				else
					unsupported_err = classify_unsupported_substitution_error(DD.obs_rhs[i][j], substituted)
					!isnothing(unsupported_err) && throw(unsupported_err)
					# If still symbolic, substitution was incomplete - this is a bug
					# Print debugging info to help diagnose
					dict_keys = collect(keys(evaluated_subst_dict))
					error("Incomplete symbolic substitution in multipoint_numerical_jacobian:\n" *
						  "  Expression: $(DD.obs_rhs[i][j])\n" *
						  "  After substitution: $(substituted)\n" *
						  "  Result type: $(typeof(val))\n" *
						  "  obs_rhs index: i=$i, j=$j\n" *
						  "  Number of derivative levels in states_lhs: $(length(DD.states_lhs))\n" *
						  "  Number of derivative levels in obs_rhs: $(length(DD.obs_rhs))\n" *
						  "  Dict keys (first 20): $(dict_keys[1:min(20, length(dict_keys))])")
				end
			end
		end
		return obs_deriv_vals
	end

	full_values = collect(values(param_dict))
	for k in eachindex(ic_dict_vector)
		append!(full_values, collect(values(ic_dict_vector[k])))
	end


	matrix = ForwardDiff.jacobian(f, full_values)
	matrix_float = map(x -> Float64(Symbolics.value(x)), matrix)
	return matrix_float, DD
end

"""
	multipoint_deriv_level_view(evaluated_jac, deriv_level, num_obs, max_num_points, deriv_count, num_points_used)

Create a view of the Jacobian matrix for specific derivative levels and points.

# Arguments
- `evaluated_jac`: Evaluated Jacobian matrix
- `deriv_level`: Dictionary of derivative levels for each observable
- `num_obs`: Number of observables
- `max_num_points`: Maximum number of points
- `deriv_count`: Total number of derivatives
- `num_points_used`: Number of points actually used

# Returns
- View of the Jacobian matrix
"""
function multipoint_deriv_level_view(evaluated_jac, deriv_level, num_obs, max_num_points, deriv_count, num_points_used)
	function linear_index(which_obs, this_deriv_level, this_point)
		return this_deriv_level * num_obs + which_obs + (this_point - 1) * num_obs * (deriv_count + 1)
	end
	view_array = []
	for k in 1:num_points_used
		for (which_observable, max_deriv_level_this) in deriv_level
			for j in 0:max_deriv_level_this
				push!(view_array, linear_index(which_observable, j, k))
			end
		end
	end
	return view(evaluated_jac, view_array, :)
end

"""
	multipoint_local_identifiability_analysis(
		model::ModelingToolkit.System,
		measured_quantities::Vector{ModelingToolkit.Equation},
		max_num_points::Int,
		reltol::Float64 = 1e-12,
		abstol::Float64 = 1e-12
	) -> Tuple{Dict{Int,Int}, Dict, Vector{Num}, DerivativeData}

Performs local identifiability analysis at multiple points.

# Arguments
- `model::ModelingToolkit.System`: The ODE system
- `measured_quantities::Vector{ModelingToolkit.Equation}`: Measured quantities
- `max_num_points::Int`: Maximum number of points to use
- `reltol::Float64`: Relative tolerance (default: 1e-12)
- `abstol::Float64`: Absolute tolerance (default: 1e-12)

# Returns
- Tuple containing:
  1. Dictionary mapping observables to their required derivative levels
  2. Dictionary of unidentifiable parameters and their values
  3. List of identifiable variables
  4. DerivativeData object containing all computed derivatives
"""
function multipoint_local_identifiability_analysis(
	model::ModelingToolkit.AbstractSystem,
	measured_quantities,
	max_num_points::Int,
	reltol::Float64 = 1e-12,
	abstol::Float64 = 1e-12,
)::Tuple{Dict, Dict, Vector, DerivativeData}
	(t, model_eq, model_states, model_ps) = unpack_ODE(model)
	varlist = Vector{Num}(vcat(model_ps, model_states))

	#println("DEBUG [multipoint_local_identifiability_analysis]: Starting analysis with ", max_num_points, " points")

	states_count = length(model_states)
	ps_count = length(model_ps)
	D = Differential(t)

	# UNTESTED FIX: Use OrderedDict to ensure consistent ordering with varlist for Jacobian columns
	# This change was made to address potential Dict ordering issues in the Jacobian construction
	# but has NOT been verified to fix the k7=0 issue in biohydrogenation
	# TODO: Test if this actually improves parameter estimation accuracy
	parameter_values = OrderedDict{Num, Float64}()
	for p in ModelingToolkit.parameters(model)
		parameter_values[p] = rand(Float64)
	end

	points_ics = []
	test_points = []
	ordered_test_points = []

	for i in 1:max_num_points
		# UNTESTED FIX: Use OrderedDict for initial conditions too
		# See comment above - this is part of the same untested fix
		initial_conditions = OrderedDict{Num, Float64}()
		for s in ModelingToolkit.unknowns(model)
			initial_conditions[s] = rand(Float64)
		end

		ordered_test_point = OrderedDict{Num, Float64}()
		for i in model_ps
			ordered_test_point[i] = parameter_values[i]
		end
		for i in model_states
			ordered_test_point[i] = initial_conditions[i]
		end

		# test_point now uses the ordered version
		test_point = ordered_test_point

		push!(points_ics, deepcopy(initial_conditions))
		push!(test_points, deepcopy(test_point))
		push!(ordered_test_points, deepcopy(ordered_test_point))
	end

	# Determine derivative order 'n'
	# EXPERIMENTAL FIX: Match PE.jl's derivative order formula to handle biohydrogenation correctly
	# Original heuristic: n = Int64(ceil((states_count + ps_count) / length(measured_quantities)) + 2)
	# PE.jl formula: diff_order = num_parameters + 1 where num_parameters = params + states
	# Using PE's formula ensures sufficient equations for zero-dimensional polynomial system
	n_pe_formula = states_count + ps_count + 1
	n_heuristic = Int64(ceil((states_count + ps_count) / length(measured_quantities)) + 2)
	n = max(n_pe_formula, n_heuristic, 3)  # Use maximum of both approaches
	deriv_level = Dict([p => n for p in 1:length(measured_quantities)])
	unident_dict = Dict()

	jac = nothing
	evaluated_jac = nothing
	DD = nothing
	unident_set = Set{Num}()

	all_identified = false
	while (!all_identified)

		temp = ordered_test_points[1]

		(evaluated_jac, DD) = (multipoint_numerical_jacobian(model, measured_quantities, n, max_num_points, unident_dict, varlist,
			parameter_values, points_ics, temp))

		# Cap n to actual derivative levels computed (may be less than requested if coefficients overflowed)
		n = length(DD.obs_rhs) - 1
		deriv_level = Dict([p => n for p in 1:length(measured_quantities)])

		# Apply matrix equilibration for numerical stability in nullspace computation
		evaluated_jac = equilibrate_jacobian(evaluated_jac)

		ns = nullspace(evaluated_jac)

		if (!isempty(ns))
			candidate_plugins_for_unidentified = OrderedDict()
			for i in eachindex(varlist)
				ns_norm = norm(ns[i, :])
				if (!isapprox(ns_norm, 0.0, atol = abstol))
					candidate_plugins_for_unidentified[varlist[i]] = test_points[1][varlist[i]]
					push!(unident_set, varlist[i])
				end
			end
			if (!isempty(candidate_plugins_for_unidentified))
				p = first(candidate_plugins_for_unidentified)
				deleteat!(varlist, findall(x -> isequal(x, p.first), varlist))
				for k in eachindex(points_ics)
					delete!(points_ics[k], p.first)
					delete!(ordered_test_points[k], p.first)
					delete!(parameter_values, p.first)
				end
				unident_dict[p.first] = p.second
			else
				all_identified = true
			end
		else
			all_identified = true
		end
	end

	max_rank = rank(evaluated_jac, rtol = reltol)
	maxn = n

	while (n > 0)
		n = n - 1
		deriv_level = Dict([p => n for p in 1:length(measured_quantities)])
		reduced_evaluated_jac = multipoint_deriv_level_view(evaluated_jac, deriv_level, length(measured_quantities), max_num_points, maxn, max_num_points)
		r = rank(reduced_evaluated_jac, rtol = reltol)
		if (r < max_rank)
			n = n + 1
			deriv_level = Dict([p => n for p in 1:length(measured_quantities)])
			break
		end
	end

	keep_looking = true
	while (keep_looking)
		improvement_found = false
		sorting = collect(deriv_level)
		sorting = sort(sorting, by = (x -> x[2]), rev = true)
		for i in keys(deriv_level)
			if (deriv_level[i] > 0)
				deriv_level[i] = deriv_level[i] - 1
				reduced_evaluated_jac = multipoint_deriv_level_view(evaluated_jac, deriv_level, length(measured_quantities), max_num_points, maxn, max_num_points)

				r = rank(reduced_evaluated_jac, rtol = reltol)
				if (r < max_rank)
					deriv_level[i] = deriv_level[i] + 1
				else
					improvement_found = true
					break
				end
			else
				temp = pop!(deriv_level, i)
				reduced_evaluated_jac = multipoint_deriv_level_view(evaluated_jac, deriv_level, length(measured_quantities), max_num_points, maxn, max_num_points)

				r = rank(reduced_evaluated_jac, rtol = reltol)
				if (r < max_rank)
					deriv_level[i] = temp
				else
					improvement_found = true
					break
				end
			end
		end
		keep_looking = improvement_found
	end

	DD.all_unidentifiable = unident_set
	return (deriv_level, unident_dict, varlist, DD)
end


"""
	lookup_value(var, var_search, soln_index::Int,
				good_udict::AbstractDict, trivial_dict::AbstractDict,
				final_varlist::Vector, trimmed_varlist::Vector,
				solns::Vector) -> Float64

Look up a variable's value from various dictionaries and solution vectors.
This is a helper function for parameter estimation.

# Arguments
- `var`: Original variable to look up (Num or SymbolicUtils.BasicSymbolic{Real})
- `var_search`: Symbolic variable to search for
- `soln_index::Int`: Index in the solutions array
- `good_udict::AbstractDict`: Dictionary of explicit fixed values
- `trivial_dict::AbstractDict`: Dictionary of trivially determined values
- `final_varlist::Vector`: Full list of variables
- `trimmed_varlist::Vector`: Reduced list of variables
- `solns::Vector`: Solutions array

# Returns
- `Float64`: Value of the variable
"""
function lookup_value(var, var_search, soln_index::Int,
	good_udict::AbstractDict, trivial_dict::AbstractDict,
	final_varlist::Vector, trimmed_varlist::Vector,
	solns::Vector)::Float64
	# First check if it's in the unidentifiable dictionary
	if var in keys(good_udict)
		return Float64(good_udict[var])
	end

	# Then check if it's in the trivial dictionary
	if var_search in keys(trivial_dict)
		return Float64(trivial_dict[var_search])
	end

	# Finally, look it up in the solution vectors
	index = findfirst(isequal(var_search), final_varlist)
	if isnothing(index)
		index = findfirst(isequal(var_search), trimmed_varlist)
	end

	# Heuristic fallback: map model-style names to SI template names
	if isnothing(index)
		# Convert x(t) -> x_0, k5 -> k5_0, k_1 -> k_1_0, xˍt -> x_1, xˍtt -> x_2, etc.
		try
			# Phase B: with the explicit template_var_map in place upstream, this
			# string-heuristic path should only fire for unmapped/legacy callers —
			# make that observable.
			@debug "lookup_value falling through to name heuristics" var_search
			# Unwrap Num or other wrappers to get the core symbol/expression
			core = try
				Symbolics.value(var_search)
			catch e
				_rethrow_if_interrupt(e)
				@debug "Symbolics.value unwrap failed, using raw variable" exception = e
				var_search
			end
			name_str = string(core)

			# Strip special tags if present:
			# - parameter tag: _tp<name>_
			# - timepoint tag: _t<idx>_<name>_
			while true
				m = match(r"^_tp(.+)_$", name_str)
				if !isnothing(m)
					name_str = m.captures[1]
					continue
				end
				m2 = match(r"^_t\d+_(.+)_$", name_str)
				if !isnothing(m2)
					name_str = m2.captures[1]
					continue
				end
				break
			end

			# Remove trailing _t token introduced by tagging x(t) -> x_t
			name_str = chopsuffix(name_str, "_t")

			# Strip (t)
			name_str = chopsuffix(name_str, "(t)")
			# Count occurrences of the derivative marker "ˍt"
			deriv_count = 0
			while occursin("ˍt", name_str)
				name_str = replace(name_str, "ˍt" => "")
				deriv_count += 1
			end
			# Candidate template names, in order:
			#   1. The FULL model-style name + "_<deriv_count>". A parameter named
			#      k_1 must map to the jet variable k_1_0 — treating its trailing
			#      _1 as a derivative order can collide distinct parameters.
			#   2. If the name already carries a _n suffix, the name verbatim
			#      (covers callers that pass jet-style names like y1_2 directly).
			has_suffix = occursin(r"_[0-9]+$", name_str)
			fallback_candidates = [name_str * string("_", deriv_count)]
			has_suffix && push!(fallback_candidates, name_str)
			for fallback_str in fallback_candidates
				fallback_sym = Symbolics.variable(Symbol(fallback_str))
				index = findfirst(isequal(fallback_sym), final_varlist)
				if isnothing(index)
					index = findfirst(isequal(fallback_sym), trimmed_varlist)
				end
				# String-based comparison for this candidate as a fallback
				if isnothing(index)
					idx_str = findfirst(i -> string(final_varlist[i]) == fallback_str, eachindex(final_varlist))
					if isnothing(idx_str)
						idx_str = findfirst(i -> string(trimmed_varlist[i]) == fallback_str, eachindex(trimmed_varlist))
					end
					isnothing(idx_str) || (index = idx_str)
				end
				isnothing(index) || break
			end
		catch e
			_rethrow_if_interrupt(e)
			@debug "Variable index fallback lookup failed" exception = e
		end
	end

	# quiet: remove verbose debug prints

	# Return the real part of the solution as a Float64
	return Float64(real(solns[soln_index][index]))
end


"""
	PolishContext

Holds all invariant data needed for polishing solutions. Built once per problem,
reused across all polish/optimization runs. This avoids redundant calls to
`complete()`, `build_function()`, and `ODEProblem()` construction.

# Fields
- `unknown_syms`: System state variable symbols (MTK ordering)
- `param_syms`: System parameter symbols (MTK ordering)
- `n_ic`: Number of initial conditions (states)
- `n_param`: Number of parameters
- `new_model`: Completed MTK system (from `complete()`)
- `obs_funcs`: Compiled observable functions (one per measured quantity)
- `data_targets`: Vector of data vectors for each observable
- `t_vector`: Time points from data sample
- `tspan`: ODE integration time span
- `solver`: ODE solver instance
- `abstol`: Absolute tolerance for ODE solver
- `reltol`: Relative tolerance for ODE solver
- `adtype`: AD backend for Optimization.jl
- `optf`: Pre-built OptimizationFunction (loss closure + AD)
- `base_ode_prob`: Template ODEProblem for `remake` inside loss
- `state_syms_out`: User-facing state symbol ordering (from PEP.ic)
- `param_syms_out`: User-facing parameter symbol ordering (from PEP.p_true)
- `state_index`: Map from state symbol → index in unknown_syms
- `param_index`: Map from param symbol → index in param_syms
- `lb`: Optional lower bounds
- `ub`: Optional upper bounds
- `polish_ode_maxiters`: ODE solver iteration cap inside loss function (fails fast on hopeless regions)
"""
Base.@kwdef struct PolishContext
	unknown_syms::Vector
	param_syms::Vector
	n_ic::Int
	n_param::Int
	new_model::Any
	obs_funcs::Vector{Function}
	data_targets::Vector{Vector{Float64}}
	data_indices::Vector{Vector{Int}} = Vector{Int}[]
	t_vector::Vector{Float64}
	tspan::Tuple{Float64, Float64}
	solver::Any
	abstol::Float64
	reltol::Float64
	adtype::Any
	optf::Optimization.OptimizationFunction
	base_ode_prob::Any
	state_syms_out::Vector
	param_syms_out::Vector
	state_index::Dict
	param_index::Dict
	lb::Union{Nothing, Vector{Float64}} = nothing
	ub::Union{Nothing, Vector{Float64}} = nothing
	internal_lb::Union{Nothing, Vector{Float64}} = nothing
	internal_ub::Union{Nothing, Vector{Float64}} = nothing
	coordinate_transforms::Vector{Symbol} = Symbol[]
	# Per-variable shift used by :shifted_log entries; 0.0 for :log/:linear entries.
	coordinate_shifts::Vector{Float64} = Float64[]
	polish_ode_maxiters::Int = 5000
	# Optional: per-variable LSO-style regularization weight (sqrt(λ) is applied to internal coords).
	regularization_lambda::Float64 = 0.0
	# Optional soft-wall penalty near bounds. When `softwall_lambda > 0`, each parameter
	# contributes one residual row that is zero in the central (1 - 2·softwall_epsilon)
	# fraction of the internal-coord interval and grows quadratically outside that band.
	softwall_lambda::Float64 = 0.0
	softwall_epsilon::Float64 = 0.05
end

"""
Internal carrier used only when estimator-aware UQ is enabled. It preserves the
exact optimizer-space point that production selected (including best-iterate or
revert-guard recovery) while forwarding ordinary optimizer-result properties to
keep the private `_polish_single_from_context` return contract compatible.
"""
struct _PolishSolveRecord{O}
	optimizer_result::O
	internal_optimum::Vector{Float64}
end

function Base.getproperty(record::_PolishSolveRecord, name::Symbol)
	if name === :optimizer_result || name === :internal_optimum
		return getfield(record, name)
	end
	return getproperty(getfield(record, :optimizer_result), name)
end

function Base.propertynames(record::_PolishSolveRecord, private::Bool = false)
	own = (:optimizer_result, :internal_optimum)
	inner = propertynames(getfield(record, :optimizer_result), private)
	return (own..., inner...)
end

_polish_raw_optimizer_result(result) = result
_polish_raw_optimizer_result(result::_PolishSolveRecord) = result.optimizer_result

function _polish_retained_internal_vector(
	ctx::PolishContext,
	result::ParameterEstimationResult,
	optimizer_result,
)
	if optimizer_result isa _PolishSolveRecord
		return copy(optimizer_result.internal_optimum)
	end
	return _polish_result_internal_vector(ctx, result)
end

const _POLISH_SHIFT_EPS = 1e-6
# A box with `lb < 0` still gets `:shifted_log` only when the negative excursion is at
# least one order of magnitude smaller than the positive reach (|lb| <= ub / RATIO) — the
# "positive variable, but let the optimizer wiggle around 0" pattern, e.g. (-0.01, 100).
# A comparable sign-straddle (e.g. a symmetric box) gets `:linear` instead, so the shift
# never blows up to the bound magnitude.
const _POLISH_SHIFTED_LOG_RATIO = 10.0
const _POLISH_VALID_TRANSFORMS = Set([:linear, :log, :shifted_log])

"""
	_choose_polish_transforms(lb, ub; policy = :auto) -> (Vector{Symbol}, Vector{Float64})

Per-variable selection of coordinate transform.

The `:auto` rule (per variable, both bounds finite unless noted):
- `lb > 0`                                   → `:log`. Forward `log(x)`, shift 0.
- `lb == 0`                                  → `:shifted_log` (clear the `log(0)` wall).
- `lb < 0` and `|lb| <= ub / RATIO` (ub > 0) → `:shifted_log`. The negative reach is ≥1 OOM
  smaller than the positive reach — the "positive variable, wiggle around 0" pattern
  (e.g. `(-0.01, 100)`). Shift `s = max(ε·M, -lb + ε·M)`, `M = max(|lb|, |ub|, 1)`, stays
  small because `|lb|` is small. Forward `log(x + s)`.
- otherwise                                  → `:linear` (shift 0): a comparable sign-straddle
  (e.g. a symmetric box) where a shift would have to equal the bound magnitude, plus any
  case with a `±Inf` bound or missing bounds.

`RATIO = _POLISH_SHIFTED_LOG_RATIO` (= 10).

`policy`:
- `:auto`         (default) — per-variable selection per the rules above.
- `:linear`       — force `:linear` for every variable (legacy linear polish).
- `:log_only`     — `:log` where bounds permit; otherwise `:linear`.
- `:shifted_log_only` — `:shifted_log` where bounds permit; otherwise `:linear`.
"""
function _choose_polish_transforms(
	lb::Union{Nothing, AbstractVector{<:Real}},
	ub::Union{Nothing, AbstractVector{<:Real}};
	policy::Symbol = :auto,
)
	(isnothing(lb) || isnothing(ub)) && throw(ArgumentError(
		"_choose_polish_transforms requires both lb and ub (use compute_default_bounds first)"))
	length(lb) == length(ub) || throw(ArgumentError("lb and ub must have equal length"))
	n = length(lb)
	transforms = Vector{Symbol}(undef, n)
	shifts = zeros(Float64, n)
	for i in 1:n
		lbi = Float64(lb[i])
		ubi = Float64(ub[i])
		bounds_finite = isfinite(lbi) && isfinite(ubi)
		if policy == :linear
			transforms[i] = :linear
		elseif policy == :log_only
			# Strict: every variable must admit `:log` (lb > 0 && finite ub).
			# This mirrors the legacy `:log_positive` semantics — error rather
			# than silently fall back to `:linear`. Use `:auto` if you want
			# graceful per-variable fallback.
			(lbi > 0.0 && bounds_finite) || throw(ArgumentError(
				"variable $i: :log_only requires strictly positive finite bounds (lb=$lbi, ub=$ubi)"))
			transforms[i] = :log
		elseif !bounds_finite
			# `:auto` and `:shifted_log_only` fall through to :linear when unbounded
			transforms[i] = :linear
		elseif policy == :shifted_log_only
			transforms[i] = :shifted_log
			shifts[i] = _shifted_log_shift(lbi, ubi)
		else
			# :auto (both bounds finite here — the !bounds_finite case fell through to
			# :linear above). See the docstring for the rule.
			if lbi > 0.0
				transforms[i] = :log
			elseif lbi == 0.0 || (lbi < 0.0 && ubi > 0.0 && abs(lbi) <= ubi / _POLISH_SHIFTED_LOG_RATIO)
				# nonnegative (lb==0), or positive-with-small-negative-wiggle (lb<0, ≥1 OOM smaller)
				transforms[i] = :shifted_log
				shifts[i] = _shifted_log_shift(lbi, ubi)
			else
				# comparable sign-straddle (e.g. symmetric box), or ub <= 0 → affine/linear
				transforms[i] = :linear
			end
		end
	end
	return transforms, shifts
end

function _shifted_log_shift(lb::Real, ub::Real)
	M = max(abs(lb), abs(ub), 1.0)
	εM = _POLISH_SHIFT_EPS * M
	# `s` keeps the shift on a meaningful scale even when `lb` is barely negative
	return max(εM, -lb + εM)
end

"""
	_polish_coordinate_bounds(transforms, shifts, lb, ub) -> (Vector, Vector)

Per-variable map from external bounds to internal bounds, given the chosen transform
and shift for each coordinate. Unbounded entries are passed through as `±Inf` —
LSO/FastLM accept those natively. Returns `(internal_lb, internal_ub)` of the same
length as `lb`/`ub`.
"""
function _polish_coordinate_bounds(
	transforms::Vector{Symbol},
	shifts::Vector{Float64},
	lb::Union{Nothing, Vector{Float64}},
	ub::Union{Nothing, Vector{Float64}},
)
	(isnothing(lb) || isnothing(ub)) && return lb, ub
	n = length(lb)
	@assert length(transforms) == n "transforms length mismatch with bounds"
	@assert length(shifts) == n "shifts length mismatch with bounds"
	internal_lb = similar(lb)
	internal_ub = similar(ub)
	for i in 1:n
		t = transforms[i]
		if t === :linear
			internal_lb[i] = lb[i]
			internal_ub[i] = ub[i]
		elseif t === :log
			lb[i] > 0.0 || throw(ArgumentError("variable $i: :log requires lb > 0 (got $(lb[i]))"))
			isfinite(ub[i]) || throw(ArgumentError("variable $i: :log requires finite ub"))
			internal_lb[i] = log(lb[i])
			internal_ub[i] = log(ub[i])
		elseif t === :shifted_log
			s = shifts[i]
			lb[i] + s > 0.0 || throw(ArgumentError("variable $i: :shifted_log shift produces non-positive lb (lb=$(lb[i]) s=$s)"))
			isfinite(ub[i]) || throw(ArgumentError("variable $i: :shifted_log requires finite ub"))
			internal_lb[i] = log(lb[i] + s)
			internal_ub[i] = log(ub[i] + s)
		else
			throw(ArgumentError("Unknown polish coordinate transform '$t' at index $i"))
		end
	end
	return internal_lb, internal_ub
end

"""
	_polish_external_to_internal(values, transforms, shifts) -> Vector

Forward map external (user-space) values to internal (optimizer-space) coordinates,
per variable.
"""
function _polish_external_to_internal(
	values::AbstractVector{<:Real},
	transforms::Vector{Symbol},
	shifts::Vector{Float64},
)
	n = length(values)
	out = similar(values, float(eltype(values)))
	@inbounds for i in 1:n
		t = transforms[i]
		v = values[i]
		out[i] = if t === :linear
			v
		elseif t === :log
			log(v)
		elseif t === :shifted_log
			log(v + shifts[i])
		else
			throw(ArgumentError("Unknown polish coordinate transform '$t' at index $i"))
		end
	end
	return out
end

"""
	_polish_internal_to_external(values, transforms, shifts) -> Vector

Inverse map internal (optimizer-space) values back to external (user-space)
coordinates, per variable.
"""
function _polish_internal_to_external(
	values::AbstractVector{<:Real},
	transforms::Vector{Symbol},
	shifts::Vector{Float64},
)
	n = length(values)
	out = similar(values, float(eltype(values)))
	@inbounds for i in 1:n
		t = transforms[i]
		v = values[i]
		out[i] = if t === :linear
			v
		elseif t === :log
			exp(v)
		elseif t === :shifted_log
			exp(v) - shifts[i]
		else
			throw(ArgumentError("Unknown polish coordinate transform '$t' at index $i"))
		end
	end
	return out
end

function _polish_result_internal_vector(ctx::PolishContext, result::ParameterEstimationResult)
	external = Float64[
		[result.states[s] for s in ctx.unknown_syms]...,
		[result.parameters[p] for p in ctx.param_syms]...,
	]
	return _polish_external_to_internal(
		external, ctx.coordinate_transforms, ctx.coordinate_shifts,
	)
end

"""
	_polish_policy_from_legacy(coordinate_transform::Symbol) -> Symbol

Map the legacy single-symbol `coordinate_transform` keyword (`:linear` or `:log_positive`)
to the new per-variable policy used by `_choose_polish_transforms`. Internal use only.
"""
function _polish_policy_from_legacy(coordinate_transform::Symbol)
	coordinate_transform == :linear && return :linear
	coordinate_transform == :log_positive && return :log_only
	coordinate_transform == :auto && return :auto
	coordinate_transform == :shifted_log_safe && return :shifted_log_only
	throw(ArgumentError("Unsupported polish coordinate transform '$coordinate_transform'. " *
		"Supported: :linear, :log_positive, :auto, :shifted_log_safe."))
end

"""
	_build_polish_context(PEP; opts) -> PolishContext

Perform all expensive one-time setup for polishing: model completion, observable
compilation, base ODEProblem construction, and OptimizationFunction assembly.
Call this once, then pass the result to `_polish_single_from_context` or
`_polish_batch_from_context` for each solution.
"""
function _build_polish_context(
	PEP::ParameterEstimationProblem;
	opts::EstimationOptions = EstimationOptions(),
	coordinate_transform::Union{Nothing, Symbol} = nothing,
)
	context_t0 = time()
	context_stages = OrderedDict{Symbol, Float64}()
	_validate_observation_options(PEP.data_sample, opts)
	# When the caller leaves `coordinate_transform` unset, residual polish methods
	# pick up `opts.polish_coordinate_policy` (default `:auto` = per-variable);
	# scalar polish methods default to `:linear` to preserve byte-equivalent legacy
	# behavior. Explicit `coordinate_transform = :linear` / `:log_positive` keeps
	# its meaning regardless of `polish_method` (used by tests and bounds-aware
	# callers in benchmark scripts).
	resolved_transform = if !isnothing(coordinate_transform)
		coordinate_transform
	elseif is_residual_polish_method(opts.polish_method)
		opts.polish_coordinate_policy
	else
		:linear
	end
	coordinate_transform = resolved_transform
	# Stable variable ordering from the system
	unknown_syms = ModelingToolkit.unknowns(PEP.model.system)
	param_syms = ModelingToolkit.parameters(PEP.model.system)
	n_ic = length(unknown_syms)
	n_param = length(param_syms)
	p_size = n_ic + n_param

	# Complete model once (not 28× per polish run)
	new_model = _timed_detail_stage!(context_stages, :complete_model) do
		complete(PEP.model.system)
	end
	t_vector = Float64.(_observation_union(PEP.data_sample))
	tspan = (Float64(_initial_time(PEP.data_sample)), t_vector[end])
	# ObservationData pools rows for identical signal expressions. Evaluate each
	# pool once even when the model gives that signal more than one observable ID.
	quantities = PEP.data_sample isa ObservationData ?
		unique(eq -> eq.rhs, PEP.measured_quantities) : PEP.measured_quantities
	data_indices = [Int[searchsortedfirst(t_vector, tx) for tx in
		observation_times(PEP.data_sample, eq.rhs)] for eq in quantities]

	# Compile observable functions once (not 28× per polish run)
	obs_funcs = _timed_detail_stage!(context_stages, :build_observable_functions) do
		Function[
			let f_raw = ModelingToolkit.build_function(eq.rhs, unknown_syms, param_syms; expression = Val(false))
				f_fun = isa(f_raw, Tuple) ? f_raw[1] : f_raw
				(u::AbstractVector{<:Real}, p::AbstractVector{<:Real}) -> f_fun(u, p)
			end for eq in quantities
		]
	end
	data_targets = _timed_detail_stage!(context_stages, :materialize_data_targets) do
		Vector{Float64}[Float64.(PEP.data_sample[eq.rhs]) for eq in quantities]
	end

	# Solver and tolerances
	solver = PEP.solver
	abstol = opts.abstol
	reltol = opts.reltol
	adtype = get_ad_backend(opts.opt_ad_backend)

	# Build base ODEProblem once — remake inside loss will swap u0/p values
	u0_default = Dict(unknown_syms .=> zeros(n_ic))
	p_default = Dict(param_syms .=> ones(n_param))
	base_ode_prob = _timed_detail_stage!(context_stages, :base_ode_problem) do
		ODEProblem(new_model, merge(u0_default, p_default), tspan)
	end

	# Bounds: use user-specified if set (length MISMATCH throws — fail-fast; the
	# old silent fallback to the ±1e6 box let polished results escape the user's
	# intended bounds, docs/internal/2026-06-19_transform_bounds_mismatch.md). Otherwise
	# auto-compute from data scale.
	# Note: only BFGS/LBFGS support Fminbox bounds wrapping; Newton-family optimizers
	# silently ignore bounds in _polish_single_from_context.
	if !isnothing(opts.opt_lb) || !isnothing(opts.opt_ub)
		_assert_bounds_length(opts, p_size, "_build_polish_context (polish bounds)",
			"[state ICs; params] of the transformed model")
	end
	lb = nothing
	ub = nothing
	if !isnothing(opts.opt_lb) && !isnothing(opts.opt_ub)
		lb = Float64.(opts.opt_lb)
		ub = Float64.(opts.opt_ub)
	else
		# No user bounds: use the NARROWER polish multiplier (1e6), not the wide 1e9
		# detection box. Symmetric ⇒ :auto picks :linear (well-conditioned, no shift).
		lb_auto, ub_auto = compute_default_bounds(PEP; multiplier = DEFAULT_POLISH_BOUND_MULTIPLIER)
		lb = lb_auto
		ub = ub_auto
	end
	policy = _polish_policy_from_legacy(coordinate_transform)
	transforms, shifts = _choose_polish_transforms(lb, ub; policy = policy)
	internal_lb, internal_ub = _polish_coordinate_bounds(transforms, shifts, lb, ub)

	# ODE solver iteration cap — fail fast on hopeless parameter regions
	ode_maxiters = opts.polish_ode_maxiters

	# Loss closure capturing all invariants — uses remake for efficiency
	function loss(p_internal)
		loss_t0 = time()
		loss_stages = OrderedDict{Symbol, Float64}()
		p_all = _polish_internal_to_external(p_internal, transforms, shifts)
		ic_guess = @view p_all[1:n_ic]
		param_guess = @view p_all[(n_ic+1):end]

		prob_opt = _timed_detail_stage!(loss_stages, :ode_remake) do
			remake(base_ode_prob; u0 = Dict(unknown_syms .=> ic_guess), p = Dict(param_syms .=> param_guess), build_initializeprob = false)
		end
		ode_threw = false
		sol_opt = _timed_detail_stage!(loss_stages, :ode_solve) do
			try
				ModelingToolkit.solve(prob_opt, solver; saveat = t_vector, abstol = abstol, reltol = reltol, maxiters = ode_maxiters)
			catch e
				_rethrow_if_interrupt(e)
				ode_threw = true
				@warn "ODE solver failed during polish" exception = (e, catch_backtrace())
				nothing
			end
		end
		if sol_opt === nothing || sol_opt.retcode != ReturnCode.Success
			_record_detailed_timing!((
				category = :polish_scalar_loss_eval,
				context = _current_detailed_timing_context(),
				total_seconds = time() - loss_t0,
				stage_seconds = copy(loss_stages),
				ode_success = false,
				ode_threw = ode_threw,
				data_point_count = length(t_vector),
				observable_count = length(obs_funcs),
			))
			return Inf
		end

		total_error = zero(eltype(p_all))
		_timed_detail_stage!(loss_stages, :observable_error_eval) do
			for (j, f) in enumerate(obs_funcs)
				data_true = data_targets[j]
				local_err = zero(eltype(p_all))
				@inbounds for (i, time_index) in enumerate(data_indices[j])
					val = f(sol_opt.u[time_index], param_guess)
					diff = val - data_true[i]
					local_err += diff * diff
				end
				total_error += local_err
			end
		end
		_record_detailed_timing!((
			category = :polish_scalar_loss_eval,
			context = _current_detailed_timing_context(),
			total_seconds = time() - loss_t0,
			stage_seconds = copy(loss_stages),
			ode_success = true,
			ode_threw = false,
			data_point_count = length(t_vector),
			observable_count = length(obs_funcs),
		))
		return total_error
	end

	optf = _timed_detail_stage!(context_stages, :optimization_function) do
		Optimization.OptimizationFunction((x, _) -> loss(x), adtype)
	end

	# User-facing symbol ordering for result construction
	state_syms_out = collect(keys(PEP.ic))
	param_syms_out = collect(keys(PEP.p_true))
	state_index = Dict(s => i for (i, s) in enumerate(unknown_syms))
	param_index = Dict(p => i for (i, p) in enumerate(param_syms))

	ctx = PolishContext(
		unknown_syms = unknown_syms,
		param_syms = param_syms,
		n_ic = n_ic,
		n_param = n_param,
		new_model = new_model,
		obs_funcs = obs_funcs,
		data_targets = data_targets,
		data_indices = data_indices,
		t_vector = t_vector,
		tspan = tspan,
		solver = solver,
		abstol = abstol,
		reltol = reltol,
		adtype = adtype,
		optf = optf,
		base_ode_prob = base_ode_prob,
		state_syms_out = state_syms_out,
		param_syms_out = param_syms_out,
		state_index = state_index,
		param_index = param_index,
		lb = lb,
		ub = ub,
		internal_lb = internal_lb,
		internal_ub = internal_ub,
		coordinate_transforms = transforms,
		coordinate_shifts = shifts,
		polish_ode_maxiters = ode_maxiters,
		regularization_lambda = opts.polish_regularization_lambda,
		softwall_lambda = opts.polish_softwall_lambda,
		softwall_epsilon = opts.polish_softwall_epsilon,
	)
	_record_detailed_timing!((
		category = :polish_context_build,
		context = _current_detailed_timing_context(),
		total_seconds = time() - context_t0,
		stage_seconds = copy(context_stages),
		state_count = n_ic,
		parameter_count = n_param,
		observable_count = length(PEP.measured_quantities),
		data_point_count = length(t_vector),
		polish_method = opts.polish_method,
	))
	return ctx
end

"""
	_trajectory_sse(ctx::PolishContext, p_external) -> Float64

THE canonical candidate fit-error: the trajectory sum-of-squared residuals
`Σ_obs Σ_t (obs_pred − data)²`, evaluated DIRECTLY at `p_external` (user-space
`[states; params]`). Every *reported* `.err` in the pipeline is this same quantity, so
candidates from different sources (algebraic, polished, synthesized, seed) are ranked
in ONE unit — the same one the optimizer minimizes ("rank by what you optimize").

Two siblings compute the identical SSE for their own reasons, NOT a competing formula:
- `process_raw_solution` computes it inline because it works from a raw `ODESolution`
  (no `PolishContext`, observables via MTK observed-indexing);
- the optimizer `loss` closure accumulates it in-place in its AD hot loop.

Evaluated on external params directly — NO search-coordinate round-trip. Going through
`ctx.optf.f` would map `external → internal → external`, and for `:shifted_log` with a
large shift that round-trip loses ~1e-7 to cancellation, making a candidate's `err`
depend on the optimizer's coordinate.
"""
function _trajectory_sse(ctx::PolishContext, p_external::AbstractVector{<:Real})
	ic_guess = @view p_external[1:ctx.n_ic]
	param_guess = @view p_external[(ctx.n_ic+1):end]
	prob_opt = remake(ctx.base_ode_prob;
		u0 = Dict(ctx.unknown_syms .=> ic_guess),
		p = Dict(ctx.param_syms .=> param_guess),
		build_initializeprob = false)
	sol_opt = try
		ModelingToolkit.solve(prob_opt, ctx.solver; saveat = ctx.t_vector,
			abstol = ctx.abstol, reltol = ctx.reltol, maxiters = ctx.polish_ode_maxiters)
	catch err
		_rethrow_if_interrupt(err)
		return Inf
	end
	(sol_opt === nothing || sol_opt.retcode != ReturnCode.Success) && return Inf
	total_error = 0.0
	for (j, f) in enumerate(ctx.obs_funcs)
		data_true = ctx.data_targets[j]
		@inbounds for (i, time_index) in enumerate(_polish_data_indices(ctx, j))
			diff = f(sol_opt.u[time_index], param_guess) - data_true[i]
			total_error += diff * diff
		end
	end
	return total_error
end

_polish_data_indices(ctx::PolishContext, j::Int) =
	isempty(ctx.data_indices) ? eachindex(ctx.t_vector) : ctx.data_indices[j]

"""
	_polish_single_from_context(ctx, p0; optimizer, maxiters, maxtime, divergence_factor, stagnation_window) -> (ParameterEstimationResult, opt_result)

Polish a single solution using a pre-built PolishContext. Only constructs
the lightweight OptimizationProblem (wrapping p0) and solves.

Safeguard callbacks automatically stop optimization when:
- Wall-clock time exceeds `maxtime` seconds
- Loss diverges beyond `initial_loss * divergence_factor`
- No improvement seen in `stagnation_window` consecutive iterations
- Loss becomes non-finite (NaN/Inf)

The best solution seen during optimization is tracked; if the optimizer
wanders past a good minimum, the best iterate is recovered.
"""
function _polish_single_from_context(
	ctx::PolishContext,
	p0::AbstractVector{<:Real};
	optimizer = BFGS(),
	polish_method::Union{Nothing, PolishMethod} = nothing,
	maxiters::Int = 200000,
	maxtime::Float64 = 300.0,
	divergence_factor::Float64 = 10.0,
	stagnation_window::Int = 50,
	lso_delta::Float64 = 10.0,
	lso_x_tol::Float64 = -1.0,
	lso_f_tol::Float64 = -1.0,
	lso_g_tol::Float64 = -1.0,
	retain_internal_optimum::Bool = false,
)
	# Residual-mode polish (LSO / FastLM) uses a separate solver path with native
	# bound support and a revert guard. Dispatch early so the rest of this body
	# remains a clean scalar-loss path.
	if !isnothing(polish_method) && is_residual_polish_method(polish_method)
		token = get_polish_optimizer(polish_method)
		isa(token, Tuple) || error("Residual polish method $(polish_method) did not return a tagged token")
		kind, factory = token
		return _polish_single_residual(
			ctx, p0;
			solver_kind = kind,
			optimizer_factory = factory,
			maxiters = maxiters,
			maxtime = maxtime,
			lso_delta = lso_delta,
			lso_x_tol = lso_x_tol,
			lso_f_tol = lso_f_tol,
			lso_g_tol = lso_g_tol,
			retain_internal_optimum = retain_internal_optimum,
		)
	end

	# Only BFGS/LBFGS support Fminbox bounds wrapping; NewtonTrustRegion doesn't.
	use_bounds = !isnothing(ctx.internal_lb) && !isnothing(ctx.internal_ub) &&
		optimizer isa Union{Optim.BFGS, Optim.LBFGS}
	has_external_bounds = !isnothing(ctx.lb) && !isnothing(ctx.ub)
	# Any non-`:linear` entry means we have an active coordinate transform that requires
	# the seed to live inside the external bounds before we can map to internal.
	any_nontrivial_transform = any(!=(:linear), ctx.coordinate_transforms)

	p0_external = if any_nontrivial_transform
		has_external_bounds ? clamp.(Float64.(p0), ctx.lb, ctx.ub) : Float64.(p0)
	else
		use_bounds ? clamp.(Float64.(p0), ctx.lb, ctx.ub) : Float64.(p0)
	end
	p0_internal = _polish_external_to_internal(p0_external, ctx.coordinate_transforms, ctx.coordinate_shifts)

	optprob = if use_bounds
		Optimization.OptimizationProblem(ctx.optf, p0_internal; lb = ctx.internal_lb, ub = ctx.internal_ub)
	else
		Optimization.OptimizationProblem(ctx.optf, p0_internal)
	end

	# Evaluate initial loss for divergence baseline
	initial_loss = try
		ctx.optf.f(p0_internal, nothing)
	catch err
		_rethrow_if_interrupt(err)
		Inf
	end

	# Safeguard state — each Threads.@spawn gets its own closure, so Refs are thread-safe
	opt_verbose = get(ENV, "ODEPE_OPT_VERBOSE", "false") == "true"
	iter_count = Ref(0)
	best_loss = Ref(isfinite(initial_loss) ? initial_loss : Inf)
	best_p = Ref(copy(p0_internal))
	iters_since_improvement = Ref(0)
	start_time = time()
	last_log_time = Ref(start_time)
	stop_reason = Ref("")

	callback = (x, l) -> begin
		iter_count[] += 1

		# Track best solution
		if isfinite(l) && l < best_loss[]
			best_loss[] = l
			best_p[] = copy(x.u)
			iters_since_improvement[] = 0
		else
			iters_since_improvement[] += 1
		end

		# Safeguard checks
		if !isfinite(l)
			stop_reason[] = "non-finite loss"
			return true
		end

		elapsed = time() - start_time
		if elapsed > maxtime
			stop_reason[] = "wall-clock timeout ($(round(elapsed; digits=1))s > $(maxtime)s)"
			return true
		end

		if isfinite(initial_loss) && initial_loss > 0 && l > initial_loss * divergence_factor
			stop_reason[] = "divergence (loss $(round(l; sigdigits=3)) > $(round(initial_loss * divergence_factor; sigdigits=3)))"
			return true
		end

		if iters_since_improvement[] >= stagnation_window
			stop_reason[] = "stagnation (no improvement in $(stagnation_window) iters)"
			return true
		end

		# Verbose logging (gated on env var)
		if opt_verbose
			now = time()
			dt = now - last_log_time[]
			last_log_time[] = now
			if (iter_count[] == 1) || (l <= best_loss[]) || (iter_count[] % 50 == 0)
				println("[polish]  iter=$(iter_count[]) loss=$(l) best=$(round(best_loss[]; sigdigits=6)) dt=$(round(dt; digits=3))s elapsed=$(round(elapsed; digits=1))s")
			end
		end

		false
	end

	if opt_verbose
		println("[polish] Starting optimization with $(typeof(optimizer))")
		println("[polish]  n_states=$(ctx.n_ic), n_params=$(ctx.n_param), data_points=$(length(ctx.t_vector))")
		println("[polish]  solver=$(typeof(ctx.solver)), abstol=$(ctx.abstol), reltol=$(ctx.reltol), maxiters=$(maxiters)")
		println("[polish]  coordinate_transforms=$(ctx.coordinate_transforms)")
		println("[polish]  safeguards: maxtime=$(maxtime)s, divergence=$(divergence_factor)x, stagnation=$(stagnation_window) iters")
		println("[polish]  initial_loss=$(initial_loss)")
	end

	result = Optimization.solve(optprob, optimizer; maxiters = maxiters, callback = callback)

	if opt_verbose && !isempty(stop_reason[])
		println("[polish]  early stop: $(stop_reason[]) after $(iter_count[]) iters")
	end

	# Best-iterate recovery: the optimizer can wander past the minimum, so we may prefer the
	# best iterate seen during the run. CRITICAL: under a bounded (Fminbox) solve the callback's
	# `best_loss[]` is the *barrier-augmented* objective — large and often NEGATIVE far inside the
	# bounds (the -Σlog(dist-to-bound) penalty times a big μ) — so it is NOT comparable to the
	# true loss and must never be reported as `err`. Re-evaluate the real objective (`ctx.optf.f`,
	# a sum-of-squares ≥ 0) at both the best iterate and the optimizer's final point, and keep
	# whichever genuinely fits better. This keeps `err`/`post_polish_error` honest (≥ 0).
	true_loss_final = try
		Float64(ctx.optf.f(result.u, nothing))
	catch err
		_rethrow_if_interrupt(err)
		Inf
	end
	true_loss_best = isfinite(best_loss[]) ? (try
		Float64(ctx.optf.f(best_p[], nothing))
	catch err
		_rethrow_if_interrupt(err)
		Inf
	end) : Inf
	use_best_iterate = true_loss_best < true_loss_final
	if opt_verbose && use_best_iterate
		println("[polish]  recovering best iterate: true loss $(round(true_loss_best; sigdigits=6)) < final $(round(true_loss_final; sigdigits=6))")
	end
	p_opt_internal = use_best_iterate ? best_p[] : result.u
	p_opt = _polish_internal_to_external(p_opt_internal, ctx.coordinate_transforms, ctx.coordinate_shifts)
	ic_opt = p_opt[1:ctx.n_ic]
	param_opt = p_opt[(ctx.n_ic+1):end]
	prob_final = remake(ctx.base_ode_prob;
		u0 = Dict(ctx.unknown_syms .=> ic_opt),
		p = Dict(ctx.param_syms .=> param_opt),
		build_initializeprob = false,
	)
	# Guarded like the in-loss solves (postcampaign review P1): a throwing
	# integrator here (e.g. SingularException on a rank-deficient candidate) sits
	# in the nothing-else-worked terminal paths and must not kill the estimation.
	sol_final = try
		ModelingToolkit.solve(prob_final, ctx.solver; saveat = ctx.t_vector, abstol = ctx.abstol, reltol = ctx.reltol, maxiters = ctx.polish_ode_maxiters)
	catch e
		_rethrow_if_interrupt(e)
		@warn "Final polish ODE solve threw; returning result without trajectory" exception = e maxlog = 10
		nothing
	end

	# Map back to user-facing ordering
	states_out = OrderedDict(s => ic_opt[ctx.state_index[s]] for s in ctx.state_syms_out if haskey(ctx.state_index, s))
	params_out = OrderedDict(p => param_opt[ctx.param_index[p]] for p in ctx.param_syms_out if haskey(ctx.param_index, p))

	# Honest sum-of-squares error at the chosen point — re-evaluated above via `ctx.optf.f`,
	# never the barrier-augmented callback value. Guaranteed ≥ 0.
	final_obj = use_best_iterate ? true_loss_best : true_loss_final

	final_result = ParameterEstimationResult(
		params_out,
		states_out,
		ctx.tspan[1],
		final_obj,
		nothing,
		length(ctx.t_vector),
		ctx.tspan[1],
		OrderedDict{Num, Float64}(),
		Set{Num}(),
		sol_final,
	)
	final_result.provenance = ResultProvenance(
		primary_method = :direct_opt,
		post_polish_error = final_obj,
	)
	sync_result_contract!(final_result)
	returned_optimizer = retain_internal_optimum ?
		_PolishSolveRecord(result, Float64.(p_opt_internal)) : result
	return final_result, returned_optimizer
end

function _polish_cluster_metadata(
	ctx::PolishContext,
	candidates::AbstractVector;
	opts::EstimationOptions = EstimationOptions(),
	cluster_threshold::Float64 = 0.001,
)
	residual_mode = is_residual_polish_method(opts.polish_method)
	# For scalar methods, only BFGS/LBFGS support Fminbox bounds wrapping. For
	# residual methods (LSO / FastLM), bounds are passed natively to the solver,
	# so we always clamp the seed to the external bounds first.
	scalar_uses_bounds = if residual_mode
		false
	else
		optimizer_type = get_polish_optimizer(opts.polish_method)
		optimizer = optimizer_type()
		optimizer isa Union{Optim.BFGS, Optim.LBFGS}
	end
	n_candidates = length(candidates)
	use_bounds = !isnothing(ctx.lb) && !isnothing(ctx.ub) &&
		(residual_mode || scalar_uses_bounds)

	clamped_p0s = Vector{Vector{Float64}}(undef, n_candidates)
	for (i, candidate) in enumerate(candidates)
		ic_vec = [candidate.states[s] for s in ctx.unknown_syms]
		param_vec = [candidate.parameters[p] for p in ctx.param_syms]
		p0 = vcat(ic_vec, param_vec)
		clamped_p0s[i] = use_bounds ? clamp.(p0, ctx.lb, ctx.ub) : Float64.(p0)
	end

	# Phase B branch-detection path: pre-polish err filter + L-inf normalized
	# clustering in identifiable-only variable space. When opts.branch_detection
	# is false, fall through to the legacy uniform-max-rel-dist code below.
	if opts.branch_detection && n_candidates >= 2
		# 1) Err filter: drop candidates with err > branch_err_factor × min_finite_err.
		# This kills algebraic_resolve_t0 blown rescues (err > 1.0 typically), which
		# dominate the pool but cannot recover via polish.
		finite_errs = Float64[]
		for c in candidates
			e = isnothing(c.err) ? Inf : c.err
			if isfinite(e)
				push!(finite_errs, e)
			end
		end
		# If we have no finite errors, fall through to legacy path
		if !isempty(finite_errs)
			min_err = minimum(finite_errs)
			err_cap = opts.branch_err_factor * max(min_err, eps(Float64))
			keep_mask = falses(n_candidates)
			for i in 1:n_candidates
				e = isnothing(candidates[i].err) ? Inf : candidates[i].err
				if isfinite(e) && e <= err_cap
					keep_mask[i] = true
				end
			end
			n_kept = count(keep_mask)
			# Safety: if filter is too aggressive (everything dropped), keep top-5 by err
			if n_kept == 0
				idx_by_err = sortperm([isnothing(candidates[i].err) ? Inf : candidates[i].err for i in 1:n_candidates])
				for i in idx_by_err[1:min(5, n_candidates)]
					keep_mask[i] = true
				end
				n_kept = count(keep_mask)
			end

			# 2) Identifiable-mask on the (states ∪ params) vector
			# all_unidentifiable lives on each candidate; use first kept candidate's set.
			first_kept = candidates[findfirst(keep_mask)]
			all_unid = first_kept.all_unidentifiable
			id_mask = Bool[]
			for s in ctx.unknown_syms
				push!(id_mask, !(s in all_unid))
			end
			for p in ctx.param_syms
				push!(id_mask, !(p in all_unid))
			end
			# If everything flagged non-id, fall through to legacy path
			if any(id_mask)
				# 3) Build per-candidate id-only vector + robust per-axis scale
				kept_indices = findall(keep_mask)
				d_total = length(clamped_p0s[1])
				d_id = count(id_mask)
				id_positions = [j for j in 1:d_total if id_mask[j]]

				# Per-axis median and MAD on kept set
				med_vec = zeros(Float64, d_id)
				for (jj, j) in enumerate(id_positions)
					vals = [clamped_p0s[i][j] for i in kept_indices]
					med_vec[jj] = median(vals)
				end
				mad_vec = zeros(Float64, d_id)
				for (jj, j) in enumerate(id_positions)
					abs_dev = [abs(clamped_p0s[i][j] - med_vec[jj]) for i in kept_indices]
					mad_vec[jj] = median(abs_dev)
				end
				scale_vec = [max(abs(med_vec[jj]), mad_vec[jj], 1e-10) for jj in 1:d_id]

				# 4) Cluster kept candidates only, using L-inf normalized id-only dist
				cluster_reps = Int[]
				candidate_cluster = zeros(Int, n_candidates)
				for i in 1:n_candidates
					if !keep_mask[i]
						# Dropped by err filter; not assigned to any cluster
						continue
					end
					merged = false
					for (k, rep) in enumerate(cluster_reps)
						dist = 0.0
						for (jj, j) in enumerate(id_positions)
							a = (clamped_p0s[i][j] - med_vec[jj]) / scale_vec[jj]
							b = (clamped_p0s[rep][j] - med_vec[jj]) / scale_vec[jj]
							diff = abs(a - b)
							if diff > dist
								dist = diff
							end
							if dist >= opts.branch_cluster_eps
								break
							end
						end
						if dist < opts.branch_cluster_eps
							candidate_cluster[i] = k
							err_i = isnothing(candidates[i].err) ? Inf : candidates[i].err
							err_rep = isnothing(candidates[rep].err) ? Inf : candidates[rep].err
							if err_i < err_rep
								cluster_reps[k] = i
							end
							merged = true
							break
						end
					end
					if !merged
						push!(cluster_reps, i)
						candidate_cluster[i] = length(cluster_reps)
					end
				end

				return (
					cluster_reps = cluster_reps,
					candidate_cluster = candidate_cluster,
					clamped_p0s = clamped_p0s,
					use_bounds = use_bounds,
					cluster_threshold = opts.branch_cluster_eps,
				)
			end
		end
	end

	# Legacy code path (branch_detection=false, or fallthrough on degeneracy)
	cluster_reps = Int[]
	candidate_cluster = zeros(Int, n_candidates)

	for i in 1:n_candidates
		merged = false
		for (k, rep) in enumerate(cluster_reps)
			dist = zero(Float64)
			for j in eachindex(clamped_p0s[i])
				a = clamped_p0s[i][j]
				b = clamped_p0s[rep][j]
				scale = max(abs(a), abs(b), 1.0)
				dist = max(dist, abs(a - b) / scale)
			end
			if dist <= cluster_threshold
				candidate_cluster[i] = k
				err_i = isnothing(candidates[i].err) ? Inf : candidates[i].err
				err_rep = isnothing(candidates[rep].err) ? Inf : candidates[rep].err
				if err_i < err_rep
					cluster_reps[k] = i
				end
				merged = true
				break
			end
		end
		if !merged
			push!(cluster_reps, i)
			candidate_cluster[i] = length(cluster_reps)
		end
	end

	return (
		cluster_reps = cluster_reps,
		candidate_cluster = candidate_cluster,
		clamped_p0s = clamped_p0s,
		use_bounds = use_bounds,
		cluster_threshold = cluster_threshold,
	)
end

"""
	_polish_batch_from_context(ctx, candidates; opts) -> Vector{ParameterEstimationResult}

Polish all candidate solutions using a shared PolishContext. For each candidate,
extracts the p0 vector, calls `_polish_single_from_context`, and returns the
combined list of original + polished results.
"""
function _polish_batch_from_context(
	ctx::PolishContext,
	candidates::AbstractVector;
	opts::EstimationOptions = EstimationOptions(),
)
	batch_t0 = time()
	batch_stages = OrderedDict{Symbol, Float64}()
	residual_mode = is_residual_polish_method(opts.polish_method)
	# Build the legacy scalar optimizer instance only when actually needed; residual
	# methods route through `_polish_single_residual` and don't take an `optimizer` arg.
	optimizer = if residual_mode
		nothing
	else
		optimizer_type = get_polish_optimizer(opts.polish_method)
		optimizer_type()
	end
	maxiters = opts.polish_maxiters
	n_candidates = length(candidates)

	cluster_meta = _timed_detail_stage!(batch_stages, :cluster_metadata) do
		_polish_cluster_metadata(ctx, candidates; opts = opts)
	end
	cluster_reps = cluster_meta.cluster_reps
	clamped_p0s = cluster_meta.clamped_p0s

	n_unique = length(cluster_reps)
	if !opts.nooutput && n_unique < n_candidates
		println("Deduplicated $n_candidates candidates → $n_unique unique starting points for polish")
	end

	# Bounded-concurrency dispatch: spawning all cluster_reps simultaneously via
	# Threads.@spawn lets Julia oversubscribe — each polish does heavy ForwardDiff
	# Jacobian assembly (residual_count × n_unknowns ODE integrations under
	# Rodas5P/AutoVern9), so N polishes on T threads each run ~N/T-times slower
	# than serial. With polish_maxtime enforced (216e548), polishes hit the wall
	# before converging. Cap at `opts.polish_max_concurrency` (default = nthreads()).
	n_workers = min(max(opts.polish_max_concurrency, 1), n_unique)

	if !opts.nooutput
		n_threads = Threads.nthreads()
		println("Polishing $n_unique solutions (maxiters=$maxiters, nthreads=$n_threads, concurrency=$n_workers)...")
	end

	polish_start = time()
	print_lock = ReentrantLock()

	# Preallocate per-task result slots so we can collect in input order.
	task_results = [ParameterEstimationResult[] for _ in 1:n_unique]
	work_chan = Channel{Tuple{Int, Int}}(n_unique)
	for (i, rep_idx) in enumerate(cluster_reps)
		put!(work_chan, (i, rep_idx))
	end
	close(work_chan)

	function _polish_one(task_idx::Int, rep_idx::Int)
		t0 = time()
		candidate = candidates[rep_idx]
		try
			p0 = clamped_p0s[rep_idx]
			polished_result, opt_result = _polish_single_from_context(
				ctx, p0;
				optimizer = isnothing(optimizer) ? BFGS() : optimizer,
				polish_method = opts.polish_method,
				maxiters = maxiters,
				maxtime = opts.polish_maxtime,
				divergence_factor = opts.polish_divergence_factor,
				stagnation_window = opts.polish_stagnation_window,
				lso_delta = opts.polish_lso_delta,
				lso_x_tol = opts.polish_lso_x_tol,
				lso_f_tol = opts.polish_lso_f_tol,
				lso_g_tol = opts.polish_lso_g_tol,
				retain_internal_optimum = opts.compute_uncertainty,
			)
			dt = time() - t0
			n_iters = try; opt_result.original.iterations; catch err; _rethrow_if_interrupt(err); -1; end
			polished_result.unident_dict = deepcopy(candidate.unident_dict)
			polished_result.all_unidentifiable = copy(candidate.all_unidentifiable)
			polished_result.provenance = copy_provenance(
				candidate.provenance;
				pre_polish_error = candidate.err,
				post_polish_error = polished_result.err,
				polish_applied = true,
				polish_source_hc_idx = rep_idx,
			)
			set_result_lineage!(
				polished_result;
				primary_method = candidate.provenance.primary_method,
				interpolator_source = candidate.provenance.interpolator_source,
				rescue_path = candidate.provenance.rescue_path,
				source_shooting_index = candidate.provenance.source_shooting_index,
				source_candidate_index = candidate.provenance.source_candidate_index,
				pre_polish_error = candidate.err,
				post_polish_error = polished_result.err,
				polish_applied = true,
				notes = candidate.provenance.notes,
				source_type = candidate.provenance.source_type,
				multipoint_time_indices = candidate.provenance.multipoint_time_indices,
				multipoint_combo_index = candidate.provenance.multipoint_combo_index,
			)
			parent_identity = ensure_result_estimator_identity!(candidate)
			identity = set_result_estimator_identity!(polished_result;
				estimator_kind = :trajectory_polish,
				data_scope = :full_trajectory,
				time_indices = collect(eachindex(ctx.t_vector)),
				time_values = copy(ctx.t_vector),
				interpolator_source = candidate.provenance.interpolator_source,
				parent_candidate_ids = Int[parent_identity.candidate_id],
			)
			if opts.compute_uncertainty
				artifact = _run_ctx_try_uq_capture("trajectory-polish artifact") do
					internal_optimum = _polish_retained_internal_vector(
						ctx, polished_result, opt_result,
					)
					PolishUQArtifact(
						ctx, internal_optimum, _polish_raw_optimizer_result(opt_result),
						parent_identity.candidate_id,
						is_residual_polish_method(opts.polish_method) ? :residual_least_squares : :trajectory_sse,
					)
				end
				!isnothing(artifact) &&
					_run_ctx_register_artifact!(identity.candidate_id, artifact)
			end
			if !opts.nooutput
				err_before = isnothing(candidate.err) ? Inf : candidate.err
				err_after = isnothing(polished_result.err) ? Inf : polished_result.err
				lock(print_lock) do
					println("  Polish $task_idx/$n_unique (candidate $rep_idx): $(round(dt; digits=1))s, $(n_iters) iters, err $(round(err_before; sigdigits=3)) → $(round(err_after; sigdigits=3))")
				end
			end
			push!(task_results[task_idx], polished_result)
		catch e
			_rethrow_if_interrupt(e)
			dt = time() - t0
			@warn "Failed to polish solution $rep_idx ($(round(dt; digits=1))s): $e"
		end
		return nothing
	end

	_timed_detail_stage!(batch_stages, :worker_polish) do
		workers = [Threads.@spawn begin
			for (task_idx, rep_idx) in work_chan
				_polish_one(task_idx, rep_idx)
			end
		end for _ in 1:n_workers]
		foreach(wait, workers)
	end

	# Collect results: all original candidates (unpolished baselines) + polished results, in input order.
	polished_results = ParameterEstimationResult[]
	polished_only = ParameterEstimationResult[]
	_timed_detail_stage!(batch_stages, :collect_results) do
		for candidate in candidates
			push!(polished_results, candidate)
		end
		for slot in task_results
			append!(polished_results, slot)
			append!(polished_only, slot)
		end
	end

	if !opts.nooutput
		println("  Polish total: $(round(time() - polish_start; digits=1))s for $n_unique unique solutions (from $n_candidates candidates)")
	end
	_record_detailed_timing!((
		category = :polish_batch,
		context = _current_detailed_timing_context(),
		total_seconds = time() - batch_t0,
		stage_seconds = copy(batch_stages),
		candidate_count = n_candidates,
		unique_start_count = n_unique,
		worker_count = n_workers,
		thread_count = Threads.nthreads(),
		residual_mode = residual_mode,
		polished_only_count = length(polished_only),
		output_count = length(polished_results),
	))
	return polished_results
end

"""
	direct_optimization_parameter_estimation(PEP; opts) -> Vector{ParameterEstimationResult}

Perform parameter estimation via direct LBFGS optimization from a random initial guess.
Uses the shared PolishContext infrastructure for consistency with the polish path.
"""
function direct_optimization_parameter_estimation(PEP::ParameterEstimationProblem;
	opts::EstimationOptions = EstimationOptions())
	_validate_observation_options(PEP.data_sample, opts)
	_run_scopes_pending(opts) &&
		return _with_run_scopes(() -> direct_optimization_parameter_estimation(PEP; opts = opts), opts)
	if _run_ctx() === nothing
		value, _ = _with_run_context(() -> direct_optimization_parameter_estimation(PEP; opts = opts))
		return value
	end
	_run_ctx_begin_uq!(opts.compute_uncertainty)
	if opts.compute_uncertainty && isnothing(_run_ctx_noise_interpolants(:agp_uq))
		configured = resolve_interpolator_list(opts)
		agp_idx = findfirst(pair -> first(pair) == InterpolatorGPUncertainty, configured)
		if !isnothing(agp_idx)
			method, custom = configured[agp_idx]
			interp_func = get_interpolator_function(
				method, custom; s3_adapt_k = opts.s3_adapt_k,
			)
			try
				interpolants = create_interpolants(
					PEP.measured_quantities, PEP.data_sample,
					Float64.(PEP.data_sample["t"]), interp_func,
				)
				_run_ctx_register_noise_interpolants!(:agp_uq, interpolants)
			catch err
				_rethrow_if_interrupt(err)
				@warn "Configured AGPUQ observation-noise provider could not be retained for direct-optimizer UQ" exception = (err, catch_backtrace())
			end
		end
	end
	ctx = _build_polish_context(PEP; opts = opts)
	p_size = ctx.n_ic + ctx.n_param

	# Generate random initial guess. Optim 2's Fminbox raises ArgumentError when the
	# initial gradient is NaN, which happens whenever the random point is far enough out
	# that the ODE solve fails (loss = Inf → ForwardDiff returns zero gradient → mu0 = 0/0).
	# Default detection bounds are ±1e9 — uniform-in-bounds almost always blows up the ODE,
	# so we draw on a small unit-scale and clamp into bounds when present, then retry until
	# the loss is finite.
	function _draw_p0(scale)
		raw = scale .* randn(p_size)
		(isnothing(ctx.lb) || isnothing(ctx.ub)) ? raw : clamp.(raw, ctx.lb, ctx.ub)
	end
	p0 = _draw_p0(1.0)
	for attempt in 1:30
		loss0 = try
			ctx.optf.f(p0, nothing)
		catch err
			_rethrow_if_interrupt(err)
			Inf
		end
		isfinite(loss0) && break
		# Widen scale modestly each retry; bounds clamp keeps things sane.
		p0 = _draw_p0(1.0 * 1.3^attempt)
	end

	if !opts.nooutput
		println("Starting direct optimization with initial guess: ", p0)
	end

	final_result, opt_result = _polish_single_from_context(
		ctx, p0;
		optimizer = LBFGS(),
		maxiters = opts.opt_maxiters,
		maxtime = opts.polish_maxtime,
		divergence_factor = opts.polish_divergence_factor,
		stagnation_window = opts.polish_stagnation_window,
		retain_internal_optimum = opts.compute_uncertainty,
	)

	if !opts.nooutput
		println("Direct optimization finished with final loss: ", opt_result.objective)
		println("Found solution: ", merge(final_result.states, final_result.parameters))
	end
	final_result.provenance = ResultProvenance(
		primary_method = :direct_opt,
		rescue_path = :none,
		source_candidate_index = 1,
		pre_polish_error = nothing,
		post_polish_error = final_result.err,
		polish_applied = true,
	)
	identity = set_result_estimator_identity!(final_result;
		estimator_kind = :direct_optimization,
		data_scope = :full_trajectory,
		time_indices = collect(eachindex(ctx.t_vector)),
		time_values = copy(ctx.t_vector),
	)
	if opts.compute_uncertainty
		artifact = _run_ctx_try_uq_capture("direct-optimization artifact") do
			PolishUQArtifact(
				ctx, _polish_retained_internal_vector(ctx, final_result, opt_result),
				_polish_raw_optimizer_result(opt_result), 0,
				:trajectory_sse,
			)
		end
		!isnothing(artifact) && _run_ctx_register_artifact!(identity.candidate_id, artifact)
	end
	sync_result_contract!(final_result)

	return [final_result]
end
