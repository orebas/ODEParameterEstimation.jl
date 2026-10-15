"""
SI.jl Integration Module for ODEParameterEstimation

This module provides functions to use StructuralIdentifiability.jl 
for equation system construction instead of iterative scanning.
"""


"""
	apply_prefixed_params_to_model(ode, measured_quantities, pre_fixed_params)

Apply pre-fixed parameter substitutions to the model before SIAN analysis.
This is used for structural representative fixing - when we fix one parameter
or initial condition and want to re-run the full SIAN analysis with that
quantity treated as a constant.

# Arguments
- `ode`: The ODE model (OrderedODESystem or ODESystem)
- `measured_quantities`: Vector of measurement equations
- `pre_fixed_params`: OrderedDict mapping parameter symbols to their fixed values

# Returns
- Modified ODE with substituted parameters
- Modified measured_quantities with substituted parameters
"""
function apply_prefixed_params_to_model(ode, measured_quantities, pre_fixed_params::OrderedDict)
	# Get the underlying system
	model = isa(ode, OrderedODESystem) ? ode.system : ode

	# Build substitution dictionary from MTK parameter symbols to values
	t = ModelingToolkit.get_iv(model)
	original_params = ModelingToolkit.parameters(model)
	original_states = ModelingToolkit.unknowns(model)

	# Create substitution dict matching parameter symbols
	subst_dict = Dict()
	params_to_remove = Set()
	for (param_sym, fix_value) in pre_fixed_params
		# Find matching parameter in model
		param_str = string(param_sym)
		for p in original_params
			p_str = string(p)
			# Handle both "param" and "param(t)" forms
			p_base = replace(p_str, "(t)" => "")
			if p_base == param_str || p_str == param_str
				subst_dict[p] = fix_value
				push!(params_to_remove, p)
				@info "[PRE-FIX] Substituting $p => $fix_value"
				break
			end
		end
	end

	if isempty(subst_dict)
		@warn "[PRE-FIX] No matching parameters found for substitution"
		return ode, measured_quantities
	end

	# Get the remaining parameters (those not being fixed)
	remaining_params = [p for p in original_params if !(p in params_to_remove)]

	# Substitute in the ODE equations
	original_eqs = ModelingToolkit.equations(model)
	new_eqs = [Symbolics.substitute(eq.lhs, subst_dict) ~ Symbolics.substitute(eq.rhs, subst_dict)
	           for eq in original_eqs]

	# Substitute in measured quantities
	new_measured = [Symbolics.substitute(mq.lhs, subst_dict) ~ Symbolics.substitute(mq.rhs, subst_dict)
	                for mq in measured_quantities]

	# Create new ODESystem with remaining parameters
	# Use a unique name to avoid conflicts
	new_name = Symbol(string(ModelingToolkit.nameof(model)) * "_prefixed")
	new_system = ModelingToolkit.ODESystem(
		new_eqs, t, original_states, remaining_params;
		name = new_name
	)

	# If original was OrderedODESystem, wrap the new system
	if isa(ode, OrderedODESystem)
		# Keep original state ordering, update parameter ordering
		new_ode = OrderedODESystem(new_system, ode.original_states, remaining_params)
	else
		new_ode = new_system
	end

	@info "[PRE-FIX] Created modified model with $(length(remaining_params)) parameters (removed $(length(params_to_remove)))"

	return new_ode, new_measured
end

"""
	convert_to_si_ode(ode, measured_quantities::Vector{ModelingToolkit.Equation}, inputs::Vector{Num} = Vector{Num}())

Convert ModelingToolkit ODESystem to StructuralIdentifiability.jl ODE format.
Based on ParameterEstimation.jl's preprocess_ode function.

Returns:
- si_ode: StructuralIdentifiability.ODE object
- input_symbols: Original symbolic variables
- gens: Nemo polynomial ring generators
"""
function convert_to_si_ode(
	ode,  # Will be OrderedODESystem but can't reference type here
	measured_quantities::Vector{ModelingToolkit.Equation},
	inputs = [],  # Vector{Num}
)
	@info "Converting ODESystem to SI.jl format"

	model = ode.system

	# Filter out output equations from the ODE system
	diff_eqs = filter(eq -> !(ModelingToolkit.isoutput(eq.lhs)),
		ModelingToolkit.equations(model))

	# Get symbolic components
	y_functions = [each.lhs for each in measured_quantities]
	state_vars = ModelingToolkit.unknowns(model)
	params = ModelingToolkit.parameters(model)

	# Get time variable
	t = ModelingToolkit.get_iv(model)

	# Collect all parameters including from measured quantities
	params_from_measured = ModelingToolkit.parameters(
		ModelingToolkit.ODESystem(measured_quantities, t, name = :DataSeries),
	)
	params = union(params, params_from_measured)

	# Create input symbols array
	input_symbols = vcat(state_vars, y_functions, inputs, params)

	# Create generator strings (remove (t) from variables)
	generators = string.(input_symbols)
	generators = map(g -> replace(g, "(t)" => ""), generators)

	# Create Nemo polynomial ring
	R, gens_ = Nemo.polynomial_ring(Nemo.QQ, generators)

	# Create dictionaries for state equations and output equations
	state_eqn_dict = Dict{Nemo.QQMPolyRingElem,
		Union{Nemo.QQMPolyRingElem, Nemo.Generic.FracFieldElem{Nemo.QQMPolyRingElem}}}()

	out_eqn_dict = Dict{Nemo.QQMPolyRingElem,
		Union{Nemo.QQMPolyRingElem, Nemo.Generic.FracFieldElem{Nemo.QQMPolyRingElem}}}()

	# Convert state equations
	subs_dict = Dict(input_symbols .=> gens_)
	for i in eachindex(diff_eqs)
		lhs_nemo = eval_at_nemo(Symbolics.value(state_vars[i]), subs_dict)
		# Route through eval_at_nemo to robustly handle constants
		rhs_nemo = eval_at_nemo(Symbolics.value(diff_eqs[i].rhs), subs_dict)
		rhs_nemo = coerce_nemo_ring_value(R, rhs_nemo; context = "state equation")
		state_eqn_dict[lhs_nemo] = rhs_nemo
	end

	# Convert output equations
	for i in 1:length(measured_quantities)
		lhs_nemo = eval_at_nemo(Symbolics.value(y_functions[i]), subs_dict)
		rhs_nemo = eval_at_nemo(Symbolics.value(measured_quantities[i].rhs), subs_dict)
		rhs_nemo = coerce_nemo_ring_value(R, rhs_nemo; context = "output equation")
		out_eqn_dict[lhs_nemo] = rhs_nemo
	end

	# Convert inputs
	inputs_ = [eval_at_nemo(Symbolics.value(each), subs_dict) for each in inputs]
	if isempty(inputs_)
		inputs_ = Vector{Nemo.QQMPolyRingElem}()
	end

	# Create SI.jl ODE object
	si_ode = ODE{Nemo.QQMPolyRingElem}(state_eqn_dict, out_eqn_dict, inputs_)

	return si_ode, input_symbols, gens_
end

function coerce_nemo_ring_value(R, value; context = "expression")
	if !(value isa Number)
		return value
	end
	if value isa AbstractFloat
		try
			return R(value)
		catch e
			@debug "Nemo ring coercion failed, retrying with rationalize" context = context exception = e
			return R(rationalize(value))
		end
	end
	return R(value)
end

function _record_si_symbolic_placeholder!(stats::Dict{Symbol, Vector{String}}, category::Symbol, var_name::AbstractString)
	category = canonicalize_si_placeholder_category(category)
	names = get!(stats, category, String[])
	length(names) < 5 && push!(names, String(var_name))
	return nothing
end

function _create_si_symbolic_placeholder!(
	extended_map::Dict,
	var,
	var_name::AbstractString,
	stats::Dict{Symbol, Vector{String}},
	category::Symbol;
	infolevel::Integer = 0,
	message_level::Symbol = :debug,
	message::AbstractString = "",
)
	category = canonicalize_si_placeholder_category(category)
	sym_var = Symbolics.variable(Symbol(var_name))
	extended_map[var] = sym_var
	_record_si_symbolic_placeholder!(stats, category, var_name)
	if infolevel > 0
		if message_level == :warn
			@warn message var_name = var_name symbolic = sym_var
		elseif message_level == :info
			@info message var_name = var_name symbolic = sym_var
		else
			@debug message var_name = var_name symbolic = sym_var
		end
	end
	return sym_var
end

is_sian_auxiliary_variable_name(var_name::AbstractString) = var_name == "z_aux"

strip_time_suffix(name::AbstractString) = replace(String(name), "(t)" => "")

function extract_symbol_base_names(expr)
	return Set(strip_time_suffix(string(v)) for v in Symbolics.get_variables(expr))
end

function _sample_collection(values; limit = 5)
	items = collect(values)
	length(items) <= limit && return items
	return items[1:limit]
end

function _log_si_polynomial_system_summary(;
	infolevel::Integer,
	initial_equation_count::Int,
	initial_variable_count::Int,
	theta_l_count::Int,
	phantom_vars,
	filtered_variable_count::Int,
	selected_equation_indices,
	dropped_equation_indices,
	reduced_equation_count::Int,
	reduced_variable_count::Int,
)
	infolevel > 0 || return nothing
	@info "[SI-TEMPLATE] Polynomial construction summary" raw_equations = initial_equation_count raw_ring_variables = initial_variable_count locally_identifiable_count = theta_l_count filtered_ring_variables = filtered_variable_count selected_equations = length(selected_equation_indices) dropped_equations = length(dropped_equation_indices)
	if !isempty(phantom_vars)
		@info "[SI-TEMPLATE] Filtered phantom ring variables that never appear in the polynomial system" count = length(phantom_vars) sample = _sample_collection(phantom_vars)
	end
	if !isempty(dropped_equation_indices)
		@info "[SI-TEMPLATE] Rank trimming removed polynomial equations" dropped_indices = dropped_equation_indices
	end
	if reduced_equation_count == reduced_variable_count
		@info "[SI-TEMPLATE] Reduced polynomial system is square before data/support-variable accounting" equations = reduced_equation_count ring_variables = reduced_variable_count
	else
		@info "[SI-TEMPLATE] Reduced polynomial system is not square before data/support-variable accounting" equations = reduced_equation_count ring_variables = reduced_variable_count
	end
	return nothing
end

function build_si_role_context(ode, measured_quantities)
	model = isa(ode, OrderedODESystem) ? ode.system : ode
	state_names = Set(strip_time_suffix(string(s)) for s in ModelingToolkit.unknowns(model))
	param_names = Set(strip_time_suffix(string(p)) for p in ModelingToolkit.parameters(model))
	t = ModelingToolkit.get_iv(model)
	params_from_measured = ModelingToolkit.parameters(
		ModelingToolkit.ODESystem(measured_quantities, t, name = :SIRoleContext),
	)
	union!(param_names, Set(strip_time_suffix(string(p)) for p in params_from_measured))
	measured_rhs_names = Set{String}()
	for mq in measured_quantities
		union!(measured_rhs_names, extract_symbol_base_names(mq.rhs))
	end
	return (
		state_names = state_names,
		param_names = param_names,
		measured_rhs_names = measured_rhs_names,
	)
end

max_required_observable_derivative_order(derivative_dict) =
	isempty(derivative_dict) ? 0 : maximum(Int(v) for v in values(derivative_dict))

function ensure_si_template_dd_support(ode, measured_quantities, DD, derivative_dict)
	required_order = max_required_observable_derivative_order(derivative_dict)
	current_order = isnothing(DD) ? -1 : length(DD.obs_lhs) - 1
	if current_order >= required_order
		return DD
	end

	model = isa(ode, OrderedODESystem) ? ode.system : ode
	target_deriv_level = max(required_order + 1, 2)
	@info "[SI-MAP] Extending DerivativeData support for SI template" current_order = current_order required_order = required_order
	return populate_derivatives(model, measured_quantities, target_deriv_level, OrderedDict();
		include_cleared = false)
end

function build_si_observable_index_map(measured_quantities)
	obs_name_to_idx = Dict{String, Int}()
	for (idx, mq) in enumerate(measured_quantities)
		lhs_val = Symbolics.value(mq.lhs)
		if Symbolics.iscall(lhs_val)
			lhs_name = string(Symbolics.operation(lhs_val))
			obs_name_to_idx[lhs_name] = idx
		end
	end
	return obs_name_to_idx
end

function resolve_si_observable_index(base_name::AbstractString, obs_name_to_idx::Dict{String, Int})
	obs_idx = get(obs_name_to_idx, base_name, nothing)
	if isnothing(obs_idx)
		m = match(r"^y(\d+)$", base_name)
		if !isnothing(m)
			obs_idx = parse(Int, m.captures[1])
		end
	end
	return obs_idx
end

function classify_si_ring_variable(
	var_name::AbstractString,
	obs_name_to_idx::Dict{String, Int},
	DD,
	role_context = nothing,
)
	if is_sian_auxiliary_variable_name(var_name)
		return (
			category = :sian_auxiliary,
			message_level = :info,
			message = "Mapped known SIAN auxiliary variable",
		)
	end

	parsed = parse_derivative_variable_name(var_name)
	if isnothing(parsed)
		return (
			category = :true_unknown_variable,
			message_level = :debug,
			message = "Mapped truly unknown SI variable to symbolic placeholder",
		)
	end

	base_name, deriv_order = parsed
	if isnothing(DD)
		return (
			category = :no_dd_derivative,
			message_level = :info,
			message = "No DD structure available; using symbolic placeholder",
		)
	end

	obs_idx = resolve_si_observable_index(base_name, obs_name_to_idx)
	if isnothing(obs_idx)
		if !isnothing(role_context)
			if startswith(base_name, "_trfn_")
				return (
					category = :transformed_analytic_support,
					message_level = :debug,
					message = "Mapped analytical transformed support variable to symbolic support variable",
				)
			end
			if base_name in role_context.param_names && deriv_order == 0
				return (
					category = :parameter_or_ic_symbol,
					message_level = :debug,
					message = "Mapped parameter/IC SI support symbol to symbolic support variable",
				)
			end
			if base_name in role_context.measured_rhs_names
				return (
					category = :measured_rhs_jet,
					message_level = :debug,
					message = "Mapped measured-RHS jet symbol to symbolic support variable",
				)
			end
			if base_name in role_context.state_names
				return (
					category = :state_jet,
					message_level = :debug,
					message = "Mapped state jet symbol to symbolic support variable",
				)
			end
		end
		return (
			category = :support_jet,
			message_level = :debug,
			message = "Mapped generic SI support jet symbol to symbolic support variable",
		)
	end

	if deriv_order == 0
		if obs_idx <= length(DD.obs_lhs[1])
			return (
				category = :observable_derivative,
				symbol = DD.obs_lhs[1][obs_idx],
				message_level = :debug,
				message = "Mapped observable to DD structure",
			)
		end
		return (
			category = :dd_observable_index_oob,
			message_level = :warn,
			message = "Observable index $obs_idx out of bounds for DD structure; using symbolic support variable",
		)
	end

	if deriv_order + 1 <= length(DD.obs_lhs) && obs_idx <= length(DD.obs_lhs[deriv_order+1])
		return (
			category = :observable_derivative,
			symbol = DD.obs_lhs[deriv_order+1][obs_idx],
			message_level = :debug,
			message = "Mapped observable derivative to DD structure",
		)
	end

	return (
		category = :observable_derivative_overflow,
		message_level = :warn,
		message = "Observable derivative exceeds DD support (order=$deriv_order); using symbolic support variable",
	)
end

function resolve_si_ring_variable(
	var,
	base_map::Dict,
	obs_name_to_idx::Dict{String, Int},
	DD;
	role_context = nothing,
	infolevel::Integer = 0,
	stats = Dict{Symbol, Vector{String}}(),
)
	haskey(base_map, var) && return (
		category = :model_symbol,
		symbol = base_map[var],
	)

	var_name = string(var)
	classification = classify_si_ring_variable(var_name, obs_name_to_idx, DD, role_context)
	category = classification.category

	if category == :observable_derivative
		if infolevel > 0
			@debug classification.message var_name = var_name symbolic = classification.symbol
		end
		return (
			category = category,
			symbol = classification.symbol,
		)
	end

	sym_var = _create_si_symbolic_placeholder!(
		base_map,
		var,
		var_name,
		stats,
		category;
		infolevel = infolevel,
		message_level = classification.message_level,
		message = classification.message,
	)

	return (
		category = category,
		symbol = sym_var,
	)
end

function build_extended_si_variable_map(
	all_vars,
	base_map::Dict,
	measured_quantities,
	DD;
	role_context = nothing,
	infolevel::Integer = 0,
)
	extended_map = Dict{Nemo.QQMPolyRingElem, Any}()
	support_var_stats = Dict{Symbol, Vector{String}}()

	for (k, v) in base_map
		extended_map[k] = v
	end

	obs_name_to_idx = build_si_observable_index_map(measured_quantities)

	for var in all_vars
		haskey(extended_map, var) && continue
		resolution = resolve_si_ring_variable(
			var,
			extended_map,
			obs_name_to_idx,
			DD;
			role_context = role_context,
			infolevel = infolevel,
			stats = support_var_stats,
		)
		extended_map[var] = resolution.symbol
	end

	if !isempty(support_var_stats)
		support_var_counts = Dict(k => length(v) for (k, v) in support_var_stats)
		@info "[SI-MAP] Classified symbolic SI support variables while converting SIAN output" counts = support_var_counts samples = support_var_stats
	end

	return extended_map, support_var_stats
end

function si_role_summary_from_stats(stats::Dict{Symbol, Vector{String}})
	counts = Dict(category => length(names) for (category, names) in stats)
	suspicious_categories = Dict(
		category => counts[category]
		for category in keys(counts)
		if category in (:true_unknown_variable, :late_map_miss)
	)
	auxiliary_variables = get(stats, :sian_auxiliary, String[])
	return (
		counts = counts,
		samples = Dict(stats),
		suspicious_categories = suspicious_categories,
		auxiliary_variables = copy(auxiliary_variables),
	)
end

function collect_used_nemo_variables(polys)
	used_vars = OrderedSet{Any}()
	for poly in polys
		for var in Nemo.vars(poly)
			push!(used_vars, var)
		end
	end
	return collect(used_vars)
end

"""
	algebraic_independence(Et::Vector{Nemo.QQMPolyRingElem},
						   indets::Vector{Nemo.QQMPolyRingElem},
						   vals)

Returns the indices of the equations in Et to be used for polynomial solving
and the variables that form a transcendence basis.

# Arguments
- `Et::Vector{Nemo.QQMPolyRingElem}`: The equations to be solved (must come from identifiability check).
- `indets::Vector{Nemo.QQMPolyRingElem}`: The indeterminates.
- `vals::Vector{Nemo.QQMPolyRingElem}`: The values of the indeterminates sampled by identifiability algorithm.
"""
function algebraic_independence(Et::Vector{Nemo.QQMPolyRingElem},
	indets::Vector{Nemo.QQMPolyRingElem},
	vals)
	@info "[DEBUG-ALG-INDEP] Input: $(length(Et)) equations, $(length(indets)) indeterminates"

	pivots = Vector{Nemo.QQMPolyRingElem}()
	Jacobian = SIANBackend.jacobi_matrix(Et, indets, vals)

	@info "[DEBUG-ALG-INDEP] Jacobian size: $(size(Jacobian))"
	@info "[DEBUG-ALG-INDEP] Jacobian rank: $(Nemo.rank(Jacobian))"

	U = Nemo.lu(Jacobian)[end]
	#find pivot columns in u
	for row_idx in 1:size(U, 1)
		row = U[row_idx, :]
		if !all(row .== 0)
			pivot_col = findfirst(row .!= 0)
			push!(pivots, indets[pivot_col])
		end
	end

	@info "[DEBUG-ALG-INDEP] Found $(length(pivots)) pivot columns"

	current_idx = 1
	output_rows = Jacobian[[current_idx], :]
	current_rank = 1
	output_ids = [1]
	for current_idx in 2:length(Et)
		current = vcat(output_rows, Jacobian[[current_idx], :])
		if Nemo.rank(current) > current_rank
			output_rows = current
			push!(output_ids, current_idx)
			current_rank += 1
		end
	end

	@info "[DEBUG-ALG-INDEP] Selected $(length(output_ids)) equations (current_rank=$current_rank)"
	@info "[DEBUG-ALG-INDEP] Returning: output_ids=$(output_ids), alg_indep has $(length(setdiff(indets, pivots))) vars"

	return output_ids, setdiff(indets, pivots)
end

"""
	eval_at_nemo(expr, subs_dict)

Evaluate a Symbolics expression in the Nemo polynomial ring.
Delegates to StructuralIdentifiability's eval_at_nemo function.
"""
function eval_at_nemo(expr, subs_dict)
	# Use StructuralIdentifiability's implementation
	return StructuralIdentifiability.eval_at_nemo(expr, subs_dict)
end

"""Run-local structural analysis, independent of data and polynomial representative substitutions.

`coordinate_basis` contains free parameter or state-at-anchor coordinates. A
locally identifiable quantity can still have finitely many branches; it must
not be fixed merely because global identifiability has not been established.
`identifiable_functions === nothing` means those functions were not computed.
"""
struct SIStructuralAnalysis{M, F}
	model::M
	observations::Vector{ModelingToolkit.Equation}
	strategy::Symbol
	probability::Float64
	classification::OrderedDict{String, Symbol}
	coordinate_basis::Vector{String}
	identifiable_functions::F
	timing::OrderedDict{Symbol, Float64}
end

"""
	analyze_si_structure(ode, measured_quantities, si_ode; strategy, p, infolevel)

# Arguments
- `ode`, `measured_quantities`: Original model and observations, retained to guard reuse.
- `si_ode`: Converted SI model; parameters and free initial states are assessed together.
- `strategy`: `:local_basis` or the comparison method `:identifiable_functions`.
- `p`: Requested SI probability. The local method preserves the full SI call's
  local-stage budget, `1 - (1 - p)/10`.

# Returns
A `SIStructuralAnalysis` reusable for builds of this same model. State basis
coordinates are substituted only as jet-0 values in the polynomial template,
never as constant state trajectories in the ODE.
"""
function analyze_si_structure(ode, measured_quantities, si_ode;
	strategy::Symbol = _OPT_STRUCT_DEFAULTS.si_fix_strategy,
	p::Float64 = 0.99, infolevel::Int = 0)
	strategy in (:local_basis, :identifiable_functions) ||
		throw(ArgumentError("Unknown si_fix_strategy: $strategy"))
	quantities = vcat(si_ode.parameters, si_ode.x_vars)
	classification = OrderedDict{String, Symbol}()
	basis = Nemo.QQMPolyRingElem[]
	timing = OrderedDict{Symbol, Float64}()
	loglevel = infolevel > 0 ? Logging.Info : Logging.Warn
	if strategy == :local_basis
		timing[:assess_local_identifiability] = @elapsed result = StructuralIdentifiability.assess_local_identifiability(
			si_ode; funcs_to_check = quantities, type = :SE, trbasis = basis,
			prob_threshold = 1 - (1 - p) * 0.1, loglevel = loglevel,
		)
		for q in quantities
			classification[string(q)] = result[q] ? :locally_identifiable : :nonidentifiable
		end
		all(q -> haskey(result, q) && !result[q], basis) ||
			error("SI returned a coordinate basis outside the nonidentifiable quantities")
		identifiable_functions = nothing
	else
		timing[:assess_identifiability] = @elapsed result = StructuralIdentifiability.assess_identifiability(
			si_ode; funcs_to_check = quantities, prob_threshold = p, loglevel = loglevel,
		)
		for q in quantities
			classification[string(q)] = result[q]
		end
		timing[:find_identifiable_functions] = @elapsed identifiable_functions =
			isempty(si_ode.parameters) ? Any[] : find_identifiable_functions(si_ode)
	end
	return SIStructuralAnalysis(ode.system, copy(measured_quantities), strategy, p,
		classification, string.(basis), identifiable_functions, timing)
end

function validate_si_analysis_reuse(analysis::SIStructuralAnalysis, ode, measured_quantities, strategy, p)
	analysis.model === ode.system && isequal(analysis.observations, measured_quantities) &&
		analysis.strategy == strategy && analysis.probability == p ||
		throw(ArgumentError("Structural analysis can only be reused for the same model, observations, SI strategy, and probability"))
	return analysis
end

"""
	_substitute_si_template_polynomials(polynomials, pre_fixed_params)

Apply structural representative assignments in the exact polynomial ring.

# Arguments
- `polynomials`: SIAN template polynomials, all in the same rational ring.
- `pre_fixed_params`: Fixed parameter or anchor-state values. Names match
  exactly first, then with the `_0` suffix; higher state jets remain free.

# Returns
Polynomials in the original order, including any resulting zeros. Inputs are
not mutated. Floating-point assignments retain their exact binary value,
matching the multiplicity calculation's conversion to rational coefficients.
"""
function _substitute_si_template_polynomials(
	polynomials::AbstractVector{Nemo.QQMPolyRingElem},
	pre_fixed_params::OrderedDict,
)
	(isempty(polynomials) || isempty(pre_fixed_params)) && return polynomials
	by_name = Dict(string(v) => v for v in collect_used_nemo_variables(polynomials))
	fixed = OrderedDict{Nemo.QQMPolyRingElem, Nemo.QQFieldElem}()
	for (name, value) in pre_fixed_params
		key = string(name)
		variable = get(by_name, key, get(by_name, key * "_0", nothing))
		if isnothing(variable)
			@warn "[PRE-FIX] Variable '$key' (or '$(key)_0') not found in template polynomials"
			continue
		end
		exact_value = value isa AbstractFloat ? Rational{BigInt}(value) : value
		fixed[variable] = Nemo.QQ(exact_value)
		@info "[PRE-FIX] Will substitute $variable => $value in Nemo"
	end
	isempty(fixed) && return polynomials
	fixed_vars, fixed_values = collect(keys(fixed)), collect(values(fixed))
	return [Nemo.evaluate(poly, fixed_vars, fixed_values) for poly in polynomials]
end

"""
	get_si_equation_system(ode, measured_quantities::Vector{ModelingToolkit.Equation}, data_sample::OrderedDict; DD=nothing, kwargs...)

Get polynomial equation system from StructuralIdentifiability.jl.
This replaces the iterative equation construction in ODEPE.

Returns:
- equations: Polynomial equations in Symbolics format (template)
- derivative_vars: Dictionary mapping derivative variables to their orders
- unidentifiable: Set of unidentifiable parameters
"""
function get_si_equation_system(
	ode,  # Will be OrderedODESystem
	measured_quantities::Vector{ModelingToolkit.Equation},
	data_sample::AbstractDict;
	DD = nothing,  # DerivativeData structure for mapping
	p = 0.99,
	p_mod = 0,
	infolevel = 0,
	pre_fixed_params::OrderedDict = OrderedDict(),  # Parameters already fixed in previous iterations
	compute_multiplicity = true,  # false on detection passes: skip the algebraic-multiplicity Groebner step
	si_fix_strategy::Symbol = _OPT_STRUCT_DEFAULTS.si_fix_strategy,
	structural_analysis::Union{Nothing, SIStructuralAnalysis} = nothing,
	kwargs...,
)
	@info "Getting equation system from StructuralIdentifiability.jl"
	equation_builder_timing = OrderedDict{Symbol, Float64}()
	if !isempty(pre_fixed_params)
		@info "[PRE-FIX] Will apply $(length(pre_fixed_params)) representative assignments to multiplicity and template equations"
	end

	# Convert to SI.jl format (pre-fixed params are applied after SIAN, at the polynomial level)
	_t_convert_to_si_ode_start = time()
	si_ode, symbol_map, gens = convert_to_si_ode(ode, measured_quantities)
	equation_builder_timing[:convert_to_si_ode] = time() - _t_convert_to_si_ode_start
	# Only the polynomial template changes between the detection and fixed
	# passes. A parameter-fixed model used by state rescue needs fresh analysis.
	if !isnothing(structural_analysis)
		validate_si_analysis_reuse(structural_analysis, ode, measured_quantities, si_fix_strategy, p)
	end

	# Get parameters for identifiability analysis 
	# Use the parameters field directly instead of SIANBackend.get_parameters to avoid dependency issues
	params_to_assess = vcat(si_ode.parameters, si_ode.x_vars)

	# Create mapping from Nemo to MTK types
	nemo2mtk = Dict(gens .=> symbol_map)

	# Explicit MTK → jet-0 template-variable map (maintainability campaign Phase B).
	# `symbol_map` (MTK symbols) and `gens` (clean ring names) are aligned by
	# construction in convert_to_si_ode, and the SI template names every jet as
	# "<clean name>_<order>" — so the jet-0 Symbolics variable for each state/param
	# is recoverable with NO string parsing of the MTK side. Threaded to result
	# processing via si_template_metadata so lookup_value's exact-match stage hits
	# directly instead of name heuristics (which collided k_1/k_2-style names;
	# see commit 4b9e14b). Convention verified empirically across model classes
	# incl. underscore-digit, transcendental (_trfn_), and partially identifiable.
	template_var_map = OrderedDict{Num, Num}()
	for (i, mtk_sym) in enumerate(symbol_map)
		template_var_map[Num(mtk_sym)] = Symbolics.variable(Symbol(string(gens[i]) * "_0"))
	end

	# Get polynomial system using SIAN
	@info "Getting polynomial system from SIAN"
	_t_get_polynomial_system_start = time()
	result = get_polynomial_system_from_sian(
		si_ode,
		params_to_assess;
		p = p,
		infolevel = infolevel,
		compute_multiplicity = compute_multiplicity,
		pre_fixed_params = pre_fixed_params,
	)
	equation_builder_timing[:get_polynomial_system_from_sian] = time() - _t_get_polynomial_system_start

	# Extract the polynomial system and derivative info
	poly_system = result["polynomial_system"]
	full_poly_system = get(result, "full_polynomial_system", poly_system)
	y_derivative_dict = result["Y_eq"]
	_t_ensure_dd_support_start = time()
	DD = ensure_si_template_dd_support(ode, measured_quantities, DD, y_derivative_dict)
	equation_builder_timing[:ensure_dd_support] = time() - _t_ensure_dd_support_start

	analysis_reused = !isnothing(structural_analysis)
	@info "Checking identifiability" strategy = si_fix_strategy reused = analysis_reused
	equation_builder_timing[:structural_analysis] = @elapsed analysis = analysis_reused ? structural_analysis :
		analyze_si_structure(ode, measured_quantities, si_ode; strategy = si_fix_strategy, p = p, infolevel = infolevel)
	if !analysis_reused
		merge!(equation_builder_timing, analysis.timing)
	end
	@info "[SI-STRUCTURAL] Analysis timing" reused = analysis_reused seconds = analysis.timing
	unidentifiable = Set(q for q in params_to_assess if analysis.classification[string(q)] == :nonidentifiable)
	identifiable_funcs = analysis.identifiable_functions
	@info "[SI-STRUCTURAL] SIAN/SI template summary" template_equation_count = length(poly_system) derivative_symbol_count = length(y_derivative_dict) max_derivative_order = (isempty(y_derivative_dict) ? 0 : maximum(values(y_derivative_dict))) structural_unidentifiable_count = length(unidentifiable) identifiable_function_count = (isnothing(identifiable_funcs) ? nothing : length(identifiable_funcs)) coordinate_basis = analysis.coordinate_basis
	if infolevel > 0
		@info "[SI-STRUCTURAL] Structural unidentifiable variables from SI" variables = unidentifiable
		@info "[SI-STRUCTURAL] Observable derivative support requested by SI template" derivative_orders = y_derivative_dict
		@info "[SI-STRUCTURAL] Identifiable functions returned by SI" functions = identifiable_funcs
	end

	# Specialize before converting to Symbolics: collecting polynomial
	# coefficients suffices to detect zero, without general-purpose rewriting.
	# Work on the full list once, then take the already selected rows in order.
	selected_equation_indices = get(result, "selected_equation_indices", collect(1:length(poly_system)))
	dropped_equation_indices = get(result, "dropped_equation_indices", Int[])
	zero_equation_indices = Int[]
	_t_prefixed_substitutions_start = time()
	if !isempty(pre_fixed_params)
		full_poly_system = _substitute_si_template_polynomials(full_poly_system, pre_fixed_params)
		zero_equation_indices = findall(iszero, full_poly_system)
		selected_equation_indices = filter(i -> !iszero(full_poly_system[i]), selected_equation_indices)
		poly_system = full_poly_system[selected_equation_indices]
		full_poly_system = filter(!iszero, full_poly_system)
		dropped_equation_indices = sort!(union(dropped_equation_indices, zero_equation_indices))
		@info "[PRE-FIX] Exact polynomial substitution complete" selected_equations = length(poly_system) removed_zero_equations = zero_equation_indices

		fixed_names = Set(string.(keys(pre_fixed_params)))
		unidentifiable = Set(u for u in unidentifiable if
			!(string(u) in fixed_names || string(u) * "_0" in fixed_names))
	end
	equation_builder_timing[:prefixed_substitutions] = time() - _t_prefixed_substitutions_start

	# Build comprehensive variable mapping including derivatives
	# SIAN uses variables like y1_0, y1_1, y1_2 for derivatives
	# We need to map these to our DD structure when available
	si_variable_role_summary = si_role_summary_from_stats(Dict{Symbol, Vector{String}}())

	# Build mappings only for variables actually used in the template polynomials.
	# SIAN's parent ring can contain a much larger jet-variable universe than the
	# final polynomial system needs, and mapping the whole ring inflates support-role
	# traffic with irrelevant high-order observable derivatives.
	_t_variable_role_mapping_start = time()
	if !isempty(poly_system)
		used_template_vars = collect_used_nemo_variables(poly_system)
		role_context = build_si_role_context(ode, measured_quantities)

		nemo2mtk, si_variable_role_stats = build_extended_si_variable_map(
			used_template_vars,
			nemo2mtk,
			measured_quantities,
			DD;
			role_context = role_context,
			infolevel = infolevel,
		)
		si_variable_role_summary = si_role_summary_from_stats(si_variable_role_stats)
		if !isempty(si_variable_role_summary.suspicious_categories)
			@warn "[SI-MAP] Suspicious SI variable-role categories encountered" counts = si_variable_role_summary.suspicious_categories samples = Dict(k => si_variable_role_summary.samples[k] for k in keys(si_variable_role_summary.suspicious_categories))
		end
		if infolevel > 0
			R = parent(poly_system[1])
			all_ring_vars = Nemo.gens(R)
			@info "[SI-MAP] Template uses $(length(used_template_vars)) of $(length(all_ring_vars)) SIAN ring variables"
		end
	end
	equation_builder_timing[:variable_role_mapping] = time() - _t_variable_role_mapping_start

	# Convert polynomial systems to Symbolics format
	full_template_equations = []
	template_equations = []

	if infolevel > 0
		@info "Converting $(length(poly_system)) polynomials from Nemo to Symbolics"
	end

	_t_nemo_to_symbolics_conversion_start = time()
	for poly in full_poly_system
		poly_sym = nemo_to_symbolics(poly, nemo2mtk)
		push!(full_template_equations, poly_sym)
	end

	for poly in poly_system
		# Convert the polynomial to Symbolics
		poly_sym = nemo_to_symbolics(poly, nemo2mtk)

		if infolevel > 0
			@debug "Converted polynomial: $(typeof(poly)) -> $(typeof(poly_sym))"
		end

		push!(template_equations, poly_sym)
	end
	equation_builder_timing[:nemo_to_symbolics_conversion] = time() - _t_nemo_to_symbolics_conversion_start

	# Debug: Final check
	if infolevel > 0 && !isempty(template_equations)
		@debug "Final template_equations[1] type: $(typeof(template_equations[1]))"
	end

	si_template_metadata = (
		selected_equation_indices = selected_equation_indices,
		dropped_equation_indices = dropped_equation_indices,
		zero_equation_indices = zero_equation_indices,
		original_equation_count = get(result, "original_equation_count", length(poly_system)),
		full_equations = full_template_equations,
		algebraic_multiplicity = get(result, "algebraic_multiplicity", nothing),
		equation_builder_timing = equation_builder_timing,
		sian_timing = get(result, "timing", OrderedDict{Symbol, Float64}()),
		algebraic_multiplicity_timing = get(result, "algebraic_multiplicity_timing", OrderedDict{Symbol, Any}()),
		template_var_map = template_var_map,
		structural_analysis = analysis,
		structural_analysis_reused = analysis_reused,
	)

	# Return identifiable_funcs as well
	return template_equations, y_derivative_dict, unidentifiable, identifiable_funcs, si_variable_role_summary, si_template_metadata
end

"""
	_eliminate_to(gb_input, ring_gens, keep_vars)

Compute a Groebner basis of the elimination ideal `I ∩ k[keep_vars]` — the
projection of `V(⟨gb_input⟩)` onto `keep_vars`, eliminating `ring_gens ∖ keep_vars`
via a block ordering that ranks the eliminated variables above the kept ones
(Elimination Theorem). Returns `(Rid, gens)`, the generators re-embedded into a
fresh ring `Rid` on `keep_vars` (`gens` may be empty, meaning the ideal is `{0}` and
every kept variable is free), or `nothing` if `keep_vars` is empty.
"""
function _eliminate_to(gb_input, ring_gens, keep_vars)
	isempty(keep_vars) && return nothing
	keepset = Set(keep_vars)
	Rid, _ = Nemo.polynomial_ring(Nemo.QQ, [string(v) for v in keep_vars]; internal_ordering = :degrevlex)
	elim_vars = [g for g in ring_gens if !(g in keepset)]
	isempty(elim_vars) && return (Rid, [SIANBackend.parent_ring_change(g, Rid) for g in gb_input])
	block = Groebner.ProductOrdering(Groebner.DegRevLex(elim_vars...), Groebner.DegRevLex(keep_vars...))
	gb_elim = Groebner.groebner(gb_input; ordering = block)
	elim_gens = [g for g in gb_elim if issubset(Set(Nemo.vars(g)), keepset)]
	return (Rid, [SIANBackend.parent_ring_change(g, Rid) for g in elim_gens])
end

"""
	_multiplicity_via_identifiable_projection(gb_input, Rjet_gb, theta_l)

Count algebraic solutions after projecting a positive-dimensional ideal onto
the finite-valued order-0 coordinates in `theta_l`. Local identifiability allows
multiple finite branches; global identifiability would require a unique value.
A coordinate varying along a continuous symmetry is not locally identifiable.

First project onto `theta_l`. If the projected ideal still has positive
dimension, retain only coordinates with a nonzero univariate elimination
ideal. Return their quotient-space dimension (counting multiplicity), or
`nothing` if no zero-dimensional count is obtained.
"""
function _multiplicity_via_identifiable_projection(gb_input, Rjet_gb, theta_l)
	ring_gens = Nemo.gens(Rjet_gb)
	theta_names = Set(string(t) for t in theta_l)
	theta_ring_vars = [g for g in ring_gens if string(g) in theta_names]

	# Stage 1: project onto the locally-identifiable order-0 coordinates.
	res = _eliminate_to(gb_input, ring_gens, theta_ring_vars)
	res === nothing && return nothing
	Rid, gens_id = res
	if !isempty(gens_id) && Groebner.dimension(gens_id) == 0
		return length(Groebner.quotient_basis(gens_id))  # theta_l already finite-valued
	end

	# Stage 2: a positive-dimensional component survives projection. Keep only the
	# finite-valued coordinates — v is finite-valued iff its univariate
	# elimination ideal I ∩ k[v] is nonzero — working in the small ring Rid.
	small_gens = Nemo.gens(Rid)
	keep = eltype(small_gens)[]
	for v in small_gens
		res_v = _eliminate_to(gens_id, small_gens, [v])
		res_v === nothing && continue
		isempty(res_v[2]) || push!(keep, v)
	end
	isempty(keep) && return nothing

	res2 = _eliminate_to(gens_id, small_gens, keep)
	res2 === nothing && return nothing
	_, gens_id2 = res2
	(!isempty(gens_id2) && Groebner.dimension(gens_id2) == 0) || return nothing
	return length(Groebner.quotient_basis(gens_id2))
end

"""
	_sian_local_coordinates(polynomials, coordinates, quantities, sample_values)

# Arguments
- `polynomials`, `coordinates`: Equations with output jets substituted, and their unknowns.
- `quantities`: Parameter/state-at-anchor coordinates to classify.
- `sample_values`: Exact values in the original jet ring, ordered by its generators.

# Returns
Locally identifiable coordinates and the full Jacobian rank at the sample.
Removing a locally identifiable coordinate lowers rank. The comparison is with
the full rank, not the equation count: dependent rows must not imply identifiability.
"""
function _sian_local_coordinates(polynomials, coordinates, quantities, sample_values)
	jacobian = SIANBackend.jacobi_matrix(polynomials, coordinates, sample_values)
	full_rank = LinearAlgebra.rank(jacobian)
	locally_identifiable = eltype(coordinates)[]
	for q in quantities
		idx = findfirst(isequal(q), coordinates)
		isnothing(idx) && continue
		other_columns = [j for j in eachindex(coordinates) if j != idx]
		if LinearAlgebra.rank(jacobian[:, other_columns]) < full_rank
			push!(locally_identifiable, q)
		end
	end
	return (; locally_identifiable, jacobian_rank = full_rank)
end

"""
	_prepare_sian_multiplicity_system(si_ode, Et, Q, X_eq, Y_eq, all_params, sample, sample_bound; pre_fixed_params)

# Arguments
- `Et`, `Q`: Full polynomial constraints and the common ODE denominator.
- `X_eq`, `Y_eq`, `all_params`: SIAN's exact jet recurrence and order-0 coordinates.
- `sample`, `sample_bound`: Generic sample and sampling bound from template construction.
- `pre_fixed_params`: Representative assignments, keyed by SI coordinate names.

# Returns
The saturated polynomial system, its smaller ring, an exact satisfying point,
and local-rank metadata. A new sample respecting the assignments is used when
fixes are present. State assignments affect only order 0; higher state jets
are generated by the original ODE. All polynomial constraints are retained.
"""
function _prepare_sian_multiplicity_system(si_ode, Et, Q, X_eq, Y_eq, all_params, sample, sample_bound;
	pre_fixed_params::OrderedDict = OrderedDict())
	Rjet = parent(Q)
	gens_Rjet = Nemo.gens(Rjet)
	n, m, u = length(si_ode.x_vars), length(si_ode.y_vars), length(si_ode.u_vars)
	s = length(all_params)
	parameter_count = length(si_ode.parameters)
	parameter_jets = all_params[1:parameter_count]
	# Production fixes use base SI names, including names with underscores/digits.
	# Remove exactly the final `_0` jet suffix from SIAN's order-0 names.
	by_base_name = Dict(chop(string(q); tail = 2) => q for q in all_params)
	fixed = OrderedDict{Nemo.QQMPolyRingElem, Nemo.QQFieldElem}()
	for (key, value) in pre_fixed_params
		name = string(key)
		haskey(by_base_name, name) || throw(ArgumentError("Unknown SI representative coordinate: $name"))
		# Nemo has no QQ(Float64) constructor. Preserve the exact assignment value.
		exact_value = value isa AbstractFloat ? Rational{BigInt}(value) : value
		fixed[by_base_name[name]] = Nemo.QQ(exact_value)
	end
	fixed_vars, fixed_values = collect(keys(fixed)), collect(values(fixed))
	substitute_fixed(poly) = isempty(fixed) ? poly : Nemo.evaluate(poly, fixed_vars, fixed_values)
	Q_fixed = substitute_fixed(Q)
	iszero(Q_fixed) && throw(ArgumentError("Structural representative assignments make the ODE denominator identically zero"))
	m_sample = if isempty(fixed)
		sample
	else
		SIANBackend.sample_point(sample_bound, si_ode.x_vars, si_ode.y_vars, Nemo.QQMPolyRingElem[],
			all_params, X_eq, Y_eq, Q; known_states_jet_form = fixed_vars, known_values = fixed_values)
	end
	u_hat, y_hat = m_sample[2], m_sample[1]
	data_vars, data_values = vcat(u_hat[1], y_hat[1]), vcat(u_hat[2], y_hat[2])
	Et_hat = [substitute_fixed(Nemo.evaluate(e, data_vars, data_values)) for e in Et]
	Q_hat = isempty(u_hat[1]) ? Q_fixed : Nemo.evaluate(Q_fixed, u_hat[1], u_hat[2])
	sample_values = SIANBackend.insert_zeros_to_vals(m_sample[4][1], m_sample[4][2])
	# Detect a mismatched synthetic jet before spending time on an inconsistent ideal.
	all(e -> iszero(Nemo.evaluate(e, sample_values)), Et_hat) ||
		error("Multiplicity sample does not satisfy the representative-fixed polynomial equations")

	remaining_parameters = filter(q -> !haskey(fixed, q), parameter_jets)
	used = Set{Nemo.QQMPolyRingElem}(Nemo.vars(Q_hat))
	for e in Et_hat
		union!(used, Nemo.vars(e))
	end
	setdiff!(used, parameter_jets)
	all_indets = Nemo.gens(si_ode.poly_ring)
	state_jets = sort(collect(used), lt = (x, y) -> SIANBackend.compare_diff_var(x, y, all_indets, n + m + u, s))
	coordinates = vcat(state_jets, remaining_parameters)
	quantities = filter(q -> !haskey(fixed, q), all_params)
	local_info = _sian_local_coordinates(Et_hat, coordinates, quantities, sample_values)
	z_aux = gens_Rjet[end - parameter_count]
	ordered_vars = vcat(state_jets, z_aux, sort(remaining_parameters, rev = true))
	ring, _ = Nemo.polynomial_ring(Nemo.QQ, string.(ordered_vars); internal_ordering = :degrevlex)
	polynomials = [SIANBackend.parent_ring_change(e, ring) for e in Et_hat]
	push!(polynomials, SIANBackend.parent_ring_change(z_aux * Q_hat, ring) - 1)
	q_at_sample = Nemo.evaluate(Q_hat, sample_values)
	iszero(q_at_sample) && error("Multiplicity sample lies on an excluded ODE denominator")
	point = [v == z_aux ? inv(q_at_sample) : sample_values[Nemo.var_index(v)] for v in ordered_vars]
	return (; polynomials, ring, point, fixed_coordinates = fixed,
		locally_identifiable = local_info.locally_identifiable, jacobian_rank = local_info.jacobian_rank)
end

"""
	get_polynomial_system_from_sian(si_ode, params_to_assess; p = 0.99, infolevel = 0, compute_multiplicity = true)

Get polynomial system using SIAN functions, adapted from PE.jl's implementation.
This now properly builds the system Et through the iterative rank-checking process.

`compute_multiplicity = false` skips the algebraic-multiplicity (M) Groebner step.
Detection/intermediate passes of the structural-fix flow pass `false`. With
`pre_fixed_params`, the final pass computes M on a separate representative-fixed
system with a consistent exact sample. Template construction and its original
structural classification still use the unfixed model.
"""
function get_polynomial_system_from_sian(si_ode, params_to_assess; p = 0.99, infolevel = 0, compute_multiplicity = true,
	pre_fixed_params::OrderedDict = OrderedDict())
	sian_timing = OrderedDict{Symbol, Float64}()
	algebraic_multiplicity_timing = OrderedDict{Symbol, Any}()
	# Get equations using SIAN
	_t_get_equations = @elapsed begin
		eqs, Q, x_eqs, y_eqs, x_vars, y_vars, u_vars, mu, all_indets, gens_Rjet = SIANBackend.get_equations(si_ode)
	end
	sian_timing[:get_equations] = _t_get_equations

	non_jet_ring = si_ode.poly_ring
	Rjet = gens_Rjet[1].parent

	n = length(x_vars)
	m = length(y_vars)
	u = length(u_vars)
	s = length(mu) + n

	# Get X and Y equations
	_t_get_x_y_eq_start = time()
	X, X_eq = SIANBackend.get_x_eq(x_eqs, y_eqs, n, m, s, u, gens_Rjet)
	Y, Y_eq = SIANBackend.get_y_eq(x_eqs, y_eqs, n, m, s, u, gens_Rjet)
	sian_timing[:get_x_y_eq] = time() - _t_get_x_y_eq_start

	# Extract parameters and state variables
	_t_variable_setup_start = time()
	not_int_cond_params = gens_Rjet[(end-length(si_ode.parameters)+1):end]
	all_params = vcat(not_int_cond_params, gens_Rjet[1:n])

	x_variables = gens_Rjet[1:n]
	for i in 1:(s+1)
		x_variables = vcat(x_variables,
			gens_Rjet[(i*(n+m+u)+1):(i*(n+m+u)+n)])
	end
	sian_timing[:variable_setup] = time() - _t_variable_setup_start

	# Compute degree bound
	_t_degree_bound_start = time()
	d0 = BigInt(maximum(vcat([Nemo.total_degree(SIANBackend.unpack_fraction(Q * eq[2])[1])
							  for eq in eqs], Nemo.total_degree(Q))))

	# Sample point for Jacobian evaluations
	D1 = floor(BigInt,
		(length(params_to_assess) + 1) * 2 * d0 * s * (n + 1) * (1 + 2 * d0 * s) /
		(1 - p))
	sian_timing[:degree_bound] = time() - _t_degree_bound_start

	# Convert empty array to proper type for u_variables
	u_empty = Vector{Nemo.QQMPolyRingElem}()
	_t_sample_point_start = time()
	sample = SIANBackend.sample_point(D1, x_vars, y_vars, u_empty, all_params, X_eq, Y_eq, Q)
	sian_timing[:sample_point] = time() - _t_sample_point_start
	all_subs = sample[4]
	u_hat = sample[2]
	y_hat = sample[1]

	# Build the polynomial system Et through iterative process
	Et = Array{Nemo.QQMPolyRingElem}(undef, 0)
	x_theta_vars = all_params
	beta = [0 for i in 1:m]
	prolongation_possible = [1 for i in 1:m]

	all_x_theta_vars_subs = SIANBackend.insert_zeros_to_vals(all_subs[1], all_subs[2])
	eqs_i_old = Array{Nemo.QQMPolyRingElem}(undef, 0)
	evl_old = Array{Nemo.QQMPolyRingElem}(undef, 0)

	# Iterative rank-based construction
	_t_rank_construction_start = time()
	while sum(prolongation_possible) > 0
		for i in 1:m
			if prolongation_possible[i] == 1
				eqs_i = vcat(Et, Y[i][beta[i]+1])
				evl = [Nemo.evaluate(eq, vcat(u_hat[1], y_hat[1]),
					vcat(u_hat[2], y_hat[2]))
					   for eq in eqs_i if !(eq in eqs_i_old)]
				evl_old = vcat(evl_old, evl)
				JacX = SIANBackend.jacobi_matrix(evl_old, x_theta_vars, all_x_theta_vars_subs)
				eqs_i_old = eqs_i

				if LinearAlgebra.rank(JacX) == length(eqs_i)
					Et = vcat(Et, Y[i][beta[i]+1])
					beta[i] = beta[i] + 1

					# Add necessary X-equations
					polys_to_process = vcat(Et, [Y[k][beta[k]+1] for k in 1:m if beta[k] < length(Y[k])])
					while length(polys_to_process) != 0
						new_to_process = Array{Nemo.QQMPolyRingElem}(undef, 0)
						vrs = Set{Nemo.QQMPolyRingElem}()
						for poly in polys_to_process
							vrs = union(vrs,
								[v for v in Nemo.vars(poly) if v in x_variables])
						end
						vars_to_add = Set{Nemo.QQMPolyRingElem}(v
																for v in vrs
																if !(v in x_theta_vars))
						for v in vars_to_add
							x_theta_vars = vcat(x_theta_vars, v)
							ord_var = SIANBackend.get_order_var2(v, all_indets, n + m + u, s)
							var_idx = Nemo.var_index(ord_var[1])
							poly = X[var_idx][ord_var[2]]
							Et = vcat(Et, poly)
							new_to_process = vcat(new_to_process, poly)
						end
						polys_to_process = new_to_process
					end
				else
					prolongation_possible[i] = 0
				end
			end
		end
	end

	# Add remaining Y equations that don't introduce new variables
	for i in 1:m
		for j in (beta[i]+1):length(Y[i])
			to_add = true
			for v in SIANBackend.get_vars(Y[i][j], x_vars, all_indets, n + m + u, s)
				if !(v in x_theta_vars)
					to_add = false
				end
			end
			if to_add
				beta[i] = beta[i] + 1
				Et = vcat(Et, Y[i][j])
			end
		end
	end
	sian_timing[:rank_construction] = time() - _t_rank_construction_start

	@info "Built polynomial system with $(length(Et)) equations"
	if infolevel > 1
		@info "[SI-TEMPLATE] Raw polynomial variables collected during SIAN construction" variables = x_theta_vars
		for (idx, eq) in enumerate(Et)
			vars_in_eq = Nemo.vars(eq)
			@info "[SI-TEMPLATE] Raw polynomial equation" index = idx ring_variable_count = length(vars_in_eq) ring_variables = vars_in_eq
		end
	end

	# Classify coordinates using rank loss, even when Et has dependent rows.
	params_to_assess_ = [SIANBackend.add_to_var(param, Rjet, 0) for param in params_to_assess]
	_t_evaluate_Et_base_start = time()
	Et_eval_base = [Nemo.evaluate(e, vcat(u_hat[1], y_hat[1]),
		vcat(u_hat[2], y_hat[2]))
					for e in Et]
	sian_timing[:evaluate_Et_base] = time() - _t_evaluate_Et_base_start
	_t_local_identifiability_start = time()
	local_info = _sian_local_coordinates(Et_eval_base, x_theta_vars, params_to_assess_, all_x_theta_vars_subs)
	theta_l = local_info.locally_identifiable
	sian_timing[:local_identifiability_jacobians] = time() - _t_local_identifiability_start
	x_theta_vars_reorder = vcat(theta_l,
		reverse([x for x in x_theta_vars if !(x in theta_l)]))

	# Count solutions of the representative-fixed ideal <Et_hat, z_aux*Q_hat-1>.
	# The synthetic output jet must come from the same representative slice.
	# Keep every equation: local row dependence does not prove ideal redundancy.
	# Detection passes skip M; the original classification/template remain unfixed.
	# Genuine Groebner failures propagate. A positive-dimensional result can still
	# be projected onto finite-valued coordinates by the existing counting path.
	algebraic_multiplicity = if !compute_multiplicity
		algebraic_multiplicity_timing[:skipped] = true
		algebraic_multiplicity_timing[:disabled_by_caller] = true
		@info "[SI-TEMPLATE] algebraic_multiplicity: skipped by caller"
		nothing
	elseif isempty(theta_l) && isempty(pre_fixed_params)
		@info "[SI-TEMPLATE] algebraic_multiplicity: skipping (theta_l empty → fully non-identifiable)"
		algebraic_multiplicity_timing[:skipped] = true
		nothing
	else
		_M_setup_t = @elapsed begin
			m_system = _prepare_sian_multiplicity_system(si_ode, Et, Q, X_eq, Y_eq, all_params, sample, D1;
				pre_fixed_params = pre_fixed_params)
			gb_input, Rjet_gb = m_system.polynomials, m_system.ring
		end
		algebraic_multiplicity_timing[:setup_seconds] = _M_setup_t
		algebraic_multiplicity_timing[:ring_variable_count] = Nemo.ngens(Rjet_gb)
		algebraic_multiplicity_timing[:polynomial_count] = length(gb_input)
		algebraic_multiplicity_timing[:total_terms] = sum(length, gb_input)
		algebraic_multiplicity_timing[:max_total_degree] = maximum(Nemo.total_degree, gb_input)
		algebraic_multiplicity_timing[:fixed_coordinates] = Dict(string(k) => string(v) for (k, v) in m_system.fixed_coordinates)
		algebraic_multiplicity_timing[:jacobian_rank] = m_system.jacobian_rank
		algebraic_multiplicity_timing[:linearized_dimension] = Nemo.ngens(Rjet_gb) - 1 - m_system.jacobian_rank
		algebraic_multiplicity_timing[:sample_verified] = true
		@info "[SI-TEMPLATE] Multiplicity input" variables = Nemo.ngens(Rjet_gb) polynomials = length(gb_input) terms = sum(length, gb_input) max_degree = maximum(Nemo.total_degree, gb_input) fixed_coordinates = algebraic_multiplicity_timing[:fixed_coordinates] linearized_dimension = algebraic_multiplicity_timing[:linearized_dimension]
		M_value = if isempty(m_system.locally_identifiable)
			@info "[SI-TEMPLATE] algebraic_multiplicity: skipping (no finite-valued coordinates after fixing)"
			nothing
		else
			_M_gb_t = @elapsed gb = Groebner.groebner(gb_input)
			_M_qb_t = @elapsed begin
				M_result = try
					d = Groebner.dimension(gb)
					algebraic_multiplicity_timing[:dimension] = d
					if d == 0
						length(Groebner.quotient_basis(gb))
					elseif d >= 1
						algebraic_multiplicity_timing[:positive_dimensional] = true
						algebraic_multiplicity_timing[:gauge_dimension] = d
						_multiplicity_via_identifiable_projection(gb_input, Rjet_gb, m_system.locally_identifiable)
					else
						nothing
					end
				catch e
					(e isa DomainError || e isa Groebner.MonomialDegreeOverflow) || rethrow()
					algebraic_multiplicity_timing[:multiplicity_error] = string(nameof(typeof(e)))
					nothing
				end
			end
			algebraic_multiplicity_timing[:groebner_seconds] = _M_gb_t
			algebraic_multiplicity_timing[:quotient_basis_seconds] = _M_qb_t
			M_result
		end
		algebraic_multiplicity_timing[:skipped] = isnothing(M_value)
		if !isnothing(M_value)
			algebraic_multiplicity_timing[:multiplicity] = M_value
			@info "[SI-TEMPLATE] algebraic_multiplicity M = $M_value" timing = algebraic_multiplicity_timing
		end
		M_value
	end

	if infolevel > 1
		@info "[SI-TEMPLATE] Variables before algebraic independence reduction" equations = length(Et_eval_base) reordered_ring_variables = length(x_theta_vars_reorder) locally_identifiable_count = length(theta_l)
		@info "[SI-TEMPLATE] Locally identifiable ring variables from SIAN Jacobian pruning" variables = theta_l
		@info "[SI-TEMPLATE] Reordered ring variable list before rank trimming" variables = x_theta_vars_reorder
	end

	# ==== Filter variables to only those that actually appear in Et ====
	# This is critical for models where some parameters/states don't affect observed outputs.
	# Without filtering, we'd have phantom variables inflating the count, causing overdetermined systems.
	_t_variable_filtering_start = time()
	vars_in_Et = Set{Nemo.QQMPolyRingElem}()
	for eq in Et
		union!(vars_in_Et, Set(Nemo.vars(eq)))
	end

	# Filter x_theta_vars to only include variables that appear in Et
	x_theta_vars_filtered = filter(v -> v in vars_in_Et, x_theta_vars)

	# Log any phantom variables that were filtered out
	phantom_vars = setdiff(Set(x_theta_vars), vars_in_Et)
	# Also filter theta_l to only include variables present in equations
	theta_l_filtered = filter(v -> v in vars_in_Et, theta_l)

	# Recompute the reordered variable list using filtered variables
	x_theta_vars_reorder_filtered = vcat(theta_l_filtered,
		reverse([x for x in x_theta_vars_filtered if !(x in theta_l_filtered)]))

	if infolevel > 1
		@info "[SI-TEMPLATE] Reordered ring variables after removing phantom support" variables = x_theta_vars_reorder_filtered
	end
	sian_timing[:variable_filtering] = time() - _t_variable_filtering_start
	# ==== End of filtering block ====

	_t_algebraic_independence_start = time()
	Et_ids, alg_indep = algebraic_independence(Et_eval_base, x_theta_vars_reorder_filtered,
		all_x_theta_vars_subs)
	sian_timing[:algebraic_independence] = time() - _t_algebraic_independence_start

	_t_reduce_system_start = time()
	dropped_ids = setdiff(1:length(Et), Et_ids)
	if infolevel > 1
		@info "[SI-TEMPLATE] Algebraic independence result" selected_equation_indices = Et_ids transcendence_basis_count = length(alg_indep) transcendence_basis = alg_indep
		for idx in dropped_ids
			@info "[SI-TEMPLATE] Dropped polynomial equation" index = idx equation = Et[idx]
		end
	end

	# Reduce the system using the computed indices
	reduced_Et = Et[Et_ids]

	# Compute the actual variables present in the reduced system
	reduced_vars = Set{Nemo.QQMPolyRingElem}()
	for eq in reduced_Et
		union!(reduced_vars, Set(Nemo.vars(eq)))
	end
	sian_timing[:reduce_system] = time() - _t_reduce_system_start

	@info "Reduced polynomial system to $(length(reduced_Et)) equations"
	_log_si_polynomial_system_summary(
		infolevel = infolevel,
		initial_equation_count = length(Et),
		initial_variable_count = length(x_theta_vars),
		theta_l_count = length(theta_l),
		phantom_vars = phantom_vars,
		filtered_variable_count = length(x_theta_vars_filtered),
		selected_equation_indices = Et_ids,
		dropped_equation_indices = dropped_ids,
		reduced_equation_count = length(reduced_Et),
		reduced_variable_count = length(reduced_vars),
	)
	if infolevel > 1
		@info "[SI-TEMPLATE] Reduced polynomial ring variables" variables = reduced_vars
		if length(reduced_Et) != length(reduced_vars)
			@info "[SI-TEMPLATE] Pre-accounting ring-variable mismatch details" filtered_not_in_equations = setdiff(Set(x_theta_vars_filtered), reduced_vars) equations_not_in_filtered = setdiff(reduced_vars, Set(x_theta_vars_filtered))
		end
	end

	# Build the derivative mapping dictionary
	_t_derivative_mapping_start = time()
	y_derivative_dict = Dict()
	for each in Y_eq
		name, order = SIANBackend.get_order_var(each[1], non_jet_ring)
		y_derivative_dict[each[1]] = order
	end
	sian_timing[:derivative_mapping] = time() - _t_derivative_mapping_start

	# Return result with the full polynomial system Et
	return Dict(
		"polynomial_system" => reduced_Et,  # Return the REDUCED system
		"full_polynomial_system" => Et,
		"Y_eq" => y_derivative_dict,  # Maps derivative variables to orders
		"X_eq" => X_eq,
		"Y" => Y,  # Keep the full Y structure for reference
		"X" => X,  # Keep X structure as well
		"x_theta_vars" => x_theta_vars_filtered,  # Variables that actually appear in the system
		"beta" => beta,  # Derivative orders used
		"non_jet_ring" => non_jet_ring,
		"selected_equation_indices" => collect(Et_ids),
		"dropped_equation_indices" => collect(dropped_ids),
		"original_equation_count" => length(Et),
		"algebraic_multiplicity" => algebraic_multiplicity,
		"timing" => sian_timing,
		"algebraic_multiplicity_timing" => algebraic_multiplicity_timing,
	)
end


"""
	parse_derivative_variable_name(var_name::String)

Parse a SIAN derivative variable name to extract base variable and derivative order.
The last `_N` suffix (where N is a non-negative integer) is always the derivative order.
Supports patterns like:
  - "y1_2" where y1 is the base and 2 is the derivative order
  - "y_2" where y is the base and 2 is the derivative order
  - "_obs_trfn_cos_5_0_cos_3" where _obs_trfn_cos_5_0_cos is the base and 3 is the order
Returns (base_name, derivative_order) or nothing if parsing fails.
"""
function parse_derivative_variable_name(var_name::String)
	# Match the last _N suffix as the derivative order
	m = match(r"^(.+)_(\d+)$", var_name)
	if !isnothing(m)
		base_name = m.captures[1]
		derivative_order = parse(Int, m.captures[2])
		return (base_name, derivative_order)
	end

	return nothing
end

"""
	parse_jet_label(label::AbstractString) -> Union{Nothing, Tuple{String, Int}}

THE single shared parser for observable/derivative labels (maintainability
campaign Phase D1). Previously three near-identical copies lived in
diagnostics.jl, sigma_d.jl and multipoint_template.jl and had to agree for
Σ_d↔sensitivity column alignment; they now delegate here.

Handles, in order:
1. Symbolics comma form  `"Differential(t, N)(var(t))"` → (var, N)
2. Symbolics bare form   `"var(t)"`                     → (var, 0)
3. SIAN jet form         `"var_N"`                      → (var, N)

Returns `nothing` when no form matches. Known locked behaviors (see
test/test_label_parsers.jl before changing ANY of these): the order-1
non-comma form `"Differential(t)(var(t))"` does NOT parse; a bare
underscore-digit name parses its suffix as an order (`"k_1"` → ("k", 1));
`_trfn_` frequency suffixes misparse (`"_trfn_sin_0_5"` → ("_trfn_sin_0", 5)).
"""
function parse_jet_label(label::AbstractString)
	s = String(label)
	m = match(r"^Differential\(t,\s*(\d+)\)\((\w+)\(t\)\)$", s)
	if !isnothing(m)
		return (String(m.captures[2]), parse(Int, m.captures[1]))
	end
	m = match(r"^(\w+)\(t\)$", s)
	if !isnothing(m)
		return (String(m.captures[1]), 0)
	end
	parsed = parse_derivative_variable_name(s)
	isnothing(parsed) && return nothing
	base_name, order = parsed
	return (String(base_name), Int(order))
end

"""
	nemo_to_symbolics(nemo_expr, var_map::Dict)

Convert a Nemo expression to a Symbolics expression using proper term-by-term reconstruction.
"""
function nemo_to_symbolics(nemo_expr, var_map::Dict)
	# Handle constants
	if nemo_expr isa Number
		return nemo_expr
	end

	# Handle Nemo QQ field elements (rationals)
	if nemo_expr isa Nemo.QQFieldElem
		num = BigInt(Nemo.numerator(nemo_expr))
		den = BigInt(Nemo.denominator(nemo_expr))
		return Rational(num, den)
	end

	# Handle fraction field elements
	if nemo_expr isa Nemo.Generic.FracFieldElem
		numer = nemo_to_symbolics(Nemo.numerator(nemo_expr), var_map)
		denom = nemo_to_symbolics(Nemo.denominator(nemo_expr), var_map)
		return numer / denom
	end

	# Handle polynomial variables directly
	if haskey(var_map, nemo_expr)
		return var_map[nemo_expr]
	end

	# For multivariate polynomials - use proper Nemo API
	if nemo_expr isa Nemo.QQMPolyRingElem
		# Handle zero polynomial
		if iszero(nemo_expr)
			return 0
		end

		# Get the parent ring and variables
		R = parent(nemo_expr)
		vars = Nemo.gens(R)

		# Build Symbolics expression by iterating over terms
		symbolic_terms = []

		# Use the correct Nemo API: iterate through indices
		for i in 1:length(nemo_expr)
			# Get coefficient for this term
			c = Nemo.coeff(nemo_expr, i)

			# Convert coefficient to Julia number
			coeff_val = if c isa Nemo.QQFieldElem
				num = BigInt(Nemo.numerator(c))
				den = BigInt(Nemo.denominator(c))
				Rational(num, den)
			elseif c isa Nemo.ZZRingElem
				BigInt(c)
			elseif c isa Integer
				c
			else
				# Fail fast: substituting a placeholder 1 for an unknown
				# coefficient silently corrupts the polynomial — strictly worse
				# than any crash. (Zero occurrences across all repro/PEB logs.)
				error("Unknown Nemo coefficient type $(typeof(c)) in nemo_to_symbolics — refusing to substitute 1. Value: $c")
			end

			# Get exponent vector for this term
			exp_vec = Nemo.exponent_vector(nemo_expr, i)

			# Build the monomial
			monomial_factors = []
			for (j, var) in enumerate(vars)
				if exp_vec[j] > 0
					# Get the Symbolics variable from the map
					sym_var = get(var_map, var, nothing)
					if isnothing(sym_var)
						var_name = string(var)
						# Late placeholder creation is still allowed here because some
						# identifiable-function outputs introduce symbols that were not
						# present in the initial SIAN→MTK map.
						placeholder_stats = Dict{Symbol, Vector{String}}()
						late_category = is_sian_auxiliary_variable_name(var_name) ? :sian_auxiliary : :late_map_miss
						late_message = late_category == :sian_auxiliary ?
							"Variable not found in SI variable map during Nemo reconstruction; preserving known SIAN auxiliary." :
							"Variable not found in SI variable map during Nemo reconstruction; creating a late symbolic placeholder."
						sym_var = _create_si_symbolic_placeholder!(
							var_map,
							var,
							var_name,
							placeholder_stats,
							late_category;
							message_level = :warn,
							message = late_message,
						)
					end

					if exp_vec[j] == 1
						push!(monomial_factors, sym_var)
					else
						push!(monomial_factors, sym_var^exp_vec[j])
					end
				end
			end

			# Construct the term
			if isempty(monomial_factors)
				# Constant term
				push!(symbolic_terms, coeff_val)
			else
				# Term with variables
				term = coeff_val
				for factor in monomial_factors
					term = term * factor
				end
				push!(symbolic_terms, term)
			end
		end

		# Sum all terms
		if isempty(symbolic_terms)
			return 0
		else
			result = symbolic_terms[1]
			for i in 2:length(symbolic_terms)
				result = result + symbolic_terms[i]
			end
			return result
		end
	end

	# Unknown type
	@error "Cannot convert Nemo expression of type $(typeof(nemo_expr))"
	return nemo_expr
end
