using ModelingToolkit
using OrderedCollections
using Logging
using LinearAlgebra
using NonlinearSolve
using OrdinaryDiffEq
using Statistics

# Use functions from the current module
using ..ODEParameterEstimation

# Shared helper relocated from consensus_estimation.jl (2026-06-09) to sever the only
# production -> research call hook. Builds a "candidate-as-truth" PEP from a result.
# Production callers: branch_completion.jl, sensitivity_seeds.jl. Also used by the
# research consensus cluster now living under src/research/.
function _consensus_candidate_pep(pep::ParameterEstimationProblem, candidate::ParameterEstimationResult)
	return ParameterEstimationProblem(
		pep.name,
		pep.model,
		pep.measured_quantities,
		pep.data_sample,
		pep.recommended_time_interval,
		pep.solver,
		OrderedDict{Num, Float64}(k => Float64(v) for (k, v) in candidate.parameters),
		OrderedDict{Num, Float64}(k => Float64(v) for (k, v) in candidate.states),
		pep.unident_count,
	)
end

const UNSUPPORTED_MODEL_CATEGORY_PRIORITY = (
	:state_trigonometric,
	:sqrt_nonlinearity,
	:unsupported_transcendental,
	:unsupported_nonlinear_function,
)

"""
	UnsupportedModelClassError

Thrown when a model is not rational: it applies a function such as `sin`, `exp`
or `sqrt` to a state or to an unknown parameter. `expressions` lists the terms
found, and `category` says which kind they are.

`sin(c * t)`, `cos(c * t)` and `exp(c * t)` with a number `c` depend on time
alone and are accepted.
"""
struct UnsupportedModelClassError <: Exception
	category::Symbol
	expressions::Vector{String}
	details::OrderedDict{Symbol, Vector{String}}
end

function Base.showerror(io::IO, err::UnsupportedModelClassError)
	print(io, "This model is not rational")
	isempty(err.expressions) || print(io, ": it contains ", join(err.expressions, ", "))
	print(io, ". A model must be built from polynomials and ratios of polynomials in its states and parameters. ")
	print(io, "sin(c*t), cos(c*t) and exp(c*t) with a number c are accepted.")
end

function _record_unsupported_expr!(
	found::OrderedDict{Symbol, Vector{String}},
	category::Symbol,
	expr,
)
	expr_text = replace(string(expr), '\n' => ' ')
	values = get!(found, category, String[])
	expr_text in values || push!(values, expr_text)
	return found
end

function _num_or_nothing(x)
	x isa Num && return x
	try
		return Num(x)
	catch err
		_rethrow_if_interrupt(err)
		return nothing
	end
end

function _collect_unsupported_model_constructs!(
	expr,
	t_var,
	found::OrderedDict{Symbol, Vector{String}},
)
	num_expr = _num_or_nothing(expr)
	isnothing(num_expr) && return found
	val = Symbolics.value(num_expr)
	Symbolics.iscall(val) || return found

	op = Symbolics.operation(val)
	args = Symbolics.arguments(val)

	if op === sqrt
		_record_unsupported_expr!(found, :sqrt_nonlinearity, num_expr)
		return found
	elseif op === sin || op === cos
		if length(args) == 1
			arg_expr = _num_or_nothing(args[1])
			if isnothing(arg_expr) || isnothing(_is_constant_times_t(arg_expr, t_var))
				_record_unsupported_expr!(found, :state_trigonometric, num_expr)
				return found
			end
		else
			_record_unsupported_expr!(found, :state_trigonometric, num_expr)
			return found
		end
	elseif op === exp
		if length(args) == 1
			arg_expr = _num_or_nothing(args[1])
			if isnothing(arg_expr) || isnothing(_is_constant_times_t(arg_expr, t_var))
				_record_unsupported_expr!(found, :unsupported_transcendental, num_expr)
				return found
			end
		else
			_record_unsupported_expr!(found, :unsupported_transcendental, num_expr)
			return found
		end
	elseif op in (log, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh)
		_record_unsupported_expr!(found, :unsupported_nonlinear_function, num_expr)
		return found
	end

	for arg in args
		_collect_unsupported_model_constructs!(arg, t_var, found)
	end

	return found
end

function collect_unsupported_model_constructs(
	model_system,
	measured_quantities,
)
	t_var = ModelingToolkit.get_iv(model_system)
	model_equations = ModelingToolkit.equations(model_system)
	found = OrderedDict{Symbol, Vector{String}}()

	for eq in model_equations
		_collect_unsupported_model_constructs!(eq.rhs, t_var, found)
	end
	for mq in measured_quantities
		_collect_unsupported_model_constructs!(mq.rhs, t_var, found)
	end

	return found
end

function primary_unsupported_model_category(found::OrderedDict{Symbol, Vector{String}})
	for category in UNSUPPORTED_MODEL_CATEGORY_PRIORITY
		haskey(found, category) && return category
	end
	return first(keys(found))
end

function validate_supported_model_class(PEP::ParameterEstimationProblem)
	found = collect_unsupported_model_constructs(PEP.model.system, PEP.measured_quantities)
	isempty(found) && return nothing
	category = primary_unsupported_model_category(found)
	expressions = get(found, category, String[])
	throw(UnsupportedModelClassError(category, copy(expressions), deepcopy(found)))
end

function classify_unsupported_substitution_error(expr, substituted)
	text = string(substituted)
	category = if occursin("sqrt(", text)
		:sqrt_nonlinearity
	elseif occursin("sin(", text) || occursin("cos(", text)
		:state_trigonometric
	elseif occursin("exp(", text)
		:unsupported_transcendental
	else
		nothing
	end
	isnothing(category) && return nothing
	details = OrderedDict{Symbol, Vector{String}}(category => [replace(string(expr), '\n' => ' ')])
	return UnsupportedModelClassError(category, copy(details[category]), details)
end

"""
	setup_identifiability(PEP::ParameterEstimationProblem; max_num_points, nooutput)

Setup phase for the standard SI workflow. Runs best-effort numerical advisory
heuristics without letting them define structural identifiability, then builds
derivative support without numerical representative substitutions.

This is the shared, interpolator-independent part of the pipeline.

# Returns
- Named tuple with identifiability data: states, params, t_vector, derivative levels, etc.
"""
function practical_status_from_advisory(advisory::Union{Nothing, NumericalIdentifiabilityAdvisory})
	isnothing(advisory) && return :not_assessed
	advisory.status == :available && return :advisory_available
	advisory.status == :failed && return :advisory_failed
	return :not_assessed
end

function default_numerical_advisory_heuristics(model, measured_quantities, max_num_points, t_vector, states, params)
	states_count = length(states)
	params_count = length(params)
	obs_count = max(length(measured_quantities), 1)
	recommended_num_points = min(length(params), max_num_points, length(t_vector))
	recommended_deriv_order = max(
		states_count + params_count + 1,
		Int(ceil((states_count + params_count) / obs_count) + 2),
		3,
	)
	recommended_deriv_level = Dict(i => recommended_deriv_order for i in eachindex(measured_quantities))
	good_DD = populate_derivatives(
		model,
		measured_quantities,
		recommended_deriv_order + 1,
		OrderedDict{Num, Float64}();
		include_cleared = false,
	)
	good_DD.all_unidentifiable = Set{Num}()
	return (
		good_num_points = recommended_num_points,
		good_deriv_level = recommended_deriv_level,
		good_udict = OrderedDict{Num, Float64}(),
		good_varlist = Vector{Num}(vcat(params, states)),
		good_DD = good_DD,
	)
end

function run_numerical_identifiability_advisory(
	model,
	measured_quantities,
	num_points_cap,
	t_vector,
	states,
	params;
	nooutput = false,
	advisory_runner = determine_optimal_points_count,
)
	full_varlist = Set{Num}(vcat(params, states))
	try
		good_num_points, good_deriv_level, _good_udict, advisory_varlist, _good_DD =
			advisory_runner(model, measured_quantities, num_points_cap, t_vector, nooutput)
		good_DD = populate_derivatives(
			model,
			measured_quantities,
			isempty(good_deriv_level) ? 2 : maximum(values(good_deriv_level)) + 1,
			OrderedDict{Num, Float64}();
			include_cleared = false,
		)
		good_DD.all_unidentifiable = Set{Num}()
		flagged_variables = setdiff(full_varlist, Set{Num}(advisory_varlist))
		advisory = NumericalIdentifiabilityAdvisory(
			status = :available,
			recommended_num_points = good_num_points,
			recommended_deriv_level = good_deriv_level,
			flagged_variables = flagged_variables,
			notes = isempty(flagged_variables) ? Symbol[] : [:rank_deficient_at_probe],
		)
		return (
			good_num_points = good_num_points,
			good_deriv_level = good_deriv_level,
			good_udict = OrderedDict{Num, Float64}(),
			good_varlist = Vector{Num}(vcat(params, states)),
			good_DD = good_DD,
			advisory = advisory,
		)
	catch err
		_rethrow_if_interrupt(err)
		@warn "Numerical identifiability advisory failed; continuing with deterministic heuristics" exception = err
		# The fallback may require deep symbolic derivatives. Build it only when
		# the advisory actually fails, not before attempting the normal path.
		fallback = default_numerical_advisory_heuristics(model, measured_quantities, num_points_cap, t_vector, states, params)
		advisory = NumericalIdentifiabilityAdvisory(
			status = :failed,
			recommended_num_points = fallback.good_num_points,
			recommended_deriv_level = fallback.good_deriv_level,
			flagged_variables = Set{Num}(),
			notes = [:heuristic_fallback],
			failure_reason = sprint(showerror, err),
		)
		return (; fallback..., advisory = advisory)
	end
end

function setup_identifiability(
	PEP::ParameterEstimationProblem;
	max_num_points = 1,
	nooutput = false,
	advisory_runner = determine_optimal_points_count,
)
	validate_supported_model_class(PEP)

	t, eqns, states, params = unpack_ODE(PEP.model.system)
	t_vector = PEP.data_sample["t"]

	num_points_cap = min(length(params), max_num_points, length(t_vector))
	advisory_data = run_numerical_identifiability_advisory(
		PEP.model.system,
		PEP.measured_quantities,
		num_points_cap,
		t_vector,
		states,
		params;
		nooutput = nooutput,
		advisory_runner = advisory_runner,
	)

	@debug "Parameter estimation using $(advisory_data.good_num_points) points"

	return (
		states = states,
		params = params,
		t_vector = t_vector,
		good_num_points = advisory_data.good_num_points,
		good_deriv_level = advisory_data.good_deriv_level,
		good_udict = advisory_data.good_udict,
		good_varlist = advisory_data.good_varlist,
		good_DD = advisory_data.good_DD,
		all_unidentifiable = Set{Num}(),
		numerical_advisory = advisory_data.advisory,
	)
end

function is_structurally_unidentifiable(var, all_unidentifiable)
	return any(entry -> isequal(entry, var) || string(entry) == string(var), all_unidentifiable)
end

function representative_completion_value(kind::Symbol)
	if kind == :parameter
		return 1.0
	elseif kind == :state
		return 0.0
	end
	error("Unknown representative completion kind: $kind")
end

function note_provenance!(provenance::ResultProvenance, note::Symbol)
	note in provenance.notes || push!(provenance.notes, note)
	return provenance
end

function apply_representative_assignment!(
	assignments::OrderedDict{Num, Float64},
	notes::Vector{Symbol},
	var,
	kind::Symbol,
	all_unidentifiable,
)
	if !is_structurally_unidentifiable(var, all_unidentifiable)
		return nothing
	end

	value = representative_completion_value(kind)
	assignments[var] = value
	:representative_completion in notes || push!(notes, :representative_completion)
	return value
end


function set_result_lineage!(
	result::ParameterEstimationResult;
	primary_method::Union{Nothing, Symbol} = nothing,
	interpolator_source = result.interpolator_source,
	rescue_path::Union{Nothing, Symbol} = nothing,
	source_shooting_index::Union{Nothing, Int} = nothing,
	source_candidate_index::Union{Nothing, Int} = nothing,
	pre_polish_error = nothing,
	post_polish_error = nothing,
	polish_applied::Union{Nothing, Bool} = nothing,
	notes::Vector{Symbol} = Symbol[],
	source_type::Union{Nothing, Symbol} = nothing,
	multipoint_time_indices::Union{Nothing, Vector{Int}} = nothing,
	multipoint_combo_index::Union{Nothing, Int} = nothing,
)
	prov = result.provenance
	if !isnothing(primary_method)
		prov.primary_method = primary_method
	end
	prov.interpolator_source = interpolator_source
	result.interpolator_source = interpolator_source
	if !isnothing(rescue_path)
		prov.rescue_path = rescue_path
	end
	if !isnothing(source_shooting_index)
		prov.source_shooting_index = source_shooting_index
	end
	if !isnothing(source_candidate_index)
		prov.source_candidate_index = source_candidate_index
	end
	if !isnothing(pre_polish_error)
		prov.pre_polish_error = Float64(pre_polish_error)
	end
	if !isnothing(post_polish_error)
		prov.post_polish_error = Float64(post_polish_error)
	end
	if !isnothing(polish_applied)
		prov.polish_applied = polish_applied
	end
	if !isnothing(source_type)
		prov.source_type = source_type
	end
	if !isnothing(multipoint_time_indices)
		prov.multipoint_time_indices = copy(multipoint_time_indices)
	end
	if !isnothing(multipoint_combo_index)
		prov.multipoint_combo_index = multipoint_combo_index
	end
	for note in notes
		note_provenance!(prov, note)
	end
	sync_result_contract!(result)
	return result
end

function _legacy_estimator_kind(prov::ResultProvenance)::Symbol
	prov.primary_method == :direct_opt && return :direct_optimization
	prov.polish_applied && return :trajectory_polish
	prov.source_type == :synthesized_aggregate && return :synthesized_aggregate
	prov.source_type == :sensitivity_seed && return :sensitivity_seed
	prov.source_type == :branch_completed && return :branch_completed
	prov.rescue_path in (:algebraic_resolve_t0, :algebraic_resolve_shoot) && return :state_resolve
	prov.source_type == :multipoint && return :multipoint_algebraic
	# `ResultProvenance()` historically defaulted to `:single_point`. Require a
	# concrete production locator before accepting that legacy classification;
	# otherwise imported/default rows remain honestly `:unknown`.
	if prov.source_type == :single_point &&
	   (!isnothing(prov.source_shooting_index) ||
		!isnothing(prov.source_candidate_index) ||
		!isnothing(prov.interpolator_source))
		return :single_point_algebraic
	end
	return :unknown
end

"""Assign a fresh canonical estimator identity after a value-changing stage."""
function set_result_estimator_identity!(
	result::ParameterEstimationResult;
	estimator_kind::Symbol,
	data_scope::Symbol,
	time_indices::AbstractVector{<:Integer} = Int[],
	time_values::AbstractVector{<:Real} = Float64[],
	interpolator_source::Union{Nothing, Symbol} = result.provenance.interpolator_source,
	parent_candidate_ids::AbstractVector{<:Integer} = Int[],
)
	identity = _run_ctx_new_identity!(
		estimator_kind = estimator_kind,
		data_scope = data_scope,
		time_indices = time_indices,
		time_values = time_values,
		interpolator_source = interpolator_source,
		parent_candidate_ids = parent_candidate_ids,
	)
	result.provenance.estimator_identity = identity
	return identity
end

"""Give legacy/imported candidates an explicit non-fabricated identity."""
function ensure_result_estimator_identity!(result::ParameterEstimationResult)
	result.provenance.estimator_identity.candidate_id > 0 && return result.provenance.estimator_identity
	kind = _legacy_estimator_kind(result.provenance)
	indices = if kind == :multipoint_algebraic && !isnothing(result.provenance.multipoint_time_indices)
		result.provenance.multipoint_time_indices
	elseif !isnothing(result.provenance.source_shooting_index)
		Int[result.provenance.source_shooting_index]
	else
		Int[]
	end
	# Legacy multipoint rows did not retain all evaluation times. Never expand a
	# single compatibility `at_time` value into a false multi-point claim.
	times = length(indices) == 1 ? Float64[result.at_time] : Float64[]
	return set_result_estimator_identity!(result;
		estimator_kind = kind,
		data_scope = kind in (:trajectory_polish, :direct_optimization) ? :full_trajectory :
			kind == :synthesized_aggregate ? :ensemble :
			kind in (:state_resolve, :branch_completed, :sensitivity_seed) ? :derived :
			kind == :unknown ? :unknown : :point_set,
		time_indices = indices,
		time_values = times,
	)
end

function si_template_lineage_kwargs(si_template)
	if isnothing(si_template)
		return NamedTuple()
	end
	structural_fix_set = hasproperty(si_template, :structural_fix_set) ? deepcopy(si_template.structural_fix_set) : OrderedDict{Num, Float64}()
	template_status_value = hasproperty(si_template, :template_status) ? si_template.template_status : nothing
	rank_trim_meta = hasproperty(si_template, :rank_trimming_metadata) ? si_template.rank_trimming_metadata : nothing
	dropped_equations = (!isnothing(rank_trim_meta) && hasproperty(rank_trim_meta, :dropped_equation_indices)) ? copy(rank_trim_meta.dropped_equation_indices) : Int[]
	numerical_advisory = hasproperty(si_template, :numerical_advisory) ? si_template.numerical_advisory : nothing
	practical_status = hasproperty(si_template, :practical_identifiability_status) ? si_template.practical_identifiability_status : practical_status_from_advisory(numerical_advisory)
	return (
		structural_fix_set = structural_fix_set,
		template_status = template_status_value,
		equations_dropped_by_rank_trimming = dropped_equations,
		practical_identifiability_status = practical_status,
		numerical_advisory = numerical_advisory,
	)
end

"""
	process_estimation_results(PEP, solution_data, lowest_time_index; polish_solutions, polish_maxiters, polish_method)

Process the raw results from the solver into a format suitable for analysis.
Backsolves for the original model parameters and creates ParameterEstimationResult objects.

# Arguments
- `PEP`: Parameter estimation problem
- `solution_data`: Data from the solution phase
- `lowest_time_index`: The lowest time index used in the estimation
- `polish_solutions`: Whether to polish solutions using optimization
- `polish_maxiters`: Maximum iterations for polishing
- `polish_method`: Optimization method for polishing

# Returns
- Vector of ParameterEstimationResult objects
"""

"""
	_assert_bounds_length(opts, expected_len, site, layout)

Fail-fast guard for user-supplied `opt_lb`/`opt_ub`: if either is set and its
length differs from `expected_len`, throw an ArgumentError naming the site, the
expected layout, and the likely cause. Wrong-length bounds are always caller
error or upstream transform drift — the old behavior (each consumer silently
degrading differently: untransformed bounds into a scaled solve, fallback to
the ±1e6 box, skipped backsolve clamp) shipped out-of-box polished results
(docs/internal/2026-06-19_transform_bounds_mismatch.md, CSTR final_v2 evidence).
No-op when bounds are unset or lengths match.
"""
function _assert_bounds_length(opts, expected_len::Int, site::AbstractString, layout::AbstractString)
	for (name, v) in ((:opt_lb, opts.opt_lb), (:opt_ub, opts.opt_ub))
		v === nothing && continue
		length(v) == expected_len && continue
		throw(ArgumentError(
			"$(site): user-supplied `$(name)` has length $(length(v)) but the current " *
			"unknown vector needs $(expected_len) ($(layout)). A problem transform " *
			"(transcendental handling, rescaling) likely changed the variable set after " *
			"these bounds were written — see docs/internal/2026-06-19_transform_bounds_mismatch.md. " *
			"Supply bounds for the TRANSFORMED system, disable the transform " *
			"(e.g. auto_handle_transcendentals=false), or unset opt_lb/opt_ub."))
	end
	return nothing
end

"""
	_clamp_params_for_backsolve(p_map, opts, param_order, n_states) -> AbstractDict

When `opts.opt_lb` and `opts.opt_ub` are provided AND have length `n_states + n_params`,
clamp the parameter portion of `p_map` to those bounds and return a new dict (same
concrete type as the input). A length MISMATCH throws (fail-fast; was a silent
no-op — see `_assert_bounds_length`). The bounds vector
is laid out as `[state_ic_bounds; param_bounds]` (matches `compute_default_bounds`),
so parameter `i` maps to bound index `n_states + i`. Complex values are reduced to
their real part before clamping.

This is an opt-in extension of the polish-side bounds (`opts.opt_lb` / `opts.opt_ub`)
to ALSO apply at backward ODE integration. Particularly useful for cases like
cstr_1_0 where HC.jl produces spurious candidates with parameters at zero (a
degenerate solution branch where a coupling vanishes), making the backsolve diverge.
"""
function _clamp_params_for_backsolve(p_map, opts, param_order, n_states::Int)
	(isnothing(opts) || isnothing(opts.opt_lb) || isnothing(opts.opt_ub)) && return p_map
	n_params = length(param_order)
	expected_len = n_states + n_params
	_assert_bounds_length(opts, expected_len, "_clamp_params_for_backsolve (backsolve clamp)",
		"[state ICs; params] of the model in use")

	out = copy(p_map)
	for (i, p) in enumerate(param_order)
		if haskey(out, p)
			idx = n_states + i
			v_in = out[p]
			v_real = isa(v_in, Complex) ? real(v_in) : v_in
			out[p] = clamp(Float64(v_real), Float64(opts.opt_lb[idx]), Float64(opts.opt_ub[idx]))
		end
	end
	return out
end

function process_estimation_results(
	PEP::ParameterEstimationProblem,
	solution_data,
	setup_data;
	opts::Union{Nothing, EstimationOptions} = nothing,
	nooutput = false,
	polish_solutions = false,
	polish_maxiters = 20,
	polish_method = NewtonTrustRegion,
)
	# If opts not provided, construct from legacy kwargs for backward compatibility
	if isnothing(opts)
		# Map legacy function types to PolishMethod enums
		pm = if polish_method isa PolishMethod
			polish_method
		elseif polish_method === NewtonTrustRegion
			PolishNewtonTrust
		elseif polish_method === BFGS
			PolishBFGS
		elseif polish_method === LBFGS
			PolishLBFGS
		else
			PolishNewtonTrust  # default
		end
		opts = EstimationOptions(
			nooutput = nooutput,
			polish_solutions = polish_solutions,
			polish_maxiters = polish_maxiters,
			polish_method = pm,
		)
	end
	# Use opts as the single source of truth from here on
	nooutput = opts.nooutput
	polish_solutions = opts.polish_solutions
	si_lineage = si_template_lineage_kwargs(hasproperty(setup_data, :si_template) ? setup_data.si_template : nothing)

	function lookup_explicit_fixed_value(dict, candidates)
		for candidate in candidates
			haskey(dict, candidate) && return Float64(real(dict[candidate]))
		end
		candidate_strings = Set(string(c) for c in candidates)
		for (k, v) in dict
			if string(k) in candidate_strings
				return Float64(real(v))
			end
		end
		return nothing
	end

	# Extract components from the solution data
	solns = solution_data.solns
	forward_subst_dict = solution_data.forward_subst_dict
	trivial_dict = solution_data.trivial_dict
	final_varlist = solution_data.final_varlist
	trimmed_varlist = solution_data.trimmed_varlist
	# Phase B: explicit MTK→jet-0 template map (absent on legacy/branch-completion
	# fixtures — hasfield-guarded like the other optional solution_data fields).
	template_var_map = hasfield(typeof(solution_data), :template_var_map) ?
					   solution_data.template_var_map : OrderedDict{Num, Num}()

	# Extract components from the problem
	t, eqns, states, params = unpack_ODE(PEP.model.system)
	t_vector = PEP.data_sample["t"]

	# Find the lowest time index
	lowest_time_index = min(setup_data.time_index_set...)

	# Create a new model for solving ODEs
	@named new_model = ODESystem(eqns, t, states, params)
	new_model = complete(new_model)

	# Get current ordering from ModelingToolkit
	current_states = ModelingToolkit.unknowns(PEP.model.system)
	current_params = ModelingToolkit.parameters(PEP.model.system)

	# Create ordered dictionaries to preserve parameter order
	param_dict = OrderedDict(current_params .=> ones(length(current_params)))
	states_dict = OrderedDict(current_states .=> ones(length(current_states)))

	# Create a template for results
	# Create a template for the result
	result_template = ParameterEstimationResult(
		param_dict, states_dict, t_vector[lowest_time_index], nothing, nothing,
		length(PEP.data_sample["t"]), t_vector[lowest_time_index],
		OrderedDict{Num, Float64}(k => Float64(v) for (k, v) in solution_data.good_udict),
		Set{Num}(setup_data.all_unidentifiable), nothing,
	)

	# Process each solution
	results_vec = []
	for soln_index in eachindex(solns)
		# Determine shooting time index early — needed for _trfn_ state IC computation below.
		shoot_idx = hasfield(typeof(solution_data), :solution_time_indices) && soln_index <= length(solution_data.solution_time_indices) ? solution_data.solution_time_indices[soln_index] : lowest_time_index
		t_shoot = t_vector[shoot_idx]
		source_interp = hasfield(typeof(solution_data), :solution_interpolator_sources) && soln_index <= length(solution_data.solution_interpolator_sources) ? solution_data.solution_interpolator_sources[soln_index] : nothing
		# Multipoint provenance
		soln_source_type = hasfield(typeof(solution_data), :solution_source_types) && soln_index <= length(solution_data.solution_source_types) ? solution_data.solution_source_types[soln_index] : :single_point
		soln_mp_time_indices = hasfield(typeof(solution_data), :solution_mp_time_indices) && soln_index <= length(solution_data.solution_mp_time_indices) ? solution_data.solution_mp_time_indices[soln_index] : nothing
		soln_mp_combo_index = hasfield(typeof(solution_data), :solution_mp_combo_indices) && soln_index <= length(solution_data.solution_mp_combo_indices) ? solution_data.solution_mp_combo_indices[soln_index] : nothing
		soln_uq_artifact = hasfield(typeof(solution_data), :solution_uq_artifacts) && soln_index <= length(solution_data.solution_uq_artifacts) ? solution_data.solution_uq_artifacts[soln_index] : nothing
		representative_assignments = OrderedDict{Num, Float64}()
		provenance_notes = Symbol[]

		# Extract initial conditions and parameter values
		initial_conditions = [1e10 for s in states]
		parameter_values = [1e10 for p in params]

		# Lookup parameters
		for i in eachindex(params)
			param_search = if !isempty(forward_subst_dict[1])
				forward_subst_dict[1][(params[i])]
			else
				# SI workflow: prefer the explicit jet-0 template variable; fall
				# back to the MTK symbol (lookup_value name heuristics) if unmapped.
				get(template_var_map, params[i], params[i])
			end
			# In SI-template workflow forward_subst_dict is empty; avoid using random good_udict values.
			use_si_workflow = isempty(forward_subst_dict[1])
			local_good_udict = use_si_workflow ? Dict{Any, Any}() : solution_data.good_udict
			try
				parameter_values[i] = lookup_value(
					params[i], param_search,
					soln_index, local_good_udict, trivial_dict, final_varlist, trimmed_varlist, solns,
				)
			catch e
				_rethrow_if_interrupt(e)
				if use_si_workflow
					explicit_fixed_value = lookup_explicit_fixed_value(solution_data.good_udict, (params[i], param_search))
					if !isnothing(explicit_fixed_value)
						parameter_values[i] = explicit_fixed_value
						@info "Parameter $(params[i]) resolved from good_udict after SI elimination"
					else
						representative_value = apply_representative_assignment!(
							representative_assignments,
							provenance_notes,
							params[i],
							:parameter,
							setup_data.all_unidentifiable,
						)
						if !isnothing(representative_value)
							parameter_values[i] = representative_value
							@info "Parameter $(params[i]) is structurally unidentifiable; using representative completion value $(representative_value)"
						else
							error("Parameter $(params[i]) is missing from the SI solution and has no explicit fixed value. Refusing to fabricate a default.")
						end
					end
				else
					rethrow(e)
				end
			end
		end

		# Lookup initial states
		for i in eachindex(states)
			# Check if this is a _trfn_ transcendental input state (e.g. _trfn_sin_5_0(t)).
			# These were substituted out during solving — their values are known analytically.
			state_name = string(states[i])
			base_name = replace(state_name, "(t)" => "")
			trfn_info = _parse_trfn_base_name(base_name)
			if !isnothing(trfn_info)
				func_type, frequency = trfn_info
				# Compute analytical value at shooting time (0th derivative = the function itself)
				if func_type == :sin
					initial_conditions[i] = sin(frequency * t_shoot)
				elseif func_type == :cos
					initial_conditions[i] = cos(frequency * t_shoot)
				elseif func_type == :exp
					initial_conditions[i] = exp(frequency * t_shoot)
				end
				@debug "Set _trfn_ state $(states[i]) = $(initial_conditions[i]) at t=$t_shoot"
				continue
			end

			model_state_search = if !isempty(forward_subst_dict[1])
				forward_subst_dict[1][(states[i])]
			else
				# SI workflow: prefer the explicit jet-0 template variable (Phase B).
				get(template_var_map, states[i], states[i])
			end
			# Convert trivial_dict to Dict{Symbol, Any} to ensure type stability
			safe_trivial_dict = Dict{Symbol, Any}(solution_data.trivial_dict)
			use_si_workflow = isempty(forward_subst_dict[1])
			try
				initial_conditions[i] = lookup_value(
					states[i], model_state_search,
					soln_index, solution_data.good_udict, safe_trivial_dict, final_varlist, trimmed_varlist, solns,
				)
			catch e
				_rethrow_if_interrupt(e)
				if use_si_workflow
					# State not found in solver vars — it may have been eliminated because it
					# equals a measured quantity (e.g. y2 ~ y means y was treated as data).
					# Try to get its value from the data sample via measured quantities.
					found_from_mq = false
					for mq in PEP.measured_quantities
						mq_rhs = Symbolics.diff2term(mq.rhs)
						if isequal(mq_rhs, states[i])
							# This state is directly observed via this measured quantity
							mq_lhs_str = string(mq.lhs)
							mq_key = replace(mq_lhs_str, "(t)" => "")
							if haskey(PEP.data_sample, mq_key)
								initial_conditions[i] = Float64(PEP.data_sample[mq_key][shoot_idx])
								@debug "State $(states[i]) resolved from measured quantity $mq_key at t=$t_shoot"
								found_from_mq = true
								break
							end
						end
					end
					if !found_from_mq
						representative_value = apply_representative_assignment!(
							representative_assignments,
							provenance_notes,
							states[i],
							:state,
							setup_data.all_unidentifiable,
						)
						if !isnothing(representative_value)
							initial_conditions[i] = representative_value
							@info "State $(states[i]) is structurally unidentifiable; using representative completion value $(representative_value)"
						else
							error("State $(states[i]) is missing from the SI solution and is not directly observed. Refusing to fabricate an initial condition.")
						end
					end
				else
					rethrow(e)
				end
			end
		end

		# Convert to arrays of the appropriate type
		initial_conditions = convert_to_real_or_complex_array(initial_conditions)
		parameter_values = convert_to_real_or_complex_array(parameter_values)

		@debug "Processing solution $soln_index"
		@debug "Constructed initial conditions: $initial_conditions"
		@debug "Constructed parameter values: $parameter_values"

		# Solve the ODE with the estimated parameters
		tspan = (t_vector[shoot_idx], _initial_time(PEP.data_sample))
		u0_map = Dict(states .=> initial_conditions)
		p_map = Dict(params .=> parameter_values)
		# Opt-in pre-backsolve parameter clamp: when opts.opt_lb / opts.opt_ub are
		# user-provided, enforce them on parameters before backward integration.
		# No-op when bounds aren't provided.
		p_map = _clamp_params_for_backsolve(p_map, opts, params, length(states))

		prob = ODEProblem(new_model, merge(u0_map, p_map), tspan)
		ode_solution = try
			ModelingToolkit.solve(prob, PEP.solver, abstol = opts.abstol, reltol = opts.reltol)
		catch e
			_rethrow_if_interrupt(e)
			@warn "ODE solve failed during final trajectory reconstruction: $e"
			nothing
		end

		# Extract state values at the initial dataset time (backsolved to t0)
		backsolved_initial_conditions = copy(initial_conditions)
		try
			if !(ode_solution === nothing)
				for (i, s) in enumerate(states)
					val0 = ode_solution(_initial_time(PEP.data_sample), idxs = s)
					backsolved_initial_conditions[i] = Float64(real(val0))
				end
			end
		catch e
			_rethrow_if_interrupt(e)
			@warn "Failed to extract backsolved initial conditions at t0: $e"
		end

		@debug "Constructed initial conditions (at shooting time): $initial_conditions"
		@debug "Backsolved initial conditions (at t0): $backsolved_initial_conditions"
		@debug "Constructed parameter values: $parameter_values"

		# We report the initial conditions at t0 (backsolved), not the state at the shooting time
		push!(results_vec, (
			raw = [backsolved_initial_conditions; parameter_values],
			shoot_idx = shoot_idx,
			source_interp = source_interp,
			source_candidate_index = soln_index,
			representative_assignments = deepcopy(representative_assignments),
			notes = copy(provenance_notes),
			source_type = soln_source_type,
			mp_time_indices = soln_mp_time_indices,
			mp_combo_index = soln_mp_combo_index,
			uq_artifact = soln_uq_artifact,
		))
	end

	# Convert raw results to ParameterEstimationResult objects
	solved_res = []
	for (i, result_entry) in enumerate(results_vec)
		if !nooutput
			@debug "Processing solution $i for final result"
		end

		# Create a copy of the template
		push!(solved_res, deepcopy(result_template))

		# Process the raw solution
		ordered_states, ordered_params, ode_solution, err = process_raw_solution(
			result_entry.raw, PEP.model, PEP.data_sample, PEP.solver, abstol = opts.abstol, reltol = opts.reltol,
		)

		# Update result with processed data
		solved_res[end].states = ordered_states
		solved_res[end].parameters = ordered_params
		solved_res[end].solution = ode_solution
		solved_res[end].err = err
		solved_res[end].at_time = Float64(t_vector[result_entry.shoot_idx])
		solved_res[end].report_time = Float64(_initial_time(PEP.data_sample))
		identity_time_indices = result_entry.source_type == :multipoint && !isnothing(result_entry.mp_time_indices) ?
			copy(result_entry.mp_time_indices) : Int[result_entry.shoot_idx]
		identity = _run_ctx_new_identity!(
			estimator_kind = result_entry.source_type == :multipoint ? :multipoint_algebraic : :single_point_algebraic,
			data_scope = :point_set,
			time_indices = identity_time_indices,
			time_values = Float64[t_vector[idx] for idx in identity_time_indices],
			interpolator_source = result_entry.source_interp,
		)
		solved_res[end].provenance = ResultProvenance(
			primary_method = :algebraic,
			interpolator_source = result_entry.source_interp,
			rescue_path = :none,
			source_shooting_index = result_entry.shoot_idx,
			source_candidate_index = result_entry.source_candidate_index,
			post_polish_error = err,
			representative_assignments = result_entry.representative_assignments,
			source_type = result_entry.source_type,
			multipoint_time_indices = result_entry.mp_time_indices,
			multipoint_combo_index = result_entry.mp_combo_index,
			estimator_identity = identity,
			; si_lineage...,
			notes = result_entry.notes,
		)
		if !isnothing(result_entry.uq_artifact)
			_run_ctx_register_artifact!(identity.candidate_id, result_entry.uq_artifact)
		end
		sync_result_contract!(solved_res[end])
	end

	# Polish solutions if requested (shared context built once, reused for all solutions)
	if polish_solutions
		ctx = _build_polish_context(PEP; opts = opts)
		solved_res = _polish_batch_from_context(ctx, solved_res; opts = opts)
	end

	# Print solutions to match ParameterEstimation.jl output when requested.
	if !nooutput
		println("\n[ODEPE SOLUTIONS]:")
		for (i, result) in enumerate(solved_res)
			sol_dict = merge(result.states, result.parameters)
			println("Solution $i: $sol_dict")
		end
	end

	return solved_res
end
