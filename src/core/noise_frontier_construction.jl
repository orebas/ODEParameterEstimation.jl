"""
Noise-first polynomial-system construction probes.

This module builds square single-point or multipoint systems from the full SI
template equation pool, then selects among rank-complete bases by:

1. lowest maximum observed-derivative order;
2. lowest actual mixed volume among candidate bases at that order;
3. support/conditioning deterministic tie-breaks.

The standard solver path uses this construction when
`system_construction_policy = :noise_frontier`; `:legacy` keeps the older
SI-template and multipoint builders.
"""

const NoiseEqMeta = @NamedTuple{point::Int, source_index::Int, max_observed_order::Int, support_score::Float64}

struct NoiseFrontierCandidate
	n_points::Int
	strategy::Symbol
	max_observed_order::Int
	selected_equation_indices::Vector{Int}
	selected_source_indices::Vector{Int}
	dropped_equation_indices::Vector{Int}
	equations::Vector{Any}
	solve_vars::Vector{Any}
	data_vars::Vector{Any}
	eq_metadata::Vector{NoiseEqMeta}
	rank::Int
	target_rank::Int
	is_square::Bool
	rank_complete::Bool
	support_score::Float64
	solve_var_count::Int
	data_var_count::Int
	condition_proxy::Float64
	sigma_max::Float64
	sigma_min::Float64
	unfloored_svd_ratio::Float64
	rank_atol::Float64
	mixed_volume::Union{Nothing, Int}
end

function NoiseFrontierCandidate(
	n_points,
	strategy,
	max_observed_order,
	selected_equation_indices,
	selected_source_indices,
	dropped_equation_indices,
	equations,
	solve_vars,
	data_vars,
	eq_metadata,
	rank,
	target_rank,
	is_square,
	rank_complete,
	support_score,
	solve_var_count,
	data_var_count,
	condition_proxy,
	mixed_volume,
)
	return NoiseFrontierCandidate(
		n_points,
		strategy,
		max_observed_order,
		selected_equation_indices,
		selected_source_indices,
		dropped_equation_indices,
		equations,
		solve_vars,
		data_vars,
		eq_metadata,
		rank,
		target_rank,
		is_square,
		rank_complete,
		support_score,
		solve_var_count,
		data_var_count,
		condition_proxy,
		NaN,
		NaN,
		Inf,
		NaN,
		mixed_volume,
	)
end

struct NoiseFrontierResult
	n_points::Int
	minimal_max_observed_order::Union{Nothing, Int}
	selected::Union{Nothing, NoiseFrontierCandidate}
	candidates::Vector{NoiseFrontierCandidate}
	frontier::Vector{NamedTuple}
	full_equation_count::Int
	combined_equation_count::Int
	target_rank::Int
end

function _noise_strip_point_suffix(name::AbstractString)
	return replace(String(name), r"_pt\d+$" => "")
end

function _noise_observable_bases(measured_quantities)
	bases = Set{String}()
	for mq in measured_quantities
		name = replace(string(mq.lhs), "(t)" => "")
		startswith(name, "_obs_trfn_") || push!(bases, name)
	end
	return bases
end

function _noise_dd_order_maps(template_DD)
	by_obj = Dict{Any, Int}()
	by_string = Dict{String, Int}()
	isnothing(template_DD) && return by_obj, by_string
	for (level_idx, level_vars) in enumerate(template_DD.obs_lhs)
		order = level_idx - 1
		for v in level_vars
			by_obj[v] = order
			by_string[string(v)] = order
		end
	end
	return by_obj, by_string
end

function _noise_var_role(v, pep::ParameterEstimationProblem, dd_order_by_obj, dd_order_by_string, obs_bases)
	vname = string(v)
	clean = _noise_strip_point_suffix(vname)
	contains(clean, "_trfn_") && return (role = :transcendental, order = 0, base = clean)
	if haskey(dd_order_by_obj, v)
		return (role = :data, order = dd_order_by_obj[v], base = clean)
	elseif haskey(dd_order_by_string, clean)
		return (role = :data, order = dd_order_by_string[clean], base = clean)
	end
	parsed = parse_derivative_variable_name(clean)
	if !isnothing(parsed)
		base, order = parsed
		if base in obs_bases
			return (role = :data, order = order, base = base)
		end
		param_bases = Set(replace(string(p), "(t)" => "") for p in pep.model.original_parameters)
		state_bases = Set(replace(string(s), "(t)" => "") for s in pep.model.original_states)
		if base in param_bases && order == 0
			return (role = :parameter, order = 0, base = base)
		elseif base in state_bases
			return (role = order == 0 ? :state_ic : :state_derivative, order = order, base = base)
		end
	end
	return (role = :unknown, order = 0, base = clean)
end

function _noise_equation_feature(eq, pep::ParameterEstimationProblem, dd_order_by_obj, dd_order_by_string, obs_bases)
	vars = collect(Symbolics.get_variables(eq))
	data_orders = Int[]
	for v in vars
		role = _noise_var_role(v, pep, dd_order_by_obj, dd_order_by_string, obs_bases)
		role.role == :data && push!(data_orders, role.order)
	end
	max_obs = isempty(data_orders) ? -1 : maximum(data_orders)
	# Cheap structural proxy. Actual MV is used for candidate selection whenever requested.
	support_score = Float64(length(vars)) + 1.0e-3 * length(string(eq))
	return max_obs, support_score
end

function _noise_template_equations(si_template)
	if hasproperty(si_template, :all_equations)
		return collect(si_template.all_equations)
	elseif hasproperty(si_template, :rank_trimming_metadata) &&
		   hasproperty(si_template.rank_trimming_metadata, :full_equations)
		return collect(si_template.rank_trimming_metadata.full_equations)
	end
	return collect(si_template.equations)
end

function _noise_template_clean_name_map(symb_template_vars, inst_var_name_set::Set{String}, template_DD, measured_quantities)
	symb_to_clean = Dict{Any, String}()
	for v in symb_template_vars
		sname = string(v)
		if sname in inst_var_name_set
			symb_to_clean[v] = sname
			continue
		end
		found = false
		if !isnothing(template_DD)
			for (level_idx, level_vars) in enumerate(template_DD.obs_lhs)
				for (obs_idx, lhs_v) in enumerate(level_vars)
					if lhs_v === v || isequal(lhs_v, v)
						deriv_order = level_idx - 1
						obs_name = replace(string(measured_quantities[obs_idx].lhs), r"\(.*\)$" => "")
						symb_to_clean[v] = "$(obs_name)_$(deriv_order)"
						found = true
						break
					end
				end
				found && break
			end
		end
		if !found
			clean = replace(sname, r"\(.*?\)" => "")
			clean = replace(clean, r"[^a-zA-Z0-9_]" => "_")
			symb_to_clean[v] = clean
		end
	end
	return symb_to_clean
end

function _noise_model_param_set(pep::ParameterEstimationProblem)
	names = Set{String}()
	for p in pep.model.original_parameters
		name = replace(string(p), r"\(.*\)$" => "")
		push!(names, name)
		push!(names, "$(name)_0")
	end
	return names
end

function _noise_is_parameter_clean_name(clean::AbstractString, param_set::Set{String})
	clean in param_set && return true
	parsed = parse_derivative_variable_name(clean)
	isnothing(parsed) && return false
	base, order = parsed
	return order == 0 && string(base) in param_set
end

function _noise_rename_symbolic_equations(symb_eqs, symb_vars, symb_to_clean, param_set::Set{String}, point::Int)
	rd = Dict{Any, Any}()
	for v in symb_vars
		sname = string(v)
		clean = get(symb_to_clean, v, sname)
		if point == 1
			if sname != clean && !_noise_is_parameter_clean_name(clean, param_set)
				rd[v] = Symbolics.variable(Symbol(clean))
			end
		else
			_noise_is_parameter_clean_name(clean, param_set) && continue
			rd[v] = Symbolics.variable(Symbol(clean * "_pt$(point)"))
		end
	end
	return isempty(rd) ? collect(symb_eqs) : [Symbolics.substitute(eq, rd) for eq in symb_eqs]
end

function _noise_probe_indices(data_sample, n_points::Int)
	t_vec = data_sample["t"]
	n_t = length(t_vec)
	fracs = n_points == 1 ? [0.5] :
		n_points == 2 ? [0.25, 0.75] :
		collect(range(0.15, 0.85; length = n_points))
	return [max(2, min(n_t - 1, round(Int, n_t * f))) for f in fracs]
end

function _noise_rank_pool_from_symbolic_equations(
	pep::ParameterEstimationProblem,
	rank_source_eqs,
	metadata_eqs,
	points::Vector{Int},
	source_indices::Vector{Int},
	template_DD,
)
	dd_order_by_obj, dd_order_by_string = _noise_dd_order_maps(template_DD)
	obs_bases = _noise_observable_bases(pep.measured_quantities)

	combined_inst_vars_set = OrderedCollections.OrderedSet{Any}()
	for eq in rank_source_eqs
		union!(combined_inst_vars_set, Symbolics.get_variables(eq))
	end
	combined_inst_vars_all = collect(combined_inst_vars_set)
	rank_vars = Any[]
	data_subst = Dict{Any, Float64}()
	for (i, v) in enumerate(combined_inst_vars_all)
		role = _noise_var_role(v, pep, dd_order_by_obj, dd_order_by_string, obs_bases)
		if role.role in (:data, :transcendental)
			data_subst[v] = 1.0 + 0.125 * i
		else
			push!(rank_vars, v)
		end
	end
	rank_inst_eqs = isempty(data_subst) ? rank_source_eqs :
		[Symbolics.substitute(eq, data_subst) for eq in rank_source_eqs]

	metadata = NoiseEqMeta[]
	for (i, eq) in enumerate(metadata_eqs)
		max_obs, support_score = _noise_equation_feature(eq, pep, dd_order_by_obj, dd_order_by_string, obs_bases)
		push!(metadata, (point = points[i], source_index = source_indices[i], max_observed_order = max_obs, support_score = support_score))
	end

	return rank_inst_eqs, rank_vars, metadata
end

function _noise_generic_interpolants(measured_quantities, max_required_deriv::Int)
	degree = max(max_required_deriv + 3, 4)
	interpolants = Dict{Any, Any}()
	for (obs_idx, obs_eqn) in enumerate(measured_quantities)
		obs_rhs = Symbolics.diff2term(obs_eqn.rhs)
		_is_trfn_observable(Symbolics.wrap(obs_rhs)) && continue
		coeffs = Float64[1.0 + 0.173 * obs_idx + 0.037 * k for k in 0:degree]
		interpolants[obs_rhs] = let coeffs = coeffs
			x -> begin
				z = Float64(x) + 0.41
				acc = 0.0
				pow = 1.0
				for c in coeffs
					acc += c * pow
					pow *= z
				end
				acc
			end
		end
	end
	return interpolants
end

function _noise_append_source_mapped_equations!(
	combined_inst_eqs,
	combined_symb_eqs,
	source_indices::Vector{Int},
	points::Vector{Int},
	inst_renamed,
	symb_renamed,
	inst_source_indices,
	point::Int;
	diagnostics::Bool = false,
	context::AbstractString = "template",
)
	length(inst_renamed) == length(inst_source_indices) ||
		error("[NOISE-FRONTIER] $context source-index mismatch at point $point: $(length(inst_renamed)) instantiated equations vs $(length(inst_source_indices)) source indices")
	if length(inst_renamed) != length(symb_renamed) && diagnostics
		@info "[NOISE-FRONTIER] $context equation-count mismatch; using retained source indices" point = point instantiated = length(inst_renamed) symbolic = length(symb_renamed)
	end
	for (local_idx, src_idx) in enumerate(inst_source_indices)
		1 <= src_idx <= length(symb_renamed) ||
			error("[NOISE-FRONTIER] $context invalid source index $src_idx at point $point; symbolic template has $(length(symb_renamed)) equations")
		push!(combined_inst_eqs, inst_renamed[local_idx])
		push!(combined_symb_eqs, symb_renamed[src_idx])
		push!(source_indices, src_idx)
		push!(points, point)
	end
	return nothing
end

function _noise_build_generic_pool(
	pep::ParameterEstimationProblem,
	setup::NamedTuple,
	si_template,
	full_symb_eqs,
	full_symb_vars,
	probes::Vector{Int},
	template_DD;
	n_points::Int,
	diagnostics::Bool = false,
)
	max_required_deriv = isempty(si_template.deriv_dict) ? 0 : maximum(values(si_template.deriv_dict))
	generic_interpolants = _noise_generic_interpolants(pep.measured_quantities, max_required_deriv)
	combined_symb_eqs = Any[]
	combined_inst_eqs = Any[]
	source_indices = Int[]
	points = Int[]

	first_inst = instantiate_si_template_equations(
		full_symb_eqs,
		pep.measured_quantities,
		pep.data_sample,
		si_template.deriv_dict,
		template_DD;
		interpolants = generic_interpolants,
		time_index = probes[1],
		diagnostics = false,
		prune_overdetermined = false,
		substitute_trfn = false,
	)
	param_set = _noise_model_param_set(pep)
	symb_to_clean = _noise_template_clean_name_map(full_symb_vars, Set(string.(first_inst.vars)), template_DD, pep.measured_quantities)

	for pt in 1:n_points
		inst = pt == 1 ? first_inst : instantiate_si_template_equations(
			full_symb_eqs,
			pep.measured_quantities,
			pep.data_sample,
			si_template.deriv_dict,
			template_DD;
			interpolants = generic_interpolants,
			time_index = probes[pt],
			diagnostics = false,
			prune_overdetermined = false,
			substitute_trfn = false,
		)
		inst_renamed, _ = _rename_per_point_variables(inst.equations, inst.vars, param_set, pt)
		symb_renamed = _noise_rename_symbolic_equations(full_symb_eqs, full_symb_vars, symb_to_clean, param_set, pt)
		_noise_append_source_mapped_equations!(
			combined_inst_eqs,
			combined_symb_eqs,
			source_indices,
			points,
			inst_renamed,
			symb_renamed,
			inst.source_indices,
			pt;
			diagnostics = diagnostics,
			context = "generic template",
		)
	end

	rank_inst_eqs, rank_vars, metadata = _noise_rank_pool_from_symbolic_equations(
		pep,
		combined_inst_eqs,
		combined_symb_eqs,
		points,
		source_indices,
		template_DD,
	)

	return (
		symbolic_equations = combined_symb_eqs,
		instantiated_equations = rank_inst_eqs,
		instantiated_vars = rank_vars,
		metadata = metadata,
		template_DD = template_DD,
		full_equation_count = length(full_symb_eqs),
		probe_indices = probes,
	)
end

function _noise_build_pool(
	pep::ParameterEstimationProblem,
	setup::NamedTuple,
	si_template;
	n_points::Int,
	diagnostics::Bool = false,
	probe_indices::Union{Nothing, Vector{Int}} = nothing,
	selection_mode::Symbol = :generic,
)
	n_points >= 1 || throw(ArgumentError("n_points must be positive"))
	template_DD = hasproperty(si_template, :template_DD) ? si_template.template_DD : setup.good_DD
	full_symb_eqs = _noise_template_equations(si_template)
	full_vars_set = OrderedCollections.OrderedSet{Any}()
	for eq in full_symb_eqs
		union!(full_vars_set, Symbolics.get_variables(eq))
	end
	full_symb_vars = collect(full_vars_set)

	probes = isnothing(probe_indices) ? _noise_probe_indices(pep.data_sample, n_points) : probe_indices
	length(probes) == n_points || throw(ArgumentError("probe_indices length must match n_points"))

	if selection_mode == :generic
		return _noise_build_generic_pool(
			pep,
			setup,
			si_template,
			full_symb_eqs,
			full_symb_vars,
			probes,
			template_DD;
			n_points = n_points,
			diagnostics = diagnostics,
		)
	elseif !(selection_mode in (:instantiated, :data_conditioned))
		throw(ArgumentError("selection_mode must be :generic, :instantiated, or :data_conditioned (got $selection_mode)"))
	end

	interpolants = get(setup, :interpolants, nothing)
	isnothing(interpolants) && throw(ArgumentError("setup.interpolants is required for instantiated noise-frontier construction probes"))

	combined_inst_eqs = Any[]
	combined_symb_eqs = Any[]
	source_indices = Int[]
	points = Int[]

	first_inst = instantiate_si_template_equations(
		full_symb_eqs,
		pep.measured_quantities,
		pep.data_sample,
		si_template.deriv_dict,
		template_DD;
		interpolants = interpolants,
		time_index = probes[1],
		diagnostics = false,
		prune_overdetermined = false,
	)
	roles = _classify_polynomial_variables(string.(first_inst.vars), pep)
	param_set = Set(vn for (vn, role) in roles if role == :parameter)
	symb_to_clean = _noise_template_clean_name_map(full_symb_vars, Set(string.(first_inst.vars)), template_DD, pep.measured_quantities)

	for pt in 1:n_points
		inst = pt == 1 ? first_inst : instantiate_si_template_equations(
			full_symb_eqs,
			pep.measured_quantities,
			pep.data_sample,
			si_template.deriv_dict,
			template_DD;
			interpolants = interpolants,
			time_index = probes[pt],
			diagnostics = false,
			prune_overdetermined = false,
		)
		inst_renamed, _ = _rename_per_point_variables(inst.equations, inst.vars, param_set, pt)
		symb_renamed = _noise_rename_symbolic_equations(full_symb_eqs, full_symb_vars, symb_to_clean, param_set, pt)
		_noise_append_source_mapped_equations!(
			combined_inst_eqs,
			combined_symb_eqs,
			source_indices,
			points,
			inst_renamed,
			symb_renamed,
			inst.source_indices,
			pt;
			diagnostics = diagnostics,
			context = "data-conditioned template",
		)
	end

	rank_inst_eqs, rank_vars, metadata = _noise_rank_pool_from_symbolic_equations(
		pep,
		combined_inst_eqs,
		combined_symb_eqs,
		points,
		source_indices,
		template_DD,
	)

	return (
		symbolic_equations = combined_symb_eqs,
		instantiated_equations = rank_inst_eqs,
		instantiated_vars = rank_vars,
		metadata = metadata,
		template_DD = template_DD,
		full_equation_count = length(full_symb_eqs),
		probe_indices = probes,
	)
end

function _noise_rank_matrix(equations, variables; rank_atol::Float64 = 1e-8, n_rank_probes::Int = 3)
	rank_t0 = time()
	rank_stages = OrderedDict{Symbol, Float64}()
	n_var = length(variables)
	n_eq = length(equations)
	if n_var == 0 || n_eq == 0
		_record_detailed_timing!((
			category = :noise_rank_matrix,
			context = _current_detailed_timing_context(),
			total_seconds = time() - rank_t0,
			stage_seconds = copy(rank_stages),
			equation_count = n_eq,
			variable_count = n_var,
			n_rank_probes = n_rank_probes,
			finite_probe_count = 0,
			best_rank = 0,
		))
		return zeros(Float64, n_eq, n_var), 0
	end
	_compile_t0 = time()
	f = _compile_system_function(equations, variables)
	rank_stages[:compile_system_function] = get(rank_stages, :compile_system_function, 0.0) + (time() - _compile_t0)
	best_J = zeros(Float64, n_eq, n_var)
	best_rank = -1
	finite_probe_count = 0
	rng = MersenneTwister(0x9f4d0329)
	for _ in 1:max(1, n_rank_probes)
		x = randn(rng, n_var) .* 3.0
		_jacobian_t0 = time()
		J = try
			# Generated systems are used for only a few rank probes. Large dual
			# chunks can spend minutes in LLVM optimization on Julia 1.13;
			# scalar chunks keep compilation small while evaluating the same
			# derivatives, at the cost of more calls to the compiled system.
			config = ForwardDiff.JacobianConfig(f, x, ForwardDiff.Chunk{1}())
			ForwardDiff.jacobian(f, x, config)
		catch err
			_rethrow_if_interrupt(err)
			zeros(Float64, n_eq, n_var)
		finally
			rank_stages[:forwarddiff_jacobian] = get(rank_stages, :forwarddiff_jacobian, 0.0) + (time() - _jacobian_t0)
		end
		all(isfinite, J) || continue
		finite_probe_count += 1
		_rank_eval_t0 = time()
		r = rank(J; atol = rank_atol)
		rank_stages[:rank_eval] = get(rank_stages, :rank_eval, 0.0) + (time() - _rank_eval_t0)
		if r > best_rank
			best_rank = r
			best_J = Float64.(J)
		end
	end
	_record_detailed_timing!((
		category = :noise_rank_matrix,
		context = _current_detailed_timing_context(),
		total_seconds = time() - rank_t0,
		stage_seconds = copy(rank_stages),
		equation_count = n_eq,
		variable_count = n_var,
		n_rank_probes = n_rank_probes,
		finite_probe_count = finite_probe_count,
		best_rank = max(best_rank, 0),
	))
	return best_J, max(best_rank, 0)
end

function _noise_rank_of_rows(J, rows; rank_atol::Float64 = 1e-8)
	isempty(rows) && return 0
	return rank(J[collect(rows), :]; atol = rank_atol)
end

function _noise_greedy_basis(J, ordered_indices, target_rank::Int; rank_atol::Float64 = 1e-8)
	selected = Int[]
	current = zeros(Float64, 0, size(J, 2))
	current_rank = 0
	for idx in ordered_indices
		trial = vcat(current, J[idx:idx, :])
		r = rank(trial; atol = rank_atol)
		if r > current_rank
			push!(selected, idx)
			current = trial
			current_rank = r
		end
		current_rank == target_rank && break
	end
	return current_rank == target_rank && length(selected) == target_rank ? selected : Int[]
end

function _noise_dynamic_basis(J, allowed, metadata, symbolic_equations, pep, dd_order_by_obj, dd_order_by_string, obs_bases,
		target_rank::Int; tie_policy::Symbol, rank_atol::Float64 = 1e-8)
	selected = Int[]
	introduced = Set{String}()
	remaining = Set(allowed)
	current = zeros(Float64, 0, size(J, 2))
	current_rank = 0
	while current_rank < target_rank
		best = nothing
		best_key = nothing
		for idx in sort(collect(remaining))
			trial = vcat(current, J[idx:idx, :])
			r = rank(trial; atol = rank_atol)
			r > current_rank || continue
			var_tokens = Set(string.(Symbolics.get_variables(symbolic_equations[idx])))
			new_tokens = setdiff(var_tokens, introduced)
			new_data_orders = Int[]
			for v in Symbolics.get_variables(symbolic_equations[idx])
				string(v) in new_tokens || continue
				role = _noise_var_role(v, pep, dd_order_by_obj, dd_order_by_string, obs_bases)
				role.role == :data && push!(new_data_orders, role.order)
			end
			max_new_data = isempty(new_data_orders) ? 999 : maximum(new_data_orders)
			marginal_count = length(new_tokens)
			key = if tie_policy == :data_order
				(max_new_data, marginal_count, metadata[idx].support_score, idx)
			else
				(marginal_count, metadata[idx].support_score, max_new_data, idx)
			end
			if isnothing(best_key) || key < best_key
				best = idx
				best_key = key
			end
		end
		isnothing(best) && break
		push!(selected, best)
		union!(introduced, Set(string.(Symbolics.get_variables(symbolic_equations[best]))))
		current = vcat(current, J[best:best, :])
		current_rank = rank(current; atol = rank_atol)
		delete!(remaining, best)
	end
	return current_rank == target_rank && length(selected) == target_rank ? selected : Int[]
end

function _noise_removal_basis(J, allowed, metadata, target_rank::Int; rank_atol::Float64 = 1e-8)
	kept = sort(collect(allowed))
	for idx in sort(kept; by = i -> (-metadata[i].max_observed_order, -metadata[i].support_score, -i))
		length(kept) <= target_rank && break
		trial = setdiff(kept, [idx])
		if _noise_rank_of_rows(J, trial; rank_atol = rank_atol) >= target_rank
			kept = trial
		end
	end
	return length(kept) == target_rank && _noise_rank_of_rows(J, kept; rank_atol = rank_atol) == target_rank ? sort(kept) : Int[]
end

function _noise_basis_score(indices, metadata)
	isempty(indices) && return Inf
	return sum(metadata[i].support_score for i in indices)
end

function _noise_exchange_candidates(J, seeds, allowed, metadata, target_rank::Int; beam_width::Int, rank_atol::Float64 = 1e-8)
	beam = unique([sort(s) for s in seeds if length(s) == target_rank])
	sort!(beam; by = s -> _noise_basis_score(s, metadata))
	length(beam) > beam_width && resize!(beam, beam_width)
	for _ in 1:2
		next = copy(beam)
		for basis in beam
			basis_set = Set(basis)
			for out_idx in basis
				for in_idx in allowed
					in_idx in basis_set && continue
					trial = sort!(vcat(setdiff(basis, [out_idx]), [in_idx]))
					if _noise_rank_of_rows(J, trial; rank_atol = rank_atol) == target_rank
						push!(next, trial)
					end
				end
			end
		end
		unique!(next)
		sort!(next; by = s -> _noise_basis_score(s, metadata))
		length(next) > beam_width && resize!(next, beam_width)
		beam = next
	end
	return beam
end

function _noise_classify_selected_vars(equations, pep, dd_order_by_obj, dd_order_by_string, obs_bases)
	vars_set = OrderedCollections.OrderedSet{Any}()
	for eq in equations
		union!(vars_set, Symbolics.get_variables(eq))
	end
	solve_vars = Any[]
	data_vars = Any[]
	for v in vars_set
		role = _noise_var_role(v, pep, dd_order_by_obj, dd_order_by_string, obs_bases)
		if role.role in (:data, :transcendental)
			push!(data_vars, v)
		else
			push!(solve_vars, v)
		end
	end
	return solve_vars, data_vars
end

function _noise_svd_diagnostics(
	J,
	selected;
	rank_atol::Float64 = 1e-8,
	rank_value::Union{Nothing, Int} = nothing,
	target_rank::Union{Nothing, Int} = nothing,
)
	length(selected) == 0 && return (
		sigma_max = NaN,
		sigma_min = NaN,
		unfloored_svd_ratio = Inf,
		condition_proxy = Inf,
		rank_atol = rank_atol,
	)
	Js = J[selected, :]
	svs = try
		svdvals(Js)
	catch err
		_rethrow_if_interrupt(err)
		return (
			sigma_max = NaN,
			sigma_min = NaN,
			unfloored_svd_ratio = Inf,
			condition_proxy = Inf,
			rank_atol = rank_atol,
		)
	end
	isempty(svs) && return (
		sigma_max = NaN,
		sigma_min = NaN,
		unfloored_svd_ratio = Inf,
		condition_proxy = Inf,
		rank_atol = rank_atol,
	)
	sigma_max = Float64(svs[1])
	sigma_min = Float64(svs[end])
	condition_proxy = size(Js, 1) < size(Js, 2) ? Inf : sigma_max / max(sigma_min, rank_atol)
	full_rank = if !isnothing(rank_value) && !isnothing(target_rank)
		rank_value >= target_rank
	else
		size(Js, 1) >= size(Js, 2) && rank(Js; atol = rank_atol) >= size(Js, 2)
	end
	unfloored_svd_ratio =
		full_rank && isfinite(sigma_max) && isfinite(sigma_min) && sigma_min > 0.0 ?
		sigma_max / sigma_min : Inf
	return (
		sigma_max = sigma_max,
		sigma_min = sigma_min,
		unfloored_svd_ratio = unfloored_svd_ratio,
		condition_proxy = condition_proxy,
		rank_atol = rank_atol,
	)
end


struct NoiseValidationCache
	usable::Bool
	data_dependent::Bool
	jacobian_oop::Any
	jacobian_ip::Any
	equation_count::Int
	variable_count::Int
	data_var_count::Int
	rank_value::Int
	condition_proxy::Float64
	sigma_max::Float64
	sigma_min::Float64
	unfloored_svd_ratio::Float64
	rank_atol::Float64
	finite_probe_count::Int
	failure_reason::Symbol
end

const _NOISE_VALIDATION_CACHE = Dict{Any, NoiseValidationCache}()
const _NOISE_VALIDATION_CACHE_LOCK = ReentrantLock()

function _noise_validation_cache_key(equations, solve_vars, data_vars, rank_atol::Float64, n_rank_probes::Int)
	# CONTENT hash (mirrors `_hc_structure_key`), NOT objectid. objectid of a
	# mutable Vector is address-derived and can be reused after the original is
	# GC'd, so an objectid-keyed entry could FALSE-HIT across estimation calls
	# in one process and return the wrong compiled Jacobian for a structurally
	# different system (→ a rank/validity decision on the wrong system). Hashing
	# the printed structure keys on content: identical systems correctly share
	# an entry, different systems never collide.
	return (
		hash(string.(equations)), length(equations),
		hash(string.(solve_vars)), length(solve_vars),
		hash(string.(data_vars)), length(data_vars),
		rank_atol,
		n_rank_probes,
	)
end

function _noise_jacobian_uses_data(J_expr, data_vars)
	isempty(data_vars) && return false
	data_names = Set(string.(data_vars))
	for expr in J_expr
		for v in Symbolics.get_variables(expr)
			string(v) in data_names && return true
		end
	end
	return false
end

function _noise_compile_jacobian_function(J_expr, all_vars)
	built = Symbolics.build_function(vec(J_expr), all_vars; expression = Val(false))
	if built isa Tuple
		return built[1], length(built) >= 2 ? built[2] : nothing
	end
	return built, nothing
end

function _noise_evaluate_compiled_jacobian(cache::NoiseValidationCache, x, data_values)
	input_values = Vector{Float64}(undef, length(x) + length(data_values))
	@inbounds begin
		for i in eachindex(x)
			input_values[i] = Float64(x[i])
		end
		offset = length(x)
		for i in eachindex(data_values)
			input_values[offset + i] = Float64(data_values[i])
		end
	end

	raw = if cache.jacobian_ip !== nothing
		buf = zeros(Float64, cache.equation_count * cache.variable_count)
		cache.jacobian_ip(buf, input_values)
		buf
	else
		cache.jacobian_oop(input_values)
	end
	vals = raw isa AbstractArray ? Float64.(raw) : Float64[Float64(raw)]
	return reshape(vals, cache.equation_count, cache.variable_count)
end

function _noise_rank_from_compiled_jacobian(
	cache::NoiseValidationCache,
	data_values;
	rank_atol::Float64 = 1e-8,
	n_rank_probes::Int = 3,
	stage_seconds = nothing,
)
	n_var = cache.variable_count
	n_eq = cache.equation_count
	if n_var == 0 || n_eq == 0
		return zeros(Float64, n_eq, n_var), 0, 0
	end
	best_J = zeros(Float64, n_eq, n_var)
	best_rank = -1
	finite_probe_count = 0
	rng = MersenneTwister(0x9f4d0329)
	for _ in 1:max(1, n_rank_probes)
		x = randn(rng, n_var) .* 3.0
		_eval_t0 = time()
		J = _noise_evaluate_compiled_jacobian(cache, x, data_values)
		if !isnothing(stage_seconds)
			stage_seconds[:evaluate_cached_jacobian] = get(stage_seconds, :evaluate_cached_jacobian, 0.0) + (time() - _eval_t0)
		end
		all(isfinite, J) || continue
		finite_probe_count += 1
		_rank_t0 = time()
		r = rank(J; atol = rank_atol)
		if !isnothing(stage_seconds)
			stage_seconds[:rank_eval] = get(stage_seconds, :rank_eval, 0.0) + (time() - _rank_t0)
		end
		if r > best_rank
			best_rank = r
			best_J = Float64.(J)
		end
	end
	return best_J, max(best_rank, 0), finite_probe_count
end

function _noise_store_validation_cache!(key, cache::NoiseValidationCache)
	lock(_NOISE_VALIDATION_CACHE_LOCK)
	try
		return get!(_NOISE_VALIDATION_CACHE, key, cache)
	finally
		unlock(_NOISE_VALIDATION_CACHE_LOCK)
	end
end

function _noise_prepare_validation_cache(
	equations,
	solve_vars,
	data_vars;
	rank_atol::Float64 = 1e-8,
	n_rank_probes::Int = 3,
)
	key = _noise_validation_cache_key(equations, solve_vars, data_vars, rank_atol, n_rank_probes)
	lock(_NOISE_VALIDATION_CACHE_LOCK)
	try
		cached = get(_NOISE_VALIDATION_CACHE, key, nothing)
		!isnothing(cached) && return cached
	finally
		unlock(_NOISE_VALIDATION_CACHE_LOCK)
	end

	cache = try
		J_expr = Symbolics.jacobian(equations, solve_vars)
		data_dependent = _noise_jacobian_uses_data(J_expr, data_vars)
		jac_oop, jac_ip = _noise_compile_jacobian_function(J_expr, vcat(solve_vars, data_vars))
		provisional = NoiseValidationCache(
			true,
			data_dependent,
			jac_oop,
			jac_ip,
			length(equations),
			length(solve_vars),
			length(data_vars),
			-1,
			Inf,
			NaN,
			NaN,
			Inf,
			rank_atol,
			0,
			:ok,
		)
		if data_dependent
			provisional
		else
			J, rank_value, finite_probe_count = _noise_rank_from_compiled_jacobian(
				provisional,
				zeros(Float64, length(data_vars));
				rank_atol = rank_atol,
				n_rank_probes = n_rank_probes,
			)
			svd_diag = _noise_svd_diagnostics(
				J,
				collect(eachindex(equations));
				rank_atol = rank_atol,
				rank_value = rank_value,
				target_rank = length(solve_vars),
			)
			NoiseValidationCache(
				true,
				false,
				jac_oop,
				jac_ip,
				length(equations),
				length(solve_vars),
				length(data_vars),
				rank_value,
				svd_diag.condition_proxy,
				svd_diag.sigma_max,
				svd_diag.sigma_min,
				svd_diag.unfloored_svd_ratio,
				svd_diag.rank_atol,
				finite_probe_count,
				:ok,
			)
		end
	catch err
		_rethrow_if_interrupt(err)
		@warn "[NOISE-FRONTIER] cached Jacobian validation unavailable; using slow validation" exception = err
		NoiseValidationCache(
			false,
			false,
			nothing,
			nothing,
			length(equations),
			length(solve_vars),
			length(data_vars),
			0,
			Inf,
			NaN,
			NaN,
			Inf,
			rank_atol,
			0,
			:cache_build_failed,
		)
	end
	return _noise_store_validation_cache!(key, cache)
end

"""
	_mixed_volume_without_progress(hc_system)

The mixed volume of a square system, as `HomotopyContinuation.mixed_volume`
computes it. That function always draws a progress meter on stderr, which a
quiet run must not, so this goes to MixedSubdivisions with the meter off.
"""
function _mixed_volume_without_progress(hc_system)
	# The homogeneous case needs HomotopyContinuation's own preparation.
	HomotopyContinuation.is_homogeneous(hc_system) && return HomotopyContinuation.mixed_volume(hc_system)
	support, _ = HomotopyContinuation.support_coefficients(hc_system)
	return HomotopyContinuation.MixedSubdivisions.mixed_volume(support; show_progress = false)
end

function _noise_mixed_volume(equations, solve_vars, data_vars)
	length(equations) == length(solve_vars) || return nothing
	try
		if isempty(data_vars)
			hc_system, _ = convert_to_hc_format(equations, solve_vars)
			return Int(_mixed_volume_without_progress(hc_system))
		end
		hc_system, _, _ = convert_to_hc_format_with_params(equations, solve_vars, data_vars)
		return Int(_mixed_volume_without_progress(hc_system))
	catch e
		_rethrow_if_interrupt(e)
		@warn "[NOISE-FRONTIER] mixed_volume failed" exception = e
		return nothing
	end
end

function _noise_candidate_from_indices(
	indices,
	strategy::Symbol,
	cap::Int,
	pool,
	J,
	pep,
	dd_order_by_obj,
	dd_order_by_string,
	obs_bases;
	compute_mixed_volume::Bool,
	rank_atol::Float64 = 1e-8,
)
	selected = sort(collect(indices))
	equations = Any[pool.symbolic_equations[i] for i in selected]
	solve_vars, data_vars = _noise_classify_selected_vars(equations, pep, dd_order_by_obj, dd_order_by_string, obs_bases)
	r = _noise_rank_of_rows(J, selected; rank_atol = rank_atol)
	target_rank = length(pool.instantiated_vars)
	is_square = length(equations) == length(solve_vars)
	rank_complete = r == target_rank
	mv = compute_mixed_volume && is_square && rank_complete ? _noise_mixed_volume(equations, solve_vars, data_vars) : nothing
	all_indices = collect(eachindex(pool.symbolic_equations))
	svd_diag = _noise_svd_diagnostics(
		J,
		selected;
		rank_atol = rank_atol,
		rank_value = r,
		target_rank = target_rank,
	)
	return NoiseFrontierCandidate(
		pool.n_points,
		strategy,
		cap,
		selected,
		Int[pool.metadata[i].source_index for i in selected],
		setdiff(all_indices, selected),
		equations,
		solve_vars,
		data_vars,
		NoiseEqMeta[pool.metadata[i] for i in selected],
		r,
		target_rank,
		is_square,
		rank_complete,
		_noise_basis_score(selected, pool.metadata),
		length(solve_vars),
		length(data_vars),
		svd_diag.condition_proxy,
		svd_diag.sigma_max,
		svd_diag.sigma_min,
		svd_diag.unfloored_svd_ratio,
		svd_diag.rank_atol,
		mv,
	)
end

function _noise_candidate_sort_key(c::NoiseFrontierCandidate)
	mv_key = isnothing(c.mixed_volume) ? typemax(Int) : c.mixed_volume
	return (mv_key, c.support_score, c.solve_var_count, c.condition_proxy, string(c.strategy), join(c.selected_equation_indices, " "))
end

function _noise_observable_index_by_name(measured_quantities)
	obs_name_to_idx = Dict{String, Int}()
	for (idx, mq_eq) in enumerate(measured_quantities)
		obs_name = replace(string(mq_eq.lhs), r"\(.*\)$" => "")
		obs_name_to_idx[obs_name] = idx
	end
	return obs_name_to_idx
end

function _noise_evaluate_data_var(v, interpolants, measured_quantities, obs_name_to_idx, t_point)
	clean_name = _noise_strip_point_suffix(string(v))
	trfn_val = evaluate_trfn_template_variable(clean_name, t_point)
	isnothing(trfn_val) && (trfn_val = evaluate_obs_trfn_template_variable(clean_name, t_point))
	if !isnothing(trfn_val)
		return Float64(trfn_val)
	end

	parsed = parse_derivative_variable_name(clean_name)
	if isnothing(parsed)
		@warn "[NOISE-FRONTIER] Cannot parse data variable; marking value nonfinite" data_var = string(v)
		return NaN
	end
	base_name, deriv_order = parsed
	obs_idx = get(obs_name_to_idx, string(base_name), nothing)
	if isnothing(obs_idx)
		m = match(r"^y(\d+)$", string(base_name))
		!isnothing(m) && (obs_idx = parse(Int, m.captures[1]))
	end

	if !isnothing(obs_idx) && obs_idx <= length(measured_quantities)
		obs_rhs = Symbolics.diff2term(measured_quantities[obs_idx].rhs)
		if haskey(interpolants, obs_rhs)
			return Float64(_estimation_derivative(
				interpolants[obs_rhs], deriv_order, t_point,
			))
		end
		obs_lhs_wrapped = Symbolics.wrap(measured_quantities[obs_idx].lhs)
		if haskey(interpolants, obs_lhs_wrapped)
			return Float64(_estimation_derivative(
				interpolants[obs_lhs_wrapped], deriv_order, t_point,
			))
		end
	end

	@warn "[NOISE-FRONTIER] No interpolant for data variable; marking value nonfinite" data_var = string(v) base = string(base_name) deriv_order obs_idx
	return NaN
end

"""
	evaluate_noise_frontier_data_vars_at_point(interpolants, data_vars, measured_quantities, t_point)

Evaluate the selected frontier candidate's data variables in the exact order
expected by HC parameter homotopy. Data variables are clean SIAN-style names
(`y1_0`, `y2_3`, possibly suffixed with `_ptK`) plus optional `_trfn_` variables.
"""
function evaluate_noise_frontier_data_vars_at_point(interpolants, data_vars, measured_quantities, t_point)
	obs_name_to_idx = _noise_observable_index_by_name(measured_quantities)
	return Float64[
		_noise_evaluate_data_var(v, interpolants, measured_quantities, obs_name_to_idx, t_point)
		for v in data_vars
	]
end

function instantiate_noise_frontier_candidate(candidate::NoiseFrontierCandidate, interpolants, data_sample, measured_quantities, time_index::Int)
	t_point = data_sample["t"][time_index]
	values = evaluate_noise_frontier_data_vars_at_point(interpolants, candidate.data_vars, measured_quantities, t_point)
	subst = Dict{Any, Any}(candidate.data_vars[i] => values[i] for i in eachindex(candidate.data_vars))
	return [Symbolics.substitute(eq, subst) for eq in candidate.equations], candidate.solve_vars
end

function _noise_validation_result(;
	valid::Bool,
	reason::Symbol,
	rank::Int,
	target_rank::Int,
	rank_atol::Float64,
	condition_proxy::Float64 = Inf,
	sigma_max::Float64 = NaN,
	sigma_min::Float64 = NaN,
	unfloored_svd_ratio::Float64 = Inf,
)
	return (
		valid = valid,
		reason = reason,
		rank = rank,
		target_rank = target_rank,
		sigma_max = sigma_max,
		sigma_min = sigma_min,
		unfloored_svd_ratio = unfloored_svd_ratio,
		condition_proxy = condition_proxy,
		rank_atol = rank_atol,
	)
end

function validate_noise_frontier_instantiation(
	equations,
	solve_vars,
	data_vars,
	data_values;
	rank_atol::Float64 = 1e-8,
	n_rank_probes::Int = 3,
)
	validation_t0 = time()
	validation_stages = OrderedDict{Symbol, Float64}()
	target_rank = length(solve_vars)
	length(equations) == target_rank || return _noise_validation_result(
		valid = false,
		reason = :nonsquare,
		rank = 0,
		target_rank = target_rank,
		rank_atol = rank_atol,
	)
	length(data_values) == length(data_vars) || return _noise_validation_result(
		valid = false,
		reason = :data_length_mismatch,
		rank = 0,
		target_rank = target_rank,
		rank_atol = rank_atol,
	)
	all(isfinite, data_values) || return _noise_validation_result(
		valid = false,
		reason = :nonfinite_data,
		rank = 0,
		target_rank = target_rank,
		rank_atol = rank_atol,
	)

	_cache_t0 = time()
	cache = _noise_prepare_validation_cache(
		equations,
		solve_vars,
		data_vars;
		rank_atol = rank_atol,
		n_rank_probes = n_rank_probes,
	)
	validation_stages[:prepare_validation_cache] = get(validation_stages, :prepare_validation_cache, 0.0) + (time() - _cache_t0)

	if cache.usable
		cached_rank_value = 0
		cached_condition_proxy = Inf
		cached_sigma_max = NaN
		cached_sigma_min = NaN
		cached_unfloored_svd_ratio = Inf
		cached_rank_atol = rank_atol
		cached_finite_probe_count = 0
		cached_mode = cache.data_dependent ? :cached_data_dependent_jacobian : :cached_data_independent_rank
		_cached_t0 = time()
		cached_ok = try
			if cache.data_dependent
				J, cached_rank_value, cached_finite_probe_count = _noise_rank_from_compiled_jacobian(
					cache,
					data_values;
					rank_atol = rank_atol,
					n_rank_probes = n_rank_probes,
					stage_seconds = validation_stages,
				)
				_condition_t0 = time()
				svd_diag = _noise_svd_diagnostics(
					J,
					collect(eachindex(equations));
					rank_atol = rank_atol,
					rank_value = cached_rank_value,
					target_rank = target_rank,
				)
				cached_condition_proxy = svd_diag.condition_proxy
				cached_sigma_max = svd_diag.sigma_max
				cached_sigma_min = svd_diag.sigma_min
				cached_unfloored_svd_ratio = svd_diag.unfloored_svd_ratio
				cached_rank_atol = svd_diag.rank_atol
				validation_stages[:condition_proxy] = get(validation_stages, :condition_proxy, 0.0) + (time() - _condition_t0)
			else
				cached_rank_value = cache.rank_value
				cached_condition_proxy = cache.condition_proxy
				cached_sigma_max = cache.sigma_max
				cached_sigma_min = cache.sigma_min
				cached_unfloored_svd_ratio = cache.unfloored_svd_ratio
				cached_rank_atol = cache.rank_atol
				cached_finite_probe_count = cache.finite_probe_count
				validation_stages[:reuse_data_independent_rank] = get(validation_stages, :reuse_data_independent_rank, 0.0) + 0.0
			end
			true
		catch err
			_rethrow_if_interrupt(err)
			@warn "[NOISE-FRONTIER] cached Jacobian validation failed; using slow validation" exception = err
			false
		finally
			validation_stages[:cached_validation] = get(validation_stages, :cached_validation, 0.0) + (time() - _cached_t0)
		end
		if cached_ok
			valid = cached_rank_value >= target_rank
			reason = valid ? :ok : :rank_deficient
			_record_detailed_timing!((
				category = :noise_frontier_validation,
				context = _current_detailed_timing_context(),
				total_seconds = time() - validation_t0,
				stage_seconds = copy(validation_stages),
				equation_count = length(equations),
				variable_count = length(solve_vars),
				data_var_count = length(data_vars),
				data_value_count = length(data_values),
				rank = cached_rank_value,
				target_rank = target_rank,
				reason = reason,
				sigma_max = cached_sigma_max,
				sigma_min = cached_sigma_min,
				unfloored_svd_ratio = cached_unfloored_svd_ratio,
				condition_proxy = cached_condition_proxy,
				rank_atol = cached_rank_atol,
				n_rank_probes = n_rank_probes,
				validation_mode = cached_mode,
				jacobian_data_dependent = cache.data_dependent,
				cache_usable = true,
				finite_probe_count = cached_finite_probe_count,
			))
			return _noise_validation_result(
				valid = valid,
				reason = reason,
				rank = cached_rank_value,
				target_rank = target_rank,
				condition_proxy = cached_condition_proxy,
				sigma_max = cached_sigma_max,
				sigma_min = cached_sigma_min,
				unfloored_svd_ratio = cached_unfloored_svd_ratio,
				rank_atol = cached_rank_atol,
			)
		end
	end

	_substitute_t0 = time()
	subst = Dict{Any, Any}(data_vars[i] => data_values[i] for i in eachindex(data_vars))
	instantiated_eqs = isempty(subst) ? equations : [Symbolics.substitute(eq, subst) for eq in equations]
	validation_stages[:substitute_data] = get(validation_stages, :substitute_data, 0.0) + (time() - _substitute_t0)

	_rank_matrix_t0 = time()
	J, rank_value = _noise_rank_matrix(instantiated_eqs, solve_vars;
		rank_atol = rank_atol, n_rank_probes = n_rank_probes)
	validation_stages[:rank_matrix] = get(validation_stages, :rank_matrix, 0.0) + (time() - _rank_matrix_t0)

	_condition_t0 = time()
	svd_diag = _noise_svd_diagnostics(
		J,
		collect(eachindex(instantiated_eqs));
		rank_atol = rank_atol,
		rank_value = rank_value,
		target_rank = target_rank,
	)
	validation_stages[:condition_proxy] = get(validation_stages, :condition_proxy, 0.0) + (time() - _condition_t0)
	valid = rank_value >= target_rank
	reason = valid ? :ok : :rank_deficient
	_record_detailed_timing!((
		category = :noise_frontier_validation,
		context = _current_detailed_timing_context(),
		total_seconds = time() - validation_t0,
		stage_seconds = copy(validation_stages),
		equation_count = length(equations),
		variable_count = length(solve_vars),
		data_var_count = length(data_vars),
		data_value_count = length(data_values),
		rank = rank_value,
		target_rank = target_rank,
		reason = reason,
		sigma_max = svd_diag.sigma_max,
		sigma_min = svd_diag.sigma_min,
		unfloored_svd_ratio = svd_diag.unfloored_svd_ratio,
		condition_proxy = svd_diag.condition_proxy,
		rank_atol = svd_diag.rank_atol,
		n_rank_probes = n_rank_probes,
		validation_mode = :slow_substitute_forwarddiff,
		jacobian_data_dependent = cache.usable ? cache.data_dependent : missing,
		cache_usable = cache.usable,
		cache_failure_reason = cache.failure_reason,
	))
	return _noise_validation_result(
		valid = valid,
		reason = reason,
		rank = rank_value,
		target_rank = target_rank,
		condition_proxy = svd_diag.condition_proxy,
		sigma_max = svd_diag.sigma_max,
		sigma_min = svd_diag.sigma_min,
		unfloored_svd_ratio = svd_diag.unfloored_svd_ratio,
		rank_atol = svd_diag.rank_atol,
	)
end

function validate_noise_frontier_candidate_at_values(
	candidate::NoiseFrontierCandidate,
	data_values;
	rank_atol::Float64 = 1e-8,
	n_rank_probes::Int = 3,
)
	return validate_noise_frontier_instantiation(
		candidate.equations,
		candidate.solve_vars,
		candidate.data_vars,
		data_values;
		rank_atol = rank_atol,
		n_rank_probes = n_rank_probes,
	)
end

function _noise_point_index_for_var(v, n_points::Int)
	name = string(v)
	for pt in 2:n_points
		endswith(name, "_pt$pt") && return pt
	end
	return 1
end

function _noise_order_data_vars_by_point(data_vars, n_points::Int)
	return sort(collect(data_vars); by = v -> (_noise_point_index_for_var(v, n_points), string(v)))
end

function _noise_per_point_indices(data_vars, n_points::Int)
	# Exact index lists (no first:last collapse — see _per_point_data_indices).
	per_point_indices = [Int[] for _ in 1:n_points]
	for (i, dv) in enumerate(data_vars)
		push!(per_point_indices[_noise_point_index_for_var(dv, n_points)], i)
	end
	return per_point_indices
end

function _noise_param_indices_and_names(solve_vars, pep::ParameterEstimationProblem)
	param_names_set = Set(replace(string(p), "(t)" => "") for p in keys(pep.p_true))
	param_indices = Int[]
	param_names = String[]
	for (i, v) in enumerate(solve_vars)
		clean = _noise_strip_point_suffix(string(v))
		clean = replace(clean, r"_0$" => "")
		clean = replace(clean, r"\(.*\)$" => "")
		if clean in param_names_set
			push!(param_indices, i)
			push!(param_names, clean)
		end
	end
	return param_indices, param_names
end

"""
	build_noise_frontier_multipoint_template(pep, setup, si_template; kwargs...) -> MultiPointTemplate

Build a production `MultiPointTemplate` from the full-equation noise-frontier
selection. This is the replacement for `build_multipoint_template` when
`system_construction_policy = :noise_frontier`.
"""
function build_noise_frontier_multipoint_template(
	pep::ParameterEstimationProblem,
	setup::NamedTuple,
	si_template;
	n_points::Int = 2,
	compute_mixed_volume::Bool = true,
	candidate_limit::Int = 64,
	beam_width::Int = 16,
	rank_atol::Float64 = 1e-8,
	diagnostics::Bool = false,
	selection_mode::Symbol = :generic,
)
	result = build_noise_frontier_system(
		pep,
		setup,
		si_template;
		n_points = n_points,
		compute_mixed_volume = compute_mixed_volume,
		candidate_limit = candidate_limit,
		beam_width = beam_width,
		rank_atol = rank_atol,
		diagnostics = diagnostics,
		selection_mode = selection_mode,
	)
	selected = result.selected
	isnothing(selected) && error("noise-frontier multipoint construction found no square full-rank candidate")

	data_vars = _noise_order_data_vars_by_point(selected.data_vars, n_points)
	param_indices, param_names = _noise_param_indices_and_names(selected.solve_vars, pep)
	eq_metadata = [(point = m.point, is_data = false, order = m.max_observed_order) for m in selected.eq_metadata]
	per_point_indices = _noise_per_point_indices(data_vars, n_points)
	if diagnostics
		@info "[NOISE-FRONTIER-MP] Selected template" n_points equations = length(selected.equations) solve_vars = length(selected.solve_vars) data_vars = length(data_vars) max_observed_order = selected.max_observed_order mixed_volume = selected.mixed_volume
	end

	return MultiPointTemplate(
		n_points,
		si_template,
		Num[eq for eq in selected.equations],
		Num[eq for eq in selected.equations],
		selected.solve_vars,
		data_vars,
		param_indices,
		param_names,
		eq_metadata,
		result.combined_equation_count,
		selected.selected_equation_indices,
		selected.dropped_equation_indices,
		per_point_indices,
		hasproperty(si_template, :template_DD) ? si_template.template_DD : setup.good_DD,
		pep.measured_quantities,
		_build_data_var_meta(data_vars, n_points, pep.measured_quantities),
	)
end

"""
	build_noise_frontier_system(pep, setup, si_template; kwargs...) -> NoiseFrontierResult

Build a noise-first square-system construction frontier. `setup` must provide
`good_DD`. The default selector is generic/structural and ignores
`setup.interpolants`; pass `selection_mode = :instantiated` to reproduce the
older data-conditioned probe behavior.
"""
function build_noise_frontier_system(
	pep::ParameterEstimationProblem,
	setup::NamedTuple,
	si_template;
	n_points::Int = 1,
	compute_mixed_volume::Bool = true,
	candidate_limit::Int = 64,
	beam_width::Int = 16,
	rank_atol::Float64 = 1e-8,
	n_rank_probes::Int = 3,
	diagnostics::Bool = false,
	probe_indices::Union{Nothing, Vector{Int}} = nothing,
	selection_mode::Symbol = :generic,
)
	candidate_limit > 0 || throw(ArgumentError("candidate_limit must be positive"))
	beam_width > 0 || throw(ArgumentError("beam_width must be positive"))
	raw_pool = _noise_build_pool(pep, setup, si_template;
		n_points = n_points, diagnostics = diagnostics, probe_indices = probe_indices,
		selection_mode = selection_mode)
	pool = (; raw_pool..., n_points = n_points)
	return _noise_select_pool(pep, pool; compute_mixed_volume, candidate_limit,
		beam_width, rank_atol, n_rank_probes, diagnostics)
end

# The selector also accepts a pool assembled from distinct experiment blocks.
# Combining the blocks before this step lets their shared parameters acquire
# rank jointly; no independent experiment needs an arbitrary representative fix.
function _noise_select_pool(pep::ParameterEstimationProblem, pool;
	compute_mixed_volume::Bool = true, candidate_limit::Int = 64,
	beam_width::Int = 16, rank_atol::Float64 = 1e-8,
	n_rank_probes::Int = 3, diagnostics::Bool = false,
	materialize_polynomials::F = identity) where {F}
	candidate_limit > 0 || throw(ArgumentError("candidate_limit must be positive"))
	beam_width > 0 || throw(ArgumentError("beam_width must be positive"))
	n_points = pool.n_points
	J, full_rank = _noise_rank_matrix(pool.instantiated_equations, pool.instantiated_vars;
		rank_atol = rank_atol, n_rank_probes = n_rank_probes)
	target_rank = length(pool.instantiated_vars)
	max_cap = maximum([m.max_observed_order for m in pool.metadata]; init = -1)
	max_cap = max(max_cap, 0)
	frontier = NamedTuple[]
	candidates = NoiseFrontierCandidate[]
	selected_cap = nothing

	dd_order_by_obj, dd_order_by_string = _noise_dd_order_maps(pool.template_DD)
	obs_bases = _noise_observable_bases(pep.measured_quantities)

	for cap in 0:max_cap
		allowed = [i for i in eachindex(pool.metadata) if pool.metadata[i].max_observed_order <= cap]
		allowed_rank = _noise_rank_of_rows(J, allowed; rank_atol = rank_atol)
		feasible = allowed_rank >= target_rank && full_rank >= target_rank
		push!(frontier, (
			max_observed_order = cap,
			allowed_count = length(allowed),
			allowed_rank = allowed_rank,
			target_rank = target_rank,
			feasible = feasible,
		))
		feasible || continue
		selected_cap = cap
		# Rational experiment jets can defer polynomial construction until rank
		# permits a solve. Materialization must preserve row order and derivative
		# metadata; support scores are computed from the resulting polynomials.
		pool = materialize_polynomials(pool)

		seeds = Vector{Vector{Int}}()
		orders = Dict{Symbol, Vector{Int}}(
			:support_greedy => sort(allowed; by = i -> (pool.metadata[i].support_score, pool.metadata[i].max_observed_order, i)),
			:index_greedy => sort(allowed),
			:reverse_support_greedy => sort(allowed; by = i -> (-pool.metadata[i].support_score, pool.metadata[i].max_observed_order, i)),
		)
		for (strategy, order) in orders
			basis = _noise_greedy_basis(J, order, target_rank; rank_atol = rank_atol)
			isempty(basis) || push!(seeds, basis)
		end
		for tie_policy in (:support, :data_order)
			basis = _noise_dynamic_basis(J, allowed, pool.metadata, pool.symbolic_equations, pep,
				dd_order_by_obj, dd_order_by_string, obs_bases, target_rank;
				tie_policy = tie_policy, rank_atol = rank_atol)
			isempty(basis) || push!(seeds, basis)
		end
		removal = _noise_removal_basis(J, allowed, pool.metadata, target_rank; rank_atol = rank_atol)
		isempty(removal) || push!(seeds, removal)

		exchanged = _noise_exchange_candidates(J, seeds, allowed, pool.metadata, target_rank;
			beam_width = beam_width, rank_atol = rank_atol)
		all_bases = unique(vcat([sort(s) for s in seeds], exchanged))
		sort!(all_bases; by = s -> _noise_basis_score(s, pool.metadata))
		length(all_bases) > candidate_limit && resize!(all_bases, candidate_limit)

		for (i, basis) in enumerate(all_bases)
			strategy = i <= length(seeds) ? Symbol(:seed_, i) : Symbol(:exchange_, i)
			push!(candidates, _noise_candidate_from_indices(
				basis,
				strategy,
				cap,
				pool,
				J,
				pep,
				dd_order_by_obj,
				dd_order_by_string,
				obs_bases;
				compute_mixed_volume = compute_mixed_volume,
				rank_atol = rank_atol,
			))
		end
		break
	end

	valid_candidates = [c for c in candidates if c.rank_complete && c.is_square]
	sort!(valid_candidates; by = _noise_candidate_sort_key)
	selected = isempty(valid_candidates) ? nothing : first(valid_candidates)
	if diagnostics
		if isnothing(selected)
			@warn "[NOISE-FRONTIER] No square full-rank candidate found" n_points target_rank full_rank
		else
			@info "[NOISE-FRONTIER] Selected candidate" n_points max_observed_order = selected.max_observed_order mixed_volume = selected.mixed_volume support_score = selected.support_score solve_vars = selected.solve_var_count data_vars = selected.data_var_count
		end
	end
	return NoiseFrontierResult(
		n_points,
		selected_cap,
		selected,
		candidates,
		frontier,
		pool.full_equation_count,
		length(pool.symbolic_equations),
		target_rank,
	)
end

function noise_frontier_rows(result::NoiseFrontierResult)
	rows = NamedTuple[]
	for c in result.candidates
		push!(rows, (
			n_points = c.n_points,
			strategy = c.strategy,
			max_observed_order = c.max_observed_order,
			rank = c.rank,
			target_rank = c.target_rank,
			is_square = c.is_square,
			solve_var_count = c.solve_var_count,
			data_var_count = c.data_var_count,
			support_score = c.support_score,
			condition_proxy = c.condition_proxy,
			sigma_max = c.sigma_max,
			sigma_min = c.sigma_min,
			unfloored_svd_ratio = c.unfloored_svd_ratio,
			rank_atol = c.rank_atol,
			mixed_volume = isnothing(c.mixed_volume) ? missing : c.mixed_volume,
			selected_equation_indices = join(c.selected_equation_indices, " "),
			selected_source_indices = join(c.selected_source_indices, " "),
		))
	end
	return rows
end

function write_noise_frontier_csv(path::AbstractString, result::NoiseFrontierResult)
	rows = noise_frontier_rows(result)
	open(path, "w") do io
		header = [
			"n_points", "strategy", "max_observed_order", "rank", "target_rank",
			"is_square", "solve_var_count", "data_var_count", "support_score",
			"condition_proxy", "sigma_max", "sigma_min", "unfloored_svd_ratio",
			"rank_atol", "mixed_volume", "selected_equation_indices",
			"selected_source_indices",
		]
		println(io, join(header, ","))
		for row in rows
			values = [replace(string(getproperty(row, Symbol(h))), "," => ";") for h in header]
			println(io, join(values, ","))
		end
	end
	return path
end
