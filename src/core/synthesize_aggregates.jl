# synthesize_aggregates.jl
#
# Inject extra "test" candidates into the polish pool by per-component aggregating
# the existing SP and MP candidates' parameters and ICs. Designed as a research
# probe with full provenance, NOT a sloppy-case fix.
#
# All aggregations are per-component (each parameter / state IC aggregated
# independently across the pool), not vector medoid (which would be in the input
# set by construction → redundant).
#
# Categories:
#   A — Global parameter-only aggregates × 3 groupings × 3 strategies = 9 candidates
#       (each requires resolve_states_with_fixed_params; cached) — Phase 2
#   B — Per-SP per-component aggregates of (params + states) × 2 strategies × n_sps
#       (no resolve needed — reuses post-backsolve t=0 ICs)
#   C — Per-SP-with-MP-anchored aggregates × 2 strategies × n_sps
#       (no resolve needed)
#   D — Cheap extras: per-interpolator median, interior-only median,
#       cross-source-spread weighted median (Phase 3, requires resolve)
#
# Cache for the resolve calls (Category A + most of D) lives below as
# `_SYNTH_RESOLVE_CACHE`; degenerate-value fallback bypasses it (Phase 4).
#
# With `diagnostics = true`, artifacts/diagnostics/<model>/synthesis_log.csv
# records all synthesized candidates' lineage for post-hoc inspection.

# ─── Per-component aggregation primitives ──────────────────────────────────────

"""
    _aggregate_scalar(values::AbstractVector{<:Real}, strategy::Symbol) -> Float64

Apply `strategy` (`:median`, `:mean`, `:trim25_mean`) to a vector of scalar values.
Returns NaN if input is empty or all-NaN. `:trim25_mean` drops the top 12.5% and
bottom 12.5% by value, then averages the middle 75%.
"""
function _aggregate_scalar(values::AbstractVector{<:Real}, strategy::Symbol)
	finite_values = Float64[v for v in values if isfinite(v)]
	isempty(finite_values) && return NaN
	if strategy == :median
		return Statistics.median(finite_values)
	elseif strategy == :mean
		return Statistics.mean(finite_values)
	elseif strategy == :trim25_mean
		n = length(finite_values)
		if n < 4
			# Too few to trim — fall back to mean
			return Statistics.mean(finite_values)
		end
		sorted = sort(finite_values)
		drop_each_side = max(1, floor(Int, n / 8))  # 12.5% per tail
		keep = sorted[(drop_each_side + 1):(n - drop_each_side)]
		return isempty(keep) ? Statistics.median(finite_values) : Statistics.mean(keep)
	else
		error("unknown aggregation strategy: $strategy")
	end
end

"""
    _per_component_aggregate(dicts::AbstractVector, strategy::Symbol) -> OrderedDict

Given a vector of OrderedDicts {Num => Float64} all sharing the same key set,
return a new OrderedDict with `_aggregate_scalar(values, strategy)` applied
per key. The resulting dict is a frankenstein vector — each component aggregated
independently across the inputs. Skips keys where all values are non-finite.
"""
function _per_component_aggregate(dicts::AbstractVector, strategy::Symbol)
	out = OrderedDict{Num, Float64}()
	isempty(dicts) && return out
	# Use the first dict's keys as the canonical ordering
	for k in keys(first(dicts))
		vals = Float64[]
		for d in dicts
			haskey(d, k) || continue
			v = d[k]
			isfinite(v) && push!(vals, Float64(v))
		end
		isempty(vals) && continue
		out[k] = _aggregate_scalar(vals, strategy)
	end
	return out
end

# ─── Candidate-grouping helpers ────────────────────────────────────────────────

"""
    _is_sp_candidate(c) -> Bool

A "single-point" candidate has provenance.source_type ∈ (:single_point,) and
came from an algebraic SP solve (not a synthesized aggregate, not imported).
"""
function _is_sp_candidate(c)
	isnothing(c.provenance) && return false
	c.provenance.source_type == :single_point || return false
	return true
end

"""
    _is_mp_candidate(c) -> Bool
"""
function _is_mp_candidate(c)
	isnothing(c.provenance) && return false
	return c.provenance.source_type == :multipoint
end

"""
    _mp_anchor_sp(c) -> Union{Nothing, Int}

Returns the leftmost time-index of an MP candidate's combo, or nothing if not MP.
Per existing convention, MP candidates' ICs are anchored at this index.
"""
function _mp_anchor_sp(c)
	_is_mp_candidate(c) || return nothing
	idxs = c.provenance.multipoint_time_indices
	(isnothing(idxs) || isempty(idxs)) && return nothing
	return minimum(idxs)
end

# ─── Build a synthetic candidate from aggregated (params, states) ──────────────

# Model-level structural fix set, copied from a source candidate's provenance (all
# candidates of one estimate share it). Guarded for a missing provenance.
function _synth_structural_fix_set(c)
	(hasproperty(c, :provenance) && c.provenance !== nothing) || return OrderedDict{Num, Float64}()
	return copy(c.provenance.structural_fix_set)
end

"""
    _build_synth_candidate(params::OrderedDict, states::OrderedDict, t0::Real,
        n_total::Int, source_indices::Vector{Int}, strategy::Symbol,
        category_notes::Vector{Symbol}) -> ParameterEstimationResult

Construct a synthetic candidate with provenance fully tagged. Uses
`source_type = :synthesized_aggregate` and the supplied strategy + source indices.
The candidate is NOT polished here; downstream polish runs on it like any other.
"""
function _build_synth_candidate(
	params::OrderedDict{Num, Float64},
	states::OrderedDict{Num, Float64},
	t0::Real,
	n_total::Int,
	source_indices::Vector{Int},
	strategy::Symbol,
	category_notes::Vector{Symbol};
	all_unidentifiable::Set{Num} = Set{Num}(),
	structural_fix_set::OrderedDict{Num, Float64} = OrderedDict{Num, Float64}(),
	parent_candidate_ids::Vector{Int} = Int[],
)
	result = ParameterEstimationResult(
		params,
		states,
		Float64(t0),
		nothing,                          # err — filled in downstream
		nothing,                          # return_code
		n_total,
		Float64(t0),
		OrderedDict{Num, Float64}(),     # known/auxiliary states
		all_unidentifiable,
		nothing,                          # ode_solution
	)
	# Note: interpolator_source is intentionally `nothing` for synthesized candidates:
	# they aren't tied to any single interpolator's interpolant cache. The
	# :synthesized_aggregate marker lives on `source_type` instead.
	result.provenance = ResultProvenance(
		primary_method = :algebraic,
		interpolator_source = nothing,
		source_type = :synthesized_aggregate,
		aggregation_strategy = strategy,
		aggregation_source_indices = source_indices,
		notes = vcat([:synthesized_aggregate], category_notes),
		polish_applied = false,
		# Structural unidentifiability is a MODEL-level property, identical across all
		# candidates of this estimate — an aggregate of them must carry it too, else a
		# winning aggregate reports an empty structural_fix_set (canary regression).
		structural_fix_set = structural_fix_set,
	)
	set_result_estimator_identity!(result;
		estimator_kind = :synthesized_aggregate,
		data_scope = :ensemble,
		parent_candidate_ids = parent_candidate_ids,
	)
	sync_result_contract!(result)
	return result
end

# ─── Category B/C: per-SP full aggregate (params + states, no resolve) ─────────

"""
    _synthesize_per_sp_full_aggregate(PEP, candidates_for_sp::Vector,
        sp_idx::Int, strategy::Symbol, source_indices::Vector{Int};
        with_mp::Bool) -> Union{Nothing, ParameterEstimationResult}

Aggregate (params, t=0 state ICs) per-component across `candidates_for_sp` using
`strategy`. Build a synthetic candidate. Returns nothing when fewer than 2
inputs (median/trim of 1 == that 1, redundant).
"""
function _synthesize_per_sp_full_aggregate(
	PEP, candidates_for_sp::Vector, sp_idx::Int, strategy::Symbol,
	source_indices::Vector{Int}; with_mp::Bool,
)
	length(candidates_for_sp) < 2 && return nothing
	param_dicts = OrderedDict{Num, Float64}[c.parameters for c in candidates_for_sp]
	state_dicts = OrderedDict{Num, Float64}[c.states for c in candidates_for_sp]
	agg_params = _per_component_aggregate(param_dicts, strategy)
	agg_states = _per_component_aggregate(state_dicts, strategy)
	(isempty(agg_params) || isempty(agg_states)) && return nothing
	t_vec = PEP.data_sample["t"]
	cat_notes = with_mp ?
		Symbol[:per_sp_full_with_mp, Symbol("sp_$sp_idx")] :
		Symbol[:per_sp_full, Symbol("sp_$sp_idx")]
	return _build_synth_candidate(
		agg_params, agg_states,
		Float64(t_vec[1]),
		length(t_vec),
		source_indices,
		strategy,
		cat_notes;
		all_unidentifiable = copy(first(candidates_for_sp).all_unidentifiable),
		structural_fix_set = _synth_structural_fix_set(first(candidates_for_sp)),
		parent_candidate_ids = Int[
			ensure_result_estimator_identity!(candidate).candidate_id
			for candidate in candidates_for_sp
		],
	)
end

# ─── Outlier rejection (per-component MAD) ─────────────────────────────────────

"""
    _filter_outliers_per_component(candidates, k_mad=3.0) -> Vector

Drop candidates where ANY parameter is more than `k_mad` MADs from the
per-parameter median. The rejection criterion is per-parameter (a candidate
is rejected if it's an outlier on any single component), more aggressive
than per-vector MAD. Returns the surviving candidates plus the indices that
were kept (relative to the input order).
"""
function _filter_outliers_per_component(candidates::Vector; k_mad::Float64 = 3.0)
	isempty(candidates) && return candidates, Int[]
	length(candidates) < 4 && return candidates, collect(1:length(candidates))
	# Per-parameter median + MAD across the pool
	param_keys = collect(keys(first(candidates).parameters))
	medians = Dict{Num, Float64}()
	mads = Dict{Num, Float64}()
	for p in param_keys
		vals = Float64[]
		for c in candidates
			haskey(c.parameters, p) || continue
			v = c.parameters[p]
			isfinite(v) && push!(vals, Float64(v))
		end
		isempty(vals) && continue
		med = Statistics.median(vals)
		mad = Statistics.median(abs.(vals .- med))
		medians[p] = med
		mads[p] = mad
	end
	keep_indices = Int[]
	keep_candidates = eltype(candidates)[]
	for (i, c) in enumerate(candidates)
		ok = true
		for p in param_keys
			haskey(medians, p) || continue
			haskey(c.parameters, p) || continue
			v = Float64(c.parameters[p])
			isfinite(v) || (ok = false; break)
			mad_p = mads[p]
			# Skip MAD check when MAD is exactly zero (all values identical → no spread to measure)
			mad_p == 0 && continue
			if abs(v - medians[p]) > k_mad * mad_p
				ok = false
				break
			end
		end
		if ok
			push!(keep_indices, i)
			push!(keep_candidates, c)
		end
	end
	return keep_candidates, keep_indices
end

# ─── Category A: global parameter-only aggregate (uses resolve) ───────────────

"""
    _synthesize_global_param_aggregate(PEP, candidates_with_indices, strategy,
        grouping, setup_data, opts) -> Union{Nothing, ParameterEstimationResult}

For Category A: aggregate parameters per-component over the supplied
candidates, then call `resolve_states_with_fixed_params` to recover state
ICs at the first shooting point. Build a candidate via the resolve result.

`candidates_with_indices` is a Vector{Tuple{Int, ParameterEstimationResult}}:
the Int is the index into the original pool (used for provenance).
"""
function _synthesize_global_param_aggregate(
	PEP, candidates_with_indices::Vector, strategy::Symbol, grouping::Symbol,
	setup_data, opts::EstimationOptions,
)
	length(candidates_with_indices) < 2 && return nothing
	# For :sp_and_mp_outlier_rejected, drop outliers first
	if grouping == :sp_and_mp_outlier_rejected
		raw_cands = [e[2] for e in candidates_with_indices]
		raw_idxs = [e[1] for e in candidates_with_indices]
		kept_cands, kept_local_idxs = _filter_outliers_per_component(raw_cands)
		length(kept_cands) < 2 && return nothing
		candidates_with_indices = [(raw_idxs[i], kept_cands[k]) for (k, i) in enumerate(kept_local_idxs)]
	end
	source_candidates = [e[2] for e in candidates_with_indices]
	source_indices = [e[1] for e in candidates_with_indices]
	# Aggregate parameters per-component
	param_dicts = OrderedDict{Num, Float64}[c.parameters for c in source_candidates]
	agg_params = _per_component_aggregate(param_dicts, strategy)
	isempty(agg_params) && return nothing
	# Convert to OrderedDict for resolve_states (needs Num keys with Float64 vals)
	known_param_dict = OrderedDict{Any, Float64}(k => Float64(v) for (k, v) in agg_params)
	# Call resolve at first SP (time_index=1)
	resolve_result = try
		_with_resolve_timing_context(:synthesis) do
			resolve_states_with_fixed_params(
				PEP.model.system,
				PEP.measured_quantities,
				PEP.data_sample,
				setup_data.good_deriv_level,
				Dict{Num, Float64}(),  # empty unident_dict for synth
				setup_data.good_varlist,
				setup_data.good_DD,
				known_param_dict,
				setup_data.interpolants;
				si_fix_strategy = opts.si_fix_strategy,
				si_probability = opts.si_probability,
				compute_multiplicity = opts.compute_algebraic_multiplicity && isnothing(opts.algebraic_multiplicity),
				time_index = 1,
				diagnostics = false,
			)
		end
	catch err
		@warn "[SYNTH-A] resolve failed for grouping=$grouping strategy=$strategy" exception = err
		return nothing
	end
	(isempty(resolve_result.solutions) || isempty(resolve_result.state_vars)) && return nothing
	# Pick first solution
	state_sol = resolve_result.solutions[1]
	state_vars = resolve_result.state_vars
	# Build a state IC dict at t=0. Match each MTK unknown to its resolve output
	# at order 0 (e.g. SIAN var "C_0" → MTK state C(t)). Fall back via existing
	# helpers: parse_derivative_variable_name pulls the base name.
	mtk_unknowns = ModelingToolkit.unknowns(PEP.model.system)
	sian_base_to_val = Dict{String, Float64}()
	for j in eachindex(state_vars)
		vname = replace(string(state_vars[j]), "(t)" => "")
		parsed = parse_derivative_variable_name(vname)
		isnothing(parsed) && continue
		base, order = parsed
		order == 0 && (sian_base_to_val[String(base)] = state_sol[j])
	end
	states_dict = OrderedDict{Num, Float64}()
	t0 = Float64(PEP.data_sample["t"][1])
	for s in mtk_unknowns
		sname = replace(string(s), "(t)" => "")
		if haskey(sian_base_to_val, sname)
			states_dict[s] = sian_base_to_val[sname]
		elseif startswith(sname, "_trfn_")
			# Auxiliary trfn states are known functions of time; reuse evaluator.
			tval = evaluate_trfn_template_variable(sname, t0)
			isnothing(tval) || (states_dict[s] = tval)
		else
			# State is missing from resolve output — skip this synth candidate
			# (matches Change 1 skip-on-failure semantics).
			return nothing
		end
	end
	cat_notes = grouping == :sp_only ?
		Symbol[:global_param_only, :sp_only] :
		grouping == :mp_only ? Symbol[:global_param_only, :mp_only] :
		Symbol[:global_param_only, :sp_and_mp, :outlier_rejected]
	return _build_synth_candidate(
		agg_params, states_dict, t0,
		length(PEP.data_sample["t"]),
		Int[source_indices...],
		strategy,
		cat_notes;
		all_unidentifiable = copy(first(source_candidates).all_unidentifiable),
		structural_fix_set = _synth_structural_fix_set(first(source_candidates)),
		parent_candidate_ids = Int[
			ensure_result_estimator_identity!(candidate).candidate_id
			for candidate in source_candidates
		],
	)
end

# ─── Category D extras ─────────────────────────────────────────────────────────

"""
    _aggregate_and_resolve_to_candidate(PEP, candidates_with_indices, strategy,
        cat_notes, setup_data, opts; weights=nothing) -> Union{Nothing, ParameterEstimationResult}

Helper used by Category D extras. Aggregates parameters per-component over
the supplied (idx, candidate) tuples using `strategy`, optionally weighted
by per-candidate weights, then calls resolve_states_with_fixed_params and
builds a candidate. Returns nothing on failure.
"""
function _aggregate_and_resolve_to_candidate(
	PEP, candidates_with_indices::Vector, strategy::Symbol, cat_notes::Vector{Symbol},
	setup_data, opts::EstimationOptions; weights::Union{Nothing, Vector{Float64}} = nothing,
)
	length(candidates_with_indices) < 2 && return nothing
	source_candidates = [e[2] for e in candidates_with_indices]
	source_indices = Int[e[1] for e in candidates_with_indices]
	param_dicts = OrderedDict{Num, Float64}[c.parameters for c in source_candidates]
	# For weighted mode: replicate each candidate's contribution proportional to
	# its weight (works for median/trim25_mean/mean uniformly). We compute weights
	# normalized to sum to length(param_dicts) for stability.
	agg_params = if isnothing(weights)
		_per_component_aggregate(param_dicts, strategy)
	else
		# Weighted aggregation: for each parameter, sort values by parameter value;
		# the weighted median is the value where cumulative weight crosses 50%.
		_per_component_weighted_aggregate(param_dicts, strategy, weights)
	end
	isempty(agg_params) && return nothing
	known_param_dict = OrderedDict{Any, Float64}(k => Float64(v) for (k, v) in agg_params)
	resolve_result = try
		_with_resolve_timing_context(:synthesis) do
			resolve_states_with_fixed_params(
				PEP.model.system, PEP.measured_quantities, PEP.data_sample,
				setup_data.good_deriv_level, Dict{Num, Float64}(),
				setup_data.good_varlist, setup_data.good_DD,
				known_param_dict, setup_data.interpolants;
				si_fix_strategy = opts.si_fix_strategy,
				si_probability = opts.si_probability,
				compute_multiplicity = opts.compute_algebraic_multiplicity && isnothing(opts.algebraic_multiplicity),
				time_index = 1, diagnostics = false,
			)
		end
	catch err
		@warn "[SYNTH-D] resolve failed for $cat_notes" exception = err
		return nothing
	end
	(isempty(resolve_result.solutions) || isempty(resolve_result.state_vars)) && return nothing
	state_sol = resolve_result.solutions[1]
	state_vars = resolve_result.state_vars
	# Map SIAN order-0 names to Float64 values
	sian_base_to_val = Dict{String, Float64}()
	for j in eachindex(state_vars)
		vname = replace(string(state_vars[j]), "(t)" => "")
		parsed = parse_derivative_variable_name(vname)
		isnothing(parsed) && continue
		base, order = parsed
		order == 0 && (sian_base_to_val[String(base)] = state_sol[j])
	end
	mtk_unknowns = ModelingToolkit.unknowns(PEP.model.system)
	t0 = Float64(PEP.data_sample["t"][1])
	states_dict = OrderedDict{Num, Float64}()
	for s in mtk_unknowns
		sname = replace(string(s), "(t)" => "")
		if haskey(sian_base_to_val, sname)
			states_dict[s] = sian_base_to_val[sname]
		elseif startswith(sname, "_trfn_")
			tval = evaluate_trfn_template_variable(sname, t0)
			isnothing(tval) || (states_dict[s] = tval)
		else
			return nothing  # missing state — skip this candidate
		end
	end
	return _build_synth_candidate(
		agg_params, states_dict, t0,
		length(PEP.data_sample["t"]),
		source_indices, strategy, cat_notes;
		all_unidentifiable = copy(first(source_candidates).all_unidentifiable),
		structural_fix_set = _synth_structural_fix_set(first(source_candidates)),
		parent_candidate_ids = Int[
			ensure_result_estimator_identity!(candidate).candidate_id
			for candidate in source_candidates
		],
	)
end

"""
    _per_component_weighted_aggregate(dicts, strategy, weights) -> OrderedDict

Weighted per-component aggregator. Weights are per-candidate; same weight
applied across all parameters. For `:median`, computes a weighted median
(value at cumulative-weight-fraction 0.5). For `:mean`, a weighted mean.
"""
function _per_component_weighted_aggregate(
	dicts::AbstractVector, strategy::Symbol, weights::Vector{Float64},
)
	@assert length(dicts) == length(weights) "weights/dicts mismatch"
	out = OrderedDict{Num, Float64}()
	isempty(dicts) && return out
	for k in keys(first(dicts))
		vals = Float64[]
		ws = Float64[]
		for (i, d) in enumerate(dicts)
			haskey(d, k) || continue
			v = Float64(d[k])
			isfinite(v) && weights[i] > 0 || continue
			push!(vals, v)
			push!(ws, weights[i])
		end
		isempty(vals) && continue
		total_w = sum(ws)
		total_w == 0 && continue
		if strategy == :mean
			out[k] = sum(vals .* ws) / total_w
		else  # :median or :trim25_mean — fall back to weighted median for both
			perm = sortperm(vals)
			cw = cumsum(ws[perm]) ./ total_w
			idx = findfirst(>=(0.5), cw)
			out[k] = vals[perm[idx]]
		end
	end
	return out
end

"""
    _category_d_per_interpolator(PEP, solved_res, setup_data, opts) -> Vector

For each interpolator that produced ≥ 2 candidates in the SP pool,
synthesize a per-component median candidate.
"""
function _category_d_per_interpolator(PEP, solved_res, setup_data, opts)
	out = ParameterEstimationResult[]
	by_interp = Dict{Symbol, Vector{Tuple{Int, Any}}}()
	for (i, c) in enumerate(solved_res)
		_is_sp_candidate(c) || continue
		src = c.provenance.interpolator_source
		isnothing(src) && continue
		push!(get!(by_interp, src, Tuple{Int, Any}[]), (i, c))
	end
	for (src, entries) in by_interp
		length(entries) < 2 && continue
		cand = try
			_aggregate_and_resolve_to_candidate(
				PEP, entries, :median,
				Symbol[:per_interpolator, Symbol(src)], setup_data, opts,
			)
		catch err
			@warn "[SYNTH-D-per-interp] $src failed" exception = err
			nothing
		end
		!isnothing(cand) && push!(out, cand)
	end
	return out
end

"""
    _category_d_interior_only(PEP, solved_res, setup_data, opts) -> Vector

Per-component median of SP candidates from interior shooting points only
(drop boundary SPs at fraction ∉ [0.10, 0.90]).
"""
function _category_d_interior_only(PEP, solved_res, setup_data, opts)
	t_vec = PEP.data_sample["t"]
	t0 = first(t_vec); t_end = last(t_vec); span = t_end - t0
	span <= 0 && return ParameterEstimationResult[]
	interior_lo = t0 + 0.10 * span
	interior_hi = t0 + 0.90 * span
	entries = Tuple{Int, Any}[]
	for (i, c) in enumerate(solved_res)
		_is_sp_candidate(c) || continue
		sp_idx = c.provenance.source_shooting_index
		isnothing(sp_idx) && continue
		(sp_idx < 1 || sp_idx > length(t_vec)) && continue
		t_sp = t_vec[sp_idx]
		(t_sp < interior_lo || t_sp > interior_hi) && continue
		push!(entries, (i, c))
	end
	length(entries) < 2 && return ParameterEstimationResult[]
	cand = try
		_aggregate_and_resolve_to_candidate(
			PEP, entries, :median,
			Symbol[:interior_only], setup_data, opts,
		)
	catch err
		@warn "[SYNTH-D-interior-only] failed" exception = err
		nothing
	end
	return isnothing(cand) ? ParameterEstimationResult[] : ParameterEstimationResult[cand]
end

"""
    _category_d_spread_weighted(PEP, solved_res, setup_data, opts) -> Vector

Cross-source-spread weighted per-component median. Per-candidate weight =
1/(1 + total_normalized_deviation_across_params). Down-weights candidates
whose parameters are outliers relative to pool consensus, without dropping
them entirely.
"""
function _category_d_spread_weighted(PEP, solved_res, setup_data, opts)
	algebraic_entries = [(i, c) for (i, c) in enumerate(solved_res)
	                    if _is_sp_candidate(c) || _is_mp_candidate(c)]
	length(algebraic_entries) < 4 && return ParameterEstimationResult[]
	candidates = [e[2] for e in algebraic_entries]
	# Compute per-candidate weight = 1 / (1 + total |z_score| across params)
	param_keys = collect(keys(first(candidates).parameters))
	medians = Dict{Num, Float64}(); mads = Dict{Num, Float64}()
	for p in param_keys
		vals = Float64[Float64(c.parameters[p]) for c in candidates if haskey(c.parameters, p) && isfinite(c.parameters[p])]
		isempty(vals) && continue
		medians[p] = Statistics.median(vals)
		mads[p] = max(Statistics.median(abs.(vals .- medians[p])), 1e-12)
	end
	weights = Float64[]
	for c in candidates
		total_dev = 0.0
		for p in param_keys
			(haskey(medians, p) && haskey(c.parameters, p)) || continue
			v = Float64(c.parameters[p])
			isfinite(v) || (total_dev = Inf; break)
			total_dev += abs(v - medians[p]) / mads[p]
		end
		push!(weights, isfinite(total_dev) ? 1.0 / (1.0 + total_dev) : 0.0)
	end
	all(iszero, weights) && return ParameterEstimationResult[]
	cand = try
		_aggregate_and_resolve_to_candidate(
			PEP, algebraic_entries, :median,
			Symbol[:cross_source_weighted], setup_data, opts; weights = weights,
		)
	catch err
		@warn "[SYNTH-D-weighted] failed" exception = err
		nothing
	end
	return isnothing(cand) ? ParameterEstimationResult[] : ParameterEstimationResult[cand]
end

# ─── Sidecar CSV writer ────────────────────────────────────────────────────────

"""
    _synthesis_sidecar_path(PEP) -> String

Per-run sidecar location. Returns "" if no model name available.
"""
function _synthesis_sidecar_path(PEP)
	name = string(PEP.name)
	isempty(name) && return ""
	return joinpath("artifacts", "diagnostics", name, "synthesis_log.csv")
end

"""
    _write_synthesis_sidecar(PEP, synth_candidates, source_pool)

Write a CSV log of every synthesized candidate's lineage. One row per candidate.
Columns: synth_idx, category, strategy, source_indices, params (per-name),
states (per-name), err_pre_polish (pre-pool-eval; usually nothing).
"""
function _write_synthesis_sidecar(PEP, synth_candidates, source_pool)
	path = _synthesis_sidecar_path(PEP)
	isempty(path) && return nothing
	mkpath(dirname(path))
	# Get param + state names from the first synth candidate
	isempty(synth_candidates) && return nothing
	pnames = collect(keys(first(synth_candidates).parameters))
	snames = collect(keys(first(synth_candidates).states))
	open(path, "w") do io
		# Header
		header_parts = String["synth_idx", "category", "strategy", "source_indices"]
		append!(header_parts, ["param_" * string(p) for p in pnames])
		append!(header_parts, ["state_" * string(s) for s in snames])
		push!(header_parts, "err")
		println(io, join(header_parts, ","))
		for (k, c) in enumerate(synth_candidates)
			cat_notes = filter(n -> n != :synthesized_aggregate, c.provenance.notes)
			cat_str = isempty(cat_notes) ? "?" : string(first(cat_notes))
			src_str = "[" * join(c.provenance.aggregation_source_indices, ";") * "]"
			row_parts = String[
				string(k),
				cat_str,
				string(c.provenance.aggregation_strategy),
				src_str,
			]
			for p in pnames
				v = get(c.parameters, p, NaN)
				push!(row_parts, isfinite(v) ? string(v) : "NaN")
			end
			for s in snames
				v = get(c.states, s, NaN)
				push!(row_parts, isfinite(v) ? string(v) : "NaN")
			end
			err_str = isnothing(c.err) ? "" : (isfinite(c.err) ? string(c.err) : "Inf")
			push!(row_parts, err_str)
			println(io, join(row_parts, ","))
		end
	end
	return path
end

# ─── Loss evaluation (matches sensitivity_seeds.jl pattern) ────────────────────

"""
    _evaluate_candidate_loss!(candidate, polish_ctx)

Compute and set `candidate.err` via the polish-context loss closure.
Mirrors the per-candidate evaluation in `sensitivity_seeds.jl` so that
downstream consumers (cluster, branch consensus, etc.) see a finite Float64
err on every synthesized candidate. Leaves `err` as nothing if evaluation
fails or returns non-finite.
"""
function _evaluate_candidate_loss!(candidate, polish_ctx)
	try
		p_external = _candidate_to_external_vector(candidate, polish_ctx)
		any(!isfinite, p_external) && return
		if !isnothing(polish_ctx.lb) && !isnothing(polish_ctx.ub)
			p_external = clamp.(p_external, polish_ctx.lb, polish_ctx.ub)
		end
		# Coordinate-INDEPENDENT scoring: evaluate the candidate's true trajectory-fit
		# loss at its own external params. Do NOT round-trip through the polish search
		# coordinate (external→internal→optf.f), which distorts err for :shifted_log
		# with a large shift and made a candidate's err depend on the optimizer's
		# coordinate. The optimizer's own loss is untouched.
		loss = Float64(_trajectory_sse(polish_ctx, p_external))
		if isfinite(loss)
			candidate.err = loss
			if !isnothing(candidate.provenance)
				candidate.provenance.pre_polish_error = loss
				candidate.provenance.post_polish_error = loss
			end
		end
	catch
		# Leave err as nothing; downstream cluster tolerates this.
	end
	return
end

# ─── Main entry point ──────────────────────────────────────────────────────────

"""
    _maybe_synthesize_aggregate_candidates(PEP, solved_res, setup_data, opts;
        interpolant_cache=nothing) -> Vector{ParameterEstimationResult}

Inject per-component aggregate "test" candidates into the pool. Returns a
vector containing the original `solved_res` entries plus all synthesized
candidates. Synthesized candidates carry `source_type = :synthesized_aggregate`
plus strategy + source-indices fields.

Each synthesized candidate's `err` is evaluated via the polish-context loss
closure (so downstream cluster/branch-consensus paths see a finite Float64
err and don't trip on `nothing`).

Categories implemented:
  A — Global parameter-only per-component aggregate × 3 strategies × 3 groupings = 9 candidates
      (each requires resolve_states_with_fixed_params to recover ICs at first SP)
  B — Per-SP aggregate of (params + states), 2 strategies × n_sps (no resolve)
  C — Same as B but including MP-anchored-here candidates per SP (no resolve)
  D — Cheap extras (each requires resolve):
      • per-interpolator per-component median (≤ n_interp candidates)
      • interior-only per-component median (1 candidate)
      • cross-source-spread weighted per-component median (1 candidate)
  (Resolve template cache added in Phase 4 to amortize the ~22 resolve calls
  in A+D across one estimation run.)
"""
function _maybe_synthesize_aggregate_candidates(
	PEP, solved_res::Vector, setup_data, opts::EstimationOptions;
	interpolant_cache = nothing,
)
	!opts.synthesize_aggregate_candidates && return solved_res
	isempty(solved_res) && return solved_res

	new_candidates = ParameterEstimationResult[]

	# ── Category B & C: per-SP aggregates (no resolve) ───────────────────────
	# Group SP candidates by source_shooting_index
	sp_groups = Dict{Int, Vector{Tuple{Int, Any}}}()  # sp_idx => [(pool_idx, candidate), ...]
	for (i, c) in enumerate(solved_res)
		_is_sp_candidate(c) || continue
		sp_idx = c.provenance.source_shooting_index
		isnothing(sp_idx) && continue
		push!(get!(sp_groups, sp_idx, Tuple{Int, Any}[]), (i, c))
	end
	# Group MP candidates by their leftmost time index (anchor SP)
	mp_groups = Dict{Int, Vector{Tuple{Int, Any}}}()
	for (i, c) in enumerate(solved_res)
		anchor = _mp_anchor_sp(c)
		isnothing(anchor) && continue
		push!(get!(mp_groups, anchor, Tuple{Int, Any}[]), (i, c))
	end

	# ── Category A: global parameter-only aggregates × 3 strategies × 3 groupings ──
	# Build SP-only, MP-only, SP+MP-outlier-rejected groupings of (idx, candidate).
	all_sp_entries = Tuple{Int, Any}[]
	for entries in values(sp_groups)
		append!(all_sp_entries, entries)
	end
	all_mp_entries = Tuple{Int, Any}[]
	for entries in values(mp_groups)
		append!(all_mp_entries, entries)
	end
	all_combined_entries = vcat(all_sp_entries, all_mp_entries)
	for (grouping, entries) in (
		(:sp_only, all_sp_entries),
		(:mp_only, all_mp_entries),
		(:sp_and_mp_outlier_rejected, all_combined_entries),
	)
		length(entries) < 2 && continue
		for strategy in (:median, :trim25_mean, :mean)
			cand = try
				_synthesize_global_param_aggregate(PEP, entries, strategy, grouping, setup_data, opts)
			catch err
				@warn "[SYNTH-A] grouping=$grouping strategy=$strategy failed" exception = err
				nothing
			end
			!isnothing(cand) && push!(new_candidates, cand)
		end
	end

	# Category B: per-SP, SP-only
	for (sp_idx, entries) in sp_groups
		length(entries) < 2 && continue
		candidates_for_sp = [e[2] for e in entries]
		source_indices = [e[1] for e in entries]
		for strategy in (:median, :trim25_mean)
			cand = try
				_synthesize_per_sp_full_aggregate(
					PEP, candidates_for_sp, sp_idx, strategy, source_indices;
					with_mp = false,
				)
			catch err
				@warn "[SYNTH-B] sp=$sp_idx strategy=$strategy failed" exception = err
				nothing
			end
			!isnothing(cand) && push!(new_candidates, cand)
		end
	end

	# Category C: per-SP, SP + MP-anchored-here
	all_anchor_sps = union(keys(sp_groups), keys(mp_groups))
	for sp_idx in all_anchor_sps
		sp_entries = get(sp_groups, sp_idx, Tuple{Int, Any}[])
		mp_entries = get(mp_groups, sp_idx, Tuple{Int, Any}[])
		# Skip when there's no MP contribution at this anchor — would duplicate Category B
		isempty(mp_entries) && continue
		combined = vcat(sp_entries, mp_entries)
		length(combined) < 2 && continue
		candidates_for_sp = [e[2] for e in combined]
		source_indices = [e[1] for e in combined]
		for strategy in (:median, :trim25_mean)
			cand = try
				_synthesize_per_sp_full_aggregate(
					PEP, candidates_for_sp, sp_idx, strategy, source_indices;
					with_mp = true,
				)
			catch err
				@warn "[SYNTH-C] sp=$sp_idx strategy=$strategy failed" exception = err
				nothing
			end
			!isnothing(cand) && push!(new_candidates, cand)
		end
	end

	# ── Category D extras ─────────────────────────────────────────────────────
	# Per-interpolator median, interior-only median, cross-source-spread
	# weighted median. Each individually robust on per-failure (returns empty
	# vec on resolve fail), so total D output ranges from 0 to ~n_interp+2.
	for (label, fn) in (
		(:per_interpolator, _category_d_per_interpolator),
		(:interior_only, _category_d_interior_only),
		(:cross_source_weighted, _category_d_spread_weighted),
	)
		extras = try
			fn(PEP, solved_res, setup_data, opts)
		catch err
			@warn "[SYNTH-D] $label failed" exception = err
			ParameterEstimationResult[]
		end
		append!(new_candidates, extras)
	end

	# Evaluate each new candidate's err via polish-context loss closure.
	# This mirrors sensitivity_seeds.jl and ensures downstream consumers see a
	# finite Float64 err rather than the candidate's initial `nothing`.
	if !isempty(new_candidates)
		try
			polish_ctx = _build_polish_context(PEP; opts = opts)
			for c in new_candidates
				_evaluate_candidate_loss!(c, polish_ctx)
			end
		catch err
			@warn "[SYNTH] loss evaluation failed; candidates retain err=nothing (non-fatal)" exception = err
		end
	end

	# Sidecar log (a debugging artifact: written only with `diagnostics = true`)
	if opts.diagnostics && !isempty(new_candidates)
		try
			_write_synthesis_sidecar(PEP, new_candidates, solved_res)
		catch err
			@warn "[SYNTH] sidecar write failed (non-fatal)" exception = err
		end
	end

	if !opts.nooutput && !isempty(new_candidates)
		println("[Synthesize aggregates] $(length(solved_res)) → $(length(solved_res) + length(new_candidates)) candidates " *
			"(A/B/C/D synthesized)")
	end

	return vcat(solved_res, new_candidates)
end
