# Optimized Parameter Estimation with Precomputed Derivatives
# This implements a more efficient workflow that computes symbolic derivatives once
# and reuses them across multiple shooting points

# Parent module functions will be available when this file is included

# ============================================================================
# Phase Profiling Helpers
# ============================================================================

# Run-scoped state (timing capture flag, timing breakdown, auto-M hand-off,
# timing sinks) lives on the scoped RunContext (run_context.jl) as of
# 2026-07-24. The reuse bundle below stays module-global BY DESIGN: it is a
# cross-run hand-off consumed after the run completes (consensus/benchmark
# flows read `_LAST_ESTIMATION_REUSE[]`).
const _LAST_ESTIMATION_REUSE = Ref{Any}(nothing)

function _accumulate_timing!(dict::OrderedDict{Symbol, Float64}, key::Symbol, seconds::Real)
	dict[key] = get(dict, key, 0.0) + Float64(seconds)
	return dict
end

function _phase_stats_to_breakdown(
	stats::Union{Nothing, OrderedDict{String, NamedTuple}},
	label::Symbol;
	details::OrderedDict{Symbol, Any} = OrderedDict{Symbol, Any}(),
)
	if isnothing(stats)
		return TimingBreakdown(label = label, details = details)
	end

	phases = TimingPhaseEntry[
		TimingPhaseEntry(String(name), Float64(entry.time), Int64(entry.bytes), Float64(entry.gctime))
		for (name, entry) in stats
	]
	total_seconds = sum(entry.seconds for entry in phases)
	return TimingBreakdown(
		label = label,
		total_seconds = total_seconds,
		phases = phases,
		details = details,
	)
end

function _capture_estimation_timing(f::Function)
	value, ctx = _with_run_context(f; capture_timing = true)
	return value, ctx.timing
end

"""
	with_estimation_timing(f) -> (value, timing)

Run an estimation workflow while capturing the package's structured timing
breakdown, even when `profile_phases=false`.
"""
function with_estimation_timing(f::Function)
	return _capture_estimation_timing(f)
end


function _timing_json_value(value)
	if value === nothing || value isa AbstractString || value isa Bool || value isa Real
		return value
	elseif value isa Symbol
		return string(value)
	elseif value isa TimingPhaseEntry
		return OrderedDict{String, Any}(
			"name" => value.name,
			"seconds" => value.seconds,
			"bytes" => value.bytes,
			"gctime" => value.gctime,
		)
	elseif value isa TimingBreakdown
		return timing_breakdown_to_dict(value)
	elseif value isa NamedTuple
		out = OrderedDict{String, Any}()
		for k in keys(value)
			out[string(k)] = _timing_json_value(getfield(value, k))
		end
		return out
	elseif value isa AbstractDict
		out = OrderedDict{String, Any}()
		for (k, v) in value
			out[string(k)] = _timing_json_value(v)
		end
		return out
	elseif value isa AbstractVector || value isa Tuple
		return Any[_timing_json_value(v) for v in value]
	else
		return string(value)
	end
end

function timing_breakdown_to_dict(timing::TimingBreakdown)
	return OrderedDict{String, Any}(
		"label" => string(timing.label),
		"total_seconds" => timing.total_seconds,
		"phases" => Any[_timing_json_value(entry) for entry in timing.phases],
		"details" => _timing_json_value(timing.details),
	)
end

function _hc_structure_key(poly_system, solve_vars, data_vars)
	return hash((string.(poly_system), string.(solve_vars), string.(data_vars)))
end

function _multipoint_combo_priority(
	combo::Vector{Int},
	time_values::AbstractVector{<:Real},
	interpolants::AbstractDict,
	mpt::MultiPointTemplate,
	strategy::Symbol,
)
	times = Float64[time_values[index] for index in combo]
	span = max(Float64(last(time_values) - first(time_values)), floatmin(Float64))
	# Preserve the historical default exactly: it ranked integer shooting indices,
	# which differs from time-distance ranking on uneven grids.
	index_spread = Float64(sum(
		abs(combo[i] - combo[j])
		for i in eachindex(combo) for j in (i + 1):length(combo)
	))
	strategy == :spread && return index_spread
	strategy == :boundary_order || throw(ArgumentError(
		"unknown multipoint pair strategy $strategy",
	))
	spread = sum(
		abs(times[i] - times[j])
		for i in eachindex(times) for j in (i + 1):length(times)
	) / span

	# Boundary-aware research strategy: retain separation while penalizing any
	# selected point whose available one-sided support is short relative to the
	# fitted GP lengthscale and the highest observed derivative order required by
	# the exact multipoint template.  It is deterministic and adds no GP fits.
	max_order = maximum(
		(meta.order for meta in mpt.data_var_meta if meta.kind == :observable_jet);
		init = 0,
	)
	lengthscales = Float64[
		interp.lengthscale for interp in values(interpolants)
		if interp isa AGPInterpolatorUQ
	]
	reference_scale = isempty(lengthscales) ? span / 10 : maximum(lengthscales)
	order_scale = sqrt(max_order + 1.0)
	boundary_quality = minimum(
		min(t - first(time_values), last(time_values) - t) /
		max(reference_scale * order_scale, floatmin(Float64))
		for t in times
	)
	return spread * clamp(boundary_quality, 0.0, 1.0)
end

# Noise-frontier selection is structural by default: row choice is based on a
# generic data/transcendental specialization, then reused across interpolators.
function _noise_frontier_run_cache_key(policy::Symbol, n_points::Int, opts::EstimationOptions; selection_mode::Symbol = :generic, interpolator::Union{Nothing, Symbol} = nothing)
	return (
		policy,
		selection_mode,
		policy == :noise_frontier && selection_mode == :generic ? nothing : interpolator,
		n_points,
		opts.construction_compute_mixed_volume,
		opts.construction_candidate_limit,
		opts.construction_beam_width,
	)
end

function _summarize_resolve_timing(records::Vector{NamedTuple})
	summary = OrderedDict{Symbol, OrderedDict{Symbol, Any}}()
	for record in records
		context = hasproperty(record, :context) ? record.context : :unspecified
		entry = get!(summary, context) do
			OrderedDict{Symbol, Any}(
				:count => 0,
				:total_seconds => 0.0,
				:sian_rerun_seconds => 0.0,
				:instantiate_seconds => 0.0,
				:hc_solve_seconds => 0.0,
				:cascading_seconds => 0.0,
				:cascade_hc_solve_seconds => 0.0,
				:max_seconds => 0.0,
				:threw_count => 0,
				:hc_success_count => 0,
			)
		end
		stages = hasproperty(record, :stage_seconds) ? record.stage_seconds : OrderedDict{Symbol, Float64}()
		total = Float64(hasproperty(record, :total_seconds) ? record.total_seconds : 0.0)
		entry[:count] += 1
		entry[:total_seconds] += total
		entry[:sian_rerun_seconds] += Float64(get(stages, :sian_rerun, 0.0))
		entry[:instantiate_seconds] += Float64(get(stages, :instantiate, 0.0))
		entry[:hc_solve_seconds] += Float64(get(stages, :hc_solve, 0.0))
		entry[:cascading_seconds] += Float64(get(stages, :cascading, 0.0))
		entry[:cascade_hc_solve_seconds] += Float64(get(stages, :cascade_hc_solve, 0.0))
		entry[:max_seconds] = max(Float64(entry[:max_seconds]), total)
		(hasproperty(record, :status) ? record.status : :unknown) == :threw && (entry[:threw_count] += 1)
		(hasproperty(record, :hc_status) ? record.hc_status : :unknown) == :success && (entry[:hc_success_count] += 1)
	end
	return summary
end

function _summarize_detailed_timing(records::Vector{NamedTuple})
	summary = OrderedDict{Symbol, OrderedDict{Symbol, Any}}()
	for record in records
		category = hasproperty(record, :category) ? record.category : :unspecified
		entry = get!(summary, category) do
			OrderedDict{Symbol, Any}(
				:count => 0,
				:total_seconds => 0.0,
				:max_seconds => 0.0,
				:stage_seconds => OrderedDict{Symbol, Float64}(),
				:context_counts => OrderedDict{Symbol, Int}(),
			)
		end
		total = Float64(hasproperty(record, :total_seconds) ? record.total_seconds : 0.0)
		entry[:count] += 1
		entry[:total_seconds] += total
		entry[:max_seconds] = max(Float64(entry[:max_seconds]), total)
		context = hasproperty(record, :context) ? record.context : :unspecified
		context_counts = entry[:context_counts]
		context_counts[context] = get(context_counts, context, 0) + 1
		stages = hasproperty(record, :stage_seconds) ? record.stage_seconds : OrderedDict{Symbol, Float64}()
		stage_seconds = entry[:stage_seconds]
		for (stage, seconds) in stages
			stage_seconds[stage] = get(stage_seconds, stage, 0.0) + Float64(seconds)
		end
	end
	return summary
end

"""
	_record_phase!(f, stats, opts, name)

Lightweight phase profiling helper. When `stats` is `nothing`, simply calls `f()`
(plus a flushed `_heartbeat` bracket). When `stats` is an OrderedDict, wraps `f()` with `@timed`
and records time, bytes, GC time, and peak RSS at phase end (+ Δ since phase
start). The RSS fields are observability-only -- they're not propagated to
`TimingPhaseEntry` (struct API stays stable) and are dropped silently by
`_phase_stats_to_breakdown`. To see them, print `phase_stats` directly via
`_print_phase_profile`.

The `f` argument comes first to support Julia's `do` block syntax:
```julia
result = _record_phase!(phase_stats, opts, "Phase name") do
    expensive_computation()
end
```
"""
function _record_phase!(f::Function, stats::Nothing, opts, name::String)
	_heartbeat(opts, name)
	t0 = time()
	result = f()
	_heartbeat(opts, name; kind = :done, extra = @sprintf("(%.1fs)", time() - t0))
	return result
end

function _record_phase!(f::Function, stats::OrderedDict{String, NamedTuple}, opts, name::String)
	_heartbeat(opts, name)
	rss_start = Sys.maxrss()
	result = @timed f()
	rss_end = Sys.maxrss()
	# Sys.maxrss returns bytes (high-water since process start, monotone).
	rss_mb = rss_end / (1024 * 1024)
	rss_delta_mb = (rss_end - rss_start) / (1024 * 1024)
	stats[name] = (
		time = result.time,
		bytes = result.bytes,
		gctime = result.gctime,
		rss_mb = rss_mb,
		rss_delta_mb = rss_delta_mb,
	)
	_heartbeat(opts, name; kind = :done, extra = @sprintf("(%.1fs)", result.time))
	return result.value
end

"""
	_format_bytes(bytes)

Format byte count as human-readable string (B, KiB, MiB, GiB).
"""
function _format_bytes(bytes::Number)
	if bytes < 1024
		return @sprintf("%.0f B", bytes)
	elseif bytes < 1024^2
		return @sprintf("%.1f KiB", bytes / 1024)
	elseif bytes < 1024^3
		return @sprintf("%.1f MiB", bytes / 1024^2)
	else
		return @sprintf("%.2f GiB", bytes / 1024^3)
	end
end

"""
	maybe_filter_interpolators_by_noise(list, pep, opts) -> Vector

Apply auto noise-based filter when `opts.auto_filter_interpolators` is true
AND the list contains a noise-sensitive method (S2AAAMLE, AAAD, AAADOld).
Otherwise pass through. Falls back to AAADGPR when filter empties the list.

Skipping the noise estimate (which is ~50ms) when no noise-sensitive method
is in the list keeps polynomial-mode users at zero cost.
"""
function maybe_filter_interpolators_by_noise(list::AbstractVector, pep, opts)
	!opts.auto_filter_interpolators && return list
	noise_sensitive_methods = (InterpolatorS2AAAMLE, InterpolatorAAA, InterpolatorAAADOld)
	has_noise_sensitive = any(entry -> entry[1] in noise_sensitive_methods, list)
	!has_noise_sensitive && return list
	σ̂ = estimate_relative_noise(pep)
	filtered = _apply_noise_filter(list, σ̂)
	if isempty(filtered)
		@warn "[INTERP-GATE] All interpolators filtered out (σ̂=$σ̂); falling back to InterpolatorAAADGPR"
		# Match the (method, custom_func_or_nothing) tuple shape from resolve_interpolator_list.
		# Built-in methods carry `nothing` in the function slot — the function is looked up
		# downstream via get_interpolator_function. Only InterpolatorCustom uses that slot.
		return Tuple{InterpolatorMethod, Union{Nothing, Function}}[(InterpolatorAAADGPR, nothing)]
	end
	return filtered
end

"""
	_print_phase_profile(stats)

Print a formatted table of phase timing and allocation statistics.
"""
function _print_phase_profile(stats::OrderedDict{String, NamedTuple})
	isempty(stats) && return

	# Compute totals
	total_time = sum(s.time for s in values(stats))
	total_bytes = sum(s.bytes for s in values(stats))
	total_gc = sum(s.gctime for s in values(stats))
	total_gc_pct = total_time > 0 ? 100.0 * total_gc / total_time : 0.0

	# RSS aggregates. Sys.maxrss is monotone non-decreasing within a process,
	# so the high-water mark across phases is just the last phase's rss_mb.
	# Total ΔRSS is the sum of per-phase deltas (== end_last - start_first).
	has_rss = !isempty(stats) && haskey(first(values(stats)), :rss_mb)
	peak_rss_mb = has_rss ? maximum(s.rss_mb for s in values(stats)) : 0.0
	total_rss_delta_mb = has_rss ? sum(s.rss_delta_mb for s in values(stats)) : 0.0

	# Column widths
	name_w = max(47, maximum(length(k) + 4 for k in keys(stats)))  # +4 for "N. " prefix

	println()
	if has_rss
		println("╔", "═"^name_w, "╤══════════╤═════════════╤═══════╤══════════╤═════════╗")
		@printf("║ %-*s│ %8s │ %11s │ %5s │ %8s │ %7s ║\n",
			name_w - 1, "Phase", "Time (s)", "Allocs", "GC %", "RSS (MB)", "ΔRSS")
		println("╠", "═"^name_w, "╪══════════╪═════════════╪═══════╪══════════╪═════════╣")

		for (i, (name, s)) in enumerate(stats)
			gc_pct = s.time > 0 ? 100.0 * s.gctime / s.time : 0.0
			label = "$i. $name"
			@printf("║ %-*s│ %8.2f │ %11s │ %4.1f%% │ %8.1f │ %+7.1f ║\n",
				name_w - 1, label, s.time, _format_bytes(s.bytes), gc_pct, s.rss_mb, s.rss_delta_mb)
		end

		println("╠", "═"^name_w, "╪══════════╪═════════════╪═══════╪══════════╪═════════╣")
		@printf("║ %-*s│ %8.2f │ %11s │ %4.1f%% │ %8.1f │ %+7.1f ║\n",
			name_w - 1, "TOTAL", total_time, _format_bytes(total_bytes), total_gc_pct, peak_rss_mb, total_rss_delta_mb)
		println("╚", "═"^name_w, "╧══════════╧═════════════╧═══════╧══════════╧═════════╝")
	else
		# Back-compat path for callers using the old 3-field NamedTuple shape.
		println("╔", "═"^name_w, "╤══════════╤═════════════╤═══════╗")
		@printf("║ %-*s│ %8s │ %11s │ %5s ║\n", name_w - 1, "Phase", "Time (s)", "Allocs", "GC %")
		println("╠", "═"^name_w, "╪══════════╪═════════════╪═══════╣")

		for (i, (name, s)) in enumerate(stats)
			gc_pct = s.time > 0 ? 100.0 * s.gctime / s.time : 0.0
			label = "$i. $name"
			@printf("║ %-*s│ %8.2f │ %11s │ %4.1f%% ║\n", name_w - 1, label, s.time, _format_bytes(s.bytes), gc_pct)
		end

		println("╠", "═"^name_w, "╪══════════╪═════════════╪═══════╣")
		@printf("║ %-*s│ %8.2f │ %11s │ %4.1f%% ║\n", name_w - 1, "TOTAL", total_time, _format_bytes(total_bytes), total_gc_pct)
		println("╚", "═"^name_w, "╧══════════╧═════════════╧═══════╝")
	end
end

function filter_finite_shooting_point_params(point_indices, param_values_list)
	valid_point_indices = Int[]
	valid_param_values_list = Vector{Vector{Float64}}()
	dropped_points = Vector{Tuple{Int, Int}}()

	for (point_idx, params) in zip(point_indices, param_values_list)
		nonfinite_count = count(value -> !isfinite(value), params)
		if nonfinite_count == 0
			push!(valid_point_indices, point_idx)
			push!(valid_param_values_list, params)
		else
			push!(dropped_points, (point_idx, nonfinite_count))
		end
	end

	return valid_point_indices, valid_param_values_list, dropped_points
end

function _resolve_missing_state_count(resolve_result)
	isempty(resolve_result.solutions) && return typemax(Int)
	return minimum(length(vars) for vars in resolve_result.missing_vars_per_solution)
end

function _algebraic_resolve_failure_notes(err)
	notes = Symbol[:algebraic_resolve_failed]
	if err isa TaskFailedException
		push!(notes, :algebraic_resolve_upstream_failure)
	else
		push!(notes, :algebraic_resolve_exception)
	end
	return notes
end

function _note_algebraic_resolve_failure!(results, indices, err)
	for idx in indices
		for note in _algebraic_resolve_failure_notes(err)
			note_provenance!(results[idx].provenance, note)
		end
		sync_result_contract!(results[idx])
	end
	return results
end

function _build_algebraic_resolve_candidate(
	PEP::ParameterEstimationProblem,
	params_dict::OrderedDict,
	good_udict,
	all_unidentifiable,
	resolve_result,
	state_vars,
	state_sol,
	resolve_time_index::Int,
	source_shoot_idx,
	source_interp,
	source_candidate_index::Int,
	state_seed_scale::Float64,
	opts::EstimationOptions;
	rescue_path::Symbol,
	parent_candidate_id::Int = 0,
)
	unknown_syms = ModelingToolkit.unknowns(PEP.model.system)
	current_params = ModelingToolkit.parameters(PEP.model.system)
	t_vector = PEP.data_sample["t"]

	sian_name_to_val = Dict{String, Float64}()
	for j in eachindex(state_vars)
		vname = replace(string(state_vars[j]), "(t)" => "")
		parsed = parse_derivative_variable_name(vname)
		if !isnothing(parsed)
			base, order = parsed
			if order == 0
				sian_name_to_val[String(base)] = state_sol[j]
			end
		end
	end

	if opts.diagnostics
		@info "[RESOLVE-MAP] sian_name_to_val: $(sian_name_to_val)"
		@info "[RESOLVE-MAP] MTK unknowns: $(string.(unknown_syms))"
	end

	raw_ic = Float64[]
	representative_assignments = OrderedDict{Num, Float64}()
	provenance_notes = Symbol[]
	append!(provenance_notes, resolve_result.notes)
	resolve_result.used_cascading && push!(provenance_notes, :cascading_substitution)
	resolve_time_index != 1 && push!(provenance_notes, :resolved_at_shooting_time)

	for s in unknown_syms
		sname = replace(string(s), "(t)" => "")
		if haskey(sian_name_to_val, sname)
			push!(raw_ic, sian_name_to_val[sname])
		elseif haskey(params_dict, s)
			push!(raw_ic, params_dict[s])
		else
			representative_value = apply_representative_assignment!(
				representative_assignments,
				provenance_notes,
				s,
				:state,
				all_unidentifiable,
			)
			if !isnothing(representative_value)
				push!(raw_ic, representative_value)
				continue
			end

			if startswith(sname, "_trfn_")
				t_shoot = Float64(t_vector[resolve_time_index])
				trfn_val = evaluate_trfn_template_variable(sname, t_shoot)
				isnothing(trfn_val) && error("Failed to reconstruct analytical _trfn_ state $sname at t=$t_shoot")
				push!(raw_ic, trfn_val)
			else
				resolved_from_measurement = false
				for mq in PEP.measured_quantities
					mq_rhs = Symbolics.diff2term(mq.rhs)
					if isequal(mq_rhs, s)
						mq_key = replace(string(mq.lhs), "(t)" => "")
						if haskey(PEP.data_sample, mq_key)
							push!(raw_ic, Float64(PEP.data_sample[mq_key][resolve_time_index]))
							resolved_from_measurement = true
							break
						end
					end
				end
				resolved_from_measurement && continue

				if opts.polish_solutions && opts.t0_state_completion == :seed_for_polish
					rng = MersenneTwister(UInt(abs(hash((PEP.name, sname, resolve_time_index, collect(values(params_dict)))))))
					seed_val = (2 * rand(rng) - 1) * state_seed_scale
					@warn "[RESOLVE-MAP] State $sname missing from the SIAN re-solve output; seeding with $seed_val for polish-assisted rescue"
					push!(raw_ic, seed_val)
				else
					error("State $sname is missing from the SIAN re-solve output and is not directly reconstructible. Refusing to fabricate a fallback value without polish.")
				end
			end
		end
	end

	raw_sol = Float64[]
	if t_vector[resolve_time_index] == _initial_time(PEP.data_sample)
		append!(raw_sol, raw_ic)
	else
		t_shoot = Float64(t_vector[resolve_time_index])
		t0 = Float64(_initial_time(PEP.data_sample))
		ordered_states_shoot = OrderedDict{Num, Float64}(s => raw_ic[i] for (i, s) in enumerate(unknown_syms))
		ordered_params_shoot = OrderedDict{Num, Float64}(p => Float64(params_dict[p]) for p in current_params)
		# Opt-in pre-backsolve parameter clamp (mirror of the main-backsolve clamp in
		# parameter_estimation_helpers.jl). No-op when opts.opt_lb / opts.opt_ub are not provided.
		ordered_params_shoot = _clamp_params_for_backsolve(
			ordered_params_shoot, opts, current_params, length(unknown_syms))
		prob = ODEProblem(complete(PEP.model.system), merge(ordered_states_shoot, ordered_params_shoot), (t_shoot, t0))
		ode_backsolve = ModelingToolkit.solve(prob, PEP.solver, abstol = opts.abstol, reltol = opts.reltol)
		for s in unknown_syms
			push!(raw_sol, Float64(real(ode_backsolve(t0, idxs = s))))
		end
	end
	append!(raw_sol, Float64[params_dict[p] for p in current_params])

	ordered_s, ordered_p, ode_solution, err = _with_detailed_timing_context(:algebraic_resolve_candidate) do
		process_raw_solution(
			raw_sol, PEP.model, PEP.data_sample, PEP.solver,
			abstol = opts.abstol, reltol = opts.reltol,
		)
	end

	candidate = ParameterEstimationResult(
		ordered_p, ordered_s,
		Float64(t_vector[resolve_time_index]),
		err, nothing,
		length(t_vector),
		Float64(_initial_time(PEP.data_sample)),
		OrderedDict{Num, Float64}(k => Float64(v) for (k, v) in good_udict),
		all_unidentifiable, ode_solution,
	)
	candidate.return_code = rescue_path
	candidate.provenance = ResultProvenance(
		primary_method = :algebraic,
		interpolator_source = source_interp,
		rescue_path = rescue_path,
		source_shooting_index = source_shoot_idx,
		source_candidate_index = source_candidate_index,
		post_polish_error = err,
		representative_assignments = representative_assignments,
		notes = unique(provenance_notes),
	)
	set_result_estimator_identity!(candidate;
		estimator_kind = :state_resolve,
		data_scope = :derived,
		time_indices = Int[resolve_time_index],
		time_values = Float64[t_vector[resolve_time_index]],
		interpolator_source = source_interp,
		parent_candidate_ids = parent_candidate_id > 0 ? Int[parent_candidate_id] : Int[],
	)
	return candidate
end


# ============================================================================
# Phase 4: Main Workflow
# ============================================================================

"""
	_derive_param_homotopy_template(si_template, good_DD, opts)

Derive the interpolator-independent parameter-homotopy polynomial system structure.

The polynomial template (equations, solve variables, data variables, _trfn_
classification) depends only on the SI template — NOT on which interpolator is
used to plug in numerical data. This is therefore computed ONCE before the
per-interpolator loop and shared across all interpolators.

Returns a NamedTuple with:
- `template_DD`: the DD used for data-variable extraction
- `data_vars`: observable-only data variables (HC parameters from DD.obs_lhs)
- `template_equations`: template equations with trfn-only equations removed
- `solve_vars`: real solve variables (HC unknowns; _trfn_ vars removed)
- `extended_data_vars`: `data_vars` plus ordered _trfn_ vars (HC parameters)
- `trfn_info`: classification dict for _trfn_ vars
- `trfn_vars_ordered`: ordered list of _trfn_ var keys
"""
function _derive_param_homotopy_template(si_template, good_DD, opts::EstimationOptions)
	# Extract data variables from DD.obs_lhs
	# These are the observable derivative variables (y1_0, y1_1, etc.)
	template_DD = hasproperty(si_template, :template_DD) ? si_template.template_DD : good_DD
	data_vars = extract_data_variables_from_DD(template_DD)

	# Get the template equations (without substituting interpolated values)
	# We need the raw template with data_vars as symbolic variables
	template_equations = si_template.equations

	# Extract solve variables (everything in template that's NOT a data var)
	all_vars_in_template = OrderedSet{Any}()
	for eq in template_equations
		union!(all_vars_in_template, Symbolics.get_variables(eq))
	end

	# Separate into solve_vars and verify data_vars are present
	data_vars_set = Set(data_vars)
	initial_solve_vars = [v for v in all_vars_in_template if !(v in data_vars_set)]

	# Move _trfn_ variables from solve_vars to data_vars.
	# These represent sin(c*t), cos(c*t) etc. — known at any time point.
	trfn_info, real_solve_vars, trfn_only_eq_indices = classify_trfn_in_template(
		initial_solve_vars, data_vars_set, template_equations
	)

	# Remove equations that only involve data_vars + _trfn_ vars (no real unknowns)
	if !isempty(trfn_only_eq_indices)
		keep_mask = trues(length(template_equations))
		for idx in trfn_only_eq_indices
			keep_mask[idx] = false
		end
		template_equations = template_equations[keep_mask]
		if !opts.nooutput
			@info "[TRFN-SOLVE] Removed $(length(trfn_only_eq_indices)) trivial _trfn_ equations, $(length(template_equations)) remaining"
		end
	end

	# Add _trfn_ vars to data_vars (they'll be evaluated numerically at each shooting point)
	trfn_vars_ordered = collect(keys(trfn_info))
	extended_data_vars = vcat(data_vars, trfn_vars_ordered)
	solve_vars = real_solve_vars

	if !opts.nooutput && !isempty(trfn_info)
		@info "[TRFN-SOLVE] Moved $(length(trfn_info)) _trfn_ vars to data; $(length(solve_vars)) real solve vars, $(length(template_equations)) equations"
	end

	return (; template_DD, data_vars, template_equations, solve_vars, extended_data_vars, trfn_info, trfn_vars_ordered)
end

"""
	optimized_multishot_parameter_estimation(PEP; kwargs...)

Optimized parameter estimation using precomputed derivatives.
"""
function optimized_multishot_parameter_estimation(PEP::ParameterEstimationProblem, opts::EstimationOptions = EstimationOptions())
	_validate_observation_options(PEP.data_sample, opts)
	_run_scopes_pending(opts) &&
		return _with_run_scopes(() -> optimized_multishot_parameter_estimation(PEP, opts), opts)
	# Direct callers receive the same run-scoped identity/artifact isolation as
	# the top-level analysis entrypoint.
	if _run_ctx() === nothing
		value, _ = _with_run_context(() -> optimized_multishot_parameter_estimation(PEP, opts))
		return value
	end
	_run_ctx_begin_uq!(opts.compute_uncertainty)
	# Carry the per-analysis HC solver config onto whatever RunContext is bound —
	# covers direct callers (consensus/benchmark flows) that bypass
	# analyze_parameter_estimation_problem; no-op when no context is bound.
	_run_ctx_set_hc_opts!(opts.hc_threading, opts.hc_compile_mode)
	# Check input validity
	_LAST_ESTIMATION_REUSE[] = nothing
	if isnothing(PEP.data_sample)
		error("No data sample provided in the ParameterEstimationProblem")
	end

	# Auto-handle transcendental functions (sin/cos/exp) if enabled
	tr_info = nothing
	if opts.auto_handle_transcendentals
		t_var = ModelingToolkit.get_iv(PEP.model.system)
		PEP, tr_info = transform_pep_for_estimation(PEP, t_var)
		if !isnothing(tr_info) && !opts.nooutput
			println("Transcendental handling: transformed $(length(tr_info.entries)) expression(s) into polynomial form")
		end
	end

	# Extract function references from options
	system_solver = get_solver_function(opts.system_solver)

	# Resolve the interpolator list: multi-interpolator if specified, else single interpolator
	interpolator_list = resolve_interpolator_list(opts)
	# Auto-filter AAA-family interpolators that are catastrophic at the data's noise level
	# (S2/AAAD/AAADOld). No-op when opts.auto_filter_interpolators is false or when no
	# noise-sensitive method is in the list. Falls back to AAADGPR if it empties the list.
	interpolator_list = maybe_filter_interpolators_by_noise(interpolator_list, PEP, opts)
	selection_recipe = _run_ctx_selection_recipe()
	if !isnothing(selection_recipe) && !isnothing(selection_recipe.interpolator_source)
		requested_source = selection_recipe.interpolator_source
		interpolator_list = [
			entry for entry in interpolator_list
			if interpolator_method_to_symbol(first(entry)) == requested_source
		]
		isempty(interpolator_list) && throw(ArgumentError(
			"fixed campaign recipe requested interpolator '$requested_source', but it is absent from the resolved pool",
		))
	end

	# Fast path: Use SI template exactly like the standard flow, but reuse it for all selected
	# shooting points in this run. This mirrors PE.jl construction.
	if opts.use_si_template
		# Phase profiling: initialize stats collector (nothing = disabled, zero overhead)
		phase_stats = (opts.profile_phases || _run_ctx_capture_timing()) ? OrderedDict{String, NamedTuple}() : nothing
		timing_details = OrderedDict{Symbol, Any}()
		interpolant_creation_seconds_by_source = OrderedDict{Symbol, Float64}()
		single_point_data_eval_seconds_by_source = OrderedDict{Symbol, Float64}()
		single_point_frontier_seconds_by_source = OrderedDict{Symbol, Float64}()
		single_point_generic_start_seconds_by_source = OrderedDict{Symbol, Float64}()
		single_point_hc_seconds_by_source = OrderedDict{Symbol, Float64}()
		single_point_system_seconds_by_source = OrderedDict{Symbol, Float64}()
		single_point_validation_seconds_by_source = OrderedDict{Symbol, Float64}()
		multipoint_template_seconds_by_source = OrderedDict{Symbol, Float64}()
		multipoint_generic_start_seconds_by_source = OrderedDict{Symbol, Float64}()
		multipoint_eval_seconds_by_source = OrderedDict{Symbol, Float64}()
		multipoint_template_eval_seconds_by_source = OrderedDict{Symbol, Float64}()
		multipoint_validation_seconds_by_source = OrderedDict{Symbol, Float64}()
		multipoint_solve_seconds_by_source = OrderedDict{Symbol, Float64}()
		raw_solver_polish_instantiate_seconds_by_source = OrderedDict{Symbol, Float64}()
		raw_solver_polish_solve_seconds_by_source = OrderedDict{Symbol, Float64}()
		reusable_system_cache = Dict{Any, Any}()
		raw_polish_kernel_cache = Dict{Tuple, PreparedRobustSystem}()
		reusable_order_cache = Dict{Any, Any}()
		resolve_timing_records = NamedTuple[]
		detailed_timing_records = NamedTuple[]
		_run_ctx_install_sinks!(resolve_timing_records, detailed_timing_records)
		noise_frontier_sp_cache_hits = 0
		noise_frontier_sp_cache_misses = 0
		noise_frontier_mp_cache_hits = 0
		noise_frontier_mp_cache_misses = 0
		sp_generic_start_cache_hits = 0
		sp_generic_start_cache_misses = 0
		mp_generic_start_cache_hits = 0
		mp_generic_start_cache_misses = 0

		try
		# Get common setup: identifiability analysis ONLY (shared across all interpolators)
		ident_data = _record_phase!(phase_stats, opts, "Setup (identifiability)") do
		setup_identifiability(
			PEP,
			max_num_points = 1,
			nooutput = opts.nooutput,
		)
		end

		states = ident_data.states
		params = ident_data.params
		t_vector = ident_data.t_vector
		good_deriv_level = ident_data.good_deriv_level
		good_udict = ident_data.good_udict
		good_varlist = ident_data.good_varlist
		good_DD = ident_data.good_DD
		numerical_advisory = ident_data.numerical_advisory

		# For backward compat, create setup_data with interpolants from the first interpolator
		# (used by process_estimation_results which needs the named tuple shape)
		first_interp_method, first_interp_custom = interpolator_list[1]
		first_interp_func = get_interpolator_function(first_interp_method, first_interp_custom; s3_adapt_k = opts.s3_adapt_k)
		first_interp_sym = interpolator_method_to_symbol(first_interp_method)
		_t_first_interp_start = time()
		first_interpolants = create_interpolants(PEP.measured_quantities, PEP.data_sample, t_vector, first_interp_func)
		_t_first_interp_elapsed = time() - _t_first_interp_start
		_accumulate_timing!(interpolant_creation_seconds_by_source, first_interp_sym, _t_first_interp_elapsed)
		setup_data = (
			states = states,
			params = params,
			t_vector = t_vector,
			interpolants = first_interpolants,
			good_num_points = ident_data.good_num_points,
			good_deriv_level = good_deriv_level,
			good_udict = good_udict,
			good_varlist = good_varlist,
			good_DD = good_DD,
			time_index_set = Int[],  # will be set after shooting point selection
			all_unidentifiable = ident_data.all_unidentifiable,
			numerical_advisory = numerical_advisory,
			si_template = nothing,
		)

		# Build the SI template once: apply structural representative fixing from
		# SI outputs, then require a square effective template.
		ordered_model = isa(PEP.model.system, OrderedODESystem) ? PEP.model.system : OrderedODESystem(PEP.model.system, states, params)

		si_template, template_equations = _record_phase!(phase_stats, opts, "SI Template (SIAN analysis)") do
		si_template, _template_structure = prepare_si_template_with_structural_fix(
			ordered_model,
			PEP.measured_quantities,
			PEP.data_sample,
			good_DD,
			opts.diagnostics;
			states = states,
			params = params,
			infolevel = opts.diagnostics ? 1 : 0,
			si_probability = opts.si_probability,
			si_fix_strategy = opts.si_fix_strategy,
			placeholder_fail_categories = opts.si_placeholder_fail_categories,
			compute_multiplicity = opts.compute_algebraic_multiplicity && isnothing(opts.algebraic_multiplicity),
		)

		@info "[DEBUG-EQ-COUNT] Final SI template: $(length(si_template.equations)) equations after structural fixing"
		template_equations = si_template.equations
		(si_template, template_equations)
		end

		# Auto-populate algebraic multiplicity from the SI template's Groebner step
		# (see si_equation_builder.jl). `analyze_parameter_estimation_problem` will
		# read this and override `opts.algebraic_multiplicity` if the caller left
		# it unset. Falls back to `nothing` on any internal failure.
		if hasproperty(si_template, :rank_trimming_metadata) &&
		   hasproperty(si_template.rank_trimming_metadata, :algebraic_multiplicity) &&
		   !isnothing(si_template.rank_trimming_metadata.algebraic_multiplicity)
			_run_ctx_set_auto_m!(si_template.rank_trimming_metadata.algebraic_multiplicity)
		end

		good_udict = OrderedDict{Num, Float64}(k => Float64(v) for (k, v) in si_template.structural_fix_set)
		good_DD.all_unidentifiable = Set{Num}(si_template.structural_unidentifiable)
		si_template = (
			; si_template...,
			practical_identifiability_status = practical_status_from_advisory(numerical_advisory),
			numerical_advisory = numerical_advisory,
		)

		setup_data = (; setup_data...,
			good_udict = good_udict,
			good_DD = good_DD,
			all_unidentifiable = si_template.structural_unidentifiable,
			si_template = si_template,
		)

		# Enable default system saving for SI-template path and save template once
		if opts.save_system
			vars_in_template = Symbolics.get_variables.(template_equations)
			varset = Set{Any}()
			for vs in vars_in_template
				for v in vs
					push!(varset, v)
				end
			end
			varlist_template = collect(varset)
			save_filepath_tpl = _saved_system_path(opts, "si_template")
			save_poly_system(save_filepath_tpl, template_equations, varlist_template,
				metadata = Dict(
					"timestamp" => string(now()),
					"num_equations" => length(template_equations),
					"num_variables" => length(varlist_template),
					"si_variable_role_counts" => string(si_template.si_variable_role_summary.counts),
					"si_auxiliary_variables" => string(si_template.si_variable_role_summary.auxiliary_variables),
					"suspicious_si_roles" => string(si_template.si_variable_role_summary.suspicious_categories),
					"structural_fix_set" => string(si_template.structural_fix_set),
					"si_fix_strategy" => string(si_template.structural_analysis.strategy),
					"template_status" => string(si_template.template_status),
					"rank_trim_dropped_equations" => string(si_template.rank_trimming_metadata.dropped_equation_indices),
					"description" => "StructuralIdentifiability template polynomial system",
				),
			)
			if !opts.nooutput
				@info "Saved SI template to $(save_filepath_tpl)"
				if !isempty(si_template.si_variable_role_summary.counts)
					@info "[SI-TEMPLATE] SI variable roles" counts = si_template.si_variable_role_summary.counts auxiliaries = si_template.si_variable_role_summary.auxiliary_variables suspicious = si_template.si_variable_role_summary.suspicious_categories
				end
			end
		end

		# Select shooting points. A research-scoped fixed recipe uses absolute
		# rows from the current data array; ordinary runs retain the adaptive path.
		if selection_recipe isa FixedSinglePointRecipe
			selection_recipe.row <= length(t_vector) || throw(ArgumentError(
				"fixed single-point row $(selection_recipe.row) exceeds $(length(t_vector)) observations",
			))
			point_indices = [selection_recipe.row]
			n_points = 1
		elseif selection_recipe isa FixedMultipointRecipe
			all(row -> row <= length(t_vector), selection_recipe.rows) ||
				throw(ArgumentError(
					"fixed multipoint rows $(selection_recipe.rows) exceed $(length(t_vector)) observations",
				))
			length(selection_recipe.rows) == opts.multipoint_n_points ||
				throw(ArgumentError(
					"fixed multipoint recipe has $(length(selection_recipe.rows)) rows, but multipoint_n_points=$(opts.multipoint_n_points)",
				))
			point_indices = copy(selection_recipe.rows)
			n_points = length(point_indices)
		elseif opts.shooting_points == 0
			# Single midpoint
			mid = max(1, min(length(t_vector), round(Int, 0.499 * length(t_vector))))
			point_indices = [mid]
			n_points = 1
		else
			n_points = min(opts.shooting_points, length(t_vector))
			point_indices = compute_shooting_indices(n_points, length(t_vector);
				warp = opts.shooting_warp, beta = opts.shooting_warp_beta)
			n_points = length(point_indices)  # may shrink after dedup
		end

		# Update setup_data with actual shooting point indices (needed by process_estimation_results)
		setup_data = (; setup_data..., time_index_set = point_indices)

		n_interpolators = length(interpolator_list)
		if !opts.nooutput
			interp_desc = n_interpolators > 1 ? " with $n_interpolators interpolators" : ""
			if n_points == 1
				println("Phase 3: Solving system using SI template at a single shooting point (t=$(t_vector[point_indices[1]]))$interp_desc...")
			elseif opts.use_parameter_homotopy && opts.system_solver == SolverHC && n_points >= 3
				println("Phase 3: Solving at $n_points shooting points using PARAMETER HOMOTOPY$interp_desc...")
			else
				println("Phase 3: Solving at $n_points shooting points using a reused SI template$interp_desc...")
			end
		end

		# Accumulators across all interpolators
		all_solutions = []
		all_hc_vars = []
		all_trivial_dicts = []
		all_trimmed_vars = []
		all_forward_subst_dicts = []
		all_reverse_subst_dicts = []
		all_final_varlists = []
		solution_time_indices = Int[]
		solution_source_types = Symbol[]           # :single_point or :multipoint
		solution_mp_time_indices = Vector{Union{Nothing, Vector{Int}}}()
		solution_mp_combo_indices = Union{Nothing, Int}[]
		solution_interpolator_sources = Symbol[]  # track which interpolator produced each solution
		solution_uq_artifacts = Union{Nothing, AbstractEstimatorArtifact}[]
		# Per-source interpolant cache for sensitivity-seed σ_d cross-interpolator spread.
		# Populated inside the per-interpolator loop; consumed by _maybe_augment_with_sensitivity_seeds
		# at the polish phase. Keyed by interpolator-source symbol (e.g. :aaad, :aaad_gpr).
		interpolant_cache_for_seeds = Dict{Symbol, AbstractDict}()

		# Check if we should use parameter homotopy and/or multi-point template.
		# Both can run — single-point always runs, multipoint adds solutions to the pool.
		use_multipoint = if selection_recipe isa FixedSinglePointRecipe
			false
		elseif selection_recipe isa FixedMultipointRecipe
			opts.system_solver == SolverHC || throw(ArgumentError(
				"fixed multipoint recipes require SolverHC",
			))
			true
		else
			opts.use_multipoint && opts.system_solver == SolverHC
		end
		use_param_homotopy = opts.use_parameter_homotopy && opts.system_solver == SolverHC && n_points >= 3

		_record_phase!(phase_stats, opts, "Equation construction + Solving") do

		# ============================================================================
		# PARAMETER-HOMOTOPY TEMPLATE (shared across interpolators)
		# The polynomial system structure and the :generic_start generic-complex
		# solve depend only on the SI template — NOT on which interpolator plugs in
		# numerical data. Compute them ONCE here, before the per-interpolator loop.
		# ============================================================================
		data_vars = template_equations = solve_vars = extended_data_vars = trfn_info = trfn_vars_ordered = template_DD = nothing
		precomputed_generic_solutions = nothing
		precomputed_generic_params = nothing
		if use_param_homotopy && opts.system_construction_policy == :legacy
			_ph = _derive_param_homotopy_template(si_template, good_DD, opts)
			template_DD = _ph.template_DD; data_vars = _ph.data_vars; template_equations = _ph.template_equations
			solve_vars = _ph.solve_vars; extended_data_vars = _ph.extended_data_vars
			trfn_info = _ph.trfn_info; trfn_vars_ordered = _ph.trfn_vars_ordered
			if opts.diagnostics
				println("  [param-homotopy template, shared across interpolators] solve_vars=$(length(solve_vars)), data_vars(+trfn)=$(length(extended_data_vars)), equations=$(length(template_equations))")
			end
			if opts.homotopy_tracking_mode == :generic_start
				_t_sp_generic_start = time()
				precomputed_generic_solutions, precomputed_generic_params = compute_generic_start_solutions(
					template_equations, solve_vars, extended_data_vars;
					gamma_seed = opts.gamma_seed, show_progress = opts.hc_show_progress, debug = opts.diagnostics)
				_accumulate_timing!(single_point_generic_start_seconds_by_source, :legacy, time() - _t_sp_generic_start)
				sp_generic_start_cache_misses += 1
			end
		elseif use_param_homotopy && opts.system_construction_policy == :noise_frontier && opts.diagnostics
			println("  [param-homotopy template] noise-frontier policy active; deriving generic structural template once and reusing it across interpolators")
		end

		sp_frontier_cache = Dict{Any, Any}()
		mp_template_cache = Dict{Any, Any}()
		sp_generic_cache = Dict{UInt64, Any}()
		mp_generic_cache = Dict{UInt64, Any}()

		# ============================================================================
		# MULTI-INTERPOLATOR LOOP
		# Each interpolator gets its own interpolants and solve pass.
		# The SI template and shooting points are shared across all interpolators.
		# ============================================================================
		for (interp_idx, (interp_method, interp_custom)) in enumerate(interpolator_list)
			interp_func = get_interpolator_function(interp_method, interp_custom; s3_adapt_k = opts.s3_adapt_k)
			interp_sym = interpolator_method_to_symbol(interp_method)
			_heartbeat(opts, "interpolator", extra = string(interp_idx, "/", length(interpolator_list), " ", interp_sym))

			# Create interpolants for this interpolator
			_t_interp_start = time()
			interpolants = create_interpolants(PEP.measured_quantities, PEP.data_sample, t_vector, interp_func)
			_t_interp_elapsed = time() - _t_interp_start
			_accumulate_timing!(interpolant_creation_seconds_by_source, interp_sym, _t_interp_elapsed)
			interpolant_cache_for_seeds[interp_sym] = interpolants
			interp_sym == :agp_uq && _run_ctx_register_noise_interpolants!(interp_sym, interpolants)

			frontier_sp_candidate = nothing
			local_template_DD = template_DD
			local_data_vars = data_vars
			local_template_equations = template_equations
			local_solve_vars = solve_vars
			local_extended_data_vars = extended_data_vars
			local_trfn_info = trfn_info
			local_trfn_vars_ordered = trfn_vars_ordered
			local_precomputed_generic_solutions = precomputed_generic_solutions
			local_precomputed_generic_params = precomputed_generic_params
			if opts.system_construction_policy == :noise_frontier
				frontier_setup = (
					good_deriv_level = good_deriv_level,
					good_udict = good_udict,
					good_varlist = good_varlist,
					good_DD = good_DD,
					interpolants = interpolants,
				)
				_t_frontier_start = time()
				_sp_frontier_key = _noise_frontier_run_cache_key(:single_point, 1, opts; selection_mode = :generic)
				_sp_frontier_hit = haskey(sp_frontier_cache, _sp_frontier_key)
				frontier_result = if _sp_frontier_hit
					noise_frontier_sp_cache_hits += 1
					sp_frontier_cache[_sp_frontier_key]
				else
					noise_frontier_sp_cache_misses += 1
					result = build_noise_frontier_system(
						PEP,
						frontier_setup,
						si_template;
						n_points = 1,
						compute_mixed_volume = opts.construction_compute_mixed_volume,
						candidate_limit = opts.construction_candidate_limit,
						beam_width = opts.construction_beam_width,
						diagnostics = opts.diagnostics,
						selection_mode = :generic,
					)
					if !isnothing(result.selected)
						sp_frontier_cache[_sp_frontier_key] = result
					end
					result
				end
				_t_frontier_elapsed = time() - _t_frontier_start
				_accumulate_timing!(single_point_frontier_seconds_by_source, interp_sym, _t_frontier_elapsed)
				frontier_sp_candidate = frontier_result.selected
				if isnothing(frontier_sp_candidate)
					@warn "[NOISE-FRONTIER] No single-point square full-rank candidate for interpolator; falling back to legacy construction" interpolator = interp_sym
					if use_param_homotopy
						_ph = _derive_param_homotopy_template(si_template, good_DD, merge_options(opts; system_construction_policy = :legacy))
						local_template_DD = _ph.template_DD
						local_data_vars = _ph.data_vars
						local_template_equations = _ph.template_equations
						local_solve_vars = _ph.solve_vars
						local_extended_data_vars = _ph.extended_data_vars
						local_trfn_info = _ph.trfn_info
						local_trfn_vars_ordered = _ph.trfn_vars_ordered
						if opts.homotopy_tracking_mode == :generic_start
							_sp_generic_key = _hc_structure_key(local_template_equations, local_solve_vars, local_extended_data_vars)
							if haskey(sp_generic_cache, _sp_generic_key)
								sp_generic_start_cache_hits += 1
								cached = sp_generic_cache[_sp_generic_key]
								local_precomputed_generic_solutions = cached.sols
								local_precomputed_generic_params = cached.params
								opts.diagnostics && println("[HC-PARAM][SP] Generic-start cache hit for fallback legacy structure → N=$(isnothing(cached.sols) ? 0 : length(cached.sols))")
							else
								sp_generic_start_cache_misses += 1
								_t_sp_generic_start = time()
								local_precomputed_generic_solutions, local_precomputed_generic_params = compute_generic_start_solutions(
									local_template_equations,
									local_solve_vars,
									local_extended_data_vars;
									gamma_seed = opts.gamma_seed,
									show_progress = opts.hc_show_progress,
									debug = opts.diagnostics,
								)
								_accumulate_timing!(single_point_generic_start_seconds_by_source, interp_sym, time() - _t_sp_generic_start)
								sp_generic_cache[_sp_generic_key] = (sols = local_precomputed_generic_solutions, params = local_precomputed_generic_params)
							end
						end
					end
				else
					local_template_DD = hasproperty(si_template, :template_DD) ? si_template.template_DD : good_DD
					local_template_equations = frontier_sp_candidate.equations
					local_solve_vars = frontier_sp_candidate.solve_vars
					local_data_vars = frontier_sp_candidate.data_vars
					local_extended_data_vars = frontier_sp_candidate.data_vars
					local_trfn_info = Dict{Any, Any}()
					local_trfn_vars_ordered = Any[]
					if opts.diagnostics || !opts.nooutput
						cache_label = _sp_frontier_hit ? "cache_hit" : "cache_miss"
						println("  [NOISE-FRONTIER][$interp_sym] SP selected K=$(frontier_sp_candidate.max_observed_order), eqs=$(length(local_template_equations)), solve_vars=$(length(local_solve_vars)), data_vars=$(length(local_extended_data_vars)), mixed_volume=$(frontier_sp_candidate.mixed_volume), build=$(round(_t_frontier_elapsed; digits=3))s, $cache_label")
					end
					if use_param_homotopy && opts.homotopy_tracking_mode == :generic_start
						_sp_generic_key = _hc_structure_key(local_template_equations, local_solve_vars, local_extended_data_vars)
						if haskey(sp_generic_cache, _sp_generic_key)
							sp_generic_start_cache_hits += 1
							cached = sp_generic_cache[_sp_generic_key]
							local_precomputed_generic_solutions = cached.sols
							local_precomputed_generic_params = cached.params
							opts.diagnostics && println("[HC-PARAM][SP] Generic-start cache hit → N=$(isnothing(cached.sols) ? 0 : length(cached.sols))")
						else
							sp_generic_start_cache_misses += 1
							_t_sp_generic_start = time()
							local_precomputed_generic_solutions, local_precomputed_generic_params = compute_generic_start_solutions(
								local_template_equations,
								local_solve_vars,
								local_extended_data_vars;
								gamma_seed = opts.gamma_seed,
								show_progress = opts.hc_show_progress,
								debug = opts.diagnostics,
							)
							_accumulate_timing!(single_point_generic_start_seconds_by_source, interp_sym, time() - _t_sp_generic_start)
							sp_generic_cache[_sp_generic_key] = (sols = local_precomputed_generic_solutions, params = local_precomputed_generic_params)
						end
					end
				end
			end

			if !opts.nooutput && n_interpolators > 1
				println("  Interpolator $interp_idx/$n_interpolators: $interp_method ($interp_sym)")
			end
			if opts.diagnostics
				println("  [$interp_sym] Interpolant creation: $(round(_t_interp_elapsed, digits=3))s")
			end

			# Per-interpolator solution accumulators
			interp_solutions = []
			interp_time_indices = Int[]
			interp_source_types = Symbol[]
			interp_mp_time_indices = Vector{Union{Nothing, Vector{Int}}}()
			interp_mp_combo_indices = Union{Nothing, Int}[]
			interp_uq_artifacts = Union{Nothing, AbstractEstimatorArtifact}[]

		try  # Catch errors in individual interpolators so one crash doesn't lose all results

		# ════════════════════════════════════════════════════════════════════
		# SINGLE-POINT PATH (always runs)
		# ════════════════════════════════════════════════════════════════════
		if use_param_homotopy
			# ============================================================================
			# PARAMETER HOMOTOPY PATH
			# Solve all shooting points together using HC parameter homotopy
			# ============================================================================
			if opts.diagnostics
				println("\n=== Using Parameter Homotopy for $(n_points) shooting points ===")
			end

			# Build parameter values for each shooting point
			_t_eval_start = time()
			param_values_list = Vector{Vector{Float64}}()
			for point_idx in point_indices
				_point_eval_t0 = time()
				_point_eval_stages = OrderedDict{Symbol, Float64}()
				t_point = t_vector[point_idx]
				param_values = if !isnothing(frontier_sp_candidate)
					_timed_detail_stage!(_point_eval_stages, :noise_frontier_data_vars) do
						evaluate_noise_frontier_data_vars_at_point(
							interpolants, local_extended_data_vars, PEP.measured_quantities, t_point
						)
					end
				else
					# Evaluate observable derivative data vars
					obs_param_values = _timed_detail_stage!(_point_eval_stages, :observable_data_vars) do
						evaluate_data_vars_at_point(
							interpolants, local_data_vars, local_template_DD, PEP.measured_quantities, t_point
						)
					end
					# Evaluate _trfn_ vars at this time point
					trfn_values = Float64[]
					_timed_detail_stage!(_point_eval_stages, :transcendental_data_vars) do
						for v in local_trfn_vars_ordered
							func_type, frequency, deriv_order = local_trfn_info[v]
							val = evaluate_trfn_template_variable(string(v), t_point)
							if isnothing(val)
								# Fallback: compute directly
								if func_type == :sin
									val = _eval_sin_derivative(frequency, t_point, deriv_order)
								elseif func_type == :cos
									val = _eval_cos_derivative(frequency, t_point, deriv_order)
								else
									val = _eval_exp_derivative(frequency, t_point, deriv_order)
								end
							end
							push!(trfn_values, val)
						end
					end
					# Concatenate: observable data + _trfn_ data
					vcat(obs_param_values, trfn_values)
				end
				push!(param_values_list, param_values)
				_record_detailed_timing!((
					category = :single_point_param_eval,
					context = interp_sym,
					total_seconds = time() - _point_eval_t0,
					stage_seconds = copy(_point_eval_stages),
					point_index = point_idx,
					time_value = Float64(t_point),
					param_count = length(param_values),
					used_noise_frontier = !isnothing(frontier_sp_candidate),
				))

				if opts.diagnostics
					println("  Point $point_idx (t=$t_point): $(length(param_values)) parameter values")
				end
			end
				_t_eval_elapsed = time() - _t_eval_start
				_accumulate_timing!(single_point_data_eval_seconds_by_source, interp_sym, _t_eval_elapsed)
				if opts.diagnostics
					println("  [$interp_sym] Param evaluation at shooting points: $(round(_t_eval_elapsed, digits=3))s")
				end

				valid_point_indices, valid_param_values_list, dropped_points =
					filter_finite_shooting_point_params(point_indices, param_values_list)

				if !isempty(dropped_points)
					if opts.diagnostics
						for (dropped_point_idx, nonfinite_count) in dropped_points
							println("  [$interp_sym] Dropping shooting point $dropped_point_idx due to $nonfinite_count nonfinite interpolated value(s)")
						end
					elseif !opts.nooutput
						println("  [$interp_sym] Dropped $(length(dropped_points)) shooting point(s) with nonfinite interpolated values before HC")
					end
				end

				if !isnothing(frontier_sp_candidate)
					rank_valid_point_indices = Int[]
					rank_valid_param_values_list = Vector{Vector{Float64}}()
					for (point_idx, param_values) in zip(valid_point_indices, valid_param_values_list)
						_t_sp_validation = time()
						validation = validate_noise_frontier_candidate_at_values(frontier_sp_candidate, param_values)
						_sp_validation_elapsed = time() - _t_sp_validation
						_accumulate_timing!(single_point_validation_seconds_by_source, interp_sym, _sp_validation_elapsed)
						_record_detailed_timing!((
							category = :single_point_noise_frontier_validation,
							context = interp_sym,
							total_seconds = _sp_validation_elapsed,
							stage_seconds = OrderedDict{Symbol, Float64}(:rank_validation => _sp_validation_elapsed),
							point_index = point_idx,
							valid = validation.valid,
							rank = validation.rank,
							target_rank = validation.target_rank,
							reason = validation.reason,
							sigma_max = validation.sigma_max,
							sigma_min = validation.sigma_min,
							unfloored_svd_ratio = validation.unfloored_svd_ratio,
							condition_proxy = validation.condition_proxy,
							rank_atol = validation.rank_atol,
						))
						if validation.valid
							push!(rank_valid_point_indices, point_idx)
							push!(rank_valid_param_values_list, param_values)
							if opts.diagnostics
								println("  [$interp_sym] Noise-frontier SP validation passed at point $point_idx: rank=$(validation.rank)/$(validation.target_rank), proxy≈$(round(validation.condition_proxy; digits=3)), sigma_min=$(round(validation.sigma_min; digits=3))")
							end
						else
							msg = "  [$interp_sym] Dropping shooting point $point_idx: generic noise-frontier subsystem validation failed ($(validation.reason), rank=$(validation.rank)/$(validation.target_rank))"
							if opts.diagnostics || !opts.nooutput
								println(msg)
							end
						end
					end
					valid_point_indices = rank_valid_point_indices
					valid_param_values_list = rank_valid_param_values_list
				end

				if isempty(valid_point_indices)
					@warn "Interpolator $interp_sym has no valid shooting points for this subsystem; skipping this branch."
					continue
				end

			# Solve using parameter homotopy
			solver_options = Dict(
				:show_progress => opts.hc_show_progress,
				:real_tol => opts.hc_real_tol,
				:debug => opts.diagnostics,
				:use_column_scaling => opts.use_column_scaling,
				:homotopy_tracking_mode => opts.homotopy_tracking_mode,
				:gamma_max_seeds => opts.gamma_max_seeds,
				:gamma_seed => opts.gamma_seed,
			)

				_t_hc_start = time()
				solutions_by_point = solve_with_hc_parameterized(
					local_template_equations, local_solve_vars, local_extended_data_vars, valid_param_values_list;
					options = solver_options,
					precomputed_generic_solutions = local_precomputed_generic_solutions,
					precomputed_generic_params = local_precomputed_generic_params,
				)
				_t_hc_elapsed = time() - _t_hc_start
				_accumulate_timing!(single_point_hc_seconds_by_source, interp_sym, _t_hc_elapsed)
				_heartbeat(opts, "HC solve (single-point)"; kind = :done,
					extra = @sprintf("%s %d pts (%.1fs)", interp_sym, length(valid_param_values_list), _t_hc_elapsed))
				if opts.diagnostics
					println("  [$interp_sym] HC solve: $(round(_t_hc_elapsed, digits=3))s")
				end

				# Process results from each shooting point
				for (i, (point_idx, point_solutions)) in enumerate(zip(valid_point_indices, solutions_by_point))
					if opts.diagnostics
						println("\n  Point $point_idx: $(length(point_solutions)) real solutions")
					end

				# Optional: polish each raw solver solution using fast NLLS if requested
				if opts.polish_solver_solutions && !isempty(point_solutions)
					_raw_polish_t0 = time()
					_raw_polish_stages = OrderedDict{Symbol, Float64}()
					# Build the instantiated system for polishing (need concrete equations)
					target_k, varlist_k = _timed_detail_stage!(_raw_polish_stages, :instantiate_system) do
						if !isnothing(frontier_sp_candidate)
							instantiate_noise_frontier_candidate(
								frontier_sp_candidate, interpolants, PEP.data_sample, PEP.measured_quantities, point_idx
							)
						else
							construct_equation_system_from_si_template(
								PEP.model.system,
								PEP.measured_quantities,
								PEP.data_sample,
								good_deriv_level,
								good_udict,
								good_varlist,
								good_DD;
								time_index_set = [point_idx],
								precomputed_interpolants = interpolants,
								si_template = si_template,
							)
						end
					end
						reusable_system_cache[(:sp_kept, interp_sym, point_idx)] = (target_k, varlist_k)
						# Noise-frontier instantiation preserves this exact template/order.
						# Legacy construction can rename/reorder unknowns; its concrete
						# system is shared by the roots at that point.
						parameterized_polish = !isnothing(frontier_sp_candidate)
						polish_data = parameterized_polish ? valid_param_values_list[i] : Float64[]
						polish_kernel = _timed_detail_stage!(_raw_polish_stages, :prepare_kernel) do
							_cached_robust_system!(raw_polish_kernel_cache,
								parameterized_polish ? local_template_equations : target_k,
								varlist_k, parameterized_polish ? local_extended_data_vars : Num[], opts)
						end

						polished_point = Vector{Vector{Float64}}()
					for sol in point_solutions
						start_pt = real.(sol)
						p_solutions, _, _, _ = _timed_detail_stage!(_raw_polish_stages, :robust_polish_solve) do
							solve_with_robust(target_k, varlist_k;
								start_point = start_pt, polish_only = true,
								prepared_system = polish_kernel, data_values = polish_data,
								options = Dict(:abstol => 1e-12, :reltol => 1e-12, :debug => false))
						end
						if !isempty(p_solutions)
							push!(polished_point, p_solutions[1])
						else
							push!(polished_point, sol)
						end
					end
					_accumulate_timing!(raw_solver_polish_instantiate_seconds_by_source, interp_sym, get(_raw_polish_stages, :instantiate_system, 0.0))
					_accumulate_timing!(raw_solver_polish_solve_seconds_by_source, interp_sym, get(_raw_polish_stages, :robust_polish_solve, 0.0))
					_record_detailed_timing!((
						category = :raw_solver_polish,
						context = interp_sym,
						total_seconds = time() - _raw_polish_t0,
						stage_seconds = copy(_raw_polish_stages),
						point_index = point_idx,
						input_solution_count = length(point_solutions),
						output_solution_count = length(polished_point),
						equation_count = length(target_k),
						variable_count = length(varlist_k),
					))
					point_solutions = polished_point
				end

				append!(interp_solutions, point_solutions)
				for sol in point_solutions
					artifact = if opts.compute_uncertainty
						_run_ctx_try_uq_capture("single-point parameter-homotopy artifact") do
							SinglePointUQArtifact(
								local_template_equations, local_solve_vars,
								local_extended_data_vars, copy(valid_param_values_list[i]),
								Float64.(real.(sol)), interpolants,
								point_idx, Float64(t_vector[point_idx]),
							)
						end
					else
						nothing
					end
					push!(interp_uq_artifacts, artifact)
				end
				for _ in 1:length(point_solutions)
					push!(interp_source_types, :single_point)
					push!(interp_mp_time_indices, nothing)
					push!(interp_mp_combo_indices, nothing)
				end
				for _ in 1:length(point_solutions)
					push!(interp_time_indices, point_idx)
				end
			end

			# Set final varlist from template solve_vars
			all_final_varlists = local_solve_vars
			all_hc_vars = local_solve_vars
			all_trimmed_vars = local_solve_vars

			_t_interp_total = _t_interp_elapsed + _t_eval_elapsed + _t_hc_elapsed
			if opts.diagnostics
				println("\n=== Parameter Homotopy complete for $interp_sym: $(length(interp_solutions)) solutions ===")
				println("  [$interp_sym] TOTAL: $(round(_t_interp_total, digits=3))s  (interp=$(round(_t_interp_elapsed, digits=3))s, eval=$(round(_t_eval_elapsed, digits=3))s, HC=$(round(_t_hc_elapsed, digits=3))s)")
			end

		else
			# ============================================================================
			# STANDARD PATH (solve independently at each shooting point)
			# ============================================================================
			for point_idx in point_indices
				if opts.diagnostics
					println("\n--- Solving at shooting point index: $point_idx (t=$(t_vector[point_idx])) ---")
				end

				# Instantiate the SI template at this time point
				_t_standard_point_start = time()
					target_k, varlist_k = if !isnothing(frontier_sp_candidate)
						instantiate_noise_frontier_candidate(
							frontier_sp_candidate, interpolants, PEP.data_sample, PEP.measured_quantities, point_idx
						)
					else
						construct_equation_system_from_si_template(
							PEP.model.system,
							PEP.measured_quantities,
							PEP.data_sample,
							good_deriv_level,
							good_udict,
							good_varlist,
							good_DD;
							time_index_set = [point_idx],
							precomputed_interpolants = interpolants,
							si_template = si_template,
						)
					end
					reusable_system_cache[(:sp_kept, interp_sym, point_idx)] = (target_k, varlist_k)

					final_target = target_k
					final_varlist_point = varlist_k

				# Optional: save system for debugging (mirror classic flow behavior)
				if opts.save_system
					# Save the instantiated polynomial system for this shooting point
					save_filepath = _saved_system_path(opts, "system_point_$(point_idx)")
					save_poly_system(
						save_filepath,
						final_target,
						final_varlist_point,
						metadata = Dict(
							"timestamp" => string(now()),
							"num_equations" => length(final_target),
							"num_variables" => length(final_varlist_point),
							"shooting_point_index" => point_idx,
							"time" => t_vector[point_idx],
							"deriv_level" => good_deriv_level,
							"description" => "SI-template instantiated system",
						),
					)
					# Also save a simple text version
					txt_filepath = replace(save_filepath, ".jl" => ".txt")
					open(txt_filepath, "w") do f
						println(f, "# Polynomial System (SI template)")
						println(f, "# Shooting point index: ", point_idx, ", t=", t_vector[point_idx])
						println(f, "# Equations: ", length(final_target))
						println(f, "# Variables: ", length(final_varlist_point))
						println(f, "# Variables list: ", final_varlist_point)
						println(f, "\n# Equations:")
						for (i, eq) in enumerate(final_target)
							println(f, "Eq", i, ": ", eq)
						end
					end
					@info "Saved SI-template system to $save_filepath"
				end

				# Solve for this point
				solver_options = Dict(
					:debug_solver => opts.debug_solver,
					:debug_cas_diagnostics => opts.debug_cas_diagnostics,
					:debug_dimensional_analysis => opts.debug_dimensional_analysis,
				)
				solutions, hc_vars, trivial_dict, trimmed_vars = system_solver(final_target, final_varlist_point; options = solver_options)
				_t_standard_point_elapsed = time() - _t_standard_point_start
				_accumulate_timing!(single_point_system_seconds_by_source, interp_sym, _t_standard_point_elapsed)

				# Optional: polish each raw solver solution using fast NLLS if requested
				if opts.polish_solver_solutions && !isempty(solutions)
					parameterized_polish = !isnothing(frontier_sp_candidate)
					polish_data = parameterized_polish ? evaluate_noise_frontier_data_vars_at_point(
						interpolants, frontier_sp_candidate.data_vars, PEP.measured_quantities,
						t_vector[point_idx]) : Float64[]
					polish_kernel = _cached_robust_system!(raw_polish_kernel_cache,
						parameterized_polish ? frontier_sp_candidate.equations : final_target,
						final_varlist_point, parameterized_polish ? frontier_sp_candidate.data_vars : Num[], opts)
					polished_point = Vector{Vector{Float64}}()
					for sol in solutions
						start_pt = real.(sol)
						p_solutions, _, _, _ = solve_with_robust(final_target, final_varlist_point; start_point = start_pt, polish_only = true, prepared_system = polish_kernel, data_values = polish_data, options = Dict(:abstol => 1e-12, :reltol => 1e-12, :debug => opts.diagnostics))
						if !isempty(p_solutions)
							push!(polished_point, p_solutions[1])
						else
							push!(polished_point, sol)
						end
					end
					solutions = polished_point
				end

				append!(interp_solutions, solutions)
				for sol in solutions
					artifact = if opts.compute_uncertainty && !isnothing(frontier_sp_candidate)
						_run_ctx_try_uq_capture("single-point noise-frontier artifact") do
							data_values = evaluate_noise_frontier_data_vars_at_point(
								interpolants, frontier_sp_candidate.data_vars,
								PEP.measured_quantities, t_vector[point_idx],
							)
							SinglePointUQArtifact(
								frontier_sp_candidate.equations, frontier_sp_candidate.solve_vars,
								frontier_sp_candidate.data_vars, data_values,
								Float64.(real.(sol)), interpolants,
								point_idx, Float64(t_vector[point_idx]),
							)
						end
					else
						nothing
					end
					push!(interp_uq_artifacts, artifact)
				end
				for _ in 1:length(solutions)
					push!(interp_source_types, :single_point)
					push!(interp_mp_time_indices, nothing)
					push!(interp_mp_combo_indices, nothing)
				end
				for _ in 1:length(solutions)
					push!(interp_time_indices, point_idx)
				end
				all_hc_vars = hc_vars
				push!(all_trivial_dicts, trivial_dict)
				push!(all_forward_subst_dicts, OrderedDict{Num, Any}())
				push!(all_reverse_subst_dicts, OrderedDict{Num, Num}())
				all_trimmed_vars = trimmed_vars
				all_final_varlists = final_varlist_point
			end
		end  # end if use_param_homotopy (single-point path)

		# ════════════════════════════════════════════════════════════════════
		# MULTI-POINT PATH (runs AFTER single-point, adds solutions to pool)
		# ════════════════════════════════════════════════════════════════════
		if use_multipoint
			_mpt_setup = (
				good_deriv_level = good_deriv_level,
				good_udict = good_udict,
				good_varlist = good_varlist,
				good_DD = good_DD,
				interpolants = interpolants,
			)
			_t_mpt_template_start = time()
			_mpt_template_key = _noise_frontier_run_cache_key(opts.system_construction_policy, opts.multipoint_n_points, opts; selection_mode = :generic, interpolator = interp_sym)
			_mpt_template_hit = haskey(mp_template_cache, _mpt_template_key)
			mpt = if _mpt_template_hit
				noise_frontier_mp_cache_hits += 1
				mp_template_cache[_mpt_template_key]
			else
				noise_frontier_mp_cache_misses += 1
				result = try
					if opts.system_construction_policy == :noise_frontier
						build_noise_frontier_multipoint_template(
							PEP,
							_mpt_setup,
							si_template;
							n_points = opts.multipoint_n_points,
							compute_mixed_volume = opts.construction_compute_mixed_volume,
							candidate_limit = opts.construction_candidate_limit,
							beam_width = opts.construction_beam_width,
							diagnostics = opts.diagnostics,
							selection_mode = :generic,
						)
					else
						build_multipoint_template(PEP, _mpt_setup, si_template;
							n_points = opts.multipoint_n_points, diagnostics = opts.diagnostics)
					end
				catch e
					_rethrow_if_interrupt(e)
					# Ungated (postcampaign review P1): a systematically failing template
					# build must not be invisible at default verbosity.
					@warn "[MULTIPOINT] Template build failed" exception = e maxlog = 10
					nothing
				end
				if !isnothing(result)
					mp_template_cache[_mpt_template_key] = result
				end
				result
			end
			_t_mpt_template_elapsed = time() - _t_mpt_template_start
			_accumulate_timing!(multipoint_template_seconds_by_source, interp_sym, _t_mpt_template_elapsed)
			if !isnothing(mpt) && (opts.diagnostics || !opts.nooutput)
				cache_label = _mpt_template_hit ? "cache_hit" : "cache_miss"
				println("  [MULTIPOINT] Template ready: eqs=$(length(mpt.stripped_equations)), solve_vars=$(length(mpt.solve_vars)), data_vars=$(length(mpt.data_vars)), build=$(round(_t_mpt_template_elapsed; digits=3))s, $cache_label")
			end

			if !isnothing(mpt) && length(mpt.stripped_equations) == length(mpt.solve_vars)
				# Generic-start: solve the (interpolator-independent) multipoint generic system ONCE per
				# distinct mpt structure; reuse across interpolators. Keyed by structure so a rare per-interp
				# structural difference recomputes safely.
				if opts.homotopy_tracking_mode == :generic_start
					_mpt_key = _hc_structure_key(mpt.stripped_equations, mpt.solve_vars, mpt.data_vars)
					if haskey(mp_generic_cache, _mpt_key)
						mp_generic_start_cache_hits += 1
						opts.diagnostics && println("[HC-PARAM][MP] Generic-start cache hit → N=$(isnothing(mp_generic_cache[_mpt_key].sols) ? 0 : length(mp_generic_cache[_mpt_key].sols))")
					else
						mp_generic_start_cache_misses += 1
						_t_mp_generic_start = time()
						_mp_sols, _mp_p0 = compute_generic_start_solutions(mpt.stripped_equations, mpt.solve_vars, mpt.data_vars;
							gamma_seed = opts.gamma_seed, show_progress = opts.hc_show_progress, debug = opts.diagnostics)
						_accumulate_timing!(multipoint_generic_start_seconds_by_source, interp_sym, time() - _t_mp_generic_start)
						mp_generic_cache[_mpt_key] = (sols = _mp_sols, params = _mp_p0)
					end
				end
				# Generate N-tuples from existing shooting points, capped at max_pairs
				n_pts = opts.multipoint_n_points
				combos = if selection_recipe isa FixedMultipointRecipe
					[copy(selection_recipe.rows)]
				else
					all_combos = Vector{Vector{Int}}()
					_generate_combinations!(all_combos, point_indices, n_pts)
					# Rank using the explicitly selected deterministic pair policy.
					time_values = Float64.(PEP.data_sample["t"])
					sort!(all_combos; by = c -> -_multipoint_combo_priority(
						c, time_values, interpolants, mpt, opts.multipoint_pair_strategy,
					))
					all_combos[1:min(opts.multipoint_max_pairs, length(all_combos))]
				end

				if !opts.nooutput || opts.diagnostics
					println("  [MULTIPOINT] $(length(combos)) $(n_pts)-point combos from $(length(point_indices)) shooting points")
				end

				_t_mpt_eval_start = time()
				evals = MultiPointEvaluation[]
					for combo in combos
						try
							_combo_eval_t0 = time()
							_combo_eval_stages = OrderedDict{Symbol, Float64}()
							ev = _with_detailed_timing_context(interp_sym) do
								_timed_detail_stage!(_combo_eval_stages, :evaluate_template) do
									evaluate_multipoint_template(mpt, combo, interpolants, PEP.data_sample)
								end
							end
							if opts.system_construction_policy == :noise_frontier
								validation = _timed_detail_stage!(_combo_eval_stages, :noise_frontier_validation) do
									validate_noise_frontier_instantiation(
										mpt.stripped_equations,
										mpt.solve_vars,
										mpt.data_vars,
										ev.data_values,
									)
								end
								_accumulate_timing!(multipoint_validation_seconds_by_source, interp_sym, get(_combo_eval_stages, :noise_frontier_validation, 0.0))
								if validation.valid
									push!(evals, ev)
									reusable_system_cache[(:mp_eval, interp_sym, Tuple(combo))] = ev
									if opts.diagnostics
										println("  [$interp_sym] Noise-frontier MP validation passed for combo $(Tuple(combo)): rank=$(validation.rank)/$(validation.target_rank), proxy≈$(round(validation.condition_proxy; digits=3)), sigma_min=$(round(validation.sigma_min; digits=3))")
									end
								elseif opts.diagnostics || !opts.nooutput
									println("  [$interp_sym] Dropping multipoint combo $(Tuple(combo)): generic noise-frontier template validation failed ($(validation.reason), rank=$(validation.rank)/$(validation.target_rank))")
								end
							elseif all(isfinite, ev.data_values)
								push!(evals, ev)
								reusable_system_cache[(:mp_eval, interp_sym, Tuple(combo))] = ev
							end
							_accumulate_timing!(multipoint_template_eval_seconds_by_source, interp_sym, get(_combo_eval_stages, :evaluate_template, 0.0))
							_record_detailed_timing!((
								category = :multipoint_combo_eval,
								context = interp_sym,
								total_seconds = time() - _combo_eval_t0,
								stage_seconds = copy(_combo_eval_stages),
								combo = Tuple(combo),
								data_value_count = length(ev.data_values),
								accepted = (opts.system_construction_policy == :noise_frontier) ? validation.valid : all(isfinite, ev.data_values),
								rank = (opts.system_construction_policy == :noise_frontier) ? validation.rank : missing,
								target_rank = (opts.system_construction_policy == :noise_frontier) ? validation.target_rank : missing,
								reason = (opts.system_construction_policy == :noise_frontier) ? validation.reason : :not_applicable,
								sigma_max = (opts.system_construction_policy == :noise_frontier) ? validation.sigma_max : missing,
								sigma_min = (opts.system_construction_policy == :noise_frontier) ? validation.sigma_min : missing,
								unfloored_svd_ratio = (opts.system_construction_policy == :noise_frontier) ? validation.unfloored_svd_ratio : missing,
								condition_proxy = (opts.system_construction_policy == :noise_frontier) ? validation.condition_proxy : missing,
								rank_atol = (opts.system_construction_policy == :noise_frontier) ? validation.rank_atol : missing,
							))
						catch e
							_rethrow_if_interrupt(e)
							# A systematic combo-eval failure must not be
							# indistinguishable from "no combos available".
							@warn "[MULTIPOINT] combo evaluation failed" exception = (e, catch_backtrace()) maxlog = 10
						end
					end
				_t_mpt_eval_elapsed = time() - _t_mpt_eval_start
				_accumulate_timing!(multipoint_eval_seconds_by_source, interp_sym, _t_mpt_eval_elapsed)

				if !isempty(evals)
					_mpt_sols0 = nothing
					_mpt_p0 = nothing
					if opts.homotopy_tracking_mode == :generic_start
						_mpt_solve_key = _hc_structure_key(mpt.stripped_equations, mpt.solve_vars, mpt.data_vars)
						if haskey(mp_generic_cache, _mpt_solve_key)
							_mpt_sols0 = mp_generic_cache[_mpt_solve_key].sols
							_mpt_p0 = mp_generic_cache[_mpt_solve_key].params
						end
					end
					_t_mpt_solve_start = time()
					solutions_by_combo = try
						solve_multipoint_parameterized(mpt, evals;
							options = Dict(:show_progress => opts.hc_show_progress, :real_tol => opts.hc_real_tol,
								:debug => opts.diagnostics,
								:use_column_scaling => opts.use_column_scaling,
								:homotopy_tracking_mode => opts.homotopy_tracking_mode,
								:gamma_max_seeds => opts.gamma_max_seeds, :gamma_seed => opts.gamma_seed),
								precomputed_generic_solutions = _mpt_sols0,
								precomputed_generic_params = _mpt_p0)
					catch e
						_rethrow_if_interrupt(e)
						@warn "[MULTIPOINT] HC solve failed" exception = e maxlog = 10
						nothing
					end
					_t_mpt_solve_elapsed = time() - _t_mpt_solve_start
					_accumulate_timing!(multipoint_solve_seconds_by_source, interp_sym, _t_mpt_solve_elapsed)
					_heartbeat(opts, "HC solve (multipoint)"; kind = :done,
						extra = @sprintf("%s (%.1fs)", interp_sym, _t_mpt_solve_elapsed))

					if !isnothing(solutions_by_combo) && !isempty(all_final_varlists)
						# Project multipoint solutions to single-point varlist ordering.
						# Build index mapping: for each var in single-point final_varlist,
						# find its position in mpt.solve_vars.
						sp_varlist = all_final_varlists
						mp_to_sp = Int[]
						for v in sp_varlist
							idx = findfirst(isequal(v), mpt.solve_vars)
							push!(mp_to_sp, isnothing(idx) ? 0 : idx)
						end

						n_projected = 0
						for (pidx, combo_sols) in enumerate(solutions_by_combo)
							for sol in combo_sols
								# Project: extract only the components matching single-point vars.
								# Unmapped vars are NaN (NEVER 0.0 — a fabricated value enters the
								# candidate pool as plausible data; the downstream finite guards
								# reject NaN cleanly). Postcampaign review P0#3, same class as
								# the Phase-C data-evaluator fixes.
								projected = Float64[idx > 0 ? sol[idx] : NaN for idx in mp_to_sp]
								push!(interp_solutions, projected)
								push!(interp_time_indices, evals[pidx].time_indices[1])
								push!(interp_source_types, :multipoint)
								push!(interp_mp_time_indices, copy(evals[pidx].time_indices))
								push!(interp_mp_combo_indices, pidx)
								artifact = opts.compute_uncertainty ?
									_run_ctx_try_uq_capture("multipoint algebraic artifact") do
										MultipointUQArtifact(
											evals[pidx], Float64.(real.(sol)), interpolants, interp_sym,
										)
									end : nothing
								push!(interp_uq_artifacts, artifact)
								n_projected += 1
							end
						end

						if !opts.nooutput || opts.diagnostics
							total = sum(length(s) for s in solutions_by_combo)
							println("  [MULTIPOINT] $(length(evals)) combos → $total solutions ($n_projected projected)")
						end
					end
				end
			end
		end  # end multipoint

		catch e
			_rethrow_if_interrupt(e)
			@error "Interpolator $interp_sym failed" exception=(e, catch_backtrace())
			println(stderr, "[HC-CRASH] Interpolator $interp_sym threw $(typeof(e)): $e")
			if @isdefined(param_values_list) && isa(param_values_list, AbstractVector)
				println(stderr, "[HC-CRASH] param_values_list had $(length(param_values_list)) points")
				for (pi, pv) in enumerate(param_values_list)
					if any(isnan, pv) || any(isinf, pv)
						println(stderr, "[HC-CRASH]   Point $pi: contains NaN/Inf: $pv")
					end
				end
			end
			# interp_solutions is empty, so accumulation below appends nothing
		end  # end try/catch for interpolator

			# A fixed multipoint campaign targets only roots of that exact
			# multipoint recipe. Single-point solves above are retained internally
			# because their variable ordering is the production projection contract,
			# but they cannot enter ranking or become the reported estimator.
			if selection_recipe isa FixedMultipointRecipe
				keep = findall(==(:multipoint), interp_source_types)
				interp_solutions = interp_solutions[keep]
				interp_time_indices = interp_time_indices[keep]
				interp_source_types = interp_source_types[keep]
				interp_mp_time_indices = interp_mp_time_indices[keep]
				interp_mp_combo_indices = interp_mp_combo_indices[keep]
				interp_uq_artifacts = interp_uq_artifacts[keep]
			end

			# Accumulate this pass's solutions into the global pool
			append!(all_solutions, interp_solutions)
			append!(solution_time_indices, interp_time_indices)
			append!(solution_source_types, interp_source_types)
			append!(solution_mp_time_indices, interp_mp_time_indices)
			append!(solution_mp_combo_indices, interp_mp_combo_indices)
			append!(solution_uq_artifacts, interp_uq_artifacts)
			for _ in 1:length(interp_solutions)
				push!(solution_interpolator_sources, interp_sym)
			end

			if !opts.nooutput
				println("    -> $interp_sym produced $(length(interp_solutions)) solutions")
			end

		end  # end for (interp_method, interp_custom) in interpolator_list
		end  # _record_phase! "Equation construction + Solving"

		# Merge and finalize
		solutions = all_solutions
		hc_vars = all_hc_vars
		trivial_dict = isempty(all_trivial_dicts) ? Dict{Any, Any}() : merge(all_trivial_dicts...)
		trimmed_vars = all_trimmed_vars
		forward_subst_dict = isempty(all_forward_subst_dicts) ? [OrderedDict{Num, Any}()] : [all_forward_subst_dicts[1]]
		reverse_subst_dict = isempty(all_reverse_subst_dicts) ? [OrderedDict{Num, Num}()] : [all_reverse_subst_dicts[1]]
		final_varlist = all_final_varlists

		if !opts.nooutput
			println("Found $(length(solutions)) solutions total" *
				(n_interpolators > 1 ? " across $n_interpolators interpolators" : ""))
		end

		# Use first interpolator's interpolants for downstream processing (resolve-states, etc.)
		interpolants = first_interpolants

		# Package as solution_data compatible with process_estimation_results
		solution_data = (
			solns = solutions,
			hcvarlist = hc_vars,
			trivial_dict = trivial_dict,
			trimmed_varlist = trimmed_vars,
			forward_subst_dict = forward_subst_dict,
			reverse_subst_dict = reverse_subst_dict,
			final_varlist = final_varlist,
			good_udict = good_udict,
			solution_time_indices = solution_time_indices,
			solution_interpolator_sources = solution_interpolator_sources,
			solution_source_types = solution_source_types,
			solution_mp_time_indices = solution_mp_time_indices,
			solution_mp_combo_indices = solution_mp_combo_indices,
			solution_uq_artifacts = solution_uq_artifacts,
			# Phase B: explicit MTK→jet-0 template map (built at SI template
			# construction; see si_equation_builder.jl). Consumed by
			# process_estimation_results so SI-workflow lookups hit the exact
			# template variable instead of guessing names.
			template_var_map = (hasproperty(si_template, :rank_trimming_metadata) &&
								hasproperty(si_template.rank_trimming_metadata, :template_var_map)) ?
							   si_template.rank_trimming_metadata.template_var_map : OrderedDict{Num, Num}(),
		)

		# Reuse existing processing pipeline (without polish — polish gets its own phase below)
		opts_no_polish = merge_options(opts; polish_solutions = false)
		solved_res = _record_phase!(phase_stats, opts, "Result processing") do
			_with_detailed_timing_context(:result_processing) do
				process_estimation_results(
					PEP,
					solution_data,
					setup_data;
					opts = opts_no_polish,
				)
			end
		end

		# PHASE: Detect blown-up backsolves and attempt algebraic re-solve at t=0
		if !isempty(solved_res) && opts.backsolve_recovery == :algebraic_resolve
			_, ub_vec = compute_default_bounds(PEP)
			bound_threshold = ub_vec[1]
			blown_indices = Int[]
			for (i, res) in enumerate(solved_res)
				ic_blown = any(v -> !isfinite(v) || abs(v) > bound_threshold, values(res.states))
				err_blown = !isnothing(res.err) && (!isfinite(res.err) || res.err > 1e15)
				if ic_blown || err_blown
					push!(blown_indices, i)
					if opts.diagnostics
						blown_states = [(k, v) for (k, v) in res.states if !isfinite(v) || abs(v) > bound_threshold]
						@info "[BACKSOLVE] Solution $i BLOWN: err=$(res.err), blown_states=$(blown_states)"
					end
				elseif opts.diagnostics
					# Log suspect-but-not-blown solutions for visibility
					max_ic = maximum(abs.(values(res.states)))
					@info "[BACKSOLVE] Solution $i OK: err=$(res.err), max_abs_IC=$max_ic"
				end
			end

			if !isempty(blown_indices)
				_heartbeat(opts, "Backsolve algebraic re-solve",
					extra = "($(length(blown_indices))/$(length(solved_res)) blown)")
				if !opts.nooutput
					println("Detected $(length(blown_indices))/$(length(solved_res)) blown backsolves, attempting algebraic re-solve at t=0")
				end

				# Deduplicate blown candidates by parameter values (many share same params from same shooting point)
				unique_param_sets = OrderedDict{UInt64, Tuple{OrderedDict, Vector{Int}}}()
				for i in blown_indices
					pvals = collect(values(solved_res[i].parameters))
					key = hash(round.(pvals; sigdigits = 10))
					if !haskey(unique_param_sets, key)
						unique_param_sets[key] = (solved_res[i].parameters, Int[])
					end
					push!(unique_param_sets[key][2], i)
				end

				if !opts.nooutput
					println("  $(length(unique_param_sets)) unique parameter set(s) to re-solve")
				end

				resolved_candidates = ParameterEstimationResult[]
				data_scale_vals = Float64[]
				for (k, v) in PEP.data_sample
					k == "t" && continue
					append!(data_scale_vals, abs.(Float64.(v)))
				end
				state_seed_scale = isempty(data_scale_vals) ? 1.0 : max(1.0, maximum(data_scale_vals))
				for (params_dict, indices) in values(unique_param_sets)
					# Build known_param_dict for parameter substitution
					known_param_dict = OrderedDict{Any, Float64}(k => Float64(v) for (k, v) in params_dict)
					base_candidate = solved_res[first(indices)]
					base_identity = ensure_result_estimator_identity!(base_candidate)
					source_shoot_idx = base_candidate.provenance.source_shooting_index
					source_interp = base_candidate.provenance.interpolator_source

					# Algebraic re-solve at t=0 with fixed parameters
					resolve_result = try
						_with_resolve_timing_context(:backsolve_recovery) do
							resolve_states_with_fixed_params(
								PEP.model.system,
								PEP.measured_quantities,
								PEP.data_sample,
								good_deriv_level,
								good_udict,
								good_varlist,
								good_DD,
								known_param_dict,
								interpolants;
								si_template = si_template,
								time_index = 1,
								diagnostics = opts.diagnostics,
								placeholder_fail_categories = opts.si_placeholder_fail_categories,
								compute_multiplicity = opts.compute_algebraic_multiplicity && isnothing(opts.algebraic_multiplicity),
							)
						end
					catch err
						_rethrow_if_interrupt(err)
						_note_algebraic_resolve_failure!(solved_res, indices, err)
						if opts.diagnostics
							@warn "[RESOLVE] Algebraic re-solve at t=0 failed; leaving original blown candidates unchanged for this parameter set." exception = err source_candidate_indices = indices
						elseif !opts.nooutput
							println("  Algebraic re-solve failed for one parameter set; keeping original blown candidate(s)")
						end
						continue
					end
					resolve_time_index = 1
					if !isnothing(source_shoot_idx) && source_shoot_idx != 1
						t0_partial = isempty(resolve_result.solutions) || _resolve_missing_state_count(resolve_result) > 0
						if t0_partial
							try
								shoot_resolve_result = _with_resolve_timing_context(:backsolve_recovery) do
									resolve_states_with_fixed_params(
										PEP.model.system,
										PEP.measured_quantities,
										PEP.data_sample,
										good_deriv_level,
										good_udict,
										good_varlist,
										good_DD,
										known_param_dict,
										interpolants;
										si_template = si_template,
										time_index = source_shoot_idx,
										diagnostics = opts.diagnostics,
										placeholder_fail_categories = opts.si_placeholder_fail_categories,
										compute_multiplicity = opts.compute_algebraic_multiplicity && isnothing(opts.algebraic_multiplicity),
									)
								end
								if !isempty(shoot_resolve_result.solutions) &&
								   _resolve_missing_state_count(shoot_resolve_result) < _resolve_missing_state_count(resolve_result)
									resolve_result = shoot_resolve_result
									resolve_time_index = source_shoot_idx
								end
							catch err
								_rethrow_if_interrupt(err)
								if opts.diagnostics
									@warn "[RESOLVE] Algebraic re-solve at shooting time failed; falling back to the existing t=0 resolve result." exception = err source_candidate_indices = indices shooting_index = source_shoot_idx
								end
							end
						end
					end
					state_solutions = resolve_result.solutions
					state_vars = resolve_result.state_vars

					if !isempty(state_solutions)
						for (state_sol_idx, state_sol) in enumerate(state_solutions)
							# Skip-on-failure: when t0_state_completion=:strict and the
							# resolve template doesn't include some MTK unknown (typically
							# because a candidate's parameters make a state unobservable —
							# e.g. a coupling parameter near zero), _build_algebraic_resolve_candidate
							# throws "missing from the SIAN re-solve output". Skip that candidate
							# instead of crashing the whole pipeline; the surviving candidates
							# in this and other parameter sets remain processable.
							candidate = try
								_build_algebraic_resolve_candidate(
									PEP,
									params_dict,
									good_udict,
									setup_data.all_unidentifiable,
									resolve_result,
									state_vars,
									state_sol,
									resolve_time_index,
									source_shoot_idx,
									source_interp,
									first(indices),
									state_seed_scale,
									opts;
									rescue_path = resolve_time_index == 1 ? :algebraic_resolve_t0 : :algebraic_resolve_shoot,
									parent_candidate_id = base_identity.candidate_id,
								)
							catch err
								if err isa ErrorException && occursin("missing from the SIAN re-solve output", err.msg)
									@warn "[RESOLVE-BUILD] Skipping candidate: $(err.msg)" source_candidate_indices = indices state_sol_idx = state_sol_idx
									continue
								else
									rethrow(err)
								end
							end
							candidate.provenance = copy_provenance(
								candidate.provenance;
								si_template_lineage_kwargs(setup_data.si_template)...,
							)
							sync_result_contract!(candidate)
							push!(resolved_candidates, candidate)
						end
					else
						if opts.polish_solutions && opts.t0_state_completion == :seed_for_polish
							if !opts.nooutput
								println("  HC re-solve found no solutions; generating polish seed from fixed parameters plus explicit/seeded ICs")
							end
							unknown_syms = ModelingToolkit.unknowns(PEP.model.system)
							raw_ic = Float64[]
							representative_assignments = OrderedDict{Num, Float64}()
							provenance_notes = Symbol[:partial_algebraic_resolve]
							append!(provenance_notes, resolve_result.notes)
							resolve_result.used_cascading && push!(provenance_notes, :cascading_substitution)
							for s in unknown_syms
								sname = replace(string(s), "(t)" => "")
								if startswith(sname, "_trfn_")
									t0 = Float64(_initial_time(PEP.data_sample))
									trfn_val = evaluate_trfn_template_variable(sname, t0)
									isnothing(trfn_val) && error("Failed to reconstruct analytical _trfn_ state $sname at t=$t0")
									push!(raw_ic, trfn_val)
									continue
								end

								resolved_from_measurement = false
								for mq in PEP.measured_quantities
									mq_rhs = Symbolics.diff2term(mq.rhs)
									if isequal(mq_rhs, s)
										mq_key = replace(string(mq.lhs), "(t)" => "")
										if haskey(PEP.data_sample, mq_key)
											push!(raw_ic, Float64(PEP.data_sample[mq_key][1]))
											resolved_from_measurement = true
											break
										end
									end
								end
								if resolved_from_measurement
									continue
								end

								representative_value = apply_representative_assignment!(
									representative_assignments,
									provenance_notes,
									s,
									:state,
									setup_data.all_unidentifiable,
								)
								if !isnothing(representative_value)
									push!(raw_ic, representative_value)
									continue
								end

								rng = MersenneTwister(UInt(abs(hash((PEP.name, sname, collect(values(params_dict)), :hc_resolve_seed)))))
								seed_val = (2 * rand(rng) - 1) * state_seed_scale
								@warn "[RESOLVE-SEED] State $sname unresolved after failed HC re-solve; seeding with $seed_val for polish-assisted rescue"
								push!(raw_ic, seed_val)
							end

							raw_sol = raw_ic
							# Parameter block in CURRENT (MTK) order — the convention
							# process_raw_solution decodes; the states above (raw_ic)
							# are already MTK-ordered (unknown_syms). Was
							# original_parameters: a MIXED-convention vector
							# (postcampaign review P0#1).
							append!(raw_sol, Float64[params_dict[p] for p in ModelingToolkit.parameters(PEP.model.system)])

							ordered_s, ordered_p, ode_solution, err = _with_detailed_timing_context(:algebraic_resolve_seeded_candidate) do
								process_raw_solution(
									raw_sol, PEP.model, PEP.data_sample, PEP.solver,
									abstol = opts.abstol, reltol = opts.reltol,
								)
							end

							candidate = ParameterEstimationResult(
								ordered_p, ordered_s,
								Float64(_initial_time(PEP.data_sample)),
								err, nothing,
								length(PEP.data_sample["t"]),
								Float64(_initial_time(PEP.data_sample)),
								OrderedDict{Num, Float64}(k => Float64(v) for (k, v) in good_udict),
								setup_data.all_unidentifiable, ode_solution,
							)
							candidate.return_code = :algebraic_resolve_seeded
							candidate.provenance = ResultProvenance(
								primary_method = :algebraic,
								interpolator_source = source_interp,
								rescue_path = :algebraic_resolve_seeded,
								source_shooting_index = source_shoot_idx,
								source_candidate_index = first(indices),
								post_polish_error = err,
								representative_assignments = representative_assignments,
								; si_template_lineage_kwargs(setup_data.si_template)...,
								notes = unique(provenance_notes),
							)
							set_result_estimator_identity!(candidate;
								estimator_kind = :state_resolve,
								data_scope = :derived,
								time_indices = Int[1],
								time_values = Float64[_initial_time(PEP.data_sample)],
								interpolator_source = source_interp,
								parent_candidate_ids = Int[base_identity.candidate_id],
							)
							sync_result_contract!(candidate)
							push!(resolved_candidates, candidate)
						elseif opts.diagnostics
							@warn "[RESOLVE] HC re-solve with fixed parameters found no state solutions at t0; leaving blown candidates unchanged because polish-assisted rescue is disabled."
						end
					end
				end

				# Replace blown candidates with resolved ones
				if !isempty(resolved_candidates)
					blown_set = Set(blown_indices)
					solved_res = [r for (i, r) in enumerate(solved_res) if !(i in blown_set)]
					append!(solved_res, resolved_candidates)

					if !opts.nooutput
						println("  After re-solve: $(length(solved_res)) total candidates ($(length(resolved_candidates)) from re-solve)")
					end
				elseif !opts.nooutput
					println("  Re-solve produced no usable rescue candidates; keeping original blown candidates for transparency")
				end

				# Log quality of resolved candidates
				if opts.diagnostics
					for (j, rc) in enumerate(resolved_candidates)
						max_ic = maximum(abs.(values(rc.states)))
						@info "[RESOLVE-RESULT] Candidate $j: err=$(rc.err), max_abs_IC=$max_ic, states=$(Dict(string(k)=>round(v;sigdigits=4) for (k,v) in rc.states))"
					end
				end
			end
		end

		# PHASE: Sensitivity-seed augmentation (off by default).
		# Runs independently of polish_solutions so the augmented pool is available
		# both for downstream polish AND for the no-polish output (each seed gets an
		# err evaluated via the polish-context loss closure inside _maybe_augment_*).
		if opts.use_sensitivity_seeds && !isempty(solved_res)
			solved_res = _record_phase!(phase_stats, opts, "Sensitivity seeds") do
				try
					_with_detailed_timing_context(:sensitivity_seeds) do
						ctx_for_seeds = _build_polish_context(PEP; opts = opts)
						_maybe_augment_with_sensitivity_seeds(
							PEP, solved_res, ctx_for_seeds, setup_data, opts;
							interpolant_cache = interpolant_cache_for_seeds,
						)
					end
				catch err
					_rethrow_if_interrupt(err)
					@warn "[Sensitivity seeds] generation failed; using unaugmented pool" exception = (err, catch_backtrace())
					solved_res
				end
			end
		end

		# PHASE: Synthesize aggregate candidates (default on; opt-out via opts.synthesize_aggregate_candidates=false).
		# Per-component median/mean/trim25 of SP and MP candidates (Categories B/C/A/D);
		# tagged with provenance.source_type = :synthesized_aggregate. Sidecar log at
		# artifacts/diagnostics/<model>/synthesis_log.csv.
		if opts.synthesize_aggregate_candidates && !isempty(solved_res)
			solved_res = _record_phase!(phase_stats, opts, "Synthesize aggregates") do
				try
					_with_detailed_timing_context(:synthesis) do
						_maybe_synthesize_aggregate_candidates(
							PEP, solved_res, setup_data, opts;
							interpolant_cache = interpolant_cache_for_seeds,
						)
					end
				catch err
					_rethrow_if_interrupt(err)
					@warn "[Synthesize aggregates] failed; using unaugmented pool" exception = (err, catch_backtrace())
					solved_res
				end
			end
		end

		# PHASE: Polish solutions (separate phase for profiling visibility)
		if opts.polish_solutions || (opts.terminal_fallback == :direct_opt && isempty(solved_res))
			solved_res = _record_phase!(phase_stats, opts, "Polish (BFGS)") do
				if isempty(solved_res)
					if opts.terminal_fallback == :direct_opt
						# Terminal rescue: keep parity with direct optimization, but make the provenance explicit.
						ctx = _with_detailed_timing_context(:polish) do
							_build_polish_context(PEP; opts = opts)
						end
						if !opts.nooutput
							println("No algebraic solutions found. Running explicit direct-optimization fallback from a random start.")
						end
						p_size = ctx.n_ic + ctx.n_param
						# Draw on unit scale and clamp into bounds when present — uniform-in-bounds
						# under ±1e9 default bounds almost always blows up the ODE and Optim 2's
						# Fminbox hard-errors on the resulting NaN initial mu.
						# Seeded per problem (postcampaign review P1): unseeded draws made
						# this terminal rescue a per-session coin flip — the same class as
						# the direct-opt canary flake (ddb06b4). Hash-derived like the
						# RESOLVE-SEED pattern above.
						_draw_rng = MersenneTwister(UInt(abs(hash((PEP.name, :direct_opt_fallback_seed)))))
						_draw = scale -> begin
							raw = scale .* randn(_draw_rng, p_size)
							(isnothing(ctx.lb) || isnothing(ctx.ub)) ? raw : clamp.(raw, ctx.lb, ctx.ub)
						end
						p0 = _draw(1.0)
						for attempt in 1:30
							loss0 = try
								ctx.optf.f(p0, nothing)
							catch err
								_rethrow_if_interrupt(err)
								Inf
							end
							isfinite(loss0) && break
							p0 = _draw(1.0 * 1.3^attempt)
						end
						result, opt_result = _polish_single_from_context(ctx, p0;
							optimizer = LBFGS(), maxiters = opts.polish_maxiters,
							maxtime = opts.polish_maxtime,
							divergence_factor = opts.polish_divergence_factor,
							stagnation_window = opts.polish_stagnation_window,
							retain_internal_optimum = opts.compute_uncertainty)
						result.provenance.rescue_path = :direct_opt_fallback
						result.provenance.primary_method = :direct_opt
						note_provenance!(result.provenance, :terminal_fallback)
						identity = set_result_estimator_identity!(result;
							estimator_kind = :direct_optimization,
							data_scope = :full_trajectory,
							time_indices = collect(eachindex(ctx.t_vector)),
							time_values = copy(ctx.t_vector),
						)
						if opts.compute_uncertainty
							artifact = _run_ctx_try_uq_capture("terminal direct-optimization artifact") do
								PolishUQArtifact(
									ctx, _polish_retained_internal_vector(ctx, result, opt_result),
									_polish_raw_optimizer_result(opt_result), 0,
									:trajectory_sse,
								)
							end
							!isnothing(artifact) &&
								_run_ctx_register_artifact!(identity.candidate_id, artifact)
						end
						sync_result_contract!(result)
						[result]
					else
						solved_res
					end
				else
					_with_detailed_timing_context(:polish) do
						ctx = _build_polish_context(PEP; opts = opts)
						_polish_batch_from_context(ctx, solved_res; opts = opts)
					end
				end
			end
		end

		if opts.branch_completion && !isempty(solved_res)
			solved_res = _record_phase!(phase_stats, opts, "Branch completion") do
				maybe_replace_with_branch_completion(PEP, solved_res, setup_data, opts)
			end
		end

		timing_details[:n_interpolators] = n_interpolators
		timing_details[:n_shooting_points] = n_points
		timing_details[:used_parameter_homotopy] = use_param_homotopy
		timing_details[:used_multipoint] = use_multipoint
		timing_details[:interpolant_creation_seconds_by_source] = copy(interpolant_creation_seconds_by_source)
		timing_details[:single_point_data_eval_seconds_by_source] = copy(single_point_data_eval_seconds_by_source)
		timing_details[:single_point_frontier_seconds_by_source] = copy(single_point_frontier_seconds_by_source)
		timing_details[:single_point_generic_start_seconds_by_source] = copy(single_point_generic_start_seconds_by_source)
		timing_details[:single_point_hc_seconds_by_source] = copy(single_point_hc_seconds_by_source)
		timing_details[:single_point_system_seconds_by_source] = copy(single_point_system_seconds_by_source)
		timing_details[:single_point_validation_seconds_by_source] = copy(single_point_validation_seconds_by_source)
		timing_details[:multipoint_template_seconds_by_source] = copy(multipoint_template_seconds_by_source)
		timing_details[:multipoint_generic_start_seconds_by_source] = copy(multipoint_generic_start_seconds_by_source)
		timing_details[:multipoint_eval_seconds_by_source] = copy(multipoint_eval_seconds_by_source)
		timing_details[:multipoint_template_eval_seconds_by_source] = copy(multipoint_template_eval_seconds_by_source)
		timing_details[:multipoint_validation_seconds_by_source] = copy(multipoint_validation_seconds_by_source)
		timing_details[:multipoint_solve_seconds_by_source] = copy(multipoint_solve_seconds_by_source)
		timing_details[:raw_solver_polish_instantiate_seconds_by_source] = copy(raw_solver_polish_instantiate_seconds_by_source)
		timing_details[:raw_solver_polish_solve_seconds_by_source] = copy(raw_solver_polish_solve_seconds_by_source)
		timing_details[:noise_frontier_sp_cache_hits] = noise_frontier_sp_cache_hits
		timing_details[:noise_frontier_sp_cache_misses] = noise_frontier_sp_cache_misses
		timing_details[:noise_frontier_mp_cache_hits] = noise_frontier_mp_cache_hits
		timing_details[:noise_frontier_mp_cache_misses] = noise_frontier_mp_cache_misses
		timing_details[:sp_generic_start_cache_hits] = sp_generic_start_cache_hits
		timing_details[:sp_generic_start_cache_misses] = sp_generic_start_cache_misses
		timing_details[:mp_generic_start_cache_hits] = mp_generic_start_cache_hits
		timing_details[:mp_generic_start_cache_misses] = mp_generic_start_cache_misses
		timing_details[:resolve_states_with_fixed_params_summary] = _summarize_resolve_timing(resolve_timing_records)
		timing_details[:resolve_states_with_fixed_params_records] = copy(resolve_timing_records)
		timing_details[:detailed_timing_summary] = _summarize_detailed_timing(detailed_timing_records)
		timing_details[:detailed_timing_records] = copy(detailed_timing_records)
		if hasproperty(si_template, :rank_trimming_metadata)
			rtm = si_template.rank_trimming_metadata
			if hasproperty(rtm, :structural_analysis)
				timing_details[:si_fix_strategy] = rtm.structural_analysis.strategy
				timing_details[:si_structural_analysis_timing] = copy(rtm.structural_analysis.timing)
				timing_details[:si_coordinate_basis] = copy(rtm.structural_analysis.coordinate_basis)
				timing_details[:si_structural_analysis_reused] = rtm.structural_analysis_reused
			end
			if hasproperty(rtm, :equation_builder_timing)
				timing_details[:si_template_equation_builder_timing] = rtm.equation_builder_timing
			end
			if hasproperty(rtm, :sian_timing)
				timing_details[:si_template_sian_timing] = rtm.sian_timing
			end
			if hasproperty(rtm, :algebraic_multiplicity_timing)
				timing_details[:si_template_algebraic_multiplicity_timing] = rtm.algebraic_multiplicity_timing
			end
		end
		timing_details[:reusable_system_cache_entries] = length(reusable_system_cache)
		timing_details[:raw_polish_kernel_cache_entries] = length(raw_polish_kernel_cache)
		_run_ctx_set_timing!(_phase_stats_to_breakdown(
			phase_stats,
			:optimized_multishot;
			details = timing_details,
		))
		_LAST_ESTIMATION_REUSE[] = (
			model_system = PEP.model.system,
			data_length = length(t_vector),
			use_multipoint = use_multipoint,
			multipoint_n_points = opts.multipoint_n_points,
			si_fix_strategy = opts.si_fix_strategy,
			si_probability = opts.si_probability,
			interpolator_sources = Symbol[interpolator_method_to_symbol(method) for (method, _) in interpolator_list],
			system_cache = copy(reusable_system_cache),
			order_cache = copy(reusable_order_cache),
		)

		# Print phase profiling table if enabled
		if opts.profile_phases && !isnothing(phase_stats)
			_print_phase_profile(phase_stats)
		end

		# The return signature is designed for compatibility with other workflows
		return (solved_res, good_udict, trivial_dict, setup_data.all_unidentifiable)
		finally
			# Timing-sink lifetime is scoped via RunContext (run_context.jl);
			# nothing to restore here anymore.
		end
	end

	# Non-SI-template path (use_si_template=false) archived 2026-06-10 →
	# deprecated/multishot_legacy.jl (postcampaign review P2). It had zero callers
	# and crashed anyway on undefined `use_adaptive_id` / `process_single_solution`.
	# The SI-template branch above always returns; reaching here means it was
	# explicitly disabled.
	error("optimized_multishot_parameter_estimation: use_si_template=false is no longer supported — the non-SI-template path was archived to deprecated/multishot_legacy.jl. Use the default use_si_template=true.")
end

# Export the main function
