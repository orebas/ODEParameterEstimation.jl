# =============================================================================
# Automatic power-of-2 problem rescaling
# =============================================================================
# Opt-in pre/post wrapper that rescales states / parameters / observables / data
# of a ParameterEstimationProblem to O(1) using ONLY powers of 2, runs the
# (unchanged) estimation, and un-rescales the results. Mirrors the transcendental
# `transform_pep_for_estimation` pattern exactly (detect -> rebuild PEP via
# create_ordered_ode_system -> return (new_pep, info), or (pep, nothing) no-op).
#
# WHY: the polynomial solve + local polish are conditioning-sensitive; an
# ill-scaled model (e.g. raw hiv: beta=2e-5 … k=50, x(0)=1000 — ~8 orders) makes
# them diverge, while an O(1) nondimensionalized version recovers. This makes the
# problem O(1) up front.
#
# SCOPE (MVP): states/params/observables/data only. TIME is NOT scaled (it would
# touch the data grid, interpolant abscissae, and the derivative template where an
# order-n derivative scales by 2^{c*n}); a separately-flagged follow-up.
#
# Powers of 2 are EXACT in Float64, so the change of variables is bit-reversible
# (the package is bit-sensitive — this is deliberate).

"""
	ScaleInfo

Power-of-2 scale factors chosen for a rescaled ParameterEstimationProblem.

# Fields
- `state_scales::OrderedDict{Num, Float64}`: state `x_i` => `2^{a_i}` (x_i_original = scale * x_i_scaled)
- `param_scales::OrderedDict{Num, Float64}`: parameter `p_j` => `2^{b_j}`
- `observable_scales::OrderedDict{Any, Float64}`: data_sample key (Num(mq.rhs)) => `2^{o_k}`
- `metadata::NamedTuple`: diagnostics (`max_const_before`, `max_const_after`, `n_nontrivial`, ...)
"""
struct ScaleInfo
	state_scales::OrderedDict{Num, Float64}
	param_scales::OrderedDict{Num, Float64}
	observable_scales::OrderedDict{Any, Float64}
	metadata::NamedTuple
end

# ---- numeric leaf helpers ---------------------------------------------------

function _rescale_asfloat(v)
	v isa Number && return Float64(v)
	try
		return Float64(v)
	catch
		return nothing
	end
end

function _rescale_asint(v)
	v = Symbolics.value(v)
	v isa Integer && return Int(v)
	(v isa Number && isfinite(v) && isinteger(v)) && return Int(v)
	return nothing
end

# ---- monomial extraction ----------------------------------------------------
# Split an expanded RHS into additive terms, then each term into (numeric
# coefficient, Dict{var_value => integer power}). Variables are recognized by
# membership in `var_set` (a Set of Symbolics.value(...) of states+params).
# Opaque factors (t, sin/cos/exp, non-integer powers) get no scale column and do
# not distort the coefficient. Mirrors the AST-walk idiom in transcendental_utils.

function _rescale_additive_terms(v)
	v = Symbolics.value(v)
	if Symbolics.iscall(v)
		op = Symbolics.operation(v)
		args = Symbolics.arguments(v)
		if op === (+)
			return collect(Iterators.flatten(_rescale_additive_terms(a) for a in args))
		elseif op === (-) && length(args) == 2
			return vcat(_rescale_additive_terms(args[1]), _rescale_additive_terms(Symbolics.value(-Num(args[2]))))
		elseif op === (-) && length(args) == 1
			return _rescale_additive_terms(Symbolics.value(-Num(args[1])))
		end
	end
	return Any[v]
end

function _rescale_accumulate!(v, power::Int, cref::Base.RefValue{Float64}, powers::Dict{Any, Int}, var_set::Set{Any})
	v = Symbolics.value(v)
	if v in var_set                      # state or param (check BEFORE iscall)
		powers[v] = get(powers, v, 0) + power
		return
	end
	if !Symbolics.iscall(v)
		c = _rescale_asfloat(v)
		if c === nothing
			powers[v] = get(powers, v, 0) + power      # opaque leaf (e.g. t)
		elseif power >= 0
			cref[] *= c^power
		else
			cref[] *= (1.0 / c)^(-power)
		end
		return
	end
	op = Symbolics.operation(v)
	args = Symbolics.arguments(v)
	if op === (*)
		for a in args
			_rescale_accumulate!(a, power, cref, powers, var_set)
		end
	elseif op === (/)
		_rescale_accumulate!(args[1], power, cref, powers, var_set)
		for a in args[2:end]
			_rescale_accumulate!(a, -power, cref, powers, var_set)
		end
	elseif op === (^)
		pe = _rescale_asint(args[2])
		if pe === nothing
			powers[v] = get(powers, v, 0) + power      # non-integer power -> opaque
		else
			_rescale_accumulate!(args[1], power * pe, cref, powers, var_set)
		end
	else
		powers[v] = get(powers, v, 0) + power          # sin/cos/exp/other -> opaque
	end
end

"""
	extract_monomials(rhs, var_set) -> Vector{Tuple{Float64, Dict{Any,Int}}}

Enumerate the additive monomials of `Symbolics.expand(rhs)` as
`(numeric_coefficient, Dict(var_value => integer_power))`. Variables are the
elements of `var_set` (Symbolics.value of each state/parameter). Zero-coefficient
terms are dropped. Used to balance equation constants for power-of-2 rescaling.
"""
function extract_monomials(rhs, var_set::Set{Any})
	expr = Symbolics.value(Symbolics.expand(rhs))
	out = Tuple{Float64, Dict{Any, Int}}[]
	for term in _rescale_additive_terms(expr)
		cref = Ref(1.0)
		powers = Dict{Any, Int}()
		_rescale_accumulate!(term, 1, cref, powers, var_set)
		if cref[] != 0.0 && isfinite(cref[])
			# drop opaque-only zero-power noise but keep constant-only terms (empty powers)
			push!(out, (cref[], powers))
		end
	end
	return out
end

# ---- scale selection --------------------------------------------------------

# eq.lhs is Differential(t)(x_i(t)); return the Symbolics.value of x_i(t).
function _rescale_diff_state(eq)
	lhs = Symbolics.value(eq.lhs)
	if Symbolics.iscall(lhs)
		args = Symbolics.arguments(lhs)
		!isempty(args) && return Symbolics.value(args[1])
	end
	return nothing
end

# data_sample key used for an observable: Num(mq.rhs) preferred, Num(mq.lhs) fallback.
function _rescale_obs_key(mq, data_sample)
	rkey = Num(mq.rhs)
	(data_sample !== nothing && haskey(data_sample, rkey)) && return rkey
	lkey = Num(mq.lhs)
	(data_sample !== nothing && haskey(data_sample, lkey)) && return lkey
	return rkey
end

function _rescale_data_magnitude(vec::AbstractVector, max_abs_exp::Int)
	a = abs.(Float64.(vec))
	filter!(isfinite, a)
	isempty(a) && return 2.0^(-max_abs_exp)
	m = quantile(a, 0.75)
	return max(m, 2.0^(-max_abs_exp))
end

"""
	choose_scales(pep; objective=:ls, max_abs_exp=60, w_data=1.0, ridge=1e-6) -> ScaleInfo

Choose power-of-2 scale exponents for states/params/observables so the rescaled
equations' constants and the data are close to O(1). Builds a small linear system
`A·θ = b` (one row per equation/observable monomial + per directly-observed-state
data anchor), solves it (least squares, `:ls`, via ridge normal equations; or
`:minmax_irls` for a true min-max), and rounds θ to integer exponents. Never errors
(degenerate systems return the identity scaling).
"""
function choose_scales(pep::ParameterEstimationProblem; objective::Symbol = :ls,
	max_abs_exp::Int = 60, w_data::Float64 = 1.0, ridge::Float64 = 1e-6)
	system = pep.model.system
	eqs = ModelingToolkit.equations(system)
	states = Num.(ModelingToolkit.unknowns(system))
	params = Num.(ModelingToolkit.parameters(system))
	data = pep.data_sample

	state_vals = [Symbolics.value(s) for s in states]
	param_vals = [Symbolics.value(p) for p in params]
	var_set = Set{Any}(vcat(state_vals, param_vals))
	# column index per variable value: states 1..n, params n+1..n+m
	col = Dict{Any, Int}()
	for (i, v) in enumerate(state_vals); col[v] = i; end
	for (j, v) in enumerate(param_vals); col[v] = length(state_vals) + j; end
	W = length(col)

	# observable normalization exponents o_k from data (robust 75th-pct magnitude)
	obs_exp = OrderedDict{Any, Int}()
	for mq in pep.measured_quantities
		key = _rescale_obs_key(mq, data)
		o = if _is_trfn_observable(Num(mq.rhs))
			0   # a known input: see the pinning below
		elseif data !== nothing && haskey(data, key)
			clamp(round(Int, log2(_rescale_data_magnitude(data[key], max_abs_exp))), -max_abs_exp, max_abs_exp)
		else
			0
		end
		obs_exp[key] = o
	end

	rows = Tuple{Dict{Int, Float64}, Float64}[]
	max_const_before = 0.0

	# (1) state-equation monomial rows: Σ e_k s_k − a_i = −log2|κ|
	for eq in eqs
		xi = _rescale_diff_state(eq)
		(xi === nothing || !haskey(col, xi)) && continue
		for (κ, powers) in extract_monomials(eq.rhs, var_set)
			max_const_before = max(max_const_before, abs(log2(abs(κ))))
			r = Dict{Int, Float64}()
			for (v, e) in powers
				haskey(col, v) && (r[col[v]] = get(r, col[v], 0.0) + e)
			end
			r[col[xi]] = get(r, col[xi], 0.0) - 1.0
			push!(rows, (r, -log2(abs(κ))))
		end
	end

	# (2) observable monomial rows: Σ e_k s_k = −log2|κ| + o_k
	for mq in pep.measured_quantities
		key = _rescale_obs_key(mq, data)
		ok = get(obs_exp, key, 0)
		for (κ, powers) in extract_monomials(mq.rhs, var_set)
			r = Dict{Int, Float64}()
			for (v, e) in powers
				haskey(col, v) && (r[col[v]] = get(r, col[v], 0.0) + e)
			end
			isempty(r) && continue
			push!(rows, (r, -log2(abs(κ)) + ok))
		end
	end

	# (3) data-anchor rows for directly-observed single states: a_i = o_k
	for mq in pep.measured_quantities
		key = _rescale_obs_key(mq, data)
		mons = extract_monomials(mq.rhs, var_set)
		if length(mons) == 1 && isapprox(mons[1][1], 1.0; atol = 1e-9) && length(mons[1][2]) == 1
			v, e = first(mons[1][2])
			if e == 1 && haskey(col, v)
				push!(rows, (Dict(col[v] => w_data), w_data * get(obs_exp, key, 0)))
			end
		end
	end

	# Known inputs keep their scale. The states that stand for sin(c*t), cos(c*t)
	# and exp(c*t) are given their exact, unscaled values when the equations are
	# solved, so a scaled copy of them would disagree with those values (a
	# sinusoid of amplitude 1/4 used to come back with its coefficient doubled).
	pinned = [i for (i, s) in enumerate(states) if _is_trfn_observable(s)]
	if !isempty(pinned)
		for (r, _) in rows
			foreach(k -> delete!(r, k), pinned)
		end
		filter!(row -> !isempty(first(row)), rows)
	end

	# assemble dense system and solve
	exps = zeros(Int, W)
	if W > 0 && !isempty(rows)
		A = zeros(Float64, length(rows), W)
		b = zeros(Float64, length(rows))
		for (i, (r, rhs)) in enumerate(rows)
			for (k, val) in r
				A[i, k] = val
			end
			b[i] = rhs
		end
		θ = _rescale_solve(A, b, objective, ridge)
		exps = clamp.(round.(Int, θ), -max_abs_exp, max_abs_exp)
		# pin columns that appear in no row to exponent 0
		for k in 1:W
			all(iszero, @view A[:, k]) && (exps[k] = 0)
		end
	end

	# post-round residual (max |log2(new constant)|) for the diagnostics
	max_const_after = 0.0
	if !isempty(rows)
		for (r, rhs) in rows
			L = -rhs
			for (k, val) in r
				L += val * exps[k]
			end
			max_const_after = max(max_const_after, abs(L))
		end
	end

	state_scales = OrderedDict{Num, Float64}()
	for (i, s) in enumerate(states)
		state_scales[s] = 2.0^exps[i]
	end
	param_scales = OrderedDict{Num, Float64}()
	for (j, p) in enumerate(params)
		param_scales[p] = 2.0^exps[length(states)+j]
	end
	observable_scales = OrderedDict{Any, Float64}()
	for (key, o) in obs_exp
		observable_scales[key] = 2.0^o
	end

	n_nontrivial = count(!=(0), exps) + count(!=(0), collect(values(obs_exp)))
	meta = (max_const_before = max_const_before, max_const_after = max_const_after,
		n_nontrivial = n_nontrivial, objective = objective)
	return ScaleInfo(state_scales, param_scales, observable_scales, meta)
end

function _rescale_solve(A::Matrix{Float64}, b::Vector{Float64}, objective::Symbol, ridge::Float64)
	W = size(A, 2)
	# ridge normal equations: robust to rank deficiency; free columns -> ~0.
	solve_ls(w) = begin
		Aw = w === nothing ? A : (w .* A)
		bw = w === nothing ? b : (w .* b)
		(Aw' * Aw + ridge * LinearAlgebra.I(W)) \ (Aw' * bw)
	end
	θ = solve_ls(nothing)
	if objective === :minmax_irls
		for _ in 1:8
			resid = A * θ .- b
			w = 1.0 ./ sqrt.(max.(abs.(resid), 1e-3))
			θ = solve_ls(w)
		end
	end
	return θ
end

# ---- wrap / un-wrap ---------------------------------------------------------

"""
	rescale_pep(pep) -> (rescaled_pep, ScaleInfo) or (pep, nothing)

Apply power-of-2 rescaling. Returns `(pep, nothing)` unchanged when scaling is
trivial (all exponents 0) — an idempotent no-op, like the transcendental wrapper.
"""
function rescale_pep(pep::ParameterEstimationProblem; kwargs...)
	info = choose_scales(pep; kwargs...)

	# no-op if every scale is 1.0
	trivial = all(==(1.0), values(info.state_scales)) &&
			  all(==(1.0), values(info.param_scales)) &&
			  all(==(1.0), values(info.observable_scales))
	trivial && return pep, nothing

	system = pep.model.system
	_, eqs, _, _ = unpack_ODE(system)        # deepcopied equations
	states = Num.(ModelingToolkit.unknowns(system))
	params = Num.(ModelingToolkit.parameters(system))

	# substitution dict: x_i -> 2^{a_i} x_i, p_j -> 2^{b_j} p_j
	sub = Dict{Any, Any}()
	state_exp_by_val = Dict{Any, Float64}()
	for s in states
		sc = info.state_scales[s]
		sub[Symbolics.value(s)] = sc * s
		state_exp_by_val[Symbolics.value(s)] = sc
	end
	for p in params
		sub[Symbolics.value(p)] = info.param_scales[p] * p
	end

	# rewrite each ODE: keep LHS D(x_i); new RHS = substitute(rhs)/2^{a_i}
	new_eqs = similar(eqs)
	for (i, eq) in enumerate(eqs)
		xi = _rescale_diff_state(eq)
		sc = (xi !== nothing && haskey(state_exp_by_val, xi)) ? state_exp_by_val[xi] : 1.0
		new_eqs[i] = eq.lhs ~ (Symbolics.substitute(eq.rhs, sub) / sc)
	end

	# rewrite measured quantities: y_k ~ substitute(rhs)/2^{o_k}
	new_measured = similar(pep.measured_quantities)
	for (i, mq) in enumerate(pep.measured_quantities)
		key = _rescale_obs_key(mq, pep.data_sample)
		os = get(info.observable_scales, key, 1.0)
		new_measured[i] = mq.lhs ~ (Symbolics.substitute(mq.rhs, sub) / os)
	end

	new_model, new_measured = create_ordered_ode_system(pep.name * "_rescaled", states, params, new_eqs, new_measured)

	# rescale data: copy "t"; RE-KEY each observable column by its NEW (scaled) rhs so
	# the downstream pipeline (which looks data up by the new measured-quantity rhs)
	# finds it, and divide by the observable scale.
	new_data = pep.data_sample
	if pep.data_sample isa ObservationData
		series = ObservationSeries[]
		new_expressions = Dict(_rescale_obs_key(omq, pep.data_sample) => nmq.rhs
			for (omq, nmq) in zip(pep.measured_quantities, new_measured))
		for s in pep.data_sample.series
			okey = s.expression
			haskey(new_expressions, okey) || throw(ArgumentError("Observation $(s.observable_id) is absent from measured_quantities"))
			os = get(info.observable_scales, okey, 1.0)
			push!(series, ObservationSeries(s.observable_id, s.experiment_id, new_expressions[okey],
				s.times, s.values ./ os;
				noise_std = isnothing(s.noise_std) ? nothing : s.noise_std ./ abs(os)))
		end
		new_data = ObservationData(series; initial_time=pep.data_sample.initial_time)
	elseif pep.data_sample !== nothing
		new_data = OrderedDict{Union{String, Num}, Vector{Float64}}()
		haskey(pep.data_sample, "t") && (new_data["t"] = pep.data_sample["t"])
		for (omq, nmq) in zip(pep.measured_quantities, new_measured)
			okey = _rescale_obs_key(omq, pep.data_sample)
			haskey(pep.data_sample, okey) || continue
			os = get(info.observable_scales, okey, 1.0)
			new_data[Num(nmq.rhs)] = pep.data_sample[okey] ./ os
		end
	end

	# rescale truth/IC (for round-tripping and benchmark scoring)
	new_p_true = OrderedDict{Num, Float64}()
	for (k, v) in pep.p_true
		new_p_true[k] = v / get(info.param_scales, k, 1.0)
	end
	new_ic = OrderedDict{Num, Float64}()
	for (k, v) in pep.ic
		new_ic[k] = v / get(info.state_scales, k, 1.0)
	end

	new_pep = ParameterEstimationProblem(
		pep.name * "_rescaled",
		new_model,
		new_measured,
		new_data,
		pep.recommended_time_interval,
		pep.solver,
		new_p_true,
		new_ic,
		pep.unident_count,
	)
	return new_pep, info
end

"""
	unrescale_results(solved_res, info::ScaleInfo)

Un-rescale a vector of ParameterEstimationResult IN PLACE: `parameters[p] *=
2^{b_j}`, `states[x] *= 2^{a_i}`. `.err` is LEFT in scaled-observable units (it is
a collapsed multi-observable sum and not invertible by a single factor); a
`:rescaled` provenance note is stamped. `.solution` is left in scaled coordinates
(documented limitation; consumers read `.parameters`/`.states`).
"""
function unrescale_results(solved_res, info::ScaleInfo)
	solved_res === nothing && return solved_res
	for r in solved_res
		r === nothing && continue
		# idempotent: skip results already un-rescaled (returned cluster reps alias
		# the solved_res objects, so this guards against a double-scale).
		if hasproperty(r, :provenance) && r.provenance !== nothing && hasproperty(r.provenance, :notes) && (:rescaled in r.provenance.notes)
			continue
		end
		if hasproperty(r, :parameters) && r.parameters !== nothing
			for (p, sc) in info.param_scales
				haskey(r.parameters, p) && (r.parameters[p] *= sc)
			end
		end
		if hasproperty(r, :states) && r.states !== nothing
			for (s, sc) in info.state_scales
				haskey(r.states, s) && (r.states[s] *= sc)
			end
		end
		if hasproperty(r, :provenance) && r.provenance !== nothing && hasproperty(r.provenance, :notes)
			(:rescaled in r.provenance.notes) || push!(r.provenance.notes, :rescaled)
		end
	end
	return solved_res
end

"""
	rescale_option_bounds(opts, info::ScaleInfo, pep) -> EstimationOptions

Transform user-supplied optimization bounds into the SCALED coordinate system so a
physical bound keeps its physical meaning under rescaling. `opt_lb`/`opt_ub` are
length `n_states + n_params`, ordered `[states; params]` (the convention every
consumer uses: polish context, backsolve clamp, ranking saturation). A physical
bound `lb ≤ v ≤ ub` with `v = 2^e v_scaled` becomes `lb/2^e ≤ v_scaled ≤ ub/2^e`,
so each entry is divided by its variable's power-of-2 scale. Returns `opts`
unchanged when no bounds are set; a length MISMATCH throws (fail-fast — the old
silent pass-through shipped a physical bound untransformed into a
scaled-coordinate solve; see `_assert_bounds_length`).
"""
function rescale_option_bounds(opts, info::ScaleInfo, pep::ParameterEstimationProblem)
	(isnothing(opts.opt_lb) && isnothing(opts.opt_ub)) && return opts
	system = pep.model.system
	states = Num.(ModelingToolkit.unknowns(system))
	params = Num.(ModelingToolkit.parameters(system))
	scales = vcat(
		Float64[get(info.state_scales, s, 1.0) for s in states],
		Float64[get(info.param_scales, p, 1.0) for p in params],
	)
	_assert_bounds_length(opts, length(scales), "rescale_option_bounds (auto_rescale)",
		"[states; params] of the model being rescaled")
	_rs(v) = v === nothing ? v : (Float64.(v) ./ scales)
	new_lb = _rs(opts.opt_lb)
	new_ub = _rs(opts.opt_ub)
	return EstimationOptions(;
		(name => getfield(opts, name) for name in fieldnames(EstimationOptions) if name !== :opt_lb && name !== :opt_ub)...,
		opt_lb = new_lb,
		opt_ub = new_ub,
	)
end
