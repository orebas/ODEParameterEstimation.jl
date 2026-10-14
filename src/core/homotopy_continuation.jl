# ── HC.jl threading ───────────────────────────────────────────────────────────
# `HomotopyContinuation.solve` defaults `threading = Threads.nthreads() > 1`, so at
# JULIA_NUM_THREADS>1 it multi-threads its path tracker / mixed-cell (polyhedral)
# computation. HC.jl is thread-safe since PR #669 (path tracking reworked onto native
# Julia threading); the HC author (P. Breiding) confirms `threading=true` should always
# work and the old "some CPUs hang" docstring is stale.
# Validated 2026-06-05: biohydrogenation_3_1em8 ran end-to-end with threading=true in
# 1.83h vs 4.81h threading-off — 2.63× faster, BIT-IDENTICAL recovery, no deadlock. The
# earlier nondeterministic "Computing mixed cells" freeze (latent/receptor at threads=8)
# did NOT reproduce on dedicated boxes and is believed pre-#669 / a rare race. We
# therefore default HC's internal threading ON (the process already keeps all threads
# for the pure-Julia polish, which yields to GC safepoints).
# CAVEAT: that validation used bioh's SMALL mixed volume (~11/37). The large-mixed-volume
# case (receptor, mixed_volume 63577) + threading is not yet proven deadlock-free; if a
# hang recurs in HC's "Computing mixed cells", set this back to `false`.
# All HomotopyContinuation.solve calls in this module route through `_hc_solve`.
const HC_SOLVE_THREADING = Ref(true)
# Compile mode for HC systems, passed EXPLICITLY per solve. Historically set
# via `HomotopyContinuation.set_default_compile(:all)`, which mutates HC's
# process-global default for every other caller/task in the process — a
# composability defect (2026-07 review). Same Ref pattern as
# HC_SOLVE_THREADING; promoting both to EstimationOptions fields is tracked
# follow-up work.
const HC_COMPILE_MODE = Ref{Symbol}(:all)
# Per-analysis overrides come from the scoped RunContext (set from
# EstimationOptions.hc_threading / hc_compile_mode at analyze entry); the Refs
# above are the fallback for bare solver calls outside any bound context.
_hc_solve(args...; kwargs...) = HomotopyContinuation.solve(args...;
	threading = something(_run_ctx_hc_threading(), HC_SOLVE_THREADING[]),
	compile = something(_run_ctx_hc_compile_mode(), HC_COMPILE_MODE[]),
	kwargs...)

"""
	solve_with_nlopt(poly_system, varlist;
					start_point=nothing,
					optimizer=BFGS(),
					polish_only=false,
					options=Dict())

Solves a polynomial system using traditional nonlinear optimization methods.
Can be used either as a standalone solver or to polish solutions from other methods.

# Arguments
- `poly_system`: System of polynomial equations to solve
- `varlist`: List of variables in the system
- `start_point`: Optional starting point. If not provided, random initialization is used
- `optimizer`: The optimization algorithm to use (default: BFGS)
- `polish_only`: If true, assumes start_point is close to solution and uses more local methods
- `options`: Dictionary of additional options for the optimizer

# Returns
- `solutions`: Array of solutions found
- `hc_variables`: Variables in HomotopyContinuation format
"""
# Shared by the NLLS solvers below (Phase D2 dedup — was copied 4×): compile the
# system via build_function, falling back to per-call substitute/value (orders of
# magnitude slower — the @warn matters). Returns (residual!, eval_count, is_compiled).
function _nlls_residual_closure(prepared_system, mangled_varlist, label::AbstractString)
	compiled! = nothing
	try
		_f_oop, _f_ip = Symbolics.build_function(prepared_system, mangled_varlist;
			expression = Val(false))
		compiled! = (res, u, p) -> (_f_ip(res, u); nothing)
	catch err
		_rethrow_if_interrupt(err)
		@warn "build_function failed in $label; falling back to substitute/value" err
	end
	eval_count = Ref(0)
	residual! = (res, u, p) -> begin
		eval_count[] += 1
		if compiled! !== nothing
			compiled!(res, u, p)
		else
			d = Dict{Num, eltype(u)}(zip(mangled_varlist, u))
			for (i, eq) in enumerate(prepared_system)
				res[i] = convert(eltype(u), Symbolics.value(Symbolics.substitute(eq, d)))
			end
		end
		return nothing
	end
	return residual!, eval_count, compiled! !== nothing
end

# Shared 5-frame solver-exception report (Phase D2 dedup — was copied 4×).
function _log_solver_exception(label::AbstractString, e, bt)
	@error "$label failed" exception = (e, bt)
	println("SOLVER_ERROR: $label threw exception:")
	println("  Type: ", typeof(e))
	println("  Message: ", e)
	st = stacktrace(bt)
	for (i, frame) in enumerate(st[1:min(5, length(st))])
		println("  [$i] ", frame)
	end
end

function solve_with_nlopt(poly_system, varlist;
	start_point = nothing,
	optimizer = NonlinearSolve.LevenbergMarquardt(),
	polish_only = false,
	options = Dict())

	# Check if we received Nemo polynomials instead of Symbolics expressions
	if !isempty(poly_system) && poly_system[1] isa Nemo.QQMPolyRingElem
		error("solve_with_nlopt received Nemo polynomials instead of Symbolics expressions. This suggests a conversion issue in SI.jl integration.")
	end

	# Prepare system for optimization
	prepared_system, mangled_varlist = (poly_system, varlist)

	# Debug controls
	debug = get(options, :debug, false)
	jac_mode = get(options, :jacobian, :none)

	# Compiled-or-fallback residual (shared helper; Phase D2 dedup)
	residual!, res_evals_nlopt, _ = _nlls_residual_closure(prepared_system, mangled_varlist, "solve_with_nlopt")

	# Set up optimization problem
	n = length(varlist)
	m = length(prepared_system)  # Number of equations
	x0 = if isnothing(start_point)
		randn(n)  # Random initialization if no start point provided
	else
		start_point
	end

	# Debug pre-solve
	if debug
		println("[NLOPT] equations=", m, " variables=", n)
		println("[NLOPT] optimizer=", typeof(optimizer))
		println("[NLOPT] jacobian_mode=", jac_mode)
		println("[NLOPT] eltype(x0)=", eltype(x0))
	end

	# Calculate initial residual
	initial_residual = zeros(m)
	residual!(initial_residual, x0, nothing)
	initial_norm = norm(initial_residual)

	# Create NonlinearLeastSquaresProblem
	prob = NonlinearLeastSquaresProblem(
		NonlinearFunction(residual!),
		x0,
		nothing,
	)

	# Set solver options based on polish_only
	solver_opts = if polish_only
		(abstol = 1e-12, reltol = 1e-12, maxiters = 1000)
	else
		(abstol = 1e-8, reltol = 1e-8, maxiters = 10000)
	end

	# Merge with user options (only recognized keywords)
	user_opts = Dict{Symbol, Any}()
	for (k, v) in options
		if k in (:abstol, :reltol, :maxiters)
			user_opts[k] = v
		end
	end
	solver_opts = merge(solver_opts, user_opts)

	# Solve the problem with exception handling
	sol = try
		NonlinearSolve.solve(prob, optimizer; solver_opts...)
	catch e
		_rethrow_if_interrupt(e)
		_log_solver_exception("solve_with_nlopt", e, catch_backtrace())
		return [], mangled_varlist, Dict(), mangled_varlist
	end

	# Check if solution is valid
	if SciMLBase.successful_retcode(sol)
		# Calculate final residual
		final_residual = zeros(m)
		residual!(final_residual, sol.u, nothing)
		final_norm = norm(final_residual)

		improvement = initial_norm - final_norm
		if debug
			if improvement > 0
				println("Optimization improved residual by $(improvement) (from $(initial_norm) to $(final_norm))")
			else
				println("Optimization did not improve residual (initial: $(initial_norm), final: $(final_norm))")
			end
		end

		# Return all four expected values: solutions, variables, trivial_dict, trimmed_varlist
		return [sol.u], mangled_varlist, Dict(), mangled_varlist
	else
		@warn "Optimization did not converge. RetCode: $(sol.retcode)"
		return [], mangled_varlist, Dict(), mangled_varlist
	end
end


"""
	solve_with_fast_nlopt(poly_system, varlist; kwargs...)

Fast NonlinearLeastSquares solver using compiled symbolic functions.
Uses Symbolics.build_function to compile the system into efficient Julia code.
Falls back to substitute/value method if compilation fails.
"""
function solve_with_fast_nlopt(poly_system, varlist;
	start_point = nothing,
	optimizer = NonlinearSolve.LevenbergMarquardt(),
	polish_only = false,
	options = Dict())


	# Prepare system for optimization
	prepared_system, mangled_varlist = (poly_system, varlist)
	m = length(prepared_system)
	n = length(mangled_varlist)
	debug = get(options, :debug, false)

	# Compiled-or-fallback residual (shared helper; Phase D2 dedup)
	residual!, eval_count, residual_is_compiled = _nlls_residual_closure(prepared_system, mangled_varlist, "solve_with_fast_nlopt")

	# Initial guess and initial residual norm
	x0 = isnothing(start_point) ? randn(n) : copy(start_point)
	initial_residual = zeros(m)
	residual!(initial_residual, x0, nothing)
	initial_norm = LinearAlgebra.norm(initial_residual)

	# Jacobian via ForwardDiff over the residual (Dual-safe)
	function jacobian!(J, u, p)
		g(u_) = begin
			r = Vector{eltype(u_)}(undef, m)
			residual!(r, u_, nothing)
			r
		end
		ForwardDiff.jacobian!(J, g, u)
		return nothing
	end

	nf = NonlinearFunction(residual!; resid_prototype = zeros(m), jac = jacobian!)
	prob = NonlinearLeastSquaresProblem(nf, x0, nothing)

	# Solver options (filter to recognized keywords)
	solver_opts = if polish_only
		(abstol = 1e-10, reltol = 1e-10, maxiters = 2000)
	else
		(abstol = 1e-8, reltol = 1e-8, maxiters = 10000)
	end
	user_opts = Dict{Symbol, Any}()
	for (k, v) in options
		if k in (:abstol, :reltol, :maxiters)
			user_opts[k] = v
		end
	end
	solver_opts = merge(solver_opts, user_opts)

	# Solve (measure wall time)
	solve_ms = 0.0
	sol = try
		local t0 = time()
		local out = NonlinearSolve.solve(prob, optimizer; solver_opts...)
		solve_ms = (time() - t0) * 1000
		out
	catch e
		_rethrow_if_interrupt(e)
		_log_solver_exception("solve_with_fast_nlopt", e, catch_backtrace())
		return [], mangled_varlist, Dict(), mangled_varlist
	end

	# Retry on MaxIters with a robust polyalgorithm
	if (!SciMLBase.successful_retcode(sol)) && (sol.retcode == SciMLBase.ReturnCode.MaxIters)
		@info "Fast NLLS hit MaxIters; retrying with FastShortcutNLLSPolyalg()"
		retry_opts = (abstol = 1e-10, reltol = 1e-10, maxiters = 5000)
		for (k, v) in options
			if k in (:abstol, :reltol, :maxiters)
				retry_opts = merge(retry_opts, (k => v,))
			end
		end
		try
			local t0 = time()
			sol = NonlinearSolve.solve(prob, NonlinearSolve.FastShortcutNLLSPolyalg(); retry_opts...)
			solve_ms += (time() - t0) * 1000
		catch e
			_rethrow_if_interrupt(e)
			_log_solver_exception("solve_with_fast_nlopt retry", e, catch_backtrace())
		end
	end

	# Compute final residual norm regardless of convergence
	final_residual = zeros(m)
	try
		residual!(final_residual, sol.u, nothing)
	catch e
		_rethrow_if_interrupt(e)
		@debug "Final residual evaluation failed, using initial residual" exception = e
		final_residual .= initial_residual
	end
	final_norm = LinearAlgebra.norm(final_residual)
	improvement = initial_norm - final_norm
	debug && println("[NLOPT_fast] residual_norm initial=", initial_norm,
		" final=", final_norm,
		" improvement=", improvement,
		" (compiled=", residual_is_compiled,
		" solve_ms=", solve_ms,
		" evals=", eval_count[], ")")

	# If solver failed but improved significantly, accept the improvement as a polished point
	if (!SciMLBase.successful_retcode(sol)) && (improvement > 0)
		@warn "Optimization did not fully converge (RetCode=$(sol.retcode)) but improved residual by $(improvement). Returning best-so-far iterate."
		return [sol.u], mangled_varlist, Dict(), mangled_varlist
	end

	if SciMLBase.successful_retcode(sol)
		return [sol.u], mangled_varlist, Dict(), mangled_varlist
	else
		@warn "Optimization did not converge. RetCode: $(sol.retcode)"
		return [], mangled_varlist, Dict(), mangled_varlist
	end
end


function sanitize_vars(varlist)
	var_names = string.(varlist)
	sanitize = name -> begin
		s = replace(name, r"[^A-Za-z0-9_]" => "_")
		startswith(s, r"[0-9]") ? "v_" * s : s
	end
	sanitized = sanitize.(var_names)
	# Injective-on-collision: the char-replacement above is lossy (xˍt and x_t
	# both become x_t), and a collision silently IDENTIFIES two variables in
	# the HC system — duplicate names pass HC's checks, one slot goes dead, and
	# positionally-returned values are garbage for the first variable (traced
	# 2026-08-12 audit). First occurrence keeps the clean name; later ones get
	# __2, __3, …
	seen = Dict{String, Int}()
	for i in eachindex(sanitized)
		s = sanitized[i]
		n = get(seen, s, 0) + 1
		seen[s] = n
		if n > 1
			candidate = string(s, "__", n)
			while haskey(seen, candidate)
				n += 1
				candidate = string(s, "__", n)
			end
			seen[candidate] = 1
			sanitized[i] = candidate
		end
	end
	return sanitized
end


"""
	convert_to_hc_format(poly_system, varlist)

Convert a polynomial system (as Symbolics expressions) to a HomotopyContinuation.System
along with its variable list. This uses a lightweight string replacement strategy
via `hmcs(::String)` to construct ModelKit variables deterministically.
"""
function convert_to_hc_format(poly_system, varlist)
	# Convert expressions to strings and replace variable names with hmcs("name")
	string_target = string.(poly_system)

	# Sanitize variable names to be HC-friendly (alphanumeric and underscores);
	# sanitize_vars is injective per call, but fail fast if that ever regresses
	# (a duplicate HC name = silent wrong values, not an HC error).
	sanitized = sanitize_vars(varlist)
	allunique(sanitized) || throw(ArgumentError(
		"convert_to_hc_format: sanitized HC names are not unique: $(sanitized)"))

	# Build mapping from original variable string to hmcs("sanitized_name") placeholder
	variable_string_mapping = Dict{String, String}()
	for (i, v) in enumerate(varlist)
		orig_name = string(v)
		sanitized_name = sanitized[i]
		variable_string_mapping[orig_name] = "hmcs(\"" * sanitized_name * "\")"
	end

	# Apply textual replacement so parsed expressions call hmcs(...) for variables.
	# Two-pass replacement: first replace variable names with unique placeholders (longest first
	# to avoid substring collisions like "x2_0" matching inside "x2_0_pt2"), then replace
	# placeholders with hmcs("...") calls.
	sorted_keys = sort(collect(keys(variable_string_mapping)); by = length, rev = true)
	placeholders = Dict{String, String}()
	for (idx, k) in enumerate(sorted_keys)
		placeholders[k] = "\x01HCVAR$(idx)\x01"
	end
	for i in eachindex(string_target)
		local s = string_target[i]
		for k in sorted_keys
			s = replace(s, k => placeholders[k])
		end
		for k in sorted_keys
			s = replace(s, placeholders[k] => variable_string_mapping[k])
		end
		string_target[i] = s
	end

	# Parse and eval into HC expressions; hmcs returns ModelKit.Variable
	parsed = eval.(Meta.parse.(string_target))
	# (compile mode is passed per solve via _hc_solve; no global mutation)

	# Build variables list in the same order as varlist for consistent output
	hc_variables = [HomotopyContinuation.ModelKit.Variable(Symbol(sanitized[i])) for i in eachindex(varlist)]

	# Construct the system (variables are provided to preserve ordering)
	hc_system = HomotopyContinuation.System(parsed, variables = hc_variables)

	return hc_system, hc_variables
end


"""
	solve_with_hc(poly_system, varlist; options=Dict())

Solve a square polynomial system using HomotopyContinuation.jl. Returns the same
tuple layout as other solvers: (solutions, hcvarlist, trivial_dict, trimmed_varlist).
Solutions are vectors of Float64 in the order of `varlist`.
"""
function solve_with_hc(poly_system, varlist; options = Dict())
	try
		# Convert to HC format
		hc_system, hc_variables = convert_to_hc_format(poly_system, varlist)

		if get(options, :debug, false)
			println("[HC] Solving system with $(length(poly_system)) equations and $(length(varlist)) variables")
			println("[HC] System to be solved:")
			println(hc_system)
		end

		# Solve (prefer real solutions first). HC.jl's `solutions` defaults to
		# `only_nonsingular=true`; ODEPE should keep singular algebraic roots too,
		# then let downstream residual/bounds/ranking filters decide usability.
		res = _hc_solve(hc_system, show_progress = false)
		real_tol = get(options, :real_tol, _OPT_STRUCT_DEFAULTS.hc_real_tol)
		sols = HomotopyContinuation.solutions(
			res;
			only_nonsingular = false,
			only_real = true,
			real_atol = real_tol,
		)

		# Map solutions to plain Float64 vectors in the same order as varlist
		solutions = Vector{Vector{Float64}}()
		for s in sols
			vals = Float64[real(s[j]) for j in 1:length(hc_variables)]
			push!(solutions, vals)
		end

		return solutions, varlist, Dict(), varlist
	catch e
		_rethrow_if_interrupt(e)
		@error "solve_with_hc failed" exception=(e, catch_backtrace())
		println("SOLVER_ERROR: HomotopyContinuation.solve threw exception:")
		println("  Type: ", typeof(e))
		println("  Message: ", e)
		# Print abbreviated stacktrace (first 5 frames)
		bt = catch_backtrace()
		st = stacktrace(bt)
		for (i, frame) in enumerate(st[1:min(5, length(st))])
			println("  [$i] ", frame)
		end
		return [], varlist, Dict(), varlist
	end
end


"""
	save_poly_system(filepath, poly_system, varlist;
					 param_map=nothing, pep=nothing, metadata=nothing)

Saves a polynomial system and its variables to a file as Julia code.

# Arguments
- `filepath`: Path to save the file.
- `poly_system`: The polynomial system (an array of symbolic expressions).
- `varlist`: The list of symbolic variables.
- `param_map` (optional): A dictionary mapping parameter names to their estimated values.
- `pep` (optional): The `ParameterEstimationProblem` object.
- `metadata` (optional): A dictionary for any other metadata to save.
"""
function save_poly_system(filepath, poly_system, varlist;
	param_map = nothing, pep = nothing, metadata = nothing)
	open(filepath, "w") do f
		# Write header
		write(f, "# Polynomial system saved on $(now())\n")
		if !isnothing(pep)
			write(f, "# Original problem: $(pep.name)\n")
		end
		write(f, "using Symbolics\n")
		write(f, "using StaticArrays\n\n")

		# Write metadata
		if !isnothing(metadata)
			write(f, "# Metadata\n")
			for (key, value) in metadata
				write(f, "# $key: $value\n")
			end
			write(f, "\n")
		end

		# Write varlist
		write(f, "# Variables\n")
		write(f, "varlist_str = \"\"\"\n$(join(varlist, "\n"))\n\"\"\"\n")
		write(f, "@variables ")
		for (i, var) in enumerate(varlist)
			write(f, string(var))
			if i < length(varlist)
				write(f, " ")
			end
		end
		write(f, "\n")
		vars_str = join(string.(varlist), ", ")
		write(f, "varlist = [" * vars_str * "]\n\n")


		# Write polynomial system
		write(f, "# Polynomial System\n")
		write(f, "poly_system = [\n")
		for (i, poly) in enumerate(poly_system)
			# Use repr to get a string representation that can be parsed back
			poly_str = repr(poly)
			write(f, "    $poly_str")
			if i < length(poly_system)
				write(f, ",\n")
			end
		end
		write(f, "\n]\n\n")

		# Write param_map if available
		if !isnothing(param_map)
			write(f, "# Parameter Map\n")
			write(f, "param_map = Dict(\n")
			for (param, val) in param_map
				write(f, "    Symbol(\"$param\") => $val,\n")
			end
			write(f, ")\n\n")
		end

		# Write PEP if available
		if !isnothing(pep)
			# This is a bit tricky, we might just save the name and key fields
			write(f, "# Original Problem Info\n")
			write(f, "problem_name = \"$(pep.name)\"\n")
			# You might need to reconstruct the PEP object manually if needed
		end
	end
end


# ============================================================================
# Parameter Homotopy Functions
# ============================================================================

"""
	convert_to_hc_format_with_params(poly_system, solve_vars, data_vars)

Convert a polynomial system to HomotopyContinuation.System with both variables and parameters.

# Arguments
- `poly_system`: System of polynomial equations (Symbolics expressions)
- `solve_vars`: Variables to solve for (unknowns - parameters and states)
- `data_vars`: Variables that become HC parameters (interpolated observable values)

# Returns
- `hc_system`: HomotopyContinuation.System with parameters
- `hc_variables`: HC variable list (corresponding to solve_vars)
- `hc_params`: HC parameter list (corresponding to data_vars)
"""
function convert_to_hc_format_with_params(poly_system, solve_vars, data_vars)
	# Convert expressions to strings
	string_target = string.(poly_system)

	# Sanitize variable names (both solve_vars and data_vars)
	sanitized_solve = sanitize_vars(solve_vars)
	sanitized_data = sanitize_vars(data_vars)

	# Data vars carry a p_ prefix in HC — which can collide ACROSS lists with a
	# solve variable literally named p_<something> (e.g. solve var `p_y` vs data
	# var `y`; 2026-08-12 audit finding). Deduplicate the prefixed data names
	# against the solve-name set, then assert global uniqueness.
	prefixed_data = ["p_" * s for s in sanitized_data]
	taken = Set{String}(sanitized_solve)
	for i in eachindex(prefixed_data)
		name = prefixed_data[i]
		if name in taken
			k = 2
			while string(name, "__", k) in taken
				k += 1
			end
			name = string(name, "__", k)
		end
		push!(taken, name)
		prefixed_data[i] = name
	end
	all_hc_names = vcat(sanitized_solve, prefixed_data)
	allunique(all_hc_names) || throw(ArgumentError(
		"convert_to_hc_format_with_params: sanitized HC names are not unique after dedup: $(all_hc_names)"))

	# Build mapping from original variable string to hmcs("sanitized_name") placeholder
	variable_string_mapping = Dict{String, String}()

	# Map solve variables
	for (i, v) in enumerate(solve_vars)
		orig_name = string(v)
		sanitized_name = sanitized_solve[i]
		variable_string_mapping[orig_name] = "hmcs(\"" * sanitized_name * "\")"
	end

	# Map data variables (parameters in HC; prefixed + deduped names)
	for (i, v) in enumerate(data_vars)
		orig_name = string(v)
		variable_string_mapping[orig_name] = "hmcs(\"" * prefixed_data[i] * "\")"
	end

	# Apply textual replacement (two-pass, longest-first to avoid substring collisions)
	sorted_keys = sort(collect(keys(variable_string_mapping)); by = length, rev = true)
	placeholders = Dict{String, String}()
	for (idx, k) in enumerate(sorted_keys)
		placeholders[k] = "\x01HCVAR$(idx)\x01"
	end
	for i in eachindex(string_target)
		local s = string_target[i]
		for k in sorted_keys
			s = replace(s, k => placeholders[k])
		end
		for k in sorted_keys
			s = replace(s, placeholders[k] => variable_string_mapping[k])
		end
		string_target[i] = s
	end

	# Parse and eval into HC expressions
	parsed = eval.(Meta.parse.(string_target))
	# (compile mode is passed per solve via _hc_solve; no global mutation)

	# Build variables list in the same order as solve_vars
	hc_variables = HomotopyContinuation.ModelKit.Variable[
		HomotopyContinuation.ModelKit.Variable(Symbol(sanitized_solve[i])) for i in eachindex(solve_vars)
	]

	# Build parameters list in the same order as data_vars (prefixed + deduped —
	# MUST match the names used in variable_string_mapping above)
	hc_params = HomotopyContinuation.ModelKit.Variable[
		HomotopyContinuation.ModelKit.Variable(Symbol(prefixed_data[i])) for i in eachindex(data_vars)
	]

	# Construct the parameterized system
	hc_system = HomotopyContinuation.System(parsed, variables = hc_variables, parameters = hc_params)

	return hc_system, hc_variables, hc_params
end

"""
	compute_column_scales(solve_vars, data_vars, param_values_list)

Per-variable column-scale vector aligned 1:1 with `solve_vars` (== HC variable order), for the
data-driven column scaling of the parameterized HC solve.

Rule (no truth needed):
  order_mag[k] = max over all shooting points and all data_vars of derivative order k of |value|
  scale(var)   = 1.0                   if order(var) == 0   (params + order-0 state ICs untouched)
               = max(order_mag[k], 1)  if order(var) == k >= 1
               = 1.0                   if no finite data of that order (fallback)

`order(data_var)` is parsed from its string form (`Differential(t, k)(...)`; order 0 = plain `y..(t)`).
`order(solve_var)` is parsed via `parse_derivative_variable_name` (trailing `_N` suffix).
"""
function compute_column_scales(solve_vars, data_vars, param_values_list)
	n = length(solve_vars)
	isempty(param_values_list) && return ones(Float64, n)

	# derivative order of each data_var. Single-point Symbolics names are "Differential(t, N)(...)";
	# MULTIPOINT/SIAN names are "y1_N" / "y1_N_ptK" (order 0 => plain "y1(t)" or "y1_0"). Try the
	# Symbolics form first (keeps the single-point path byte-exact), else fall back to the SIAN/multipoint
	# parser, which strips any _ptK suffix. (Was: regex only => silently order-0 => all scales 1.0 on the
	# multipoint path, i.e. column scaling was an inert no-op whenever use_multipoint=true.)
	data_orders = Vector{Int}(undef, length(data_vars))
	for (j, dv) in enumerate(data_vars)
		s = string(dv)
		m = match(r"Differential\(t,\s*(\d+)\)", s)
		data_orders[j] = isnothing(m) ? _multipoint_deriv_order(s) : parse(Int, m.captures[1])
	end

	# order_mag[k] = max |value| over points, per order (finite values only)
	order_mag = Dict{Int, Float64}()
	for pv in param_values_list
		ncols = min(length(data_vars), length(pv))  # leading block aligns with data_vars
		for j in 1:ncols
			v = pv[j]
			isfinite(v) || continue
			k = data_orders[j]
			order_mag[k] = max(get(order_mag, k, 0.0), abs(Float64(v)))
		end
	end

	# map onto solve_vars by parsed order
	scales = ones(Float64, n)
	for (i, sv) in enumerate(solve_vars)
		# strips _ptK then parses the SIAN _N suffix (was parse_derivative_variable_name, which returned
		# nothing => order 0 on _ptK-suffixed multipoint solve vars — the same silent no-op as the data side)
		ord = _multipoint_deriv_order(string(sv))
		if ord != 0
			mag = get(order_mag, ord, NaN)
			scales[i] = (isfinite(mag) && mag > 0.0) ? max(mag, 1.0) : 1.0
		end
	end
	return scales
end

"""
	column_scalable(solve_vars)

Which unknowns column scaling can scale: the derivatives of order one and above. Parameters and order-0
states keep a scale of one whatever the data (the rule in `compute_column_scales`), so this depends on
the unknowns alone.
"""
column_scalable(solve_vars) = Bool[_multipoint_deriv_order(string(sv)) != 0 for sv in solve_vars]

"""
	scale_hc_system(hc_system, scaled_variables)

Return a System in which each of `scaled_variables` is replaced by s_i*v_i (a pure coordinate rescale;
Newton polytopes / mixed volume unchanged), where the scales s_i are new parameters listed after the
system's own. A solve passes the scale values after the data, `target_parameters = vcat(data, scales)`.

The scales are parameters, not numbers substituted into the expressions, so that the result is the same
system for every data set and every interpolator. HomotopyContinuation compiles each distinct system it
is given, and with data-derived numbers in the coefficients every call handed it a system it had not
seen before.
"""
function scale_hc_system(hc_system, scaled_variables)
	isempty(scaled_variables) && return hc_system
	taken = Set{String}(string.(vcat(hc_system.variables, hc_system.parameters)))
	scale_vars = HomotopyContinuation.ModelKit.Variable[]
	for i in eachindex(scaled_variables)
		name = "colscale_$(i)"
		while name in taken
			name *= "_"
		end
		push!(taken, name)
		push!(scale_vars, HomotopyContinuation.ModelKit.Variable(Symbol(name)))
	end
	rescaled = [scale_vars[i] * scaled_variables[i] for i in eachindex(scaled_variables)]
	scaled_exprs = HomotopyContinuation.ModelKit.subs(hc_system.expressions, scaled_variables => rescaled)
	return HomotopyContinuation.System(scaled_exprs, variables = hc_system.variables,
		parameters = vcat(hc_system.parameters, scale_vars))
end

"""
	_track_gamma_straight(hc_system, starts, p_start, p_target; show_progress, rng, max_seeds, target_count)

Track `starts` (solutions of `F(·;p_start)`) to the solutions of `F(·;p_target)` using HC.jl's
fixed-system `StraightLineHomotopy` WITH the γ-trick: `H(x,t) = γ·t·F(x;p_start) + (1−t)·F(x;p_target)`.
A random γ makes the system-space path generic so it generically misses the discriminant; `t=0` is
exactly `F(·;p_target)`, so it lands at the real target. Tries up to `max_seeds` random γ, keeping the
result with the most solutions; early-stops once `target_count` is reached. Returns the best HC result.
This is distinct from `ParameterHomotopy` (the straight real parameter path, no γ), which can cross the
real discriminant and stall (`terminated_max_steps`).
"""
function _track_gamma_straight(hc_system, starts, p_start, p_target;
	show_progress = false, rng = MersenneTwister(0x67a44a01), max_seeds = 5, target_count = length(starts))
	# Deterministic default rng (postcampaign review P1): all in-package callers
	# pass rng explicitly; an unseeded default made direct calls run-to-run random.
	ps = ComplexF64.(p_start)
	pt = ComplexF64.(p_target)
	best_res = nothing
	best_n = -1
	for _ in 1:max(1, max_seeds)
		γ = cis(2π * rand(rng))
		res = _hc_solve(hc_system, hc_system, starts;
			start_parameters = ps, target_parameters = pt, gamma = γ, show_progress = show_progress)
		n = length(HomotopyContinuation.solutions(res; only_nonsingular = false))
		if n > best_n
			best_n = n
			best_res = res
		end
		n >= target_count && break
	end
	return best_res
end

"""
	compute_generic_start_solutions(poly_system, solve_vars, data_vars; gamma_seed=0, show_progress=false, debug=false)

Solve the UNSCALED HC system ONCE at a generic complex p0. The system structure is interpolator-
independent (interpolators only change the data plugged in), so the expensive polyhedral solve can be
done once and its solution set fanned out (γ-straight) to every real shooting point across ALL
interpolators. Returns `(unscaled_solutions, p0)` or `(nothing, nothing)` on empty/throw. The p0 seed
is structural (from solve_vars) ⇒ reproducible and data-independent.
"""
function compute_generic_start_solutions(poly_system, solve_vars, data_vars; gamma_seed = 0, show_progress = false, debug = false)
	hc_system, hc_variables, hc_params = convert_to_hc_format_with_params(poly_system, solve_vars, data_vars)
	p0_rng = gamma_seed < 0 ? MersenneTwister() :
		MersenneTwister(gamma_seed == 0 ? hash(string.(solve_vars)) : UInt64(gamma_seed))
	p0 = randn(p0_rng, ComplexF64, length(hc_params))
	try
		gres = isempty(hc_params) ?
			_hc_solve(hc_system; show_progress = show_progress) :
			_hc_solve(hc_system; target_parameters = p0, show_progress = show_progress)
		sols0 = HomotopyContinuation.solutions(gres; only_nonsingular = false)
		isempty(sols0) && return (nothing, nothing)
		debug && println("[HC-PARAM] Generic-start (hoisted): UNSCALED generic p0 → N=$(length(sols0)) solutions (solved ONCE for all interpolators)")
		return (sols0, p0)
	catch e
		_rethrow_if_interrupt(e)
		debug && println("[HC-PARAM] Generic-start (hoisted): generic solve threw ($(e))")
		return (nothing, nothing)
	end
end

"""
	solve_with_hc_parameterized(poly_system, solve_vars, data_vars, param_values_list; options=Dict())

Solve a polynomial system at multiple parameter values using parameter homotopy.

This function provides significant speedup when solving the same polynomial structure
at multiple shooting points, as it tracks existing solutions instead of computing
fresh start systems for each point.

# Arguments
- `poly_system`: Symbolic equations (structure fixed, coefficients vary)
- `solve_vars`: Variables to solve for (parameters and states)
- `data_vars`: Variables that become HC parameters (interpolated observables like y1(t), y1'(t))
- `param_values_list`: Vector of parameter value vectors, one per shooting point

# Keyword Arguments
- `options::Dict`: Solver options (e.g., :show_progress, :real_tol)

# Returns
Vector of vectors of real solutions, one per shooting point.

# Algorithm
1. First point: Fresh solve with HC.solve(), collect ALL solutions (real + complex)
2. Subsequent points: Parameter homotopy tracking ALL solutions from previous point
3. At each point: Filter to real solutions for output
4. Track ALL solutions to next point (solutions can transition between real/complex)

# Fallback
If parameter homotopy tracking loses solutions, falls back to fresh solve at that point
and emits a warning.
"""
function solve_with_hc_parameterized(poly_system, solve_vars, data_vars, param_values_list; options = Dict(),
	precomputed_generic_solutions = nothing, precomputed_generic_params = nothing)
	# Convert to HC format with parameters
	hc_system, hc_variables, hc_params = convert_to_hc_format_with_params(
		poly_system, solve_vars, data_vars
	)

	# Get options
	# Fallbacks come from _OPT_STRUCT_DEFAULTS (single source of truth) so a
	# caller passing a partial options Dict gets the documented struct defaults,
	# not drifted literals (this cluster used to silently downgrade to
	# use_column_scaling=false / :gamma_straight for partial dicts).
	show_progress = get(options, :show_progress, _OPT_STRUCT_DEFAULTS.hc_show_progress)
	real_tol = get(options, :real_tol, _OPT_STRUCT_DEFAULTS.hc_real_tol)
	debug = get(options, :debug, false)
	use_column_scaling = get(options, :use_column_scaling, _OPT_STRUCT_DEFAULTS.use_column_scaling)
	tracking_mode = get(options, :homotopy_tracking_mode, _OPT_STRUCT_DEFAULTS.homotopy_tracking_mode)
	gamma_max_seeds = get(options, :gamma_max_seeds, _OPT_STRUCT_DEFAULTS.gamma_max_seeds)
	gamma_seed = get(options, :gamma_seed, _OPT_STRUCT_DEFAULTS.gamma_seed)
	# Deterministic by default: gamma_seed==0 derives a STABLE seed from the problem inputs (solve_vars +
	# data values), so runs are reproducible and A/B arms see the same γ stream, while different problems
	# still get different γ. A fixed γ sequence stays generic w.r.t. any one problem's discriminant
	# (measure-zero collision). gamma_seed>0 forces that exact seed; gamma_seed<0 ⇒ entropy (nondeterministic).
	# Separate RNGs for the generic-start p0 draw vs the per-point γ stream, so changing gamma_max_seeds or
	# whether p0 is drawn does NOT shift the γ sequence (keeps mode A/B comparisons controlled).
	if gamma_seed < 0
		p0_rng = MersenneTwister()
		gamma_rng = MersenneTwister()
	else
		base_seed = gamma_seed == 0 ? hash((string.(solve_vars), param_values_list)) : UInt64(gamma_seed)
		p0_rng = MersenneTwister(base_seed)
		gamma_rng = MersenneTwister(base_seed ⊻ 0x9e3779b97f4a7c15)
	end

	# Data-driven column scaling. Compute ONE scale vector for ALL points (aggregated over
	# param_values_list) so the parameter homotopy stays in a single coordinate system; scale the
	# system once here and unscale each solution at the extraction step below.
	# When off, col_scales == ones ⇒ the unscaled system is solved and the unscale step is 1.0*x.
	# A system without data parameters has no data to take scales from, so its scales stay at one.
	col_scales = use_column_scaling && !isempty(hc_params) ?
		compute_column_scales(solve_vars, data_vars, param_values_list) : ones(Float64, length(hc_variables))
	# Keep the UNSCALED system: data-derived column scales are tuned for the REAL shooting points and are
	# meaningless at the generic complex p0, where they only hurt conditioning (they made the anchor solve
	# under-count, 14 vs the true 18). So the :generic_start anchor solve below runs on hc_system_unscaled,
	# then maps its physical roots into scaled coords for the (well-conditioned) scaled fan-out.
	hc_system_unscaled = hc_system
	# The scaled system takes its scales as parameters, after the data (see `scale_hc_system`). Only
	# derivative unknowns are ever scaled, so only they get a scale parameter.
	# `with_scales` turns the data at a point into the parameters of whichever system is being solved.
	scalable = column_scalable(solve_vars)
	all(col_scales[i] == 1.0 || scalable[i] for i in eachindex(col_scales)) || throw(ArgumentError(
		"solve_with_hc_parameterized: column scaling gave a scale other than one to an unknown that is not a derivative"))
	scaled = any(!=(1.0), col_scales)
	scale_params = col_scales[scalable]
	if scaled
		hc_system = scale_hc_system(hc_system, hc_variables[scalable])
	end
	with_scales(p) = scaled ? vcat(p, scale_params) : p
	if debug && use_column_scaling
		println("[HC-PARAM] Column scaling ON: scale range $(extrema(col_scales)), nontrivial $(count(!=(1.0), col_scales))/$(length(col_scales))")
	end

	if isempty(param_values_list)
		return Vector{Vector{Vector{Float64}}}()
	end

	# HC.jl treats `target_parameters = []` differently from a parameter-free system and can throw a
	# BoundsError internally. For fixed systems, solve once and reuse the same roots at every point.
	if isempty(hc_params)
		debug && println("[HC-PARAM] No HC parameters; solving fixed system once for $(length(param_values_list)) point(s)")
		result = _hc_solve(hc_system; show_progress = show_progress)
		real_solutions_hc = HomotopyContinuation.solutions(result, only_real = true, real_tol = real_tol)
		if isempty(real_solutions_hc)
			real_solutions_hc = HomotopyContinuation.solutions(result)
			debug && println("[HC-PARAM] Fixed system: no real solutions, projecting $(length(real_solutions_hc)) complex solutions to real parts")
		end

		fixed_real_solutions = Vector{Vector{Float64}}()
		for s in real_solutions_hc
			vals = Float64[col_scales[j] * real(s[j]) for j in 1:length(hc_variables)]
			push!(fixed_real_solutions, vals)
		end
		return [copy.(fixed_real_solutions) for _ in param_values_list]
	end

	all_real_results = Vector{Vector{Vector{Float64}}}()
	prev_all_solutions = nothing  # Track ALL solutions (real + complex)
	prev_params = nothing
	initial_solution_count = 0

	# Generic-start ("ab-initio") seeding (opt-in, :generic_start). Solve ONCE at a generic COMPLEX
	# parameter point p0 — off the discriminant with probability 1 ⇒ the full generic root count N,
	# well-conditioned — then fan out below by tracking p0→p_i (γ-straight) to EVERY real shooting point.
	# Robust to a deficient point-1 real-data solve; makes the count target the true generic N (fixing the
	# initial_solution_count anchor). If the generic solve itself returns empty / throws (nasty-everywhere),
	# leave generic_start_solutions = nothing ⇒ degrade to the per-point fresh + γ-straight chain below.
	generic_start_solutions = nothing
	generic_start_params = precomputed_generic_params
	if tracking_mode == :generic_start && !isempty(param_values_list)
		# UNSCALED generic-complex solutions: use the precomputed set when supplied (hoisted ONCE over all
		# interpolators in optimized_multishot), else solve here. Either way map the physical roots into THIS
		# call's scaled coords (x̂ = x ⊘ scales) as valid starts for the scaled fan-out. col_scales==ones ⇒ no-op.
		sols0 = precomputed_generic_solutions
		if isnothing(sols0)
			generic_start_params = randn(p0_rng, ComplexF64, length(first(param_values_list)))
			try
				gres = _hc_solve(hc_system_unscaled; target_parameters = generic_start_params, show_progress = show_progress)
				sols0 = HomotopyContinuation.solutions(gres; only_nonsingular = false)
			catch e
				_rethrow_if_interrupt(e)
				debug && println("[HC-PARAM] Generic-start: generic solve threw ($(e)) → degrading to per-point fresh+track")
				sols0 = nothing
			end
		end
		if isnothing(sols0) || isempty(sols0)
			debug && println("[HC-PARAM] Generic-start: no generic solutions → degrading to per-point fresh+track")
		else
			generic_start_solutions = use_column_scaling ? [s ./ col_scales for s in sols0] : sols0
			debug && println("[HC-PARAM] Generic-start: N=$(length(generic_start_solutions)) generic solutions (scaled coords)$(isnothing(precomputed_generic_solutions) ? " [solved here]" : " [precomputed once]")")
		end
	end

	for (i, current_params) in enumerate(param_values_list)
		try
		if tracking_mode == :generic_start && !isnothing(generic_start_solutions)
			# Generic-start FAN-OUT: track the COMPLETE generic solution set p0→this real point. No chaining,
			# so a weak point cannot poison the others, and truth is in the seeded set wherever it survives.
			if i == 1
				initial_solution_count = length(generic_start_solutions)  # the true generic count N
			end
			debug && println("[HC-PARAM] Point $i: generic-start fan-out — tracking N=$(length(generic_start_solutions)) generic solutions to this real point")
			result = _track_gamma_straight(hc_system, generic_start_solutions, with_scales(generic_start_params), with_scales(current_params);
				show_progress = show_progress, rng = gamma_rng, max_seeds = gamma_max_seeds,
				target_count = length(generic_start_solutions))
			# Keep singular + complex FINITE solutions as candidates (matches the main solve path at ~687).
			# For the completeness DECISION, also count singular + at-infinity endpoints (only_finite=false):
			# a path that merely went singular or diverged is NOT a failed/lost path, so it must not trigger
			# a needless fresh solve. At-infinity stays OUT of the kept set (real(s) would be Inf downstream).
			all_solutions = HomotopyContinuation.solutions(result; only_nonsingular = false)
			n_accounted = length(HomotopyContinuation.solutions(result; only_nonsingular = false, only_finite = false))
			if n_accounted < initial_solution_count || isempty(all_solutions)
				if debug
					reason = n_accounted < initial_solution_count ?
						"genuinely short ($n_accounted accounted < $initial_solution_count; finite-kept=$(length(all_solutions)))" :
						"no finite endpoints ($n_accounted accounted; finite-kept=0)"
					println("[HC-PARAM] Point $i: fan-out $reason → fresh solve")
				end
				result = _hc_solve(hc_system; target_parameters = with_scales(current_params), show_progress = show_progress)
				all_solutions = HomotopyContinuation.solutions(result; only_nonsingular = false)
				# Completeness target must stay a SOLUTION count (≈ the generic count N), NOT the fresh
				# solve's only_finite=false endpoint count. The fan-out only ever tracks N generic
				# solutions, so n_accounted ≤ N; bumping the target to the fresh solve's path/endpoint
				# count (the homotopy path bound ~Bezout/BKK incl. at-infinity — e.g. 393) made
				# `n_accounted < target` permanently true ⇒ EVERY subsequent point needlessly fresh-solved
				# (the 15h-cstr blowup; report 2026-06-10). Use the FRESH FINITE count: it raises the
				# target only on genuine generic-anchor undercoverage (fresh finds > N real solutions),
				# the one case where fan-outs of N really are incomplete and should keep fresh-solving.
				fresh_finite = length(all_solutions)
				if fresh_finite > initial_solution_count
					@warn "[HC-PARAM] Fresh solve found more finite solutions ($fresh_finite) than the generic-start anchor count ($initial_solution_count): generic anchor undercovered; subsequent fan-outs will fresh-solve." maxlog = 3
				end
				initial_solution_count = max(initial_solution_count, fresh_finite)
			elseif debug
				real_count = length(HomotopyContinuation.solutions(result, only_real = true, real_tol = real_tol))
				println("[HC-PARAM] Point $i: fan-out complete ($n_accounted accounted, $(length(all_solutions)) finite, $real_count real)")
			end
		elseif i == 1 || isnothing(prev_all_solutions) || isempty(prev_all_solutions)
			# Fresh solve at first point - get ALL solutions
			if debug
				println("[HC-PARAM] Point $i: Fresh solve with $(length(current_params)) parameters")
			end

			result = _hc_solve(hc_system;
				target_parameters = with_scales(current_params),
				show_progress = show_progress)

			all_solutions = HomotopyContinuation.solutions(result; only_nonsingular = false)  # ALL, not just real
			initial_solution_count = length(all_solutions)

			if debug
				real_count = length(HomotopyContinuation.solutions(result, only_real = true, real_tol = real_tol))
				println("[HC-PARAM] Point $i: Fresh solve found $(length(all_solutions)) total solutions ($real_count real)")
			end
		else
			# Track ALL solutions from the previous point to this one.
			if tracking_mode == :gamma_straight || tracking_mode == :generic_start
				if debug
					println("[HC-PARAM] Point $i: γ-straight tracking $(length(prev_all_solutions)) solutions")
				end
				result = _track_gamma_straight(hc_system, prev_all_solutions, with_scales(prev_params), with_scales(current_params);
					show_progress = show_progress, rng = gamma_rng, max_seeds = gamma_max_seeds,
					target_count = initial_solution_count)
				all_solutions = HomotopyContinuation.solutions(result; only_nonsingular = false)
			else
				# :parameter (also the first leg of :gamma_straight_fallback): straight parameter homotopy
				if debug
					println("[HC-PARAM] Point $i: Parameter homotopy tracking $(length(prev_all_solutions)) solutions")
				end
				result = _hc_solve(hc_system, prev_all_solutions;
					start_parameters = with_scales(prev_params),
					target_parameters = with_scales(current_params),
					show_progress = show_progress)
				all_solutions = HomotopyContinuation.solutions(result; only_nonsingular = false)
			end

			# :gamma_straight_fallback — if the parameter path lost paths, try γ-straight before a fresh solve.
			if tracking_mode == :gamma_straight_fallback && length(all_solutions) < initial_solution_count
				if debug
					println("[HC-PARAM] Point $i: parameter path lost paths ($(length(all_solutions)) < $initial_solution_count) → γ-straight")
				end
				result = _track_gamma_straight(hc_system, prev_all_solutions, with_scales(prev_params), with_scales(current_params);
					show_progress = show_progress, rng = gamma_rng, max_seeds = gamma_max_seeds,
					target_count = initial_solution_count)
				all_solutions = HomotopyContinuation.solutions(result; only_nonsingular = false)
			end

			# Fresh solve fallback (last resort, all modes) if still below the initial count.
			if length(all_solutions) < initial_solution_count
				if debug
					println("[HC-PARAM] Point $i: still short ($(length(all_solutions)) < $initial_solution_count). Fresh solve.")
				end
				result = _hc_solve(hc_system;
					target_parameters = with_scales(current_params),
					show_progress = show_progress)
				all_solutions = HomotopyContinuation.solutions(result; only_nonsingular = false)
				initial_solution_count = max(initial_solution_count, length(all_solutions))
				if debug
					println("[HC-PARAM] Point $i: Fresh solve found $(length(all_solutions)) solutions")
				end
			elseif debug
				real_count = length(HomotopyContinuation.solutions(result, only_real = true, real_tol = real_tol))
				println("[HC-PARAM] Point $i: tracked $(length(all_solutions)) solutions ($real_count real)")
			end
		end

		# Filter for REAL solutions at this point (for output)
		real_solutions_hc = HomotopyContinuation.solutions(result, only_real = true, real_tol = real_tol)

		# Fallback: if no real solutions, project ALL solutions to real parts
		# (mirrors solve_with_hc behavior — projected points often polish to genuine solutions)
		if isempty(real_solutions_hc)
			# Explicit kwargs (== installed HC.jl defaults, verified result.jl): keep
			# singular roots, exclude at-infinity endpoints. Pinned here so an upstream
			# default change can't silently alter this fallback.
			real_solutions_hc = HomotopyContinuation.solutions(result; only_nonsingular = false, only_finite = true)
			if debug
				println("[HC-PARAM] Point $i: No real solutions, projecting $(length(real_solutions_hc)) complex solutions to real parts")
			end
		end

		# Convert to Float64 vectors (unscale column-scaled coords; col_scales is ones when off ⇒ no-op)
		real_solutions = Vector{Vector{Float64}}()
		for s in real_solutions_hc
			vals = Float64[col_scales[j] * real(s[j]) for j in 1:length(hc_variables)]
			# Defensive: never emit a non-finite "solution" downstream.
			all(isfinite, vals) || continue
			push!(real_solutions, vals)
		end
		push!(all_real_results, real_solutions)

		# Track ALL solutions to next point (real + complex)
		prev_all_solutions = all_solutions
		prev_params = current_params

		catch e
			_rethrow_if_interrupt(e)
			@error "[HC-PARAM] Point $i failed" exception=(e, catch_backtrace())
			println(stderr, "[HC-CRASH] Point $i threw $(typeof(e)): $e")
			println(stderr, "[HC-CRASH] current_params = $current_params")
			if any(isnan, current_params) || any(isinf, current_params)
				println(stderr, "[HC-CRASH] WARNING: current_params contains NaN/Inf!")
			end
			# Push empty results for this point
			push!(all_real_results, Vector{Vector{Float64}}())
			# Reset tracking so next point does a fresh solve
			prev_all_solutions = nothing
			prev_params = nothing
		end
	end

	return all_real_results
end

"""
	extract_data_variables_from_DD(DD::DerivativeData)

Extract all observable derivative variables from the DerivativeData structure.
These are the variables that should become HC parameters in parameter homotopy.

# Returns
Vector of Symbolics variables representing y_i^(j) for all observables and derivative levels.
"""
function extract_data_variables_from_DD(DD)
	data_vars = Vector{Any}()

	if isnothing(DD)
		return data_vars
	end

	# DD.obs_lhs[level+1][obs_idx] gives the variable for derivative level of observable obs_idx
	for level_vars in DD.obs_lhs
		for v in level_vars
			push!(data_vars, v)
		end
	end

	return data_vars
end

"""
	evaluate_data_vars_at_point(interpolants, data_vars, DD, measured_quantities, t_point)

Evaluate all data variables (observable derivatives) at a specific time point.

# Arguments
- `interpolants`: Dict mapping observable RHS to interpolation functions
- `data_vars`: Vector of data variables (from extract_data_variables_from_DD)
- `DD`: DerivativeData structure containing obs_lhs mapping
- `measured_quantities`: Vector of measured quantity equations
- `t_point`: Time point at which to evaluate

# Returns
Vector of Float64 values corresponding to data_vars order.
"""
function evaluate_data_vars_at_point(interpolants, data_vars, DD, measured_quantities, t_point)
	values = Vector{Float64}()

	# Build a mapping from data_var -> (obs_idx, deriv_level)
	var_to_obs = Dict{Any, Tuple{Int, Int}}()
	for (level_idx, level_vars) in enumerate(DD.obs_lhs)
		deriv_level = level_idx - 1  # 0-indexed derivative level
		for (obs_idx, v) in enumerate(level_vars)
			var_to_obs[v] = (obs_idx, deriv_level)
		end
	end

	for v in data_vars
		trfn_val = evaluate_known_trfn_variable(string(v), Float64(t_point))
		if !isnothing(trfn_val)
			push!(values, Float64(trfn_val))
			continue
		end

		if haskey(var_to_obs, v)
			obs_idx, deriv_level = var_to_obs[v]

			# Get the interpolant for this observable
			obs_rhs = Symbolics.diff2term(measured_quantities[obs_idx].rhs)

			if haskey(interpolants, obs_rhs)
				interp_func = interpolants[obs_rhs]
				val = _estimation_derivative(interp_func, deriv_level, t_point)
				push!(values, Float64(val))
			else
				# Try with wrapped LHS
				obs_lhs_wrapped = Symbolics.wrap(measured_quantities[obs_idx].lhs)
				if haskey(interpolants, obs_lhs_wrapped)
					interp_func = interpolants[obs_lhs_wrapped]
					val = _estimation_derivative(interp_func, deriv_level, t_point)
					push!(values, Float64(val))
				else
					# Analytic transformed observables (y ~ _trfn_* / _obs_trfn_*)
					# legitimately have no data-fitted interpolant — derive the
					# value in closed form from the observable RHS. (The quoll-era
					# pipeline named these jets y<k>_<n>, bypassed the trfn check
					# above, and injected 0.0 here in place of REAL sin/cos
					# derivative values — 34k hits across cstr/bicycle benchmark
					# logs. Current naming resolves them at the trfn check; this
					# branch is defense-in-depth for y-named analytic jets.)
					rhs_base = replace(string(measured_quantities[obs_idx].rhs), r"\(t\)$" => "")
					analytic_name = deriv_level == 0 ? rhs_base : string(rhs_base, "_", deriv_level)
					aval = evaluate_known_trfn_variable(analytic_name, Float64(t_point))
					isnothing(aval) && (aval = evaluate_obs_trfn_template_variable(analytic_name, Float64(t_point)))
					if !isnothing(aval)
						push!(values, Float64(aval))
					else
						# Fail fast: a missing interpolant must not masquerade as
						# data = 0.0; NaN fails visibly downstream.
						@warn "No interpolant found for observable $obs_idx at derivative level $deriv_level"
						push!(values, NaN)
					end
				end
			end
		else
			@warn "Data variable $v not found in DD.obs_lhs mapping"
			push!(values, NaN)
		end
	end

	return values
end
