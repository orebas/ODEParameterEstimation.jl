"""
	solve_with_robust(poly_system, varlist; kwargs...)

A robust polynomial system solver that uses the best algorithms from our benchmarking.
Supports both solving from scratch and polishing mode.

It is selected with `system_solver = SolverRobust`, and the estimation also uses
it with `polish_only = true` to refine the roots homotopy continuation finds.

# Arguments
- `poly_system`: Vector of polynomial equations (Symbolics expressions)
- `varlist`: Vector of variables (Symbolics variables)

# Keywords
- `start_point=nothing`: Initial guess. If nothing, uses random or multistart
- `polish_only=false`: If true, only does quick local refinement
- `prepared_system=nothing`: Optional reusable kernel from `prepare_robust_system`.
- `data_values=Float64[]`: Numerical data in the prepared kernel's declared order.
- `options=Dict()`: Additional options including:
  - `:debug => true/false`: Print debug information
  - `:jacobian => :forwarddiff/:symbolic/:finitediff/:none`: Jacobian method (default
    :forwarddiff for an unprepared call; otherwise the prepared kernel's method)
  - `:forwarddiff_chunk_size => 1`: AD directions per chunk; zero selects automatically
  - `:abstol => 1e-8`: Absolute tolerance
  - `:reltol => 1e-6`: Relative tolerance
  - `:maxiters => 1000`: Maximum iterations
  - `:algorithm => :auto/:trustregion/:bfgs/:bobyqa/:levenberg`: Force specific algorithm
  - `:multistart => true/false`: Use multiple starting points
  - `:timeout => 300.0`: Wall-clock budget in seconds. Enforced between starting
    points AND per solver call (NonlinearSolve `maxtime`, checked per iteration —
    a single pathological iteration can still overrun; the deadline-Ref-in-residual
    pattern from polish_residual.jl:22-28 is the escalation lever if that bites)

# Returns
`(solutions, varlist, Dict(), varlist)`, the format of the other system solvers,
where `solutions` is a vector of solution vectors in `varlist` order (empty if
no start converged).
"""
function solve_with_robust(poly_system, varlist;
	start_point = nothing,
	polish_only = false,
	options = Dict(),
	prepared_system::Union{Nothing, PreparedRobustSystem} = nothing,
	data_values::AbstractVector = Float64[])


	robust_t0 = time()
	robust_stages = OrderedDict{Symbol, Float64}()
	timing_enabled = _run_ctx_detailed_sink() !== nothing
	residual_call_count = 0
	residual_seconds = 0.0
	jacobian_call_count = 0
	jacobian_seconds = 0.0
	starts_attempted = 0
	starts_skipped_bad_residual = 0
	successful_start_count = 0
	algorithm_failure_count = 0
	maxtime_hit_count = 0

	# Extract options
	debug = get(options, :debug, false)
	jac_mode = get(options, :jacobian,
		isnothing(prepared_system) ? :forwarddiff : prepared_system.jacobian_mode)
	abstol = get(options, :abstol, polish_only ? 1e-6 : 1e-8)
	reltol = get(options, :reltol, polish_only ? 1e-4 : 1e-6)
	maxiters = get(options, :maxiters, polish_only ? 100 : 1000)
	algorithm = get(options, :algorithm, :auto)
	multistart = get(options, :multistart, !polish_only && isnothing(start_point))
	timeout = get(options, :timeout, 300.0)

	# System dimensions
	m = length(poly_system)
	n = length(varlist)

	if debug
		println("[ROBUST] System: $m equations, $n variables")
		println("[ROBUST] Polish mode: $polish_only")
		println("[ROBUST] Algorithm: $algorithm")
		println("[ROBUST] Jacobian: $jac_mode")
	end

	# A supplied kernel is shared by all roots/points with the same symbolic
	# structure. Only numerical data and solve-local buffers change.
	_build_residual_t0 = time()
	using_prepared = !isnothing(prepared_system)
	system = if using_prepared
		prepared_system.equation_count == m || throw(DimensionMismatch("Prepared polynomial equation count does not match"))
		isequal(prepared_system.variables, Num.(varlist)) ||
			throw(ArgumentError("Prepared polynomial unknown order does not match varlist"))
		jac_mode == prepared_system.jacobian_mode ||
			throw(ArgumentError("Jacobian option conflicts with the prepared polynomial kernel"))
		if haskey(options, :forwarddiff_chunk_size)
			options[:forwarddiff_chunk_size] == prepared_system.forwarddiff_chunk_size ||
				throw(ArgumentError("ForwardDiff chunk option conflicts with the prepared polynomial kernel"))
		end
		prepared_system
	else
		prepare_robust_system(poly_system, varlist; jacobian = jac_mode,
			forwarddiff_chunk_size = get(options, :forwarddiff_chunk_size, 1))
	end
	length(data_values) == length(system.data_variables) ||
		throw(DimensionMismatch("Expected $(length(system.data_variables)) polynomial data values, got $(length(data_values))"))
	# Bind a private copy; simultaneous solves using one prepared kernel cannot
	# overwrite one another's data or derivative workspace.
	bound_data = Float64.(data_values)
	jac_mode = system.jacobian_mode
	robust_stages[:build_residual_function] = time() - _build_residual_t0

	function residual!(res, u, p = nothing)
		if timing_enabled
			_residual_t0 = time()
			try
				system.residual!(res, u, bound_data)
			finally
				residual_call_count += 1
				residual_seconds += time() - _residual_t0
			end
		else
			system.residual!(res, u, bound_data)
		end
		return nothing
	end

	function objective(u)
		res = similar(u, m)
		residual!(res, u)
		return 0.5 * sum(abs2, res)
	end

	_jacobian_setup_t0 = time()
	raw_jacobian! = _bind_robust_jacobian(system, residual!, bound_data)
	jac_func = if isnothing(raw_jacobian!)
		nothing
	else
		(J, u) -> begin
			if timing_enabled
				_jac_t0 = time()
				try
					raw_jacobian!(J, u)
				finally
					jacobian_call_count += 1
					jacobian_seconds += time() - _jac_t0
				end
			else
				raw_jacobian!(J, u)
			end
			nothing
		end
	end
	# The least-squares gradient is JᵀF. Reuse the Jacobian kernel instead of
	# generating and compiling a separate expanded symbolic gradient.
	grad_func = if isnothing(jac_func)
		nothing
	else
		r_grad, J_grad = zeros(m), zeros(m, n)
		(g, u) -> begin
			residual!(r_grad, u)
			jac_func(J_grad, u)
			LinearAlgebra.mul!(g, transpose(J_grad), r_grad)
			nothing
		end
	end
	robust_stages[:jacobian_setup] = time() - _jacobian_setup_t0

	# Generate starting points
	_generate_starts_t0 = time()
	if multistart
		# Use diverse starting points
		starts = [
			isnothing(start_point) ? randn(n) : start_point,
			ones(n),
			zeros(n) .+ 0.1,
			ones(n) * 0.5,
			randn(n) * 0.1,
			rand(n) * 2.0 .- 1.0,
		]
	else
		starts = [isnothing(start_point) ? randn(n) : start_point]
	end
	robust_stages[:generate_starts] = get(robust_stages, :generate_starts, 0.0) + (time() - _generate_starts_t0)

	# Select algorithm based on mode and options
	function select_algorithm()
		if algorithm != :auto
			return algorithm
		end

		if polish_only
			# For polishing, use fast local methods
			if jac_func !== nothing
				return :trustregion  # Most robust with Jacobian
			else
				return :bobyqa  # Fast derivative-free
			end
		else
			# For solving from scratch
			if jac_func !== nothing
				return :trustregion  # Most robust overall
			else
				return :bobyqa  # Best derivative-free we found
			end
		end
	end

	selected_algo = select_algorithm()

	# Storage for solutions
	all_solutions = []
	best_solution = nothing
	best_residual = Inf

	# Try each starting point
	start_time = time()
	for (idx, x0) in enumerate(starts)
		starts_attempted += 1
		if time() - start_time > timeout
			if debug
				println("[ROBUST] Timeout reached")
			end
			break
		end

		if debug && length(starts) > 1
			println("[ROBUST] Trying start point $idx/$(length(starts))")
		end

		# Test initial residual
		_initial_residual_t0 = time()
		res0 = zeros(m)
		try
			residual!(res0, x0)
		finally
			robust_stages[:initial_residual] = get(robust_stages, :initial_residual, 0.0) + (time() - _initial_residual_t0)
		end
		if any(isnan, res0) || any(isinf, res0)
			starts_skipped_bad_residual += 1
			continue
		end

		sol = nothing
		success = false

		_nonlinear_solve_t0 = time()
		_nonlinear_solve_recorded = false
		try
			if selected_algo == :trustregion
				# Use NonlinearSolve.TrustRegion (most robust)
				if jac_func !== nothing
					nf = NonlinearFunction(residual!; jac = (J, u, p) -> jac_func(J, u),
						resid_prototype = zeros(m), jac_prototype = zeros(m, n))
				else
					nf = NonlinearFunction(residual!; resid_prototype = zeros(m))
				end

				prob = if m == n
					NonlinearProblem(nf, x0)
				else
					NonlinearLeastSquaresProblem(nf, x0)
				end

				sol = NonlinearSolve.solve(prob, TrustRegion();
					abstol = abstol,
					reltol = reltol,
					maxiters = maxiters,
					maxtime = max(timeout - (time() - start_time), 1.0))

				success = SciMLBase.successful_retcode(sol)
				if !success && sol.retcode == SciMLBase.ReturnCode.MaxTime
					maxtime_hit_count += 1
				end

			elseif selected_algo == :bfgs
				# Use Optim.BFGS (very robust for optimization)
				if grad_func !== nothing
					result = Optim.optimize(objective,
						(g, u) -> grad_func(g, u), x0,
						Optim.BFGS(linesearch = Optim.LineSearches.BackTracking()),
						Optim.Options(g_tol = abstol, iterations = maxiters,
							time_limit = max(timeout - (time() - start_time), 1.0)))
				else
					result = Optim.optimize(objective, x0, Optim.BFGS(),
						Optim.Options(time_limit = max(timeout - (time() - start_time), 1.0)))
				end

				sol = (u = result.minimizer,
					resid = sqrt(2 * result.minimum),
					retcode = Optim.converged(result) ? :Success : :MaxIters)
				success = Optim.converged(result)

			elseif selected_algo == :bobyqa
				# Use NLopt.BOBYQA (best derivative-free)
				opt = NLopt.Opt(:LN_BOBYQA, n)
				opt.min_objective = (x, grad) -> objective(x)
				opt.lower_bounds = fill(-100.0, n)
				opt.upper_bounds = fill(100.0, n)
				opt.ftol_abs = abstol^2  # Since we're minimizing ||f||^2
				opt.maxeval = maxiters
				opt.maxtime = max(timeout - (time() - start_time), 1.0)

				(minf, minx, ret) = NLopt.optimize(opt, x0)

				sol = (u = minx, resid = sqrt(2 * minf),
					retcode = ret == :SUCCESS ? :Success : Symbol(ret))
				success = (ret == :SUCCESS || ret == :FTOL_REACHED || ret == :XTOL_REACHED)

			elseif selected_algo == :levenberg
				# Use NonlinearSolve.LevenbergMarquardt
				if jac_func !== nothing
					nf = NonlinearFunction(residual!; jac = (J, u, p) -> jac_func(J, u),
						resid_prototype = zeros(m), jac_prototype = zeros(m, n))
				else
					nf = NonlinearFunction(residual!; resid_prototype = zeros(m))
				end

				prob = NonlinearLeastSquaresProblem(nf, x0)
				sol = NonlinearSolve.solve(prob, NonlinearSolve.LevenbergMarquardt();
					abstol = abstol,
					reltol = reltol,
					maxiters = maxiters,
					maxtime = max(timeout - (time() - start_time), 1.0))

				success = SciMLBase.successful_retcode(sol)
				if !success && sol.retcode == SciMLBase.ReturnCode.MaxTime
					maxtime_hit_count += 1
				end

			else
				# Fallback: NonlinearSolve.TrustRegion
				if jac_func !== nothing
					nf = NonlinearFunction(residual!; jac = (J, u, p) -> jac_func(J, u),
						resid_prototype = zeros(m), jac_prototype = zeros(m, n))
				else
					nf = NonlinearFunction(residual!; resid_prototype = zeros(m))
				end

				prob = if m == n
					NonlinearProblem(nf, x0)
				else
					NonlinearLeastSquaresProblem(nf, x0)
				end

				sol = NonlinearSolve.solve(prob, TrustRegion();
					abstol = abstol,
					reltol = reltol,
					maxiters = maxiters,
					maxtime = max(timeout - (time() - start_time), 1.0))

				success = SciMLBase.successful_retcode(sol)
				if !success && sol.retcode == SciMLBase.ReturnCode.MaxTime
					maxtime_hit_count += 1
				end
			end
			robust_stages[:nonlinear_solve] = get(robust_stages, :nonlinear_solve, 0.0) + (time() - _nonlinear_solve_t0)
			_nonlinear_solve_recorded = true

			# Check solution quality
			if success && !isnothing(sol)
				successful_start_count += 1
				_final_residual_t0 = time()
				res_final = zeros(m)
				final_norm = try
					residual!(res_final, sol.u)
					norm(res_final)
				finally
					robust_stages[:final_residual] = get(robust_stages, :final_residual, 0.0) + (time() - _final_residual_t0)
				end

				if final_norm < best_residual
					best_residual = final_norm
					best_solution = sol.u

					# Create solution dictionary
					sol_dict = Dict(zip(varlist, sol.u))
					push!(all_solutions, sol_dict)

					if debug
						println("[ROBUST] Found solution with residual: $final_norm")
					end

					# If we found a good solution, maybe stop (unless multistart)
					if final_norm < abstol && !multistart
						break
					end
				end
			end

		catch e
			_rethrow_if_interrupt(e)
			algorithm_failure_count += 1
			if !_nonlinear_solve_recorded
				robust_stages[:nonlinear_solve] = get(robust_stages, :nonlinear_solve, 0.0) + (time() - _nonlinear_solve_t0)
			end
			@error "[ROBUST] Algorithm failed" exception=(e, catch_backtrace())
			println("SOLVER_ERROR: solve_with_robust algorithm threw exception:")
			println("  Type: ", typeof(e))
			println("  Message: ", e)
			bt = catch_backtrace()
			st = stacktrace(bt)
			for (i, frame) in enumerate(st[1:min(5, length(st))])
				println("  [$i] ", frame)
			end
		end
	end

	function _record_robust_timing!(unique_count)
		_record_detailed_timing!((
			category = :solve_with_robust,
			context = _current_detailed_timing_context(),
			total_seconds = time() - robust_t0,
			stage_seconds = copy(robust_stages),
			equation_count = m,
			variable_count = n,
			polish_only = polish_only,
			algorithm = selected_algo,
			jacobian = jac_mode,
			multistart = multistart,
			start_count = length(starts),
			starts_attempted = starts_attempted,
			starts_skipped_bad_residual = starts_skipped_bad_residual,
			successful_start_count = successful_start_count,
			algorithm_failure_count = algorithm_failure_count,
			maxtime_hit_count = maxtime_hit_count,
			residual_call_count = residual_call_count,
			residual_seconds = residual_seconds,
			jacobian_call_count = jacobian_call_count,
			jacobian_seconds = jacobian_seconds,
			raw_solution_count = length(all_solutions),
			unique_solution_count = unique_count,
			best_residual = isfinite(best_residual) ? best_residual : nothing,
			used_compiled_residual = system.compiled_residual,
			used_prepared_system = using_prepared,
			data_value_count = length(bound_data),
			forwarddiff_chunk_size = system.forwarddiff_chunk_size,
		))
		return nothing
	end

	# Prepare output in the format of the other system solvers
	if isempty(all_solutions)
		if debug
			println("[ROBUST] No solutions found")
		end
		_record_robust_timing!(0)
		return (Vector{Vector{Float64}}(), varlist, Dict(), varlist)
	end

	# Remove duplicate solutions
	_deduplicate_t0 = time()
	unique_solutions = []
	for sol in all_solutions
		is_duplicate = false
		for unique_sol in unique_solutions
			diff = norm([sol[v] - unique_sol[v] for v in varlist])
			if diff < 1e-6
				is_duplicate = true
				break
			end
		end
		if !is_duplicate
			push!(unique_solutions, sol)
		end
	end
	robust_stages[:deduplicate_solutions] = get(robust_stages, :deduplicate_solutions, 0.0) + (time() - _deduplicate_t0)

	if debug
		println("[ROBUST] Found $(length(unique_solutions)) unique solution(s)")
		println("[ROBUST] Best residual: $best_residual")
	end

	# Convert solutions from Dicts to Vectors to match other solvers' output format
	_materialize_vectors_t0 = time()
	solutions_as_vectors = Vector{Vector{Float64}}()
	for sol_dict in unique_solutions
		# Ensure the order is correct according to varlist
		sol_vec = [sol_dict[v] for v in varlist]
		push!(solutions_as_vectors, sol_vec)
	end
	robust_stages[:materialize_solution_vectors] = get(robust_stages, :materialize_solution_vectors, 0.0) + (time() - _materialize_vectors_t0)

	_record_robust_timing!(length(unique_solutions))

	# Return in same format as other solvers: (solutions, varlist, trivial_dict, trimmed_varlist)
	return (solutions_as_vectors, varlist, Dict(), varlist)
end
