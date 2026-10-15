# Residual-mode polish path: bounded LeastSquaresOptim LM and FastLevenbergMarquardt
# in per-variable transformed coordinates.
#
# This path shares PolishContext setup (observable funcs, data_targets, base ODEProblem,
# coordinate transforms) with the legacy scalar polisher. The only difference is the
# inner solve: a residual-vector LM call rather than Optimization.solve on a scalar loss.
#
# Includes:
# - revert guard (return p0 when ‖r(p_solved)‖ > ‖r(p0)‖)
# - sentinel residual fill (1e6) for failed ODE solves — `Inf` would poison LM Jacobian
# - optional λ‖x_internal‖² regularization via augmented residual
# - optional soft-wall penalty near bounds via augmented residual
#   (zero in central interval, grows quadratically as parameters approach either bound)
# - native bound support (LSO/FastLM accept `lower=`/`upper=` directly)

const _RESIDUAL_SENTINEL_VALUE = 1.0e6

"""
	_residual_sentinel(eltype) -> sentinel value of correct type

The residual fill value used when an ODE solve fails inside the LM loop.
"""
_residual_sentinel(res::AbstractVector{T}) where {T} = convert(T, _RESIDUAL_SENTINEL_VALUE)

# Thrown from inside `residual!` when the wall-clock deadline passes. Caught
# around the LSO/FastLM call below; on catch, the polish returns the best
# iterate seen so far. LeastSquaresOptim and FastLevenbergMarquardt expose no
# iteration callback or time limit of their own, so this is the only way to
# enforce `maxtime` on the residual-mode polish path. Its scope is intentionally
# private to this file.
struct _PolishTimeoutSignal <: Exception end

"""
	_polish_single_residual(ctx, p0; ...) -> (ParameterEstimationResult, opt_result)

Residual-mode polish using `LeastSquaresOptim.LevenbergMarquardt()` (`solver_kind = :lso_direct`)
or `FastLevenbergMarquardt.lmsolve!` (`solver_kind = :fastlm_direct`). The residual
vector is `r(p_internal) = [observed - model] ++ optional λ-augmentation`.

Bounds are passed natively to LSO/FastLM. The `revert guard` ensures we never return
a polished point whose final residual norm exceeds the seed's — the seed is returned
in that case.

Optional λ regularization (`ctx.regularization_lambda > 0`) appends `√λ · x_internal`
rows to the residual. Default is off (λ = 0) per the recommendation memo.

Optional soft-wall penalty (`ctx.softwall_lambda > 0`) appends one row per parameter:
zero in the central `(1 - 2·ε_sw)·halfrange` band of the internal-coord interval,
quadratic in SSR (linear in residual) as the parameter approaches either bound.
Targets bound-saturation pathologies (e.g. biohydrogenation k10) without biasing
interior solutions. Default off (λ_sw = 0).
"""
function _polish_single_residual(
	ctx::PolishContext,
	p0::AbstractVector{<:Real};
	solver_kind::Symbol = :lso_direct,
	optimizer_factory = () -> LeastSquaresOptim.LevenbergMarquardt(),
	maxiters::Int = 1000,
	maxtime::Float64 = 300.0,
	lso_delta::Float64 = 10.0,
	lso_x_tol::Float64 = -1.0,
	lso_f_tol::Float64 = -1.0,
	lso_g_tol::Float64 = -1.0,
	retain_internal_optimum::Bool = false,
)
	polish_t0 = time()
	polish_stages = OrderedDict{Symbol, Float64}()
	residual_call_count = Ref(0)
	residual_float_call_count = Ref(0)
	residual_dual_call_count = Ref(0)
	residual_ode_solve_seconds = Ref(0.0)
	residual_model_eval_seconds = Ref(0.0)
	residual_success_count = Ref(0)
	residual_sentinel_count = Ref(0)
	jacobian_call_count = Ref(0)
	jacobian_seconds = Ref(0.0)
	# --- Coordinate transform setup ---
	has_external_bounds = !isnothing(ctx.lb) && !isnothing(ctx.ub)
	p0_external = has_external_bounds ? clamp.(Float64.(p0), ctx.lb, ctx.ub) : Float64.(p0)
	p0_internal = _polish_external_to_internal(p0_external, ctx.coordinate_transforms, ctx.coordinate_shifts)

	internal_lb = ctx.internal_lb
	internal_ub = ctx.internal_ub

	# --- Residual layout ---
	λ = max(ctx.regularization_lambda, 0.0)
	use_regularization = λ > 0.0
	penalty_scale = use_regularization ? sqrt(λ) : 0.0
	n_unknowns = length(p0_internal)
	n_obs_residual = sum(length(target) for target in ctx.data_targets)

	# Soft-wall penalty (one residual row per parameter when active).
	# Active only when bounds exist (internal_lb / internal_ub set) AND λ_sw > 0.
	λ_sw = max(ctx.softwall_lambda, 0.0)
	ε_sw = ctx.softwall_epsilon
	use_softwall = λ_sw > 0.0 && !isnothing(internal_lb) && !isnothing(internal_ub) && 0.0 <= ε_sw < 0.5
	softwall_scale = use_softwall ? sqrt(λ_sw) : 0.0
	# Pre-compute per-parameter midpoint, halfrange and activation threshold in internal coords.
	if use_softwall
		sw_midpoint = Vector{Float64}(undef, n_unknowns)
		sw_halfrange = Vector{Float64}(undef, n_unknowns)
		sw_threshold = Vector{Float64}(undef, n_unknowns)
		@inbounds for i in 1:n_unknowns
			sw_midpoint[i] = (internal_lb[i] + internal_ub[i]) / 2
			sw_halfrange[i] = (internal_ub[i] - internal_lb[i]) / 2
			sw_threshold[i] = (1.0 - ε_sw) * sw_halfrange[i]
		end
	else
		sw_midpoint = Float64[]
		sw_halfrange = Float64[]
		sw_threshold = Float64[]
	end

	residual_count = n_obs_residual + (use_regularization ? n_unknowns : 0) + (use_softwall ? n_unknowns : 0)
	residual_count == 0 && throw(ArgumentError("Residual polish requires at least one observed datum"))

	# Wall-clock deadline + best-iterate tracking. The deadline is held in a Ref
	# so we can disarm it (set to `Inf`) before the post-solve revert-guard
	# residual evaluation. `best_p_seen` records the best Float64-path iterate so
	# that on timeout we return that iterate rather than discarding all progress;
	# Dual-typed Jacobian evaluations are skipped from the comparison.
	deadline_ref = Ref(Inf)
	best_norm_seen = Ref(Inf)
	best_p_seen = copy(p0_internal)

	# --- Residual closure ---
	function residual!(res, p_internal, _ = nothing)
		residual_call_count[] += 1
		if eltype(res) === Float64 && eltype(p_internal) === Float64
			residual_float_call_count[] += 1
		else
			residual_dual_call_count[] += 1
		end
		if time() > deadline_ref[]
			throw(_PolishTimeoutSignal())
		end

		p_all = _polish_internal_to_external(p_internal, ctx.coordinate_transforms, ctx.coordinate_shifts)
		ic_guess = @view p_all[1:ctx.n_ic]
		param_guess = @view p_all[(ctx.n_ic + 1):end]

		prob_opt = remake(
			ctx.base_ode_prob;
			u0 = Dict(ctx.unknown_syms .=> ic_guess),
			p = Dict(ctx.param_syms .=> param_guess),
			build_initializeprob = false,
		)

		# `unstable_check` lets the integrator bail mid-step the moment the polish
		# deadline passes. Without this, one ForwardDiff Jacobian call (Dual-typed
		# ODE solve at tight tolerances) can burn many seconds past `deadline_ref`
		# before the next residual!-entry check fires. With it, the integrator
		# returns ReturnCode.Unstable promptly and we sentinel-fill the residual.
		ode_t0 = time()
		sol_opt = try
			ModelingToolkit.solve(
				prob_opt,
				ctx.solver;
				saveat = ctx.t_vector,
				abstol = ctx.abstol,
				reltol = ctx.reltol,
				maxiters = ctx.polish_ode_maxiters,
				unstable_check = (dt, u, p, ti) -> time() > deadline_ref[],
			)
		catch err
			_rethrow_if_interrupt(err)
			# A solve that throws is treated as a failed trial point. Say so once:
			# when every solve throws, the polish silently returns its seed.
			@warn "ODE solve inside the residual polish threw; treating the point as failed" exception = (err, catch_backtrace()) maxlog = 1
			residual_ode_solve_seconds[] += time() - ode_t0
			residual_sentinel_count[] += 1
			fill!(res, _residual_sentinel(res))
			return nothing
		end
		residual_ode_solve_seconds[] += time() - ode_t0

		if sol_opt.retcode != ReturnCode.Success
			residual_sentinel_count[] += 1
			fill!(res, _residual_sentinel(res))
			return nothing
		end
		residual_success_count[] += 1

		model_eval_t0 = time()
		idx = 1
		@inbounds for (j, f) in enumerate(ctx.obs_funcs)
			data_true = ctx.data_targets[j]
			for (i, time_index) in enumerate(_polish_data_indices(ctx, j))
				res[idx] = f(sol_opt.u[time_index], param_guess) - data_true[i]
				idx += 1
			end
		end
		residual_model_eval_seconds[] += time() - model_eval_t0
		if use_regularization
			@inbounds for i in eachindex(p_internal)
				res[idx] = penalty_scale * p_internal[i]
				idx += 1
			end
		end
		if use_softwall
			# Per-parameter soft-wall: zero in the central (1 - 2ε_sw)·halfrange band,
			# grows linearly in the augmented residual (= quadratically in SSR) as
			# the parameter moves past `threshold = (1 - ε_sw)·halfrange` toward
			# either bound. Penalty in SSR is λ_sw · over² where
			# over = (|p_internal - midpoint| - threshold) / halfrange.
			@inbounds for i in eachindex(p_internal)
				deviation = abs(p_internal[i] - sw_midpoint[i])
				thresh = sw_threshold[i]
				half = sw_halfrange[i]
				if deviation > thresh && half > 0
					over = (deviation - thresh) / half
					res[idx] = softwall_scale * over
				else
					res[idx] = zero(eltype(res))
				end
				idx += 1
			end
		end

		# Update best-seen only on the Float64 trial-evaluation path; ForwardDiff
		# Jacobian calls use Dual eltypes and would compare apples-to-oranges.
		if eltype(res) === Float64 && eltype(p_internal) === Float64
			full_norm = zero(Float64)
			@inbounds for k in 1:residual_count
				full_norm += res[k] * res[k]
			end
			full_norm = sqrt(full_norm)
			if isfinite(full_norm) && full_norm < best_norm_seen[]
				best_norm_seen[] = full_norm
				copyto!(best_p_seen, p_internal)
			end
		end
		return nothing
	end

	residual_vec(p_internal) = (r = Vector{eltype(p_internal)}(undef, residual_count); residual!(r, p_internal); r)

	function jacobian!(J, p_internal, _ = nothing)
		jacobian_call_count[] += 1
		jac_t0 = time()
		_outer_jacobian!(J, residual_vec, p_internal)
		jacobian_seconds[] += time() - jac_t0
		return nothing
	end

	# --- Initial residual for revert guard ---
	# Run with the deadline disarmed so an absurdly small `maxtime` doesn't fire
	# before we've even evaluated the seed. The Float64-path tracker inside
	# residual! seeds best_norm_seen / best_p_seen from this call.
	initial_residual = zeros(residual_count)
	_timed_detail_stage!(polish_stages, :initial_residual) do
		residual!(initial_residual, p0_internal)
	end
	initial_norm = norm(initial_residual)

	# --- Arm the deadline ---
	# `maxtime <= 0` or non-finite disables enforcement (Inf deadline). Otherwise
	# residual! throws _PolishTimeoutSignal once `time() > deadline_ref[]`.
	deadline_ref[] = (isfinite(maxtime) && maxtime > 0) ? time() + maxtime : Inf

	# --- Solve ---
	candidate_internal = p0_internal
	solver_result = nothing
	timed_out = false

	if solver_kind === :fastlm_direct
		J0 = zeros(eltype(p0_internal), residual_count, n_unknowns)
		f0 = zeros(eltype(p0_internal), residual_count)
		workspace = FastLevenbergMarquardt.LMWorkspace(copy(p0_internal), f0, J0)
		linear_solver = n_unknowns <= residual_count ? :qr : :cholesky
		fastlm_residual! = (res, u, p) -> begin
			residual!(res, u, p)
			return res
		end
		fastlm_jacobian! = (J, u, p) -> begin
			jacobian!(J, u, p)
			return J
		end
		_timed_detail_stage!(polish_stages, :optimizer_solve) do
			try
				result = FastLevenbergMarquardt.lmsolve!(
					fastlm_residual!,
					fastlm_jacobian!,
					workspace,
					nothing,
					internal_lb,
					internal_ub;
					solver = linear_solver,
					xtol = lso_x_tol > 0 ? lso_x_tol : ctx.reltol,
					ftol = lso_f_tol > 0 ? lso_f_tol : ctx.reltol,
					gtol = lso_g_tol > 0 ? lso_g_tol : ctx.abstol,
					maxit = maxiters,
				)
				solver_result = result
				candidate_internal = result[1]
			catch e
				isa(e, _PolishTimeoutSignal) || rethrow(e)
				timed_out = true
				solver_result = nothing
				candidate_internal = best_p_seen
			end
		end

	elseif solver_kind === :lso_direct
		problem = LeastSquaresOptim.LeastSquaresProblem(
			x = copy(p0_internal),
			f! = (out, x) -> residual!(out, x, nothing),
			g! = (J, x) -> jacobian!(J, x, nothing),
			output_length = residual_count,
		)
		opt = optimizer_factory()
		_timed_detail_stage!(polish_stages, :optimizer_solve) do
			try
				result = LeastSquaresOptim.optimize!(
					problem,
					opt;
					x_tol = lso_x_tol > 0 ? lso_x_tol : ctx.reltol,
					f_tol = lso_f_tol > 0 ? lso_f_tol : ctx.reltol,
					g_tol = lso_g_tol > 0 ? lso_g_tol : ctx.abstol,
					iterations = maxiters,
					Δ = lso_delta,
					lower = isnothing(internal_lb) ? eltype(p0_internal)[] : internal_lb,
					upper = isnothing(internal_ub) ? eltype(p0_internal)[] : internal_ub,
				)
				solver_result = result
				candidate_internal = result.minimizer
			catch e
				isa(e, _PolishTimeoutSignal) || rethrow(e)
				timed_out = true
				solver_result = nothing
				candidate_internal = best_p_seen
			end
		end
	else
		throw(ArgumentError("Unknown residual polish solver_kind '$solver_kind' (expected :lso_direct or :fastlm_direct)"))
	end

	# --- Revert guard: keep the seed if the solver made things worse ---
	# Disarm the deadline first; we want this final residual evaluation to run to
	# completion regardless of how much wall-clock the LSO/FastLM call burned.
	deadline_ref[] = Inf
	final_residual = similar(initial_residual)
	_timed_detail_stage!(polish_stages, :final_residual) do
		residual!(final_residual, candidate_internal)
	end
	final_norm = norm(final_residual)

	# Fail fast: if NO residual evaluation ever solved the ODE successfully
	# (every call sentinel-filled), both norms are sentinel artifacts — the
	# comparison below is meaningless and previously kept the optimizer's
	# wander-point while reporting a plausible-looking objective. Revert to the
	# seed and mark the result instead.
	polish_all_sentinel = residual_success_count[] == 0

	p_opt_internal, objective_residual = if polish_all_sentinel
		p0_internal, initial_residual
	elseif isfinite(final_norm) && final_norm <= initial_norm
		candidate_internal, final_residual
	else
		p0_internal, initial_residual
	end

	p_opt = _polish_internal_to_external(p_opt_internal, ctx.coordinate_transforms, ctx.coordinate_shifts)
	ic_opt = p_opt[1:ctx.n_ic]
	param_opt = p_opt[(ctx.n_ic + 1):end]

	prob_final = remake(
		ctx.base_ode_prob;
		u0 = Dict(ctx.unknown_syms .=> ic_opt),
		p = Dict(ctx.param_syms .=> param_opt),
		build_initializeprob = false,
	)
	sol_final = _timed_detail_stage!(polish_stages, :final_ode_solve) do
		ModelingToolkit.solve(
			prob_final,
			ctx.solver;
			saveat = ctx.t_vector,
			abstol = ctx.abstol,
			reltol = ctx.reltol,
		)
	end

	# Use ONLY the trajectory-fit portion of the residual when reporting err — exclude
	# the λ-augmented rows so the err is comparable across λ values.
	final_obj = sum(abs2, @view objective_residual[1:n_obs_residual])
	# All-sentinel polish: the model was never evaluated — report an honest Inf,
	# not a sentinel-norm that looks like a (terrible but real) fit.
	polish_all_sentinel && (final_obj = Inf)

	states_out = OrderedDict(s => ic_opt[ctx.state_index[s]] for s in ctx.state_syms_out if haskey(ctx.state_index, s))
	params_out = OrderedDict(p => param_opt[ctx.param_index[p]] for p in ctx.param_syms_out if haskey(ctx.param_index, p))

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
	if timed_out
		push!(final_result.provenance.notes, :polish_maxtime_exceeded)
	end
	if polish_all_sentinel
		push!(final_result.provenance.notes, :polish_all_sentinel)
	end
	sync_result_contract!(final_result)
	_record_detailed_timing!((
		category = :polish_single_residual,
		context = _current_detailed_timing_context(),
		total_seconds = time() - polish_t0,
		stage_seconds = copy(polish_stages),
		solver_kind = solver_kind,
		residual_count = residual_count,
		unknown_count = n_unknowns,
		residual_call_count = residual_call_count[],
		residual_float_call_count = residual_float_call_count[],
		residual_dual_call_count = residual_dual_call_count[],
		residual_success_count = residual_success_count[],
		residual_sentinel_count = residual_sentinel_count[],
		residual_ode_solve_seconds = residual_ode_solve_seconds[],
		residual_model_eval_seconds = residual_model_eval_seconds[],
		jacobian_call_count = jacobian_call_count[],
		jacobian_seconds = jacobian_seconds[],
		initial_norm = initial_norm,
		final_norm = final_norm,
		timed_out = timed_out,
		reverted_to_seed = polish_all_sentinel || !(isfinite(final_norm) && final_norm <= initial_norm),
		polish_all_sentinel = polish_all_sentinel,
		final_obj = final_obj,
	))
	returned_optimizer = retain_internal_optimum ?
		_PolishSolveRecord(solver_result, Float64.(p_opt_internal)) : solver_result
	return final_result, returned_optimizer
end
