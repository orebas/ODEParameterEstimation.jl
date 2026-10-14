using Symbolics
# Refactor safety net — added 2026-06-09, BEFORE the code-review cleanup
# (see docs/internal/2026-06-09_code_review.md and docs/internal/2026-06-09_test_safety_net.md).
#
# Purpose: give the upcoming refactors (dead-code removal, dedup, type annotations,
# file splits) a behavior lock, and give the P0 bug fixes a concrete target.
#
# Convention:
#   @test        -> behavior to LOCK (green now, must stay green through refactors)
#   @test_broken -> CORRECT behavior the P0 fix should produce (red now -> flips on fix)

using ODEParameterEstimation
using Test
using LinearAlgebra
import ModelingToolkit

@testset "Refactor safety net (pre-cleanup)" begin

	# --- P0#4 (caller-side): lock the parser's documented behavior ------------
	# The parser is correct per its docstring ("last _N is always the derivative
	# order"); the real bug is callers applying it to bare param/state names that
	# legitimately end in _<digit>. Lock current behavior so the caller-side fix
	# (classify_si_ring_variable / _multipoint_deriv_order checking known params
	# first) is a deliberate, visible change.
	@testset "parse_derivative_variable_name documented behavior" begin
		P = ODEParameterEstimation.parse_derivative_variable_name
		@test P("y1_2") == ("y1", 2)
		@test P("y_2") == ("y", 2)
		@test P("y1_0") == ("y1", 0)
		@test P("_obs_trfn_cos_5_0_cos_3") == ("_obs_trfn_cos_5_0_cos", 3)
		@test P("y1") === nothing            # no _N suffix -> nothing
		# Ambiguity at the heart of P0#4: a parameter literally named k_2 parses as
		# d^2/dt^2 of k. Documented here so the fix lands caller-side, not here.
		@test P("k_2") == ("k", 2)
	end

	# --- P0#3: lock AAA interior accuracy so the baryEval tolerance fix can't
	# regress the common case (normal grid spacing, well clear of the ~5.6e-4
	# special-case window). ----------------------------------------------------
	@testset "aaad/baryEval interior accuracy lock" begin
		x = collect(range(0.0, 2.0; length = 41))   # spacing 0.05 >> 5.6e-4
		f = sin.(2.0 .* x)
		F = ODEParameterEstimation.aaad(x, f)
		for z in (0.123, 0.777, 1.314, 1.951)        # interior, off-node
			@test F(z) ≈ sin(2.0 * z) atol = 1e-7
		end
		@test F(x[20]) ≈ f[20] atol = 1e-10          # exactly on a node
		# first derivative via the production accessor
		@test ODEParameterEstimation.nth_deriv_at(F, 1, 0.777) ≈ 2.0 * cos(2.0 * 0.777) atol = 1e-5
	end

	# --- process_raw_solution fit error = canonical SSE ------------------------
	# `err` is the trajectory sum-of-squared residuals (Σ over every observable and
	# time point), matching `_trajectory_sse` so algebraic candidates rank in the same
	# unit the optimizer minimizes. harmonic_oscillator() has TWO observables; offset both by a known
	# δ so the residual is large and clean, then recompute SSE from the SAME ode_solution
	# the function returned (robust to solver tolerance).
	@testset "process_raw_solution fit error (SSE)" begin
		opts = EstimationOptions()
		sampled = ODEParameterEstimation.sample_problem_data(ODEParameterEstimation.Examples.harmonic_oscillator(), opts)
		current_states = ModelingToolkit.unknowns(sampled.model.system)
		current_params = ModelingToolkit.parameters(sampled.model.system)
		raw_sol = vcat(
			[Float64(sampled.ic[s]) for s in current_states],
			[Float64(sampled.p_true[p]) for p in current_params],
		)

		δ = 0.1
		offset_data = deepcopy(sampled.data_sample)
		obs_keys = [k for k in keys(offset_data) if !isequal(k, "t")]
		n_obs = length(obs_keys)                     # == 2 for harmonic_oscillator()
		@test n_obs == 2
		for k in obs_keys
			offset_data[k] = offset_data[k] .+ δ
		end

		_, _, ode_solution, err = ODEParameterEstimation.process_raw_solution(
			raw_sol, sampled.model, offset_data, sampled.solver,
			abstol = opts.abstol, reltol = opts.reltol,
		)
		@test ode_solution !== nothing

		# Recompute the error exactly as the source does, from the SAME ode_solution.
		t = offset_data["t"]
		my_sum = 0.0
		for k in obs_keys
			my_sum += sum(abs2, ode_solution(t)[k] .- offset_data[k])
		end

		# 2026-06-11: `err` is now the canonical SSE — Σ over every observable and time
		# point of squared residuals (see `_trajectory_sse`), so every candidate (algebraic
		# / polished / synthesized) is reported and ranked in ONE unit. No `/N`, no
		# `/n_obs` divisor — the earlier divisor bug is moot because there is no divisor.
		@test err ≈ my_sum rtol = 1e-6
	end

	# --- Phase C1: legacy-path homotopy data evaluator ---------------------
	# Locks two behaviors of evaluate_data_vars_at_point:
	#  (1) analytic transformed observables (y ~ _trfn_*) with no data-fitted
	#      interpolant resolve in CLOSED FORM from the observable RHS — the
	#      quoll-era pipeline injected 0.0 for these (34k hits on cstr/bicycle
	#      benchmark logs), silently corrupting the homotopy data;
	#  (2) a genuinely unresolvable data variable yields NaN (visible failure),
	#      never 0.0 (fake data).
	@testset "evaluate_data_vars_at_point: analytic observables + NaN fail-fast" begin
		t = ModelingToolkit.t_nounits
		ModelingToolkit.@variables y1(t) y3(t) x1(t) _trfn_cos_0_5(t)
		mq = [y1 ~ x1, y3 ~ _trfn_cos_0_5]
		jets = ModelingToolkit.@variables y1_0 y1_1 y1_2 y3_0 y3_1 y3_2 zz_unmapped
		NumT = ModelingToolkit.Num
		DD = ODEParameterEstimation.DerivativeData(
			Vector{Vector{NumT}}(), Vector{Vector{NumT}}(),
			Vector{Vector{NumT}}(), Vector{Vector{NumT}}(),
			Vector{Vector{NumT}}(), Vector{Vector{NumT}}(),
			Vector{Vector{NumT}}(), Vector{Vector{NumT}}(),
			Set{NumT}(),
		)
		DD.obs_lhs = [[jets[1], jets[4]], [jets[2], jets[5]], [jets[3], jets[6]]]
		# Interpolant ONLY for observable 1 (y1 ~ x1); observable 2 is analytic.
		interps = Dict{Any, Any}(Symbolics.diff2term(mq[1].rhs) => (x -> 2.0 * x))
		data_vars = Any[jets[1], jets[2], jets[3], jets[4], jets[5], jets[6], jets[7]]
		tp = 0.7
		vals = ODEParameterEstimation.evaluate_data_vars_at_point(interps, data_vars, DD, mq, tp)

		@test vals[1] ≈ 2.0 * tp atol = 1e-10          # y1_0 via interpolant
		@test vals[2] ≈ 2.0 atol = 1e-8                # y1_1 = d/dt 2t
		@test vals[3] ≈ 0.0 atol = 1e-8                # y1_2 = 0
		@test vals[4] ≈ cos(0.5 * tp) atol = 1e-10     # y3_0 analytic (cos(0.5t))
		@test vals[5] ≈ -0.5 * sin(0.5 * tp) atol = 1e-10   # y3_1 analytic
		@test vals[6] ≈ -0.25 * cos(0.5 * tp) atol = 1e-10  # y3_2 analytic
		@test isnan(vals[7])                           # unmapped → NaN, never 0.0
	end

	@testset "Levenberg–Marquardt solver solves a simple polynomial system" begin
		vars = ModelingToolkit.@variables nl_x nl_y
		sys = [vars[1]^2 - 4.0, vars[2] - vars[1]]
		sols, _, _, _ = ODEParameterEstimation.solve_with_fast_nlopt(sys, [vars[1], vars[2]]; start_point = [1.5, 0.0])
		@test length(sols) == 1
		@test sols[1][1] ≈ 2.0 atol = 1e-5
		@test sols[1][2] ≈ 2.0 atol = 1e-5
		# x² + 1 = 0 has no real solution; a least-squares solver converges to the
		# minimum of the residual, x = 0, and returns it.
		inconsistent, _, _, _ = ODEParameterEstimation.solve_with_fast_nlopt([vars[1]^2 + 1.0], [vars[1]]; start_point = [0.5])
		@test only(only(inconsistent)) ≈ 0.0 atol = 1e-6
	end
end
