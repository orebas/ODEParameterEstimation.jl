# Contract tests for the scoped RunContext (run_context.jl):
# per-run isolated state replacing the run-state globals (auto-M hand-off,
# timing capture, timing sinks). Pure/fast — no estimation runs.

using ODEParameterEstimation
using Random
using Test

const _OPE = ODEParameterEstimation

@testset "RunContext scoped state" begin
	# With no context bound, every helper is a safe no-op.
	@test _OPE._run_ctx() === nothing
	@test _OPE._run_ctx_capture_timing() === false
	@test _OPE._run_ctx_take_auto_m!() === nothing
	_OPE._run_ctx_set_auto_m!(3)              # no-op without a context
	@test _OPE._run_ctx_take_auto_m!() === nothing
	@test _OPE._run_ctx_resolve_sink() === nothing
	@test _OPE._run_ctx_detailed_sink() === nothing
	@test _OPE._run_ctx_selection_recipe() === nothing

	# Research fixed-selection recipes are validated and scoped to exactly one
	# run; they never become ambient EstimationOptions state.
	sp_recipe = _OPE.FixedSinglePointRecipe(7; interpolator_source = :agp_uq)
	_, ctx_recipe = _OPE._with_run_context(selection_recipe = sp_recipe) do
		@test _OPE._run_ctx_selection_recipe() === sp_recipe
		nothing
	end
	@test ctx_recipe.selection_recipe === sp_recipe
	@test _OPE._run_ctx_selection_recipe() === nothing
	mp_recipe = _OPE.FixedMultipointRecipe([3, 11])
	@test mp_recipe.rows == [3, 11]
	@test_throws ArgumentError _OPE.FixedSinglePointRecipe(0)
	@test_throws ArgumentError _OPE.FixedMultipointRecipe(Int[])
	@test_throws ArgumentError _OPE.FixedMultipointRecipe([2, 2])

	# Scoped set/take + consume-once semantics.
	value, ctx = _OPE._with_run_context() do
		@test _OPE._run_ctx() !== nothing
		_OPE._run_ctx_set_auto_m!(2)
		@test _OPE._run_ctx_take_auto_m!() == 2
		@test _OPE._run_ctx_take_auto_m!() === nothing   # consumed
		_OPE._run_ctx_set_auto_m!(5)
		:done
	end
	@test value === :done
	@test ctx.auto_m == 5                     # ctx object outlives the scope
	@test _OPE._run_ctx() === nothing         # scope exited cleanly

	# Sinks install onto the bound context and clear with the scope.
	_, ctx_s = _OPE._with_run_context() do
		recs = NamedTuple[]
		dets = NamedTuple[]
		_OPE._run_ctx_install_sinks!(recs, dets)
		@test _OPE._run_ctx_resolve_sink() === recs
		@test _OPE._run_ctx_detailed_sink() === dets
		nothing
	end
	@test _OPE._run_ctx_detailed_sink() === nothing

	# Child tasks inherit the binding (ScopedValues semantics) — this is what
	# lets spawned polish workers see the run's sinks.
	inherited, _ = _OPE._with_run_context() do
		fetch(Threads.@spawn _OPE._run_ctx() !== nothing)
	end
	@test inherited === true

	# Two concurrent contexts are fully isolated (the cross-run contamination
	# class this refactor removes).
	results = Dict{Int, Union{Nothing, Int}}()
	lk = ReentrantLock()
	@sync for k in (10, 20)
		Threads.@spawn _OPE._with_run_context() do
			_OPE._run_ctx_set_auto_m!(k)
			sleep(0.05)                        # overlap the two scopes
			got = _OPE._run_ctx_take_auto_m!()
			lock(() -> results[k] = got, lk)
		end
	end
	@test results[10] == 10
	@test results[20] == 20

	# Public API contract: with_estimation_timing passes the value through and
	# yields timing === nothing when f performs no estimation.
	v, timing = _OPE.with_estimation_timing(() -> 42)
	@test v == 42
	@test timing === nothing

	# HC solver-config carry: nothing-fallback without a context, carried values
	# inside one, cleared with the scope.
	@test _OPE._run_ctx_hc_threading() === nothing
	@test _OPE._run_ctx_hc_compile_mode() === nothing
	_OPE._run_ctx_set_hc_opts!(false, :none)          # no-op without a context
	@test _OPE._run_ctx_hc_threading() === nothing
	_, ctx_hc = _OPE._with_run_context() do
		_OPE._run_ctx_set_hc_opts!(false, :none)
		@test _OPE._run_ctx_hc_threading() === false
		@test _OPE._run_ctx_hc_compile_mode() === :none
		nothing
	end
	@test ctx_hc.hc_compile_mode === :none
	@test _OPE._run_ctx_hc_threading() === nothing    # scope exited

	# Timing context-label stacks: per-run when bound, module fallback bare.
	@test _OPE._current_detailed_timing_context() === :unspecified
	_OPE._with_run_context() do
		_OPE._with_detailed_timing_context(:outer) do
			@test _OPE._current_detailed_timing_context() === :outer
			_OPE._with_detailed_timing_context(:inner) do
				@test _OPE._current_detailed_timing_context() === :inner
			end
			@test _OPE._current_detailed_timing_context() === :outer
		end
		@test _OPE._current_detailed_timing_context() === :unspecified
		nothing
	end
	# Bare-call fallback still functions (module stacks).
	_OPE._with_resolve_timing_context(:bare) do
		@test _OPE._current_resolve_timing_context() === :bare
	end
	@test _OPE._current_resolve_timing_context() === :unspecified
end

@testset "Scoped seed leaves the caller's RNG untouched" begin
	rng = Random.default_rng()
	Random.seed!(rng, 2026)
	before = copy(rng)

	# Seeded, reproducible, and restored afterwards.
	@test !_OPE._ESTIMATION_SEED_ACTIVE[]
	draws = _OPE._with_estimation_seed(7) do
		@test _OPE._ESTIMATION_SEED_ACTIVE[]
		rand(3)
	end
	@test copy(rng) == before
	@test !_OPE._ESTIMATION_SEED_ACTIVE[]
	@test draws == _OPE._with_estimation_seed(() -> rand(3), 7)
	@test draws != _OPE._with_estimation_seed(() -> rand(3), 8)
	@test draws == rand(Random.Xoshiro(7), 3)

	# The caller's stream is also restored when the body throws.
	@test_throws ErrorException _OPE._with_estimation_seed(() -> error("stop"), 7)
	@test copy(rng) == before

	# Only the outermost entry point seeds: a nested one continues the stream.
	@test _OPE._estimation_seed_pending((; seed = 7))
	@test !_OPE._estimation_seed_pending((; seed = nothing))
	nested = _OPE._with_estimation_seed(7) do
		@test !_OPE._estimation_seed_pending((; seed = 99))
		(rand(), rand())
	end
	@test nested == _OPE._with_estimation_seed(() -> (rand(), rand()), 7)

	# Tasks forked inside a seeded scope are reproducible too.
	forked() = _OPE._with_estimation_seed(() -> fetch(Threads.@spawn rand(3)), 7)
	@test forked() == forked()
	@test copy(rng) == before

	# Synthetic noise has its own stream, and no seed means the caller's stream.
	@test _OPE._with_noise_seed(() -> rand(3), 7) == _OPE._with_noise_seed(() -> rand(3), 7)
	@test _OPE._with_noise_seed(() -> rand(3), 7) != draws
	@test copy(rng) == before
	@test _OPE._with_noise_seed(() -> rand(3), nothing) == rand(copy(before), 3)
	@test copy(rng) != before
end
