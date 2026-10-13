# Test: polish_maxtime is enforced for residual-mode polish (PolishLSOBoundedLog,
# PolishFastLMBoundedLog). Pre-fix this was a no-op; post-fix the polish wall-
# clock is bounded by `maxtime + small slack`.
#
# Run: julia --startup-file=no /home/orebas/.julia/dev/ODEParameterEstimation/repro/test_polish_maxtime.jl

using ODEParameterEstimation
using ODEParameterEstimation.Examples
using ODEParameterEstimation: _build_polish_context, _polish_single_from_context,
    sample_problem_data, merge_options
using Test
using Printf

function build_context_for_simple(; polish_method, maxiters, datasize)
    base_opts = EstimationOptions(
        datasize = datasize,
        noise_level = 0.0,
        system_solver = SolverHC,
        flow = FlowStandard,
        use_si_template = true,
        shooting_points = 0,
        nooutput = true,
        diagnostics = false,
        save_system = false,
        use_parameter_homotopy = false,
        polish_solver_solutions = false,
        polish_solutions = false,
    )
    pep = sample_problem_data(simple(), base_opts)
    p_size = length(pep.ic) + length(pep.p_true)
    polish_opts = merge_options(
        base_opts;
        polish_method = polish_method,
        opt_lb = fill(1e-3, p_size),
        opt_ub = fill(50.0, p_size),
        polish_maxiters = maxiters,
        # Production-tight tolerances so LSO doesn't converge in microseconds and
        # we actually exercise the wall-clock cap.
        abstol = 1e-12,
        reltol = 1e-12,
    )
    return pep, _build_polish_context(pep; opts = polish_opts)
end

function run_polish(ctx, p_seed; polish_method, maxiters, maxtime)
    t0 = time()
    polished, _ = _polish_single_from_context(
        ctx, p_seed;
        polish_method = polish_method,
        maxiters = maxiters,
        maxtime = maxtime,
    )
    elapsed = time() - t0
    return polished, elapsed
end

function bad_seed(pep)
    # Seed far from truth so the LM trajectory has work to do but bounded box keeps it sane.
    p_true_vec = vcat(Float64.(collect(values(pep.ic))), Float64.(collect(values(pep.p_true))))
    return clamp.(0.5 .* p_true_vec, 1e-3, 50.0)
end

@testset "polish maxtime enforcement" begin
    # Use a moderately large datasize so each ODE solve isn't trivially fast.
    datasize = 201

    # JIT warmup. The very first Dual-typed ODE solve in a Julia process can take
    # 20-30 s as the AutoVern9(Rodas5P()) algorithm compiles for ForwardDiff
    # `Dual{...}` element types. That cost dwarfs any reasonable maxtime; the
    # *fix* enforces maxtime correctly, but the warmup bound on cold start is
    # JIT, not maxtime. In production this is amortized over the polish batch
    # (one cold call per Julia process), so the test forces a warmup polish first
    # so the wall-clock assertion below measures the steady-state path.
    let
        pep_w, ctx_w = build_context_for_simple(
            polish_method = PolishLSOBoundedLog,
            maxiters = 1,
            datasize = datasize,
        )
        run_polish(
            ctx_w, bad_seed(pep_w);
            polish_method = PolishLSOBoundedLog,
            maxiters = 1,
            maxtime = 600.0,  # generous so JIT fits inside
        )
    end

    @testset "LSO: maxtime caps wall-clock" begin
        pep, ctx = build_context_for_simple(
            polish_method = PolishLSOBoundedLog,
            maxiters = 100_000,
            datasize = datasize,
        )
        p_seed = bad_seed(pep)

        # Untimed reference: with maxtime=Inf, polish runs to natural completion or
        # maxiters. We don't actually run this — just measure with a short cap.

        maxtime_cap = 2.0
        polished, elapsed = run_polish(
            ctx, p_seed;
            polish_method = PolishLSOBoundedLog,
            maxiters = 100_000,
            maxtime = maxtime_cap,
        )
        @printf("LSO cap=%.1fs elapsed=%.2fs err=%.3g notes=%s\n",
            maxtime_cap, elapsed, polished.err, polished.provenance.notes)

        # Allow generous slack: deadline check happens at the START of each
        # residual!. The current call may take an additional residual evaluation
        # (~ODE solve) plus the throw + revert-guard final-eval to wind down.
        # Slack accounts for: one ODE solve up to the deadline (`unstable_check` cuts
        # it on the next step), plus the post-solve revert-guard residual eval.
        @test elapsed <= maxtime_cap + 4.0
        @test isfinite(polished.err)
        # If LSO converged before the cap fired, that's fine — the test only fails
        # if the cap was supposed to fire and didn't bound runtime.
        # We *expect* the cap to fire on this hard tolerance, so check the note
        # exists as a sanity signal (not a strict requirement).
        if elapsed > maxtime_cap * 0.5
            @test :polish_maxtime_exceeded in polished.provenance.notes
        end
    end

    @testset "FastLM: maxtime caps wall-clock" begin
        pep, ctx = build_context_for_simple(
            polish_method = PolishFastLMBoundedLog,
            maxiters = 100_000,
            datasize = datasize,
        )
        p_seed = bad_seed(pep)

        maxtime_cap = 2.0
        polished, elapsed = run_polish(
            ctx, p_seed;
            polish_method = PolishFastLMBoundedLog,
            maxiters = 100_000,
            maxtime = maxtime_cap,
        )
        @printf("FastLM cap=%.1fs elapsed=%.2fs err=%.3g notes=%s\n",
            maxtime_cap, elapsed, polished.err, polished.provenance.notes)

        # Slack accounts for: one ODE solve up to the deadline (`unstable_check` cuts
        # it on the next step), plus the post-solve revert-guard residual eval.
        @test elapsed <= maxtime_cap + 4.0
        @test isfinite(polished.err)
    end

    @testset "LSO: maxtime=Inf (disabled) still works" begin
        # Confirm the change is transparent when maxtime is effectively disabled —
        # the polish should still converge on this clean simple() problem.
        pep, ctx = build_context_for_simple(
            polish_method = PolishLSOBoundedLog,
            maxiters = 200,
            datasize = 51,
        )
        p_true_vec = vcat(Float64.(collect(values(pep.ic))), Float64.(collect(values(pep.p_true))))

        polished, elapsed = run_polish(
            ctx, p_true_vec;
            polish_method = PolishLSOBoundedLog,
            maxiters = 200,
            maxtime = Inf,
        )
        @printf("LSO Inf-cap (warm seed) elapsed=%.2fs err=%.3g\n", elapsed, polished.err)
        @test isfinite(polished.err)
        @test polished.err < 1e-6
        @test !(:polish_maxtime_exceeded in polished.provenance.notes)
    end
end
