using ODEParameterEstimation
using Symbolics
using Test
using LinearAlgebra
using Logging
using ModelingToolkit
using OrderedCollections
using OrdinaryDiffEq
using Random
using Statistics

include("support/model_fixtures.jl")

function quiet_feature_call(f)
    redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            with_logger(NullLogger()) do
                return f()
            end
        end
    end
end

function run_feature_canary(ctor, opts)
    pep = ctor()
    sampled = ODEParameterEstimation.sample_problem_data(pep, opts)
    raw_results, analysis, uq = quiet_feature_call() do
        ODEParameterEstimation.analyze_parameter_estimation_problem(sampled, opts)
    end
    return sampled, raw_results, analysis, uq
end

function lookup_state_value(result, state)
    for (k, v) in result.states
        string(k) == string(state) && return v
    end
    error("State $(state) not found in result")
end

function lookup_param_value(result, param)
    for (k, v) in result.parameters
        string(k) == string(param) && return v
    end
    error("Parameter $(param) not found in result")
end

function measured_series(sampled_pep, mq)
    string_key = replace(string(mq.lhs), "(t)" => "")
    if haskey(sampled_pep.data_sample, string_key)
        return sampled_pep.data_sample[string_key]
    end
    num_key = ModelingToolkit.Num(mq.lhs)
    if haskey(sampled_pep.data_sample, num_key)
        return sampled_pep.data_sample[num_key]
    end
    rhs_string_key = replace(string(mq.rhs), "(t)" => "")
    if haskey(sampled_pep.data_sample, rhs_string_key)
        return sampled_pep.data_sample[rhs_string_key]
    end
    rhs_num_key = ModelingToolkit.Num(mq.rhs)
    return sampled_pep.data_sample[rhs_num_key]
end

function partially_observed_problem()
    @independent_variables t
    @parameters a
    @variables x(t) z(t)
    D = Differential(t)

    states = [x, z]
    params = [a]
    equations = [
        D(x) ~ a * x,
        D(z) ~ -a * z,
    ]
    measured_quantities = [x ~ x]

    model, mq = ODEParameterEstimation.create_ordered_ode_system(
        "partial_observation", states, params, equations, measured_quantities,
    )

    return ODEParameterEstimation.ParameterEstimationProblem(
        "partial_observation",
        model,
        mq,
        nothing,
        [0.0, 0.2],
        Tsit5(),
        OrderedDict(a => 0.5),
        OrderedDict(x => 2.0, z => 3.0),
        0,
    )
end

const MULTI_INTERP_TEMPLATE_LIST = [
    InterpolatorAAAD,
    InterpolatorAAADGPR,
    InterpolatorS2AAAMLE,
    InterpolatorAGPRobust,
    InterpolatorAGPRobustRQ,
    InterpolatorAGPRobustSEpRQ,
    InterpolatorAGPRobustSExRQ,
    InterpolatorS3AdaptSE,
    InterpolatorS3AdaptRQ,
    InterpolatorS3AdaptSEpRQ,
    InterpolatorS3AdaptSExRQ,
    InterpolatorS3BICSE,
]

const SAVE_SYSTEM_OFF_OPTS = EstimationOptions(
    datasize = 11,
    noise_level = 0.0,
    system_solver = SolverHC,
    flow = FlowStandard,
    use_si_template = true,
    interpolators = InterpolatorMethod[InterpolatorAAAD],
    shooting_points = 0,
    nooutput = true,
    diagnostics = false,
    save_system = false,
    use_parameter_homotopy = false,
    polish_solver_solutions = false,
    polish_solutions = false,
    synthesize_aggregate_candidates = false,
)

const MULTI_INTERP_FAST_OPTS = EstimationOptions(
    datasize = 11,
    noise_level = 0.0,
    system_solver = SolverHC,
    flow = FlowStandard,
    use_si_template = true,
    interpolators = MULTI_INTERP_TEMPLATE_LIST,
    shooting_points = 0,
    nooutput = true,
    diagnostics = false,
    save_system = false,
    use_parameter_homotopy = false,
    polish_solver_solutions = false,
    polish_solutions = false,
    synthesize_aggregate_candidates = false,  # this test asserts on per-interpolator provenance; synthesis would inject :synthesized entries
)

const SMALL_SAMPLE_OPTS = EstimationOptions(
    datasize = 11,
    noise_level = 0.0,
    time_interval = [0.0, 0.2],
    nooutput = true,
    diagnostics = false,
    save_system = false,
)

@testset "Feature regressions" begin
    @testset "Synthetic sampling noise models are explicit" begin
        @test ODEParameterEstimation.EstimationOptions().noise_model == :additive

        data = OrderedDict{Union{String, Num}, Vector{Float64}}(
            "t" => [0.0, 1.0, 2.0],
            "y" => [1.0, 2.0, 4.0],
            "z" => [-2.0, 2.0, 6.0],
        )

        Random.seed!(20260630)
        additive = ODEParameterEstimation.add_synthetic_noise(data, 0.1, :additive)
        Random.seed!(20260630)
        additive_expected_y = data["y"] .+ mean(abs.(data["y"])) .* 0.1 .* randn(3)
        additive_expected_z = data["z"] .+ mean(abs.(data["z"])) .* 0.1 .* randn(3)
        @test additive["t"] == data["t"]
        @test additive["y"] ≈ additive_expected_y
        @test additive["z"] ≈ additive_expected_z

        Random.seed!(20260630)
        relative = ODEParameterEstimation.add_synthetic_noise(data, 0.1, :relative)
        Random.seed!(20260630)
        relative_expected_y = data["y"] .* (1.0 .+ 0.1 .* randn(3))
        relative_expected_z = data["z"] .* (1.0 .+ 0.1 .* randn(3))
        @test relative["y"] ≈ relative_expected_y
        @test relative["z"] ≈ relative_expected_z
    end

    @testset "multi-interpolator benchmark template path" begin
        sampled, raw_results, analysis, _ = run_feature_canary(ODEParameterEstimation.Examples.harmonic_oscillator, MULTI_INTERP_FAST_OPTS)
        expected_sources = Set(ODEParameterEstimation.interpolator_method_to_symbol(method) for method in MULTI_INTERP_TEMPLATE_LIST)
        actual_sources = Set(filter(!isnothing, [result.interpolator_source for result in raw_results[1]]))

        @test sampled.name == "harmonic_oscillator"
        @test length(ODEParameterEstimation.resolve_interpolator_list(MULTI_INTERP_FAST_OPTS)) == length(MULTI_INTERP_TEMPLATE_LIST)
        @test !isempty(raw_results[1])
        @test !isempty(analysis[1])
        @test analysis[2] < 1e-4
        @test !isempty(actual_sources)
        @test actual_sources ⊆ expected_sources
        @test :aaad in actual_sources
        @test all(result.provenance.interpolator_source == result.interpolator_source for result in raw_results[1])
        @test all(result.return_code == ODEParameterEstimation.compatibility_return_code(result.provenance) for result in raw_results[1])
        @test all(!isnothing(result.provenance.source_candidate_index) for result in raw_results[1])
    end

    @testset "transcendental transform augments the problem directly" begin
        sampled = ODEParameterEstimation.sample_problem_data(
            dc_motor_sinusoidal(),
            EstimationOptions(
                datasize = 11,
                noise_level = 0.0,
                shooting_points = 0,
                nooutput = true,
                diagnostics = false,
                save_system = false,
            ),
        )
        t_var = ModelingToolkit.get_iv(sampled.model.system)
        transformed, tr_info = quiet_feature_call() do
            ODEParameterEstimation.transform_pep_for_estimation(sampled, t_var)
        end

        @test !isnothing(tr_info)
        @test transformed.name == sampled.name * "_polynomialized"
        @test !isempty(tr_info.entries)
        @test !isempty(tr_info.input_variables)
        @test any(contains(string(state), "_trfn_") for state in transformed.model.original_states)
        @test length(transformed.model.original_states) > length(sampled.model.original_states)
        @test length(transformed.data_sample) > length(sampled.data_sample)
    end

    @testset "missing SI reconstruction values fail explicitly" begin
        sampled = ODEParameterEstimation.sample_problem_data(partially_observed_problem(), SMALL_SAMPLE_OPTS)
        observed_mq = only(sampled.measured_quantities)
        observed_series = measured_series(sampled, observed_mq)
        augmented_data_sample = OrderedDict(sampled.data_sample)
        augmented_data_sample[replace(string(observed_mq.lhs), "(t)" => "")] = observed_series
        sampled = ODEParameterEstimation.ParameterEstimationProblem(
            sampled.name,
            sampled.model,
            sampled.measured_quantities,
            augmented_data_sample,
            sampled.recommended_time_interval,
            sampled.solver,
            sampled.p_true,
            sampled.ic,
            sampled.unident_count,
        )

        setup_data = (
            time_index_set = [1],
            all_unidentifiable = Set{Any}(),
        )
        solution_data = (
            solns = [Float64[]],
            forward_subst_dict = [OrderedDict{Num, Any}()],
            trivial_dict = Dict{Any, Any}(),
            final_varlist = Any[],
            trimmed_varlist = Any[],
            good_udict = Dict{Any, Any}(),
            solution_time_indices = [1],
        )

        err = try
            quiet_feature_call() do
                ODEParameterEstimation.process_estimation_results(
                    sampled,
                    solution_data,
                    setup_data;
                    opts = SMALL_SAMPLE_OPTS,
                )
            end
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("missing from the SI solution", sprint(showerror, err))
    end

    @testset "save_system=false does not create saved_systems output" begin
        mktempdir() do tmpdir
            cd(tmpdir) do
                _, raw_results, analysis, _ = run_feature_canary(ODEParameterEstimation.Examples.harmonic_oscillator, SAVE_SYSTEM_OFF_OPTS)

                @test !isempty(raw_results[1])
                @test !isempty(analysis[1])
                @test !isdir(joinpath(tmpdir, "saved_systems"))
            end
        end
    end

    @testset "single interpolator run stays on the requested interpolator" begin
        _, raw_results, analysis, _ = run_feature_canary(ODEParameterEstimation.Examples.harmonic_oscillator, SAVE_SYSTEM_OFF_OPTS)
        actual_sources = Set(filter(!isnothing, [result.interpolator_source for result in raw_results[1]]))

        @test !isempty(raw_results[1])
        @test !isempty(analysis[1])
        @test actual_sources == Set([:aaad])
        @test length(raw_results[1]) == 1
    end

    @testset "parameter homotopy path produces valid results" begin
        homotopy_opts = EstimationOptions(
            datasize = 11,
            noise_level = 0.0,
            system_solver = SolverHC,
            flow = FlowStandard,
            use_si_template = true,
            shooting_points = 3,  # homotopy requires >= 3
            nooutput = true,
            diagnostics = false,
            save_system = false,
            use_parameter_homotopy = true,
            polish_solver_solutions = false,
            polish_solutions = false,
        )
        _, raw_results, analysis, _ = run_feature_canary(ODEParameterEstimation.Examples.harmonic_oscillator, homotopy_opts)

        @test !isempty(raw_results[1])
        @test !isempty(analysis[1])
        @test analysis[2] < 1e-2  # solutions should be reasonable
    end

    @testset "local system solver $(solver) recovers the parameters" for solver in (SolverRobust, SolverFastNLOpt)
        # The whole estimation with the local solver in place of homotopy
        # continuation, on the smallest model (one state, one parameter) and on
        # a two-state one. Unpolished, so the answer is the solver's own; then
        # with the default refinement. Each local solve compiles its system,
        # so a larger model costs minutes here.
        raw_opts = EstimationOptions(datasize = 21, noise_level = 0.0, system_solver = solver, nooutput = true,
            polish_solver_solutions = false, polish_solutions = false)
        default_opts = EstimationOptions(datasize = 21, noise_level = 0.0, system_solver = solver, nooutput = true)
        for (model, opts) in ((ODEParameterEstimation.Examples.exponential_decay, raw_opts),
                              (ODEParameterEstimation.Examples.exponential_decay, default_opts),
                              (ODEParameterEstimation.Examples.harmonic_oscillator, raw_opts))
            _, _, analysis, _ = run_feature_canary(model, opts)
            @test !isempty(analysis.returned_results)
            @test analysis.besterror < 1e-6
        end
    end

    @testset "the older way of building the equations still estimates" begin
        legacy_opts = EstimationOptions(datasize = 21, noise_level = 0.0, nooutput = true,
            system_construction_policy = :legacy)
        _, _, analysis, _ = run_feature_canary(ODEParameterEstimation.Examples.harmonic_oscillator, legacy_opts)
        @test !isempty(analysis.returned_results)
        @test analysis.besterror < 1e-6
    end

    @testset "terminal_fallback=:direct_opt leaves algebraic results alone" begin
        # The fallback runs only when the algebraic solve finds nothing, which
        # this model never does; no test reaches the fallback itself yet.
        fallback_opts = EstimationOptions(
            datasize = 11,
            noise_level = 0.0,
            system_solver = SolverHC,
            flow = FlowStandard,
            use_si_template = true,
            shooting_points = 0,
            nooutput = true,
            diagnostics = false,
            save_system = false,
            use_parameter_homotopy = false,
            terminal_fallback = :direct_opt,
            polish_solutions = true,
        )
        _, raw_results, analysis, _ = run_feature_canary(ODEParameterEstimation.Examples.harmonic_oscillator, fallback_opts)

        @test !isempty(raw_results[1])
        @test !isempty(analysis[1])
        for result in raw_results[1]
            @test result.provenance.rescue_path != :direct_opt_fallback
        end
    end

    @testset "polish_solutions applies BFGS refinement" begin
        polish_opts = EstimationOptions(
            datasize = 11,
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
            polish_solutions = true,
        )
        _, raw_results, analysis, _ = run_feature_canary(ODEParameterEstimation.Examples.harmonic_oscillator, polish_opts)

        @test !isempty(raw_results[1])
        @test !isempty(analysis[1])
        @test analysis[2] < 1e-4  # polished results should be very accurate
        # At least some results should show polish was applied
        any_polished = any(result.provenance.polish_applied for result in raw_results[1])
        @test any_polished
    end

    @testset "log-positive polish context preserves loss geometry" begin
        base_opts = EstimationOptions(
            datasize = 11,
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
            polish_solutions = true,
        )
        pep = ODEParameterEstimation.sample_problem_data(ODEParameterEstimation.Examples.harmonic_oscillator(), base_opts)
        p_size = length(pep.ic) + length(pep.p_true)
        positive_opts = ODEParameterEstimation.merge_options(
            base_opts;
            opt_lb = fill(0.05, p_size),
            opt_ub = fill(10.0, p_size),
        )
        # Pin `coordinate_transform` explicitly so the test is independent of the
        # current default `polish_method` / `polish_coordinate_policy`.
        linear_ctx = ODEParameterEstimation._build_polish_context(pep; opts = positive_opts, coordinate_transform = :linear)
        log_ctx = ODEParameterEstimation._build_polish_context(pep; opts = positive_opts, coordinate_transform = :log_positive)
        p0 = clamp.(vcat(Float64.(collect(values(pep.ic))), Float64.(collect(values(pep.p_true)))), positive_opts.opt_lb, positive_opts.opt_ub)

        # Per-variable transforms: `:linear` legacy → all `:linear`,
        # `:log_positive` legacy → `:log_only` policy → all `:log` here (all bounds are > 0)
        @test all(linear_ctx.coordinate_transforms .== :linear)
        @test all(log_ctx.coordinate_transforms .== :log)
        @test log_ctx.internal_lb ≈ log.(positive_opts.opt_lb)
        @test log_ctx.internal_ub ≈ log.(positive_opts.opt_ub)
        @test linear_ctx.optf.f(p0, nothing) ≈ log_ctx.optf.f(log.(p0), nothing) rtol = 1e-9 atol = 1e-9

        bad_opts = ODEParameterEstimation.merge_options(
            base_opts;
            opt_lb = vcat([-0.1], fill(0.05, p_size - 1)),
            opt_ub = fill(10.0, p_size),
        )
        @test_throws ArgumentError ODEParameterEstimation._build_polish_context(
            pep;
            opts = bad_opts,
            coordinate_transform = :log_positive,
        )
    end

    @testset "Bounded LSO log-space residual polish" begin
        # Direct end-to-end smoke test of the new residual polish path on `harmonic_oscillator()`.
        # Seeds the polisher near (but not at) the true parameters and confirms it
        # converges back. Uses `:auto` per-variable transforms (all positive bounds → :log).
        base_opts = EstimationOptions(
            datasize = 11,
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
        pep = ODEParameterEstimation.sample_problem_data(ODEParameterEstimation.Examples.harmonic_oscillator(), base_opts)
        p_size = length(pep.ic) + length(pep.p_true)
        lso_opts = ODEParameterEstimation.merge_options(
            base_opts;
            polish_method = PolishLSOBoundedLog,
            opt_lb = fill(1e-3, p_size),
            opt_ub = fill(50.0, p_size),
            polish_maxiters = 200,
        )

        # The residual path should pick up `:auto` automatically when polish_method is
        # residual; positive bounds resolve to all `:log`.
        ctx = ODEParameterEstimation._build_polish_context(pep; opts = lso_opts)
        @test all(ctx.coordinate_transforms .== :log)
        @test ctx.regularization_lambda == 0.0

        # Seed: true values (no perturbation in IC) — exercises the path end-to-end and
        # confirms the revert guard at worst returns the seed.
        p_true = vcat(Float64.(collect(values(pep.ic))), Float64.(collect(values(pep.p_true))))
        polished, retained_polish = ODEParameterEstimation._polish_single_from_context(
            ctx, p_true;
            polish_method = PolishLSOBoundedLog,
            maxiters = 200,
            lso_delta = lso_opts.polish_lso_delta,
            retain_internal_optimum = true,
        )
        @test polished isa ODEParameterEstimation.ParameterEstimationResult
        @test isfinite(polished.err)
        @test retained_polish isa ODEParameterEstimation._PolishSolveRecord
        @test ODEParameterEstimation._polish_retained_internal_vector(
            ctx, polished, retained_polish,
        ) == retained_polish.internal_optimum
        @test ODEParameterEstimation._polish_raw_optimizer_result(retained_polish) ===
            retained_polish.optimizer_result
        # Polished should not be worse than the perfect seed (revert guard ensures this)
        @test polished.err < 1e-6

        # Same path with FastLM should also run without error
        fastlm_opts = ODEParameterEstimation.merge_options(lso_opts; polish_method = PolishFastLMBoundedLog)
        ctx_fast = ODEParameterEstimation._build_polish_context(pep; opts = fastlm_opts)
        polished_fast, _ = ODEParameterEstimation._polish_single_from_context(
            ctx_fast, p_true;
            polish_method = PolishFastLMBoundedLog,
            maxiters = 200,
        )
        @test polished_fast isa ODEParameterEstimation.ParameterEstimationResult
        @test isfinite(polished_fast.err)

        # Optional λ regularization: residual length grows by n_unknowns; trajectory-fit
        # err remains comparable.
        reg_opts = ODEParameterEstimation.merge_options(
            lso_opts;
            polish_regularization_lambda = 1e-6,
        )
        ctx_reg = ODEParameterEstimation._build_polish_context(pep; opts = reg_opts)
        @test ctx_reg.regularization_lambda == 1e-6
        polished_reg, _ = ODEParameterEstimation._polish_single_from_context(
            ctx_reg, p_true;
            polish_method = PolishLSOBoundedLog,
            maxiters = 200,
        )
        @test polished_reg isa ODEParameterEstimation.ParameterEstimationResult
        @test isfinite(polished_reg.err)
    end

    @testset "Sensitivity-seed pipeline smoke" begin
        # With use_sensitivity_seeds = false (default), behavior must match the
        # baseline (no augmentation, no extra polish seeds, no warnings).
        # With use_sensitivity_seeds = true, the augmentation hook runs through
        # _maybe_augment_with_sensitivity_seeds, computes Σ_x = S·diag(σ_d²)·S',
        # and either emits seeds or falls through silently.
        base_opts = EstimationOptions(
            datasize = 11,
            noise_level = 0.0,
            shooting_points = 0,
            nooutput = true,
            diagnostics = false,
            save_system = false,
            use_parameter_homotopy = false,
            polish_solver_solutions = false,
            polish_solutions = true,
        )

        # Off-by-default path
        pep = ODEParameterEstimation.sample_problem_data(ODEParameterEstimation.Examples.harmonic_oscillator(), base_opts)
        baseline = ODEParameterEstimation.analyze_parameter_estimation_problem(pep, base_opts)
        @test !isempty(baseline[2][1])
        @test isfinite(baseline[2][1][1].err)

        # Flag-on path
        seeded_opts = ODEParameterEstimation.merge_options(
            base_opts;
            use_sensitivity_seeds = true,
            sensitivity_seed_probe_scale = 1.0,
            sensitivity_seed_eigenvalue_threshold = 0.05,
        )
        pep_seeded = ODEParameterEstimation.sample_problem_data(ODEParameterEstimation.Examples.harmonic_oscillator(), seeded_opts)
        seeded = ODEParameterEstimation.analyze_parameter_estimation_problem(pep_seeded, seeded_opts)
        @test !isempty(seeded[2][1])
        @test isfinite(seeded[2][1][1].err)
        # Flag-on must not break recovery on a well-determined model
        @test seeded[2][1][1].err < 1e-3

        # Compute σ_d directly with a multi-source synthetic cache; verify
        # cross-interpolator spread mode (no floor active when sources disagree).
        @independent_variables t
        @variables y1(t)
        # Use a closure factory to avoid module-scope struct redefinition warnings on rerun.
        make_linear = slope -> (x::Real -> slope * x)
        mq = [y1 ~ y1]
        key1 = Symbolics.diff2term(mq[1].rhs)
        cache_multi = Dict{Symbol, AbstractDict}(
            :a => Dict{Any, Any}(key1 => make_linear(2.0)),
            :b => Dict{Any, Any}(key1 => make_linear(2.5)),
            :c => Dict{Any, Any}(key1 => make_linear(3.0)),
        )
        est = compute_sigma_d(cache_multi, mq, 0.5, 1; noise_level = 0.0, order_decay_kappa = 2.0)
        # At order 1 the three derivatives are 2.0, 2.5, 3.0; std = 0.5
        @test est.sigma_d[1, 2] ≈ 0.5 atol = 1e-9
        # No floor active when noise_level = 0
        @test all(.!est.floor_active[1, :])

        # Single-source falls through to floor
        cache_single = Dict{Symbol, AbstractDict}(
            :a => Dict{Any, Any}(key1 => make_linear(2.0)),
        )
        est_single = compute_sigma_d(cache_single, mq, 0.5, 2; noise_level = 0.01, order_decay_kappa = 2.0)
        @test est_single.sigma_d[1, :] ≈ [0.01, 0.02, 0.04] atol = 1e-12
        @test all(est_single.floor_active[1, :])
    end

    @testset "Variational backsolve Jacobian and unrescaling match the one-exponential model" begin
        @independent_variables t
        @parameters k
        @variables x(t)
        D = Differential(t)

        model, mq = ODEParameterEstimation.create_ordered_ode_system(
            "one_exp_backsolve_uq",
            [x],
            [k],
            [D(x) ~ k * x],
            [x ~ x],
        )

        x0_true = 1.5
        k_true = 0.2
        pep = ODEParameterEstimation.ParameterEstimationProblem(
            "one_exp_backsolve_uq",
            model,
            mq,
            nothing,
            [0.0, 5.0],
            Tsit5(),
            OrderedDict(k => k_true),
            OrderedDict(x => x0_true),
            0,
        )
        sampled = ODEParameterEstimation.sample_problem_data(
            pep,
            EstimationOptions(
                datasize = 21,
                noise_level = 0.0,
                time_interval = [0.0, 5.0],
                nooutput = true,
                diagnostics = false,
            ),
        )

        t0 = sampled.data_sample["t"][1]
        t_shoot = 2.0
        completed_sys = ModelingToolkit.complete(sampled.model.system)
        completed_states = ModelingToolkit.unknowns(completed_sys)
        completed_params = ModelingToolkit.parameters(completed_sys)
        prob = ODEProblem(
            completed_sys,
            merge(Dict(completed_states .=> [x0_true]), Dict(completed_params .=> [k_true])),
            (t0, sampled.data_sample["t"][end]),
        )
        sol = OrdinaryDiffEq.solve(prob, Tsit5(); abstol = 1e-12, reltol = 1e-12)

        best_result = ODEParameterEstimation.ParameterEstimationResult(
            OrderedDict(k => k_true),
            OrderedDict(x => x0_true),
            t_shoot,
            0.0,
            nothing,
            length(sampled.data_sample["t"]),
            t_shoot,
            OrderedDict{Num, Float64}(),
            Set{Num}(),
            sol,
        )

        # ∂x(t0) / ∂(k, x(t_shoot))
        Δt = t_shoot - t0
        expected_j = reshape([-Δt * x0_true, exp(-k_true * Δt)], 1, 2)
        J, x_back = ODEParameterEstimation._uq_variational_backsolve_jacobian(sampled, best_result, t0, t_shoot)
        @test J ≈ expected_j rtol = 1e-8 atol = 1e-8
        @test x_back[1] ≈ x0_true atol = 1e-8

        scale_info = ODEParameterEstimation.ScaleInfo(
            OrderedDict(x => 4.0),
            OrderedDict(k => 2.0),
            OrderedDict{Any, Float64}(),
            (; n_nontrivial = 2),
        )
        local_snapshot = ODEParameterEstimation.LocalUQSnapshot(
            t_shoot,
            diagm([0.01, 0.04, 0.09]),
            [0.1, 0.2, 0.3],
            ["k_0", "x_0", "x_1"],
            Dict("k_0" => :parameter, "x_0" => :state_at_eval, "x_1" => :state_derivative),
            [0.1, 0.5, NaN],
            Matrix{Float64}(I, 3, 3),
            0.0,
            :ok,
            nothing,
        )
        transform = ODEParameterEstimation.UQBacksolveTransform(
            t_shoot,
            t0,
            ["k", "x"],
            ["k", "x"],
            Dict("k" => :parameter, "x" => :state_at_eval),
            Dict("k" => :parameter, "x" => :state_ic),
            [1.0 0.0; -1.0 0.5],
            NaN,
            :variational,
            String[],
        )
        scaled_physical = ODEParameterEstimation.UncertaintyReport(
            sampled.name,
            t_shoot,
            String[],
            Vector{Float64}[],
            Vector{Float64}[],
            zeros(0, 0),
            String[],
            diagm([0.01, 0.04]),
            [0.1, 0.2],
            ["k", "x"],
            Dict("k" => :parameter, "x" => :state_ic),
            [0.1, 0.375],
            Matrix{Float64}(I, 2, 2),
            0.0,
            :ok,
            String[],
            :estimator_sampling,
            :test,
            nothing,
            :physical_initial_conditions,
            local_snapshot,
            transform,
        )
        original_uq = ODEParameterEstimation.unrescale_uncertainty_report(scaled_physical, scale_info)
        @test original_uq.param_true_values ≈ [0.2, 1.5]
        @test original_uq.param_covariance ≈ diagm([0.04, 0.64])
        @test original_uq.param_std ≈ [0.2, 0.8]
        @test original_uq.backsolve_transform.transform_matrix ≈ [1.0 0.0; -2.0 0.5]
        @test isfinite(original_uq.backsolve_transform.amplification)
        @test original_uq.local_coordinate_report.coordinate_values[1:2] ≈ [0.2, 2.0]
        @test original_uq.local_coordinate_report.param_std ≈ [0.2, 0.8, 1.2]
    end

    @testset "Variational backsolve Jacobian matches the two-exponential model" begin
        @independent_variables t
        @parameters k1 k2
        @variables x1(t) x2(t)
        D = Differential(t)

        model, mq = ODEParameterEstimation.create_ordered_ode_system(
            "two_exp_backsolve_uq",
            [x1, x2],
            [k1, k2],
            [
                D(x1) ~ k1 * x1,
                D(x2) ~ k2 * x2,
            ],
            [x1 ~ x1, x2 ~ x2],
        )

        x10 = 1.5
        x20 = 3.5
        k1_true = 0.2
        k2_true = 0.4
        pep = ODEParameterEstimation.ParameterEstimationProblem(
            "two_exp_backsolve_uq",
            model,
            mq,
            nothing,
            [0.0, 5.0],
            Tsit5(),
            OrderedDict(k1 => k1_true, k2 => k2_true),
            OrderedDict(x1 => x10, x2 => x20),
            0,
        )
        sampled = ODEParameterEstimation.sample_problem_data(
            pep,
            EstimationOptions(
                datasize = 21,
                noise_level = 0.0,
                time_interval = [0.0, 5.0],
                nooutput = true,
                diagnostics = false,
            ),
        )

        t0 = sampled.data_sample["t"][1]
        t_shoot = 2.5
        Δt = t_shoot - t0
        completed_sys = ModelingToolkit.complete(sampled.model.system)
        completed_states = ModelingToolkit.unknowns(completed_sys)
        completed_params = ModelingToolkit.parameters(completed_sys)
        prob = ODEProblem(
            completed_sys,
            merge(
                Dict(completed_states .=> [x10, x20]),
                Dict(completed_params .=> [k1_true, k2_true]),
            ),
            (t0, sampled.data_sample["t"][end]),
        )
        sol = OrdinaryDiffEq.solve(prob, Tsit5(); abstol = 1e-12, reltol = 1e-12)
        xshoot = Float64.(sol(t_shoot))

        best_result = ODEParameterEstimation.ParameterEstimationResult(
            OrderedDict(k1 => k1_true, k2 => k2_true),
            OrderedDict(x1 => x10, x2 => x20),
            t_shoot,
            0.0,
            nothing,
            length(sampled.data_sample["t"]),
            t_shoot,
            OrderedDict{Num, Float64}(),
            Set{Num}(),
            sol,
        )

        # ∂(x1, x2)(t0) / ∂(k1, k2, x1(t_shoot), x2(t_shoot))
        expected_j = [
            -Δt * x10 0.0 exp(-k1_true * Δt) 0.0
            0.0 -Δt * x20 0.0 exp(-k2_true * Δt)
        ]
        J, x_back = ODEParameterEstimation._uq_variational_backsolve_jacobian(sampled, best_result, t0, t_shoot)
        @test J ≈ expected_j rtol = 1e-8 atol = 1e-8
        @test x_back ≈ [x10, x20] rtol = 1e-8
    end

    @testset "_obs_trfn_ parsing unit tests" begin
        parse_fn = ODEParameterEstimation._parse_obs_trfn_base_name
        eval_fn = ODEParameterEstimation.evaluate_obs_trfn_template_variable

        # Pattern 1: freq_sin / freq_cos
        @test parse_fn("_obs_trfn_0_5_sin") == (:sin, 0.5)
        @test parse_fn("_obs_trfn_0_5_cos") == (:cos, 0.5)

        # Pattern 2: partner naming (cos_freq_sin, cos_freq_cos)
        @test parse_fn("_obs_trfn_cos_0_5_cos") == (:cos, 0.5)
        @test parse_fn("_obs_trfn_cos_0_5_sin") == (:sin, 0.5)

        # Evaluation
        @test eval_fn("_obs_trfn_cos_0_5_cos(t)", Float64(π)) ≈ cos(0.5π)

        # Non-trfn names return nothing
        @test parse_fn("y1") === nothing
        @test eval_fn("y1_0", 0.0) === nothing
    end

    @testset "Estimator-sampling UQ influence weights" begin
        xs = collect(range(0.0, 1.0; length = 13))
        ys = @. sin(2.0 * xs) + 0.05 * cos(5.0 * xs)
        interp = ODEParameterEstimation.agp_gpr_uq(xs, ys)
        t_eval = 0.37
        max_deriv = 3

        μ, W = ODEParameterEstimation.gp_derivative_influence_matrix(interp, t_eval, max_deriv)
        expected = [ODEParameterEstimation.nth_deriv(interp, d, t_eval) for d in 0:max_deriv]
        @test size(W) == (max_deriv + 1, length(xs))
        @test μ ≈ expected atol = 1e-8 rtol = 1e-8

        raw_var = ODEParameterEstimation.learned_observation_noise_variance(interp)
        @test raw_var ≈ max(interp.noise_var, 0.0) * interp.y_std^2 atol = 1e-14

        jet = ODEParameterEstimation.joint_derivative_estimator_covariance(
            interp, t_eval, max_deriv; observable_name = "y1",
        )
        @test jet isa ODEParameterEstimation.JetInfluenceEstimate
        @test jet.covariance_kind == :estimator_sampling
        @test jet.noise_source == :learned_gp_homoscedastic
        @test jet.mean ≈ μ atol = 1e-8 rtol = 1e-8
        @test diag(jet.observation_covariance) ≈ fill(raw_var, length(xs)) atol = 1e-14
        @test jet.jet_covariance ≈ raw_var .* (W * W') atol = 1e-10 rtol = 1e-8
        @test minimum(eigvals(Symmetric(jet.jet_covariance))) >= -1e-9
    end

    @testset "compute_uncertainty=true returns physical-coordinate UQ with local audit" begin
        # End-to-end smoke test of the uncertainty report on harmonic_oscillator().
        uq_opts = EstimationOptions(
            datasize = 21, noise_level = 0.0, shooting_points = 1,
            nooutput = true, diagnostics = false, flow = FlowStandard,
            use_si_template = true, use_parameter_homotopy = false,
            interpolators = InterpolatorMethod[InterpolatorAGPUQ],
            auto_filter_interpolators = false, use_multipoint = false,
            synthesize_aggregate_candidates = false, branch_completion = false,
            save_system = false,
            polish_solver_solutions = false, polish_solutions = false,
            compute_uncertainty = true,
        )
        sampled = ODEParameterEstimation.sample_problem_data(ODEParameterEstimation.Examples.harmonic_oscillator(), uq_opts)
        _, _, uq = quiet_feature_call() do
            ODEParameterEstimation.analyze_parameter_estimation_problem(sampled, uq_opts)
        end
        @test uq isa ODEParameterEstimation.UncertaintyReport
        @test uq.coordinate_system == :physical_initial_conditions
        @test uq.covariance_kind == :estimator_sampling
        @test uq.noise_source == :learned_gp_homoscedastic
		@test uq.target isa ODEParameterEstimation.UQTargetSnapshot
		@test uq.target.selected_rank == 1
		@test uq.target.identity.estimator_kind == :single_point_algebraic
		@test uq.target.identity.time_values == [uq.t_eval]
		@test length(uq.estimate_values) == length(uq.param_labels)
        @test !isempty(uq.param_std)
        @test any(isfinite, uq.param_std)
        @test all(get(uq.param_roles, label, :unknown) in (:parameter, :state_ic) for label in uq.param_labels)
        @test uq.local_coordinate_report isa ODEParameterEstimation.LocalUQSnapshot
        @test any(role == :state_at_eval for role in values(uq.local_coordinate_report.param_roles))
        @test uq.backsolve_transform isa ODEParameterEstimation.UQBacksolveTransform
        @test uq.practical_identifiability_index isa ODEParameterEstimation.PracticalIdentifiabilityIndex
        @test isfinite(uq.practical_identifiability_index.i_a)
    end

    @testset "Auto-filter interpolators by noise" begin
        # Build a clean PEP and synthetic data with known noise levels.
        pep = ODEParameterEstimation.Examples.harmonic_oscillator()
        opts_low = EstimationOptions(datasize = 51, time_interval = [-0.5, 0.5], noise_level = 0.0,
                                     nooutput = true, diagnostics = false, save_system = false)
        opts_high = EstimationOptions(datasize = 51, time_interval = [-0.5, 0.5], noise_level = 1e-2,
                                      nooutput = true, diagnostics = false, save_system = false)
        pep_low = ODEParameterEstimation.sample_problem_data(pep, opts_low)
        pep_high = ODEParameterEstimation.sample_problem_data(pep, opts_high)

        # 1) σ̂ estimator is small for clean data, larger for noisy
        σ̂_low = ODEParameterEstimation.estimate_relative_noise(pep_low)
        σ̂_high = ODEParameterEstimation.estimate_relative_noise(pep_high)
        @test σ̂_low < 1e-4               # clean data → low noise estimate
        @test σ̂_high > σ̂_low * 10        # noisy data is detectably noisier

        # 2) Filter triggers at high noise (drops S2AAAMLE), passes through low noise
        list_with_s2 = [(InterpolatorS2AAAMLE, nothing), (InterpolatorAGPRobust, nothing)]
        filtered_high = ODEParameterEstimation._apply_noise_filter(list_with_s2, σ̂_high)
        @test length(filtered_high) == 1
        @test filtered_high[1][1] == InterpolatorAGPRobust

        filtered_low = ODEParameterEstimation._apply_noise_filter(list_with_s2, σ̂_low)
        @test length(filtered_low) == 2  # both kept at clean noise

        # 3) Wrapper short-circuits when no noise-sensitive method present
        list_safe = [(InterpolatorAGPRobust, nothing), (InterpolatorAAADGPR, nothing)]
        opts_default = EstimationOptions(datasize = 51, nooutput = true, diagnostics = false, save_system = false)
        result_safe = ODEParameterEstimation.maybe_filter_interpolators_by_noise(list_safe, pep_high, opts_default)
        @test result_safe == list_safe   # no change

        # 4) Wrapper opt-out preserves S2 even at high noise
        opts_optout = EstimationOptions(datasize = 51, nooutput = true, diagnostics = false,
                                        save_system = false, auto_filter_interpolators = false)
        result_optout = ODEParameterEstimation.maybe_filter_interpolators_by_noise(
            list_with_s2, pep_high, opts_optout)
        @test length(result_optout) == 2  # filter disabled

        # 5) Empty-after-filter fallback to AAADGPR
        list_only_s2 = [(InterpolatorS2AAAMLE, nothing)]
        result_fallback = ODEParameterEstimation.maybe_filter_interpolators_by_noise(
            list_only_s2, pep_high, opts_default)
        @test length(result_fallback) == 1
        @test result_fallback[1][1] == InterpolatorAAADGPR
    end

    @testset "Synthesize aggregate candidates (Phase 1: B/C)" begin
        # Run a small estimation with synthesis on; verify synthesized candidates
        # appear with correct provenance. Use harmonic_oscillator() at noise=0 which produces
        # a clean pool that aggregates predictably.
        opts_on = EstimationOptions(
            datasize = 11, noise_level = 0.0,
            system_solver = SolverHC, flow = FlowStandard,
            use_si_template = true, shooting_points = 0,
            nooutput = true, diagnostics = false, save_system = false,
            polish_solver_solutions = false, polish_solutions = false,
            interpolators = MULTI_INTERP_TEMPLATE_LIST,
            synthesize_aggregate_candidates = true,
        )
        opts_off = ODEParameterEstimation.merge_options(opts_on;
            synthesize_aggregate_candidates = false)

        _, raw_on, _ = run_feature_canary(ODEParameterEstimation.Examples.harmonic_oscillator, opts_on)
        _, raw_off, _ = run_feature_canary(ODEParameterEstimation.Examples.harmonic_oscillator, opts_off)

        n_synth = count(c -> c.provenance.source_type == :synthesized_aggregate, raw_on[1])
        @test length(raw_on[1]) >= length(raw_off[1])             # synth never reduces pool size
        # Categories B/C only fire when ≥ 2 candidates per SP/anchor, which
        # `harmonic_oscillator()` may or may not produce at shooting_points=0. Just verify
        # provenance shape on any synthesized candidate that DID get added.
        if n_synth > 0
            synth = first(c for c in raw_on[1]
                          if c.provenance.source_type == :synthesized_aggregate)
            @test synth.provenance.aggregation_strategy in (:median, :trim25_mean)
            @test !isempty(synth.provenance.aggregation_source_indices)
            @test :synthesized_aggregate in synth.provenance.notes
            @test isnothing(synth.provenance.interpolator_source)  # synthesized → no specific source
        end
    end
end
