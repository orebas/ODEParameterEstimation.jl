using ODEParameterEstimation
using Test
using Logging
using ModelingToolkit
using OrderedCollections
using Random

include("estimation_helpers.jl")
include("support/model_fixtures.jl")

@testset "Example canaries" begin
    @testset "simple recovers parameters" begin
        pep, raw_results, analysis, _ = run_canary(ODEParameterEstimation.Examples.harmonic_oscillator, FAST_STANDARD_OPTS)
        best = best_cluster_solution(analysis)

        @test !isempty(raw_results[1])
        @test !isnothing(best)
        @test analysis[2] < 1e-6
        @test all(string(param) != "z_aux" for param in keys(best.parameters))
        @test all(string(state) != "z_aux" for state in keys(best.states))
        @test all(string(var) != "z_aux" for var in best.all_unidentifiable)

        for (param, true_value) in pep.p_true
            @test best.parameters[param] ≈ true_value atol = 1e-6 rtol = 1e-6
        end
    end

    @testset "underscore-digit names recover distinctly (k_1/k_2, x_1/x_2)" begin
        # Regression for review P0#4 (fixed 2026-06-09): lookup_value's base-name
        # fallback treated a parameter's own trailing _1 as a jet suffix, so k_1
        # and k_2 BOTH resolved to template variable k_2_0 — k_1 silently received
        # k_2's value. Underscore-digit names are common chemistry/biology naming.
        function underscore_digit_model()
            t = ModelingToolkit.t_nounits
            D = ModelingToolkit.D_nounits
            parameters = @parameters k_1 k_2
            states = @variables x_1(t) x_2(t)
            @variables y1(t) y2(t)
            equations = [
                D(x_1) ~ -k_1 * x_2,
                D(x_2) ~ k_2 * x_1,
            ]
            measured_quantities = [y1 ~ x_1, y2 ~ x_2]
            model, mq = ODEParameterEstimation.create_ordered_ode_system(
                "underscore_digit", states, parameters, equations, measured_quantities)
            return ODEParameterEstimation.ParameterEstimationProblem(
                "underscore_digit", model, mq, nothing, nothing, nothing,
                OrderedDict(parameters .=> [0.4, 0.8]),
                OrderedDict(states .=> [0.333, 0.667]),
                0,
            )
        end

        pep, raw_results, analysis, _ = run_canary(underscore_digit_model, FAST_STANDARD_OPTS)
        best = best_cluster_solution(analysis)

        @test !isempty(raw_results[1])
        @test !isnothing(best)
        @test analysis[2] < 1e-6
        for (param, true_value) in pep.p_true
            @test best.parameters[param] ≈ true_value atol = 1e-6 rtol = 1e-6
        end
        # The bug's signature: k_1 and k_2 collapsed to one value. Assert distinct.
        param_by_name = Dict(replace(string(k), "(t)" => "") => Float64(v) for (k, v) in best.parameters)
        @test !isapprox(param_by_name["k_1"], param_by_name["k_2"]; rtol = 1e-3)
    end

    @testset "noisy data still recovers (1% additive homoskedastic noise)" begin
        # Phase A canary: the rest of CI is ~uniformly zero-noise; this locks the
        # noisy path through construction/solve/aggregation with loose thresholds.
        # Seeded: sampling noise uses the global RNG.
        Random.seed!(20260609)
        noisy_opts = EstimationOptions(
            datasize = 21,
            noise_level = 1e-2,
            shooting_points = 0,
            nooutput = true,
            diagnostics = false,
            flow = FlowStandard,
            use_si_template = true,
            use_parameter_homotopy = false,
            interpolators = [InterpolatorAAAD],
            save_system = false,
            polish_solver_solutions = false,
            polish_solutions = false,
        )
        pep, raw_results, analysis, _ = run_canary(ODEParameterEstimation.Examples.harmonic_oscillator, noisy_opts)
        best = best_cluster_solution(analysis)

        @test !isempty(raw_results[1])
        @test !isnothing(best)
        @test isfinite(analysis[2])
        @test analysis[2] < 0.2
        for (param, true_value) in pep.p_true
            @test isapprox(best.parameters[param], true_value; rtol = 0.25)
        end
    end

    @testset "lotka-volterra has an accurate solution cluster" begin
        pep, raw_results, analysis, _ = run_canary(ODEParameterEstimation.Examples.lotka_volterra, FAST_STANDARD_OPTS)
        best = best_cluster_solution(analysis)

        @test !isempty(raw_results[1])
        @test !isnothing(best)
        @test analysis[2] < 1e-5

        for (param, true_value) in pep.p_true
            @test best.parameters[param] ≈ true_value atol = 1e-5 rtol = 1e-5
        end
    end

    @testset "trivial unidentifiability is surfaced" begin
        pep, raw_results, analysis, _ = run_canary(ODEParameterEstimation.Examples.sum_of_rates, FAST_STANDARD_OPTS)
        best = best_cluster_solution(analysis)

        @test !isempty(raw_results[1])
        @test !isnothing(best)
        @test analysis[2] < 1e-6
        @test !isempty(best.all_unidentifiable)
        @test Set(best.all_unidentifiable) == Set(keys(pep.p_true))
        @test !isempty(best.provenance.structural_fix_set)
        @test isempty(best.provenance.representative_assignments)
    end

    @testset "measured-quantity linear combinations stay identifiable" begin
        pep, raw_results, analysis, _ = run_canary(ODEParameterEstimation.Examples.oscillator_mixed_outputs, FAST_STANDARD_OPTS)
        best = best_cluster_solution(analysis)

        @test !isempty(raw_results[1])
        @test !isnothing(best)
        @test analysis[2] < 1e-8

        for (param, true_value) in pep.p_true
            @test best.parameters[param] ≈ true_value atol = 1e-8 rtol = 1e-8
        end
    end

    @testset "nonlinear observable stays estimable" begin
        pep, raw_results, analysis, _ = run_canary(ODEParameterEstimation.Examples.decay_cubed_output, FAST_STANDARD_OPTS)
        best = best_cluster_solution(analysis)

        @test !isempty(raw_results[1])
        @test !isnothing(best)
        @test analysis[2] < 1e-10

        for (param, true_value) in pep.p_true
            @test best.parameters[param] ≈ true_value atol = 1e-10 rtol = 1e-10
        end
    end

    @testset "multiple nonlinear observables stay estimable" begin
        pep, raw_results, analysis, _ = run_canary(threesp_cubed, FAST_STANDARD_OPTS)
        best = best_cluster_solution(analysis)

        @test !isempty(raw_results[1])
        @test !isnothing(best)
        @test analysis[2] < 1e-10

        for (param, true_value) in pep.p_true
            @test best.parameters[param] ≈ true_value atol = 1e-10 rtol = 1e-10
        end
    end

    @testset "transcendental input path stays alive" begin
        pep, raw_results, analysis, _ = run_canary(dc_motor_sinusoidal, FAST_STANDARD_OPTS)
        best = best_cluster_solution(analysis)

        @test pep.name == "dc_motor_sinusoidal"
        @test !isempty(raw_results[1])
        @test !isnothing(best)
        @test isfinite(analysis[2])
        @test state_has_transformed_input(best)
        @test !isempty(best.all_unidentifiable)
        @test !isempty(best.provenance.structural_fix_set)
        @test best.provenance.template_status == :determined
        @test best.provenance.practical_identifiability_status == :advisory_available
        @test !isnothing(best.provenance.numerical_advisory)
        @test best.provenance.numerical_advisory.status == :available
    end

    @testset "unsupported raw model classes fail explicitly" begin
        for (ctor, expected_category) in (
            (cart_pole, :state_trigonometric),
            (ODEParameterEstimation.Examples.tank_level, :sqrt_nonlinearity),
        )
            pep = ctor()
            sampled = ODEParameterEstimation.sample_problem_data(pep, FAST_STANDARD_OPTS)
            err = try
                ODEParameterEstimation.setup_identifiability(sampled; max_num_points = 1, nooutput = true)
                nothing
            catch caught
                caught
            end
            @test err isa ODEParameterEstimation.UnsupportedModelClassError
            @test err.category == expected_category
        end
    end

    @testset "direct optimization smoke test" begin
        # Seeded: the direct-opt flow draws unseeded randn starts, and Julia's
        # default RNG is seeded randomly per session — this canary was a coin
        # flip on every CI run (probe 2026-06-10: pass/pass/empty across three
        # trials in one session; Phase C4's honest Inf made the failing face
        # visible). Seed 20260610 verified converging; 42 verified NOT.
        Random.seed!(20260610)
        _, raw_results, analysis, _ = run_canary(ODEParameterEstimation.Examples.oscillator_mixed_outputs, FAST_DIRECT_OPTS)
        best = best_cluster_solution(analysis)

        @test !isempty(raw_results[1])
        @test !isnothing(best)
        @test isfinite(analysis[2])
        @test best.provenance.primary_method == :direct_opt
        @test best.provenance.polish_applied
        @test best.return_code == :direct_opt
    end
end
