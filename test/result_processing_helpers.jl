using ODEParameterEstimation
using Test
using Logging
using ModelingToolkit
using Symbolics
using OrderedCollections

function quiet_result_call(f)
    redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            with_logger(NullLogger()) do
                return f()
            end
        end
    end
end

const HELPER_STANDARD_OPTS = EstimationOptions(
    datasize = 11,
    noise_level = 0.0,
    shooting_points = 0,
    nooutput = true,
    diagnostics = false,
    flow = FlowStandard,
    use_si_template = true,
    use_parameter_homotopy = false,
    save_system = false,
    polish_solver_solutions = false,
    polish_solutions = false,
)

function build_solution_fixture(sampled; shoot_idx = 2, empty_forward_subst = false)
    current_states = ModelingToolkit.unknowns(sampled.model.system)
    current_params = ModelingToolkit.parameters(sampled.model.system)
    raw_sol = vcat(
        [Float64(sampled.data_sample[state][shoot_idx]) for state in current_states],
        [Float64(sampled.p_true[param]) for param in current_params],
    )

    forward_subst = if empty_forward_subst
        OrderedDict{Num, Any}()
    else
        OrderedDict{Num, Any}(var => var for var in vcat(current_states, current_params))
    end

    solution_data = (
        solns = [raw_sol],
        forward_subst_dict = [forward_subst],
        trivial_dict = Dict{Any, Any}(),
        final_varlist = Any[vcat(current_states, current_params)...],
        trimmed_varlist = Any[],
        good_udict = Dict{Any, Any}(),
        solution_time_indices = [shoot_idx],
    )
    setup_data = (
        time_index_set = [shoot_idx],
        all_unidentifiable = Set{Any}(),
    )

    return solution_data, setup_data
end

@testset "Result processing helpers" begin
    @testset "lookup_value resolution order" begin
        @independent_variables t
        @parameters a
        @variables x(t)

        @test ODEParameterEstimation.lookup_value(
            a,
            a,
            1,
            Dict{Any, Any}(a => 1.25),
            Dict{Any, Any}(a => 9.0),
            Any[a],
            Any[],
            [[3.0]],
        ) == 1.25

        @test ODEParameterEstimation.lookup_value(
            a,
            a,
            1,
            Dict{Any, Any}(),
            Dict{Any, Any}(a => 2.5),
            Any[a],
            Any[],
            [[3.0]],
        ) == 2.5

        @test ODEParameterEstimation.lookup_value(
            a,
            a,
            1,
            Dict{Any, Any}(),
            Dict{Any, Any}(),
            Any[a],
            Any[],
            [[3.5]],
        ) == 3.5

        @test ODEParameterEstimation.lookup_value(
            a,
            a,
            1,
            Dict{Any, Any}(),
            Dict{Any, Any}(),
            Any[],
            Any[a],
            [[4.5]],
        ) == 4.5

        @test ODEParameterEstimation.lookup_value(
            x,
            x,
            1,
            Dict{Any, Any}(),
            Dict{Any, Any}(),
            Any[Symbolics.variable(:x_0)],
            Any[],
            [[7.5]],
        ) == 7.5

        @test_throws Exception ODEParameterEstimation.lookup_value(
            a,
            a,
            1,
            Dict{Any, Any}(),
            Dict{Any, Any}(),
            Any[],
            Any[],
            [Float64[]],
        )
    end

    @testset "process_raw_solution preserves model ordering" begin
        sampled = ODEParameterEstimation.sample_problem_data(ODEParameterEstimation.Examples.harmonic_oscillator(), HELPER_STANDARD_OPTS)
        current_states = ModelingToolkit.unknowns(sampled.model.system)
        current_params = ModelingToolkit.parameters(sampled.model.system)
        raw_sol = vcat(
            [Float64(sampled.ic[state]) for state in current_states],
            [Float64(sampled.p_true[param]) for param in current_params],
        )

        ordered_states, ordered_params, ode_solution, err = quiet_result_call() do
            ODEParameterEstimation.process_raw_solution(
                raw_sol,
                sampled.model,
                sampled.data_sample,
                sampled.solver,
                abstol = HELPER_STANDARD_OPTS.abstol,
                reltol = HELPER_STANDARD_OPTS.reltol,
            )
        end

        @test string.(collect(keys(ordered_states))) == string.(sampled.model.original_states)
        @test string.(collect(keys(ordered_params))) == string.(sampled.model.original_parameters)
        @test all(state -> isapprox(ordered_states[state], sampled.ic[state]; atol = 1e-10, rtol = 1e-10), keys(sampled.ic))
        @test all(param -> isapprox(ordered_params[param], sampled.p_true[param]; atol = 1e-10, rtol = 1e-10), keys(sampled.p_true))
        @test !isnothing(ode_solution)
        @test err < 1e-8
    end

    @testset "process_raw_solution supports symbolic observable keys" begin
        sampled = ODEParameterEstimation.sample_problem_data(ODEParameterEstimation.Examples.oscillator_mixed_outputs(), HELPER_STANDARD_OPTS)
        current_states = ModelingToolkit.unknowns(sampled.model.system)
        current_params = ModelingToolkit.parameters(sampled.model.system)
        raw_sol = vcat(
            [Float64(sampled.ic[state]) for state in current_states],
            [Float64(sampled.p_true[param]) for param in current_params],
        )

        _, _, _, err = quiet_result_call() do
            ODEParameterEstimation.process_raw_solution(
                raw_sol,
                sampled.model,
                sampled.data_sample,
                sampled.solver,
                abstol = HELPER_STANDARD_OPTS.abstol,
                reltol = HELPER_STANDARD_OPTS.reltol,
            )
        end

        @test any(
            key -> !isequal(key, "t") && all(state -> !isequal(key, state), sampled.model.original_states),
            keys(sampled.data_sample),
        )
        @test err < 1e-8
    end

    @testset "process_estimation_results reconstructs standard solutions" begin
        sampled = ODEParameterEstimation.sample_problem_data(ODEParameterEstimation.Examples.harmonic_oscillator(), HELPER_STANDARD_OPTS)
        solution_data, setup_data = build_solution_fixture(sampled; shoot_idx = 2, empty_forward_subst = false)

        results = quiet_result_call() do
            ODEParameterEstimation.process_estimation_results(sampled, solution_data, setup_data; opts = HELPER_STANDARD_OPTS)
        end
        result = only(results)

        @test all(isfinite, values(result.states))
        @test all(isfinite, values(result.parameters))
        @test all(state -> isapprox(result.states[state], sampled.ic[state]; atol = 1e-5, rtol = 1e-5), keys(sampled.ic))
        @test all(param -> isapprox(result.parameters[param], sampled.p_true[param]; atol = 1e-8, rtol = 1e-8), keys(sampled.p_true))
        @test !isnothing(result.solution)
        @test isfinite(result.err)
        @test result.err < 1e-6
        @test result.provenance.primary_method == :algebraic
        @test result.provenance.rescue_path == :none
        @test result.provenance.source_shooting_index == 2
        @test result.provenance.source_candidate_index == 1
        @test !result.provenance.polish_applied
        @test isempty(result.provenance.representative_assignments)
    end

    @testset "process_estimation_results tolerates empty forward_subst_dict when variables line up" begin
        sampled = ODEParameterEstimation.sample_problem_data(ODEParameterEstimation.Examples.harmonic_oscillator(), HELPER_STANDARD_OPTS)
        solution_data, setup_data = build_solution_fixture(sampled; shoot_idx = 2, empty_forward_subst = true)

        results = quiet_result_call() do
            ODEParameterEstimation.process_estimation_results(sampled, solution_data, setup_data; opts = HELPER_STANDARD_OPTS)
        end
        result = only(results)

        @test all(state -> isapprox(result.states[state], sampled.ic[state]; atol = 1e-5, rtol = 1e-5), keys(sampled.ic))
        @test all(param -> isapprox(result.parameters[param], sampled.p_true[param]; atol = 1e-8, rtol = 1e-8), keys(sampled.p_true))
        @test result.err < 1e-6
        @test result.provenance.source_shooting_index == 2
    end

    @testset "lookup_value SIAN jet-name resolution contract" begin
        # Synthetic SI-workflow varlist (jet-suffixed names), mirroring the P0#4
        # scenario: MTK params k_1/k_2 and states x_1/x_2 must resolve to THEIR
        # OWN jet-0 variables, never a neighbor's (the old base-name fallback
        # mapped both k_1 and k_2 onto k_2_0).
        t = ModelingToolkit.t_nounits
        params = ModelingToolkit.@parameters k_1 k_2
        states = ModelingToolkit.@variables x_1(t) x_2(t)
        jetvars = ModelingToolkit.@variables x_2_0 x_1_0 k_2_0 x_2_1 k_1_0 x_1_1
        varlist = Any[jetvars...]
        solns = [[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]]
        lv(var) = ODEParameterEstimation.lookup_value(
            var, var, 1, Dict{Any, Any}(), Dict{Any, Any}(), varlist, Any[], solns)

        @test lv(params[1]) == 50.0   # k_1   → k_1_0 (NOT k_2_0)
        @test lv(params[2]) == 30.0   # k_2   → k_2_0
        @test lv(states[1]) == 20.0   # x_1(t) → x_1_0
        @test lv(states[2]) == 10.0   # x_2(t) → x_2_0
        # Exact template-var search short-circuits the heuristics entirely.
        @test lv(jetvars[6]) == 60.0  # x_1_1 verbatim
        # Unresolvable search throws — no silent fabrication of a value.
        unresolvable = only(ModelingToolkit.@variables nonexistent_q)
        @test_throws Exception lv(unresolvable)

        # A missing exact jet must never borrow a similarly prefixed variable.
        # The removed fallback stripped k_1 to k, then returned the first k_*;
        # in this case it silently returned k_10_0's value for k_1.
        prefix_neighbor = only(ModelingToolkit.@variables k_10_0)
        @test_throws Exception ODEParameterEstimation.lookup_value(
            params[1],
            params[1],
            1,
            Dict{Any, Any}(),
            Dict{Any, Any}(),
            Any[prefix_neighbor],
            Any[],
            [[99.0]],
        )
    end

    @testset "template_var_map drives SI-workflow lookups (Phase B)" begin
        # Discriminating design: the template variables here are named so that
        # lookup_value's name heuristics can NEVER find them (zz_* bears no
        # relation to the model names). Exact recovery therefore PROVES the
        # explicit map was consumed end-to-end through process_estimation_results.
        sampled = ODEParameterEstimation.sample_problem_data(ODEParameterEstimation.Examples.harmonic_oscillator(), HELPER_STANDARD_OPTS)
        current_states = ModelingToolkit.unknowns(sampled.model.system)
        current_params = ModelingToolkit.parameters(sampled.model.system)
        zz = ModelingToolkit.@variables zz_alpha zz_beta zz_gamma zz_delta
        tvm = OrderedDict{Num, Num}(
            Num(current_states[1]) => zz[1],
            Num(current_states[2]) => zz[2],
            Num(current_params[1]) => zz[3],
            Num(current_params[2]) => zz[4],
        )
        shoot_idx = 2
        raw = [
            Float64(sampled.data_sample[current_states[1]][shoot_idx]),
            Float64(sampled.data_sample[current_states[2]][shoot_idx]),
            Float64(sampled.p_true[current_params[1]]),
            Float64(sampled.p_true[current_params[2]]),
        ]
        solution_data = (
            solns = [raw],
            forward_subst_dict = [OrderedDict{Num, Any}()],   # empty → SI workflow
            trivial_dict = Dict{Any, Any}(),
            final_varlist = Any[zz...],
            trimmed_varlist = Any[],
            good_udict = Dict{Any, Any}(),
            solution_time_indices = [shoot_idx],
            template_var_map = tvm,
        )
        setup_data = (time_index_set = [shoot_idx], all_unidentifiable = Set{Any}())

        results = quiet_result_call() do
            ODEParameterEstimation.process_estimation_results(sampled, solution_data, setup_data; opts = HELPER_STANDARD_OPTS)
        end
        result = only(results)
        @test all(param -> isapprox(result.parameters[param], sampled.p_true[param]; atol = 1e-8, rtol = 1e-8), keys(sampled.p_true))
        @test all(state -> isapprox(result.states[state], sampled.ic[state]; atol = 1e-5, rtol = 1e-5), keys(sampled.ic))
        @test result.err < 1e-6
    end
end
