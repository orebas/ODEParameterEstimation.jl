using ODEParameterEstimation
using Test

include("estimation_helpers.jl")

const EXAMPLES_ROOT = joinpath(@__DIR__, "..", "src", "examples")

function include_example(parts...)
    return Base.include(@__MODULE__, joinpath(EXAMPLES_ROOT, parts...))
end

function assert_example_result(result_tuple)
    raw_results, analysis, _ = result_tuple
    @test !isempty(raw_results[1])
    @test !isempty(analysis[1])
    @test isfinite(analysis[2])
    return first(analysis[1])
end

@testset "Example script smoke tests" begin
    include_example("control_investigations", "01_hello_world_constant_input.jl")
    include_example("control_investigations", "02_rc_circuit_voltage_step.jl")

    @testset "hello world smoke" begin
        result = quiet_call() do
            run_hello_world(smoke = true)
        end
        best = assert_example_result(result)
        @test best.err <= 1e-6
    end

    @testset "rc circuit smoke" begin
        results = quiet_call() do
            run_rc_circuit_examples(smoke = true)
        end
        best_identifiable = assert_example_result(results[1])
        best_nonident = assert_example_result(results[2])
        @test best_identifiable.err <= 1e-6
        @test !isempty(best_nonident.all_unidentifiable)
    end
end
