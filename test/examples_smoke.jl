# The scripts in `examples/` run, and say what they are meant to show.

using ODEParameterEstimation
using Test

include("estimation_helpers.jl")

const EXAMPLE_SCRIPTS = joinpath(@__DIR__, "..", "examples")

# Run a script in a module of its own and hand back that module.
function run_example_script(name)
    script_module = Module(gensym(:ExampleScript))
    quiet_call() do
        Base.include(script_module, joinpath(EXAMPLE_SCRIPTS, name))
    end
    return script_module
end

estimated(result, name) = result[Symbol(name)]

@testset "Example scripts" begin
    @testset "every script is tested" begin
        scripts = filter(endswith(".jl"), readdir(EXAMPLE_SCRIPTS))
        @test Set(scripts) == Set(["first_example.jl", "constant_input.jl", "rc_circuit.jl"])
    end

    @testset "first_example.jl" begin
        script = run_example_script("first_example.jl")
        best = first(script.results)
        for (name, value) in ("α" => 1.1, "β" => 0.4, "γ" => 0.4, "δ" => 0.1, "prey" => 1.0, "predators" => 0.5)
            @test estimated(best, name) ≈ value rtol = 1e-4
        end
    end

    @testset "constant_input.jl" begin
        script = run_example_script("constant_input.jl")
        best = first(script.results)
        @test estimated(best, "a") ≈ 0.5 rtol = 1e-6
        @test estimated(best, "u") ≈ 1.0 rtol = 1e-6
        @test isempty(best.all_unidentifiable)
    end

    @testset "rc_circuit.jl" begin
        script = run_example_script("rc_circuit.jl")
        best = first(script.results_tau)
        @test estimated(best, "tau") ≈ 1.0 rtol = 1e-6
        @test estimated(best, "V_in") ≈ 5.0 rtol = 1e-6
        @test isempty(best.all_unidentifiable)

        separate = first(script.results_separate)
        @test Set(string.(separate.all_unidentifiable)) == Set(["R", "C"])
        @test estimated(separate, "V_in") ≈ 5.0 rtol = 1e-6
    end
end
