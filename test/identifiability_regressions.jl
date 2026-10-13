using ODEParameterEstimation
using Test

include("estimation_helpers.jl")
include("support/model_fixtures.jl")

function result_value_by_name(dict_like, name::AbstractString)
    for (k, v) in dict_like
        string(k) == name && return v
    end
    error("Entry $(name) not found")
end

function result_name_set(values_iter)
    return Set(string.(collect(values_iter)))
end

@testset "Identifiability regressions ($strategy)" for strategy in (:local_basis, :identifiable_functions)
    identifiability_options = merge_options(FAST_STANDARD_OPTS; si_fix_strategy=strategy)
    @testset "substring-heavy parameter names still recover accurately" begin
        pep, raw_results, analysis, _ = run_canary(substr_test, identifiability_options)
        best = best_cluster_solution(analysis)

        @test !isempty(raw_results[1])
        @test !isnothing(best)
        @test analysis[2] < 1e-8
        @test isempty(best.all_unidentifiable)
        @test result_value_by_name(best.parameters, "a") ≈ pep.p_true[first(keys(pep.p_true))] atol = 1e-8 rtol = 1e-8
        @test result_value_by_name(best.parameters, "b") ≈ pep.p_true[collect(keys(pep.p_true))[2]] atol = 1e-8 rtol = 1e-8
        @test result_value_by_name(best.parameters, "beta") ≈ pep.p_true[collect(keys(pep.p_true))[3]] atol = 1e-6 rtol = 1e-6
    end

    @testset "global unidentifiable states and parameters are surfaced together" begin
        pep, raw_results, analysis, _ = run_canary(ODEParameterEstimation.Examples.partly_observed_chain, identifiability_options)
        best = best_cluster_solution(analysis)

        @test !isempty(raw_results[1])
        @test !isnothing(best)
        @test isfinite(analysis[2])
        @test analysis[2] < 0.35
        @test result_name_set(best.all_unidentifiable) == Set(["b", "c", "d", "x3(t)"])
        @test result_value_by_name(best.parameters, "a") ≈ pep.p_true[first(keys(pep.p_true))] atol = 1e-8 rtol = 1e-8
    end

    @testset "summed-observation unidentifiability stays explicit" begin
        pep, raw_results, analysis, _ = run_canary(ODEParameterEstimation.Examples.hidden_scale, identifiability_options)
        best = best_cluster_solution(analysis)

        @test !isempty(raw_results[1])
        @test !isnothing(best)
        @test isfinite(analysis[2])
        @test analysis[2] < 0.2
        @test result_name_set(best.all_unidentifiable) == Set(["c", "x1(t)", "x2(t)"])
        @test result_name_set(keys(best.provenance.structural_fix_set)) == Set(["c"])
        @test best.provenance.template_status == :determined
        @test best.provenance.practical_identifiability_status == :advisory_available
        @test !isnothing(best.provenance.numerical_advisory)
        @test best.provenance.numerical_advisory.status == :available
        @test any(
            isapprox(result_value_by_name(result.parameters, "a"), pep.p_true[first(keys(pep.p_true))]; atol = 1e-6, rtol = 1e-6) &&
            isapprox(result_value_by_name(result.parameters, "b"), pep.p_true[collect(keys(pep.p_true))[2]]; atol = 1e-6, rtol = 1e-6)
            for result in analysis[1]
        )
        # The default ranking uses trajectory fit. The accurate-parameter
        # branch above must be retained, but truth is not a selection input.
        @test best.err == minimum(result.err for result in analysis.returned_results)
    end
end
