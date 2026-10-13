# Regression test: multipoint through the full estimation pipeline
#
# Run: julia --startup-file=no -e 'using ODEParameterEstimation; include("test/test_multipoint_pipeline.jl")'

using ODEParameterEstimation
using Test

function flatten_r(x)
    out = ParameterEstimationResult[]
    x isa ParameterEstimationResult && (push!(out, x); return out)
    x isa Tuple && (for i in x; append!(out, flatten_r(i)); end; return out)
    x isa Vector && (for i in x; append!(out, flatten_r(i)); end; return out)
    out
end

@testset "Multipoint Pipeline" begin
    pep = ODEParameterEstimation.Examples.lotka_volterra()
    pep_data = ODEParameterEstimation.sample_problem_data(pep,
        EstimationOptions(datasize = 201, noise_level = 0.0, nooutput = true))

    # Single-point baseline
    raw_sp = analyze_parameter_estimation_problem(pep_data,
        EstimationOptions(
            datasize = 201, noise_level = 0.0, nooutput = true,
            interpolators = InterpolatorMethod[InterpolatorAGPUQ],
            auto_filter_interpolators = false, shooting_points = 4,
            use_multipoint = false, diagnostics = false,
            synthesize_aggregate_candidates = false,
            branch_completion = false, polish_solutions = false,
            save_system = false,
        ))
    results_sp = flatten_r(raw_sp)

    # With multipoint (N=2, both paths run)
    raw_mp = analyze_parameter_estimation_problem(pep_data,
        EstimationOptions(use_multipoint = true, multipoint_n_points = 2,
            multipoint_max_pairs = 3, shooting_points = 4,
            datasize = 201, noise_level = 0.0, nooutput = true,
            interpolators = InterpolatorMethod[InterpolatorAGPUQ],
            auto_filter_interpolators = false, diagnostics = false,
            synthesize_aggregate_candidates = false,
            branch_completion = false, polish_solutions = false,
            save_system = false))
    results_mp = flatten_r(raw_mp)

    # Candidate counts are not monotone after clustering and filtering. Verify
    # the actual contract: at least one retained raw row came from an MP solve.
    @test any(r -> r.provenance.source_type == :multipoint, results_mp)

    # Both should produce valid results with finite errors
    errs_sp = [r.err for r in results_sp if !isnothing(r.err) && isfinite(r.err)]
    errs_mp = [r.err for r in results_mp if !isnothing(r.err) && isfinite(r.err)]
    @test !isempty(errs_sp)
    @test !isempty(errs_mp)

    # Multipoint best error should be no worse than 10x single-point
    # (in practice it's equal or better)
    @test minimum(errs_mp) <= minimum(errs_sp) * 10

    # Research-scoped fixed MP recipes may use the ordinary SP solve internally
    # for projection ordering, but only roots from the exact requested pair may
    # enter ranking or become the returned estimator.
    fixed_rows = [20, 180]
    fixed_opts = EstimationOptions(
        use_multipoint = true,
        multipoint_n_points = 2,
        multipoint_max_pairs = 1,
        shooting_points = 2,
        datasize = 201,
        noise_level = 0.0,
        nooutput = true,
        diagnostics = false,
        synthesize_aggregate_candidates = false,
        branch_completion = false,
        polish_solutions = false,
        save_system = false,
        interpolators = InterpolatorMethod[InterpolatorAGPUQ],
        auto_filter_interpolators = false,
    )
    fixed_value, _ = ODEParameterEstimation._with_run_context(
        selection_recipe = ODEParameterEstimation.FixedMultipointRecipe(
            fixed_rows; interpolator_source = :agp_uq,
        ),
    ) do
        ODEParameterEstimation._analyze_parameter_estimation_problem_impl(
            deepcopy(pep_data), fixed_opts,
        )
    end
    _, fixed_analysis, _ = fixed_value
    @test !isempty(fixed_analysis.returned_results)
    for result in fixed_analysis.returned_results
        identity = result.provenance.estimator_identity
        @test identity.estimator_kind == :multipoint_algebraic
        @test identity.time_indices == fixed_rows
    end
end
