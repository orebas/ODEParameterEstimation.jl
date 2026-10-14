using Test, ODEParameterEstimation, ModelingToolkit, OrderedCollections
const ODEPE = ODEParameterEstimation
using ODEParameterEstimation: AbstractInterpolator, agp_gpr_robust, create_interpolants,
    direct_optimization_parameter_estimation, nth_deriv_at,
    optimized_multishot_parameter_estimation, package_wide_default_ode_solver, rescale_pep

struct GridRecordingInterpolator <: AbstractInterpolator
    times::Vector{Float64}
    values::Vector{Float64}
end

@testset "Independent observation grids and physical initial epoch" begin
    @parameters a b
    t = ModelingToolkit.t_nounits
    D = ModelingToolkit.D_nounits
    @variables x(t) z(t) y1(t) y2(t)
    tx = [1.0, 1.5, 1.5, 2.0]
    tz = [0.5, 1.25, 2.5]
    vx = 2 .* exp.(-0.4 .* tx)
    vz = 3 .* exp.(-0.7 .* tz)
    # Repeated rows are retained even when their measured values disagree.
    vx[3] += 0.2
    sx = ObservationSeries("x", "one", x, tx, vx; noise_std=fill(0.1, 4))
    sz = ObservationSeries("z", "one", z, tz, vz)
    data = ObservationData([sx, sz]; initial_time=0.0)
    @test data["t"] == [1.0, 1.25, 1.5, 2.0]
    @test observation_times(data, x) == tx
    @test observation_times(data, z) == tz
    @test data[x] == vx
    @test length(data.series[1].times) == 4
    tx[1] = -100.0
    @test first(observation_times(data, x)) == 1.0
    @test_throws ArgumentError ObservationData([sx, sz]; initial_time=0.6)
    @test_throws ArgumentError ObservationSeries("x", "one", x, [NaN], [1.0])
    @test_throws DimensionMismatch ObservationSeries("x", "one", x, [1.0], [1.0, 2.0])

    model, measured = create_ordered_ode_system("ragged_decay", [x, z], [a, b],
        [D(x) ~ -a*x, D(z) ~ -b*z], [y1 ~ x, y2 ~ z])
    pep = ParameterEstimationProblem("ragged_decay", model, measured, data,
        [0.0, 2.5], package_wide_default_ode_solver,
        OrderedDict(a => 0.4, b => 0.7), OrderedDict(x => 2.0, z => 3.0), 0)
    interps = create_interpolants(measured, data, data["t"],
        (ts, ys) -> GridRecordingInterpolator(copy(ts), copy(ys)))
    @test interps[x].times == sx.times
    @test interps[z].times == sz.times
    @test interps[x].values[3] != interps[x].values[2]
    gp = agp_gpr_robust(sx.times, sx.values)
    @test isfinite(gp(1.5))
    @test isfinite(nth_deriv_at(gp, 1, 1.5))

    # The only residual is the perturbed replicate: no interpolation or missing
    # observation enters the score, and the ODE starts before either series.
    raw = [2.0, 3.0, 0.4, 0.7]
    _, _, sol, err = ODEPE.process_raw_solution(raw, model, data, pep.solver)
    @test sol.prob.tspan == (0.0, 2.5)
    @test err ≈ 0.04 atol=1e-8
    ctx = ODEPE._build_polish_context(pep; opts=EstimationOptions())
    @test ctx.tspan == (0.0, 2.5)
    @test ODEPE._trajectory_sse(ctx, raw) ≈ err atol=1e-8
    internal = ODEPE._polish_external_to_internal(raw, ctx.coordinate_transforms, ctx.coordinate_shifts)
    @test ctx.optf.f(internal, nothing) ≈ err atol=1e-8
    for solver_kind in (:lso_direct, :fastlm_direct)
        polished, _ = ODEPE._polish_single_residual(ctx, raw;
            solver_kind=solver_kind, maxiters=1, maxtime=60.0)
        estimated = [polished.states[x], polished.states[z], polished.parameters[a], polished.parameters[b]]
        @test isfinite(polished.err)
        @test polished.err <= err + 1e-8
        @test polished.err ≈ ODEPE._trajectory_sse(ctx, estimated) atol=1e-8
        @test polished.report_time == 0.0
    end

    scaled, info = rescale_pep(pep)
    @test scaled.data_sample isa ObservationData
    @test scaled.data_sample.initial_time == 0.0
    @test scaled.data_sample.series[1].times == sx.times
    @test length(scaled.data_sample.series[1].values) == 4
    # Two observable IDs for the same expression must not duplicate their
    # already pooled data during rescaling or trajectory refinement.
    aliased = ParameterEstimationProblem(pep.name, pep.model,
        [measured; measured[1]], data, pep.recommended_time_interval,
        pep.solver, pep.p_true, pep.ic, pep.unident_count)
    scaled_alias, _ = rescale_pep(aliased)
    @test length(scaled_alias.data_sample.series) == length(data.series)
    alias_ctx = ODEPE._build_polish_context(aliased)
    @test ODEPE._trajectory_sse(alias_ctx, raw) ≈ err atol=1e-8
    @test_throws ArgumentError optimized_multishot_parameter_estimation(pep,
        EstimationOptions(compute_uncertainty=true))
    @test_throws ArgumentError direct_optimization_parameter_estimation(pep;
        opts=EstimationOptions(compute_uncertainty=true))
end
