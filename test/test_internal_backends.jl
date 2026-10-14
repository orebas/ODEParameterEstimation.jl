using ODEParameterEstimation, Test, TOML, Random, LinearAlgebra, Statistics, ForwardDiff, TaylorDiff
using ODEParameterEstimation: aaad_gpr_pivot, get_interpolator_function, merge_options
using ODEParameterEstimation.Examples: harmonic_oscillator
using ODEParameterEstimation: Nemo, StructuralIdentifiability
include("support/backend_fixtures.jl")
const SB = ODEParameterEstimation.SIANBackend
const GB = ODEParameterEstimation.GPBackend
const REFERENCE = TOML.parsefile(joinpath(@__DIR__,"fixtures","internal_backends","backends.toml"))

@testset "Extracted symbolic backend matches upstream exact results" begin
    for rational in (false,true)
        key = rational ? "sian_rational" : "sian_polynomial"
        @test sian_fixture(SB,rational) == REFERENCE[key]
    end
    R,(a,a_10,x) = Nemo.polynomial_ring(Nemo.QQ,["a","a_10","x"])
    S,(xx,aa_10,aa) = Nemo.polynomial_ring(Nemo.QQ,["x","a_10","a"])
    @test SB.parent_ring_change(a+2a_10*x,S) == aa+2aa_10*xx
    T,_ = Nemo.polynomial_ring(Nemo.QQ,["a"])
    @test_throws ArgumentError SB.parent_ring_change(x,T)
    # Derivative order has priority; names containing underscores round-trip.
    jet = SB.create_jet_ring([x],[a,a_10],3)
    @test SB.get_order_var(SB.add_to_var(a_10,jet,0),R) == [a_10,0]
    @test SB.get_order_var(SB.add_to_var(x,jet,3),R) == [x,3]
end

function se_evaluation_allocations(n)
    xs = collect(range(0.,1.;length=n))
    ys = sin.(xs)
    parameters = [-2.,0.,0.]
    workspace = GB.SEWorkspace(n)
    GB.evaluate_se!(workspace,xs,ys,parameters;gradient=true)
    return @allocated GB.evaluate_se!(workspace,xs,ys,parameters;gradient=true)
end

@testset "GP workspace reuse and ownership" begin
    data = REFERENCE["gp"]
    xs,ys,p = data["xs"],data["ys"],data["log_parameters"]
    workspace = GB.SEWorkspace(length(xs))
    first_state = GB.evaluate_se!(workspace,xs,ys,p;gradient=true)
    expected = (; covariance=copy(first_state.covariance),
        factor=copy(first_state.factor.factors), alpha=copy(first_state.alpha),
        gradient=copy(first_state.gradient), nll=first_state.nll)

    # Overwrite every scratch array, then interrupt a later trial during its
    # factorization. Returning to the original parameters must fully recover.
    GB.evaluate_se!(workspace,xs,ys,p .+ 0.1;gradient=true)
    @test_throws PosDefException GB.evaluate_se!(workspace,zeros(length(xs)),ys,
        [-1000.,0.,0.];gradient=true)
    recovered = GB.evaluate_se!(workspace,xs,ys,p;gradient=true)
    @test recovered.covariance == expected.covariance
    @test recovered.factor.factors == expected.factor
    @test recovered.alpha == expected.alpha
    @test recovered.gradient == expected.gradient
    @test recovered.nll == expected.nll
    @test recovered.gradient ≈ data["gradient"] rtol=1e-10 atol=1e-10
    @test_throws DimensionMismatch GB.evaluate_se!(GB.SEWorkspace(length(xs)+1),xs,ys,p)

    # The allocating convenience API must keep its returned state independent
    # of later evaluations, and separate fits must retain independent posteriors.
    state = GB.evaluate_se(xs,ys,p;gradient=true)
    GB.evaluate_se(xs,ys,p .+ 0.1;gradient=true)
    @test state.alpha == expected.alpha
    fit = GB.fit_se(xs,ys)
    prediction = GB.predict_mean(fit,0.4)
    other = GB.fit_se(xs,ys .+ 1.)
    @test GB.predict_mean(fit,0.4) == prediction
    @test abs(GB.predict_mean(other,0.4)-prediction) > 0.5

    # Warm the measurement helper too. A single 128×128 Float64 scratch matrix
    # costs 128 KiB; this budget catches its accidental per-evaluation return.
    se_evaluation_allocations(128)
    @test se_evaluation_allocations(128) < 4096
end

function fixed_fit(data)
    p = Float64.(data["log_parameters"])
    state = GB.evaluate_se(data["xs"],data["ys"],p;gradient=true)
    fit = GB.SEGPFit(data["xs"],p,state.covariance,state.factor,state.alpha,state.nll,true,0)
    return fit,state
end

@testset "Extracted GP fixed-parameter equivalence and analytic derivatives" begin
    data = REFERENCE["gp"]
    fit,state = fixed_fit(data)
    @test vec(state.covariance) ≈ data["covariance"] rtol=1e-12 atol=1e-14
    @test state.nll ≈ data["nll"] rtol=1e-11 atol=1e-11
    @test state.gradient ≈ data["gradient"] rtol=1e-10 atol=1e-10
    # A centered difference checks the independently coded analytic gradient.
    for i in 1:3
        hi,lo = copy(fit.log_parameters),copy(fit.log_parameters)
        hi[i] += 1e-5; lo[i] -= 1e-5
        fd = (GB.evaluate_se(fit.xs,data["ys"],hi).nll-GB.evaluate_se(fit.xs,data["ys"],lo).nll)/2e-5
        @test state.gradient[i] ≈ fd rtol=1e-5 atol=1e-6
    end
    f(x) = GB.predict_mean(fit,x)
    ell = exp(fit.log_parameters[2])
    signal = exp(2fit.log_parameters[3])
    for (j,x) in enumerate(data["points"]), order in 0:6
        value = order==0 ? f(x) : TaylorDiff.derivative(f,x,Val(order))
        @test value ≈ data["jets"][order+1][j] rtol=1e-8 atol=1e-9
        # Probabilists' Hermite recurrence gives an independent SE derivative.
        expected = sum(eachindex(fit.xs)) do i
            z = (x-fit.xs[i])/ell
            prev,h = one(z),z
            for k in 2:order
                prev,h = h,z*h-(k-1)*prev
            end
            polynomial = order==0 ? one(z) : h
            fit.alpha[i]*signal*exp(-z^2/2)*(-1)^order*polynomial/ell^order
        end
        @test value ≈ expected rtol=1e-8 atol=1e-9
        order==1 && @test ForwardDiff.derivative(f,x) ≈ value rtol=1e-10 atol=1e-10
    end
end

@testset "GP fitting edge cases and public adapter" begin
    xs = collect(range(0.,1.;length=11))
    @test aaad_gpr_pivot(xs,fill(3.,11))(0.3) == 3.
    @test aaad_gpr_pivot(reshape(xs,1,:),sin.(xs))(0.3) ≈ aaad_gpr_pivot(xs,sin.(xs))(0.3)
    repeated = GB.fit_se([0.,0.,0.5,1.],[0.,0.01,0.5,1.])
    @test isfinite(GB.predict_mean(repeated,.2))
    @test_throws ArgumentError GB.fit_se([0.,NaN],[1.,2.])
    @test_throws DimensionMismatch GB.fit_se([0.,1.],[1.])
    @test_throws ArgumentError GB.evaluate_se(xs,sin.(xs),[NaN,0.,0.])
    @test_throws PosDefException GB.evaluate_se([0.,0.],[1.,1.],[-1000.,0.,0.])
    @test get_interpolator_function(InterpolatorAAADGPR) === aaad_gpr_pivot
    @test InterpolatorAAADGPR in EstimationOptions().interpolators
    # No imported legacy package binding or global PDMats method patch remains.
    @test !isdefined(ODEParameterEstimation,:GaussianProcesses)
    @test !isdefined(ODEParameterEstimation,:SIAN)
    manifest = TOML.parsefile(joinpath(dirname(Base.active_project()),"Manifest.toml"))
    @test !haskey(manifest["deps"],"GaussianProcesses")
    @test !haskey(manifest["deps"],"SIAN")
end

@testset "Optimized GP fit equivalence and clean-curve accuracy" begin
    cases = TOML.parsefile(joinpath(@__DIR__,"fixtures","internal_backends","fits.toml"))["cases"]
    for data in cases
        @testset "$(data["case"])" begin
            f = aaad_gpr_pivot(data["xs"],data["ys"])
            for order in 0:6
                actual = [order==0 ? f(x) : TaylorDiff.derivative(f,x,Val(order)) for x in data["points"]]
                if data["case"] == "smooth_clean" && order >= 4
                    # Near-zero noise optima vary with BLAS CPU kernels, even
                    # in upstream GP. Check the known generating curve here;
                    # fixed-parameter tests above tightly constrain the kernel
                    # and derivatives. The measured worst order-six relative
                    # error is 1.0213e-3 (Sandybridge), versus 6.6408e-4 in CI.
                    expected = [1.4^order * sin(1.4x + order*π/2) +
                        0.2 * 3.1^order * cos(3.1x + order*π/2) for x in data["points"]]
                    tolerance = order == 6 ? 2e-3 : 1e-3
                    @test actual ≈ expected rtol=tolerance atol=1e-7
                else
                    @test actual ≈ data["jets"][order+1] rtol=1e-5 atol=1e-7
                end
            end
        end
    end
end

include("estimation_helpers.jl")
@testset "Internal backends recover through the GP-only estimator route" begin
    opts = merge_options(FAST_STANDARD_OPTS; datasize=31,
        interpolators=[InterpolatorAAADGPR], compute_uncertainty=false)
    Random.seed!(20261004)
    sampled = sample_problem_data(harmonic_oscillator(),opts)
    raw,analysis,_ = quiet_call() do
        analyze_parameter_estimation_problem(sampled,opts)
    end
    @test !isempty(raw[1])
    @test any(r -> r.interpolator_source == :aaad_gpr, raw[1])
    @test !isempty(analysis[1])
    if !isempty(analysis[1])
        best = first(analysis[1])
        @test best.err < 1e-7
        @test maximum(abs(best.parameters[p]-v)/max(abs(v),1e-12) for (p,v) in sampled.p_true) < 1e-3
    end
end
