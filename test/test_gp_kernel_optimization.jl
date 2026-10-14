using Test
using ODEParameterEstimation
using Random

@testset "GP kernel optimization preserves results" begin
    Random.seed!(42)

    # Test data - 101 points (enough to be meaningful but fast)
    xs = collect(0.0:0.1:10.0)
    ys = sin.(xs) .+ 0.01 .* randn(length(xs))

    for kernel_type in [:se, :rq]
        @testset "kernel_type=$kernel_type" begin
            # Call the function (uses the optimized in-place kernel computation)
            interp = ODEParameterEstimation.agp_gpr_robust(xs, ys; kernel_type=kernel_type)

            # Verify predictions at training points are reasonable
            max_train_error = 0.0
            for (x, y) in zip(xs, ys)
                pred = interp.mean_function(x)
                max_train_error = max(max_train_error, abs(pred - y))
            end
            @test max_train_error < 0.5  # Loose tolerance for noisy data

            # Verify predictions at interpolation points
            test_xs = [0.05, 2.55, 5.05, 7.55, 9.95]
            for x in test_xs
                pred = interp.mean_function(x)
                @test isfinite(pred)
                @test abs(pred - sin(x)) < 1.0  # Should be close to true function
            end

            # Verify std predictions are valid
            for x in test_xs
                std_val = interp.std_function(x)
                @test isfinite(std_val)
                @test std_val >= 0.0
            end
        end
    end

    @testset "Edge cases" begin
        # Test with constant data (should return constant interpolator)
        const_ys = fill(5.0, length(xs))
        interp_const = ODEParameterEstimation.agp_gpr_robust(xs, const_ys)
        @test interp_const.mean_function(5.0) ≈ 5.0

        # Test with minimal data points
        short_xs = [0.0, 1.0, 2.0]
        short_ys = [0.0, 1.0, 0.0]
        interp_short = ODEParameterEstimation.agp_gpr_robust(short_xs, short_ys)
        @test isfinite(interp_short.mean_function(1.0))
    end
end

@testset "GP kernel mathematical equivalence" begin
    # This test verifies that our manual kernel computation matches KernelFunctions.jl
    using KernelFunctions

    Random.seed!(123)
    n = 50
    xs = collect(range(0.0, 10.0, length=n))

    # Test parameters
    l = 1.5
    σ² = 2.0

    # Precompute squared distances (as done in optimized code)
    D² = [abs2(xs[i] - xs[j]) for i in 1:n, j in 1:n]

    @testset "SqExponential kernel" begin
        # KernelFunctions.jl computation
        k_kf = σ² * (SqExponentialKernel() ∘ ScaleTransform(1.0 / l))
        K_kf = kernelmatrix(k_kf, xs)

        # Vectorized computation (as in optimized code)
        inv_2l² = 1.0 / (2.0 * l * l)
        K_manual = @. σ² * exp(-D² * inv_2l²)

        @test K_manual ≈ K_kf atol=1e-10
    end
end
