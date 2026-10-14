using ODEParameterEstimation
using ODEParameterEstimation: agp_gpr_uq, gp_factorization_diagnostics
using Test
using LinearAlgebra
using ModelingToolkit
using Symbolics: Num

@testset "SE kernel and GP factorization consistency" begin
    @testset "shared SE recipe agrees with a high-precision reference" begin
        xs = collect(range(-0.73, 0.81; length = 31))
        D_sq = [abs2(xs[i] - xs[j]) for i in eachindex(xs), j in eachindex(xs)]
        lengthscale = 0.173
        signal_variance = 1.7
        K = ODEParameterEstimation._se_covariance_matrix(
            D_sq, lengthscale, signal_variance,
        )

        K_reference = setprecision(BigFloat, 256) do
            xs_big = BigFloat.(xs)
            lengthscale_big = BigFloat(lengthscale)
            signal_big = BigFloat(signal_variance)
            [
                signal_big * exp(
                    -abs2(xs_big[i] - xs_big[j]) /
                    (2 * lengthscale_big * lengthscale_big),
                )
                for i in eachindex(xs_big), j in eachindex(xs_big)
            ]
        end
        @test K == K'
        @test K ≈ Float64.(K_reference) rtol = 5e-15 atol = 5e-15
    end

    @testset "adaptive jitter is scale-relative" begin
        # This matrix is symmetric but slightly indefinite, forcing one retry.
        K = [1.0 1.0 + 4eps(Float64); 1.0 + 4eps(Float64) 1.0]
        C1, jitter1 = ODEParameterEstimation._cholesky_adaptive(
            K; relative_jitter = true,
        )
        scale = 2.0^30
        C2, jitter2 = ODEParameterEstimation._cholesky_adaptive(
            scale .* K; relative_jitter = true,
        )
        @test jitter1 > 0
        @test jitter2 / jitter1 ≈ scale rtol = 1e-12
        @test ODEParameterEstimation._cholesky_relative_residual(C1, K, jitter1) < 1e-12
        @test ODEParameterEstimation._cholesky_relative_residual(
            C2, scale .* K, jitter2,
        ) < 1e-12
    end

    @testset "retained AGPUQ factor matches its optimizer kernel recipe" begin
        xs = collect(range(-0.5, 0.5; length = 41))
        ys = @. sin(2.3 * xs) + 0.15 * cos(5.1 * xs)
        interp = agp_gpr_uq(xs, ys)
        D_sq = [abs2(xs[i] - xs[j]) for i in eachindex(xs), j in eachindex(xs)]
        K_train = ODEParameterEstimation._se_covariance_matrix(
            D_sq, interp.lengthscale, interp.signal_var,
        )
        K_noisy = ODEParameterEstimation._add_diagonal_variance(
            K_train, interp.noise_var + interp.cholesky_jitter,
        )
        reconstructed = Matrix(interp.chol.L * interp.chol.L')
        @test reconstructed ≈ K_noisy rtol = 2e-13 atol = 2e-13
        @test interp.alpha ≈ interp.chol \ interp.ys_train rtol = 1e-13

        diagnostics = gp_factorization_diagnostics(interp)
        @test diagnostics.jitter_to_noise == interp.cholesky_jitter_ratio
        @test diagnostics.factorization_residual == interp.factorization_residual
        @test diagnostics.factorization_residual < 1e-12
        @test diagnostics.status in (:ok, :material_regularization)

        # Production estimation must consume the same analytic fixed-smoother
        # derivative whose raw-data influence is propagated by estimator-aware
        # UQ. Wrapping AGPUQ in an anonymous function used to defeat this
        # dispatch and introduce an avoidable low-noise calculus discrepancy.
        t_eval = 0.137
        for order in 0:3
            @test ODEParameterEstimation._estimation_derivative(
                interp, order, t_eval,
            ) == ODEParameterEstimation.nth_deriv(interp, order, t_eval)
        end
        @test ODEParameterEstimation._estimation_derivative(sin, 2, t_eval) ≈
            ODEParameterEstimation.nth_deriv(x -> sin(x), 2, t_eval)

        metadata = ODEParameterEstimation.DataVarMeta[
            ODEParameterEstimation.DataVarMeta("y_0", 1, 0, 1, :observable_jet),
            ODEParameterEstimation.DataVarMeta("y_3", 1, 3, 1, :observable_jet),
        ]
        equation_metadata = NamedTuple{
            (:point, :is_data, :order), Tuple{Int, Bool, Int}
        }[]
        template = ODEParameterEstimation.MultiPointTemplate(
            2, nothing, Num[], Num[], Any[], Any[], Int[], String[],
            equation_metadata, 0, Int[], Int[], [Int[], Int[]], nothing,
            ModelingToolkit.Equation[], metadata,
        )
        interpolants = Dict(:y => interp)
        endpoint_combo = [1, length(xs)]
        interior_combo = [5, length(xs) - 4]
        @test ODEParameterEstimation._multipoint_combo_priority(
            endpoint_combo, xs, interpolants, template, :spread,
        ) > ODEParameterEstimation._multipoint_combo_priority(
            interior_combo, xs, interpolants, template, :spread,
        )
        @test ODEParameterEstimation._multipoint_combo_priority(
            endpoint_combo, xs, interpolants, template, :boundary_order,
        ) < ODEParameterEstimation._multipoint_combo_priority(
            interior_combo, xs, interpolants, template, :boundary_order,
        )

        # Historical :spread ranks shooting indices, not physical time spans.
        # Preserve that default even on an uneven observation grid: [1,3] has
        # the larger index separation although [3,4] spans much more time.
        uneven_times = [0.0, 0.001, 0.002, 1.0]
        @test ODEParameterEstimation._multipoint_combo_priority(
            [1, 3], uneven_times, interpolants, template, :spread,
        ) > ODEParameterEstimation._multipoint_combo_priority(
            [3, 4], uneven_times, interpolants, template, :spread,
        )
    end
end
