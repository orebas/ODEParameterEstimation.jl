using ODEParameterEstimation
# _ift_solve contract (degrade loudly): factorized IFT
# sensitivity, no silent pinv. The old cond>1e6 pinv fallback returned the
# minimum-norm derivative — suppressing weak-direction sensitivity and making
# downstream covariance overconfident exactly when conditioning is worst.
# Pure numerics; fast_unit tier.

using Test
using Logging
using LinearAlgebra

@testset "_ift_solve (factorized IFT, loud degradation)" begin
	@testset "well-conditioned: plain factorized solve" begin
		J_x = [2.0 0.0; 0.0 3.0]
		J_d = Matrix{Float64}(I, 2, 2)
		S, c, degraded = ODEParameterEstimation._ift_solve(J_x, J_d)
		@test S ≈ [-0.5 0.0; 0.0 -1/3] rtol = 1e-12
		@test c ≈ 1.5 rtol = 1e-12
		@test !degraded
	end

	@testset "scaled but stable: raw condition warns without false degradation" begin
		J_x = [1.0 0.0; 0.0 1e-9]
		J_d = Matrix{Float64}(I, 2, 2)
		audit = with_logger(NullLogger()) do
			ODEParameterEstimation._ift_solve_assessment(J_x, J_d)
		end
		@test !audit.degraded
		@test audit.reason == :scale_sensitive_conditioning
		@test audit.raw_condition > 1e8
		@test audit.equilibrated_condition < 1e3
		@test audit.backward_error <= 1e-10
		S_big = setprecision(256) do
			Float64.(-(BigFloat.(J_x) \ BigFloat.(J_d)))
		end
		@test norm(audit.sensitivity - S_big) / norm(S_big) <= 1e-10
	end

	@testset "intrinsically ill-conditioned: degraded, amplification visible" begin
		J_x = [1.0 1.0; 1.0 1.0 + 1e-9]
		J_d = Matrix{Float64}(I, 2, 2)
		audit = with_logger(NullLogger()) do
			ODEParameterEstimation._ift_solve_assessment(J_x, J_d)
		end
		@test audit.degraded
		@test audit.reason == :intrinsic_ill_conditioning
		@test audit.raw_condition > 1e8
		@test audit.equilibrated_condition > 1e6
		@test maximum(abs, audit.sensitivity) > 1e8
	end

	@testset "large solve residual is independently degraded" begin
		A = [2.0 0.0; 0.0 3.0]
		B = Matrix{Float64}(I, 2, 2)
		bad_X = zeros(2, 2)
		audit = ODEParameterEstimation._linear_solve_assessment(A, bad_X, B)
		@test audit.degraded
		@test audit.reason == :large_backward_error
		@test audit.backward_error > 1e-10
	end

	@testset "exactly singular: empty S + degraded (no minimum-norm fiction)" begin
		J_x = [1.0 0.0; 0.0 0.0]
		J_d = Matrix{Float64}(I, 2, 2)
		S, c, degraded = with_logger(NullLogger()) do
			ODEParameterEstimation._ift_solve(J_x, J_d)
		end
		@test degraded
		@test isempty(S)   # no local IFT derivative exists — say so, don't invent one
	end
end
