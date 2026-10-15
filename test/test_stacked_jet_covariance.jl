using ODEParameterEstimation
# Stacked multi-time estimator-sampling covariance Σ = W_stack Σ_y W_stackᵀ:
# the cross-time Σ_d machinery. Checked: fixed-hyperparameter Monte Carlo,
# repeated-time singularity preservation, scalar-method consistency,
# heteroscedastic Σ_y, cross-block orientation symmetry.

using Test
using LinearAlgebra
using Random
using Statistics

Random.seed!(31415)

@testset "stacked jet estimator covariance (cross-time Σ_d core)" begin
	ts_train = collect(0.0:0.02:1.0)
	y_train = sin.(2π .* ts_train) .+ 0.05 .* randn(length(ts_train))
	interp = ODEParameterEstimation.agp_gpr_uq(ts_train, y_train)
	n = length(ts_train)
	max_d = 2
	d = max_d + 1

	est = ODEParameterEstimation.joint_derivative_estimator_covariance(
		interp, [0.3, 0.7], max_d; observable_name = "y1")

	@testset "shape, labels, indexing, kind" begin
		@test est isa ODEParameterEstimation.StackedJetInfluenceEstimate
		@test size(est.W_stack) == (2d, n)
		@test size(est.jet_covariance) == (2d, 2d)
		@test length(est.labels) == 2d
		@test length(est.mean) == 2d
		@test est.covariance_kind == :estimator_sampling
		@test ODEParameterEstimation.stacked_jet_index(est, 1, 0) == 1
		@test ODEParameterEstimation.stacked_jet_index(est, 2, 1) == d + 2
		@test_throws ArgumentError ODEParameterEstimation.stacked_jet_index(est, 3, 0)
		@test_throws ArgumentError ODEParameterEstimation.stacked_jet_index(est, 1, max_d + 1)
	end

	@testset "fixed-hyperparameter Monte Carlo matches W_stack Σ_y W_stackᵀ" begin
		σ = sqrt(ODEParameterEstimation.learned_observation_noise_variance(interp))
		B = 6000
		draws = Matrix{Float64}(undef, 2d, B)
		for b in 1:B
			draws[:, b] = est.W_stack * (σ .* randn(n))
		end
		Σ_emp = cov(draws; dims = 2)
		@test norm(Σ_emp - est.jet_covariance) / max(norm(est.jet_covariance), 1e-300) < 0.15
	end

	@testset "cross-block orientation: C_ij == C_jiᵀ and consistency" begin
		Σ = est.jet_covariance
		A = Σ[1:d, (d+1):2d]
		Bblk = Σ[(d+1):2d, 1:d]
		@test A ≈ Bblk' atol = 1e-12
		# Diagonal blocks equal the single-time construction from the same W
		μ1, W1 = ODEParameterEstimation.gp_derivative_influence_matrix(interp, 0.3, max_d)
		Σ11 = W1 * est.observation_covariance * W1'
		@test Σ[1:d, 1:d] ≈ Σ11 rtol = 1e-10
	end

	@testset "scalar method consistency on the diagonal block" begin
		scalar = ODEParameterEstimation.joint_derivative_estimator_covariance(
			interp, 0.3, max_d; observable_name = "y1")
		@test est.jet_covariance[1:d, 1:d] ≈ scalar.jet_covariance rtol = 1e-10
		@test est.mean[1:d] ≈ scalar.mean rtol = 1e-12
	end

	@testset "repeated time → genuinely singular, preserved (not shifted)" begin
		est_rep = ODEParameterEstimation.joint_derivative_estimator_covariance(
			interp, [0.5, 0.5], max_d; observable_name = "y1")
		Σ = est_rep.jet_covariance
		@test Σ[1:d, 1:d] ≈ Σ[(d+1):2d, (d+1):2d] rtol = 1e-12
		@test Σ[1:d, (d+1):2d] ≈ Σ[1:d, 1:d] rtol = 1e-12
		evs = eigvals(Symmetric(Σ))
		@test minimum(evs) >= -1e-10 * maximum(abs, evs)          # PSD up to roundoff
		@test minimum(abs.(evs)) <= 1e-8 * maximum(abs, evs)      # singular, NOT repaired
	end

	@testset "heteroscedastic Σ_y propagates exactly" begin
		vars = collect(range(0.5, 2.0; length = n)) .* 1e-3
		est_h = ODEParameterEstimation.joint_derivative_estimator_covariance(
			interp, [0.3, 0.7], max_d;
			observable_name = "y1", observation_covariance = Diagonal(vars))
		@test est_h.noise_source == :user_supplied_covariance
		Σ_direct = est_h.W_stack * Diagonal(vars) * est_h.W_stack'
		@test est_h.jet_covariance ≈ Σ_direct rtol = 1e-10
		B = 6000
		draws = Matrix{Float64}(undef, 2d, B)
		sds = sqrt.(vars)
		for b in 1:B
			draws[:, b] = est_h.W_stack * (sds .* randn(n))
		end
		Σ_emp = cov(draws; dims = 2)
		@test norm(Σ_emp - est_h.jet_covariance) / max(norm(est_h.jet_covariance), 1e-300) < 0.15
	end

	@testset "empty ts rejected" begin
		@test_throws ArgumentError ODEParameterEstimation.joint_derivative_estimator_covariance(
			interp, Float64[], max_d)
	end
end
