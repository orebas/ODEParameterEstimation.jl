using Symbolics
# Stream B phase 1 (2026-08-13): estimate-conditioned Taylor propagation cores.
# Pure numerics — no ODE solves, no SIAN, no GP fits. fast_unit tier.
#
# _propagate_state_taylor is the shared jet recursion behind BOTH
# compute_oracle_taylor_coefficients (truth) and
# compute_estimate_taylor_coefficients (θ̂/x̂); interpolant_taylor_coefficients
# is the GP-jet view of obs_taylor the estimate-conditioned path consumes.

using Test
using ODEParameterEstimation
using ModelingToolkit
using OrderedCollections
using LinearAlgebra

@testset "taylor propagation cores (estimate-conditioned UQ)" begin
	pep_tp = ODEParameterEstimation.Examples.harmonic_oscillator()   # ẋ1 = -a·x2, ẋ2 = b·x1 → jets are Aᵏ·x
	sys_tp = pep_tp.model.system
	params_tp = ModelingToolkit.parameters(sys_tp)
	states_tp = ModelingToolkit.unknowns(sys_tp)

	a_val, b_val = 0.4, 0.8
	x_val = [0.333, 0.667]
	t_eval_tp = 0.7
	max_order_tp = 6

	param_vals_tp = Dict{Num, Float64}(params_tp[1] => a_val, params_tp[2] => b_val)

	@testset "_propagate_state_taylor matches matrix-power jets" begin
		coeffs = ODEParameterEstimation._propagate_state_taylor(
			sys_tp, x_val, param_vals_tp, t_eval_tp, max_order_tp)
		# For the linear system ẋ = A·x with A = [0 -a; b 0], the k-th derivative
		# vector is Aᵏ·x, so the Taylor coefficient is (Aᵏ·x)/k!.
		A = [0.0 -a_val; b_val 0.0]
		d = copy(x_val)
		for k in 0:max_order_tp
			for (si, s) in enumerate(states_tp)
				@test coeffs[s][k+1] ≈ d[si] / factorial(k) rtol = 1e-12 atol = 1e-14
			end
			d = A * d
		end
	end

	@testset "_propagate_state_taylor rejects mis-sized anchors" begin
		@test_throws ErrorException ODEParameterEstimation._propagate_state_taylor(
			sys_tp, [1.0], param_vals_tp, t_eval_tp, 3)
	end

	@testset "interpolant_taylor_coefficients round-trips known jets" begin
		# harmonic_oscillator() observes y1 ~ x1, y2 ~ x2; interpolants are keyed by the
		# diff2term'd observable RHS (same convention as precomputed_interpolants).
		k1 = Symbolics.diff2term(pep_tp.measured_quantities[1].rhs)
		k2 = Symbolics.diff2term(pep_tp.measured_quantities[2].rhs)
		f_poly = t -> 2.0 * t^3 + t          # jets: [2t³+t, 6t²+1, 12t, 12, 0]
		f_exp = t -> exp(0.5 * t)            # jets: 0.5ᵏ · e^{t/2}
		interps = Dict{Any, Any}(k1 => f_poly, k2 => f_exp)

		tc = ODEParameterEstimation.interpolant_taylor_coefficients(pep_tp, interps, t_eval_tp, 4)
		@test haskey(tc, k1) && haskey(tc, k2)
		@test length(tc[k1]) == 5

		expected_poly = [2 * t_eval_tp^3 + t_eval_tp, 6 * t_eval_tp^2 + 1, 12 * t_eval_tp, 12.0, 0.0]
		for k in 0:4
			# Consumers recover derivative values as tc[k+1] * k! — assert on that contract.
			@test tc[k1][k+1] * factorial(k) ≈ expected_poly[k+1] rtol = 1e-8 atol = 1e-10
			@test tc[k2][k+1] * factorial(k) ≈ 0.5^k * exp(0.5 * t_eval_tp) rtol = 1e-8
		end
	end
end
