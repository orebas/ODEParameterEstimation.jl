using Symbolics
# The Taylor-coefficient recursion behind compute_oracle_taylor_coefficients.
# Pure numerics — no ODE solves, no SIAN, no GP fits. fast_unit tier.

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
end
