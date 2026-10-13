# Derivatives of an ODE solve carry a tag whose place in ForwardDiff's order is
# fixed, whatever was compiled first (2026-10-13). See src/core/nested_ad.jl.

using ODEParameterEstimation
using Test

const _NESTED_AD_ODEPE = ODEParameterEstimation
const _NESTED_AD_FD = _NESTED_AD_ODEPE.ForwardDiff

struct _SomeInnerTag end

@testset "The outer tag is outer to every other tag" begin
	outer = _NESTED_AD_FD.Tag{_NESTED_AD_ODEPE._OuterADTag, Float64}
	dual = _NESTED_AD_FD.Dual{outer, Float64, 4}
	# A tag made inside the differentiated function, as an ODE solver makes one.
	inner = _NESTED_AD_FD.Tag{_SomeInnerTag, dual}
	# Give the inner tag its number first: with ForwardDiff's own rule that
	# would make it the outer one.
	_NESTED_AD_FD.tagcount(inner)
	@test _NESTED_AD_FD.:≺(outer, inner)
	@test !_NESTED_AD_FD.:≺(inner, outer)
	@test !_NESTED_AD_FD.:≺(outer, outer)
	unrelated = _NESTED_AD_FD.Tag{_SomeInnerTag, Float64}
	@test _NESTED_AD_FD.:≺(outer, unrelated)
	@test !_NESTED_AD_FD.:≺(unrelated, outer)
end

@testset "A Jacobian through a nested derivative is right" begin
	# f differentiates inside itself, as a stiff solver does: f(x) = d/ds of
	# (s * x.^2) at s = 3, which is x.^2, so the Jacobian is diag(2x).
	inner_derivative(x) = [_NESTED_AD_FD.derivative(s -> s * xi^2, 3.0) for xi in x]
	x = [1.0, 2.0, 3.0]
	@test _NESTED_AD_ODEPE._outer_jacobian(inner_derivative, x) ≈ [2.0 0 0; 0 4.0 0; 0 0 6.0]
	J = zeros(3, 3)
	@test _NESTED_AD_ODEPE._outer_jacobian!(J, inner_derivative, x) === J
	@test J ≈ [2.0 0 0; 0 4.0 0; 0 0 6.0]
end
