# HC name sanitization contracts: sanitize_vars is
# injective per call, and the parameterized convert deduplicates the p_-prefixed
# data names against the solve names — a duplicate HC name is NOT an HC error;
# it silently dead-slots one variable and returns garbage positionally.

using ODEParameterEstimation
using Test
using Symbolics

@testset "HC name sanitization (injective + cross-list dedup)" begin
	# Lossy char-replacement collisions become distinct names; first keeps clean.
	s = ODEParameterEstimation.sanitize_vars(["x.t", "x_t", "x t"])
	@test allunique(s)
	@test s[1] == "x_t"
	@test startswith(s[2], "x_t__")
	@test startswith(s[3], "x_t__")

	# Leading digit still gets the v_ prefix.
	@test ODEParameterEstimation.sanitize_vars(["2x"])[1] == "v_2x"

	# Clean input is untouched.
	@test ODEParameterEstimation.sanitize_vars(["a", "b_1", "c"]) == ["a", "b_1", "c"]

	# Cross-list collision: solve var `p_y` and
	# data var `y` used to both become HC name p_y.
	@variables p_y y q
	sys = [p_y + y - 1.0, q * y - 2.0]
	hc_system, hc_vars, hc_params = ODEParameterEstimation.convert_to_hc_format_with_params(
		sys, [p_y, q], [y])
	names = string.(vcat(hc_vars, hc_params))
	@test allunique(names)
	@test length(hc_vars) == 2
	@test length(hc_params) == 1
end
