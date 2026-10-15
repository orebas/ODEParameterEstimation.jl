# Every built-in interpolator fits a smooth curve from exact samples and gives
# its value and first derivative.

using ODEParameterEstimation
using ODEParameterEstimation: InterpolatorMethod, get_interpolator_function, nth_deriv_at
using Test

@testset "Every built-in interpolator fits a smooth curve" begin
	xs = collect(range(-0.5, 0.5; length = 21))
	ys = sin.(2 .* xs)
	for method in instances(InterpolatorMethod)
		method == InterpolatorCustom && continue
		@testset "$method" begin
			fit = get_interpolator_function(method)(xs, ys)
			@test fit(0.1) ≈ sin(0.2) atol = 1e-3
			@test nth_deriv_at(fit, 1, 0.1) ≈ 2cos(0.2) atol = 1e-2
		end
	end
end

