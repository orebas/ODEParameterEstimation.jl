# Regression: InterruptException (Ctrl-C) must PROPAGATE out of the estimation
# flow — broad numerical-failure catches may not swallow cancellation.
# `_rethrow_if_interrupt` guards every default-path catch. Before it, no catch
# filtered InterruptException, which turned Ctrl-C into "log, return empty, keep
# trying fallbacks".

using ODEParameterEstimation
using Test

@testset "InterruptException propagation (Ctrl-C not swallowed)" begin
	# Helper semantics: non-interrupts pass through untouched...
	@test ODEParameterEstimation._rethrow_if_interrupt(ArgumentError("x")) === nothing

	# ...and an interrupt rethrows from within a catch block.
	@test_throws InterruptException try
		throw(InterruptException())
	catch e
		ODEParameterEstimation._rethrow_if_interrupt(e)
		error("unreachable: _rethrow_if_interrupt must rethrow InterruptException")
	end

	# End-to-end contract: a cancellation raised deep inside the default flow
	# (here: from the interpolator, via InterpolatorCustom) escapes
	# analyze_parameter_estimation_problem instead of degrading into an
	# empty-result fallback chain.
	# NOTE: the plural `interpolators` list is what the flow consumes; the
	# singular `interpolator` field is honored only when the plural is empty
	# (resolve_interpolator_list, estimation_options.jl:799).
	opts = ODEParameterEstimation.EstimationOptions(
		datasize = 11, noise_level = 0.0, shooting_points = 0,
		nooutput = true, diagnostics = false,
		interpolators = [ODEParameterEstimation.InterpolatorCustom],
		custom_interpolators = Function[(args...) -> throw(InterruptException())],
		polish_solver_solutions = false, polish_solutions = false,
	)
	pep = ODEParameterEstimation.Examples.hidden_scale()
	sampled = ODEParameterEstimation.sample_problem_data(pep, opts)
	@test_throws InterruptException ODEParameterEstimation.analyze_parameter_estimation_problem(sampled, opts)
end
