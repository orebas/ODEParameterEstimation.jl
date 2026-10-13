# Estimate example models as a user would: default options, 201 exact data
# points over the model's own time interval. example_expectations.jl says what
# each one must return.

using ODEParameterEstimation
using ODEParameterEstimation.Examples
using Random
using Test
using Printf

include(joinpath(@__DIR__, "..", "estimation_helpers.jl"))
include(joinpath(@__DIR__, "..", "example_expectations.jl"))

_example_relative_error(estimate, truth) = abs(truth) > 1e-8 ? abs(estimate - truth) / abs(truth) : abs(estimate - truth)

# The largest error of `result` over everything it does not call undetermined.
function example_result_error(result, problem)
	undetermined = Set(string.(result.all_unidentifiable))
	estimates = merge(Dict(string(key) => value for (key, value) in result.parameters),
		Dict(string(key) => value for (key, value) in result.states))
	worst = 0.0
	for (variable, truth) in merge(problem.p_true, problem.ic)
		name = string(variable)
		name in undetermined && continue
		error = haskey(estimates, name) ? _example_relative_error(estimates[name], truth) : Inf
		worst = max(worst, isnan(error) ? Inf : error)
	end
	return worst
end

function test_example_estimation(name::Symbol)
	expected = EXAMPLE_EXPECTATIONS[name]
	problem = example_problem(name)
	Random.seed!(20261013)
	if expected.tier == :refused
		seconds = @elapsed begin
			@test_throws UnsupportedModelClassError quiet_call() do
				estimate(sample_problem_data(problem; datasize = 201))
			end
		end
		return seconds
	end
	sampled = sample_problem_data(problem; datasize = 201)
	local results
	seconds = @elapsed results = quiet_call(() -> estimate(sampled))
	@test length(results) == expected.answers
	isempty(results) && return seconds
	@test Set(string.(first(results).all_unidentifiable)) == expected.undetermined
	@test minimum(example_result_error(result, problem) for result in results) <= expected.tolerance
	return seconds
end

function test_example_tier(tiers...)
	selected = [name for name in example_models() if EXAMPLE_EXPECTATIONS[name].tier in tiers]
	timings = Pair{Symbol, Float64}[]
	@testset "$name" for name in selected
		@info "Estimating example model" name
		flush(stderr)
		push!(timings, name => test_example_estimation(name))
	end
	println("\nSeconds per example model")
	for (name, seconds) in sort(timings; by = last, rev = true)
		@printf("  %-32s %8.1f\n", name, seconds)
	end
	@printf("  %-32s %8.1f\n", "total", sum(last, timings; init = 0.0))
	flush(stdout)
	return timings
end
