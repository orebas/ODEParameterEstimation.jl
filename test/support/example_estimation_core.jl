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

function _print_example_timings(timings)
	println("\nSeconds per example model")
	for (name, seconds) in sort(timings; by = last, rev = true)
		@printf("  %-32s %8.1f\n", name, seconds)
	end
	@printf("  %-32s %8.1f\n", "total", sum(last, timings; init = 0.0))
	flush(stdout)
	return nothing
end

# Seconds a slow model may take in a process of its own before it is stopped
# and counted as failed. The slowest takes about half an hour on a desktop.
const SLOW_EXAMPLE_TIME_LIMIT = parse(Float64, get(ENV, "ODEPE_SLOW_EXAMPLE_TIME_LIMIT", "9000"))

# Estimate one model in a new Julia process. Each slow model gets its own:
# estimated one after another in one process, a later model can take many
# times longer than it does alone.
function test_example_estimation_isolated(name::Symbol; time_limit::Float64 = SLOW_EXAMPLE_TIME_LIMIT)
	script = """
		include($(repr(@__FILE__)))
		@testset "$name" begin
			test_example_estimation($(repr(name)))
		end
		"""
	command = `$(Base.julia_cmd()) --startup-file=no --threads=$(Threads.nthreads()) --project=$(Base.active_project()) -e $script`
	output = IOBuffer()
	process = run(pipeline(ignorestatus(command); stdout = output, stderr = output); wait = false)
	seconds = @elapsed finished = timedwait(() -> process_exited(process), time_limit; pollint = 1.0)
	if finished != :ok
		kill(process, Base.SIGKILL)
		wait(process)
		@error "Example model was stopped at its time limit" name time_limit
		@test false
		return seconds
	end
	passed = success(process)
	passed || println(String(take!(output)))
	@test passed
	return seconds
end

function test_example_tier_isolated(tiers...)
	selected = [name for name in example_models() if EXAMPLE_EXPECTATIONS[name].tier in tiers]
	timings = Pair{Symbol, Float64}[]
	@testset "$name" for name in selected
		@info "Estimating example model in its own process" name
		flush(stderr)
		push!(timings, name => test_example_estimation_isolated(name))
	end
	_print_example_timings(timings)
	return timings
end

function test_example_tier(tiers...)
	selected = [name for name in example_models() if EXAMPLE_EXPECTATIONS[name].tier in tiers]
	timings = Pair{Symbol, Float64}[]
	@testset "$name" for name in selected
		@info "Estimating example model" name
		flush(stderr)
		push!(timings, name => test_example_estimation(name))
	end
	_print_example_timings(timings)
	return timings
end
