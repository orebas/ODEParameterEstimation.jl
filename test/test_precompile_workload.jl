# The estimation the package runs while it is precompiled. It is a
# default run like any other: quiet and accurate, and it leaves the caller's
# random stream alone. A freshly loaded package carries nothing over from it.

using ODEParameterEstimation
using Logging
using Random
using Test

include(joinpath(@__DIR__, "support", "observe_run.jl"))

const _ODEPE = ODEParameterEstimation

@testset "The precompile workload is an ordinary default run" begin
	rng = Random.default_rng()
	Random.seed!(rng, 53)
	# The comparison is made inside the observed run: making a temporary
	# directory, as `_observe_run` does, moves the random stream by itself.
	run = _observe_run() do
		before = copy(rng)
		results = _ODEPE._precompile_workload()
		(; results, rng_untouched = copy(rng) == before)
	end
	@test run.value.rng_untouched
	@test isempty(run.printed)
	@test isempty(run.logs)
	@test isempty(run.files)

	result = only(run.value.results)
	for (name, truth) in ("k1" => 0.6, "k2" => 0.3, "k3" => 0.8, "k4" => 0.2, "x1" => 1.0, "x2" => 0.5)
		@test isapprox(result[name], truth; rtol = 1e-6)
	end
end

@testset "Session state can be cleared" begin
	# The run above left its reuse bundle behind, as every estimation does.
	@test !isnothing(_ODEPE._LAST_ESTIMATION_REUSE[])
	_ODEPE._reset_session_state!()
	@test isnothing(_ODEPE._LAST_ESTIMATION_REUSE[])
	@test isnothing(_ODEPE._LAST_BRANCH_COMPLETION_DEBUG[])
	@test isempty(_ODEPE._NOISE_VALIDATION_CACHE)
end

@testset "A freshly loaded package carries nothing over from precompilation" begin
	# Earlier test files have run estimations in this process, so ask a new one.
	script = """
		using ODEParameterEstimation
		const O = ODEParameterEstimation
		print(O._WORKLOAD_STATUS[], " ", isnothing(O._LAST_ESTIMATION_REUSE[]), " ",
			isnothing(O._LAST_BRANCH_COMPLETION_DEBUG[]), " ", length(O._NOISE_VALIDATION_CACHE))
		"""
	command = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $script`
	# `skipped` is what a developer who turned the workload off sees.
	@test read(command, String) in ("ok true true 0", "skipped true true 0")
end

@testset "The registry's install check skips the workload" begin
	withenv("JULIA_REGISTRYCI_AUTOMERGE" => "true") do
		@test _ODEPE._in_registry_check()
	end
	for value in (nothing, "false", "")
		withenv("JULIA_REGISTRYCI_AUTOMERGE" => value) do
			@test !_ODEPE._in_registry_check()
		end
	end
end

@testset "The final polish works in a freshly loaded package" begin
	# The polish differentiates an ODE solve whose solver differentiates too.
	# ForwardDiff orders the two layers by a number fixed when each is first
	# compiled, and the workload fixed ours at precompile time, ahead of the
	# solver's. Every polish then failed quietly and returned its seed.
	# Only a new process starts from the precompiled state, and
	# this model is one whose answer before the polish is good to 1e-5 only.
	script = """
		using ODEParameterEstimation, ODEParameterEstimation.Examples, Random
		problem = brusselator()
		Random.seed!(20261013)
		best = first(estimate(sample_problem_data(problem; datasize = 201)))
		truth = merge(Dict(string(k) => v for (k, v) in problem.p_true), Dict(string(k) => v for (k, v) in problem.ic))
		found = merge(Dict(string(k) => v for (k, v) in best.parameters), Dict(string(k) => v for (k, v) in best.states))
		print(best.provenance.polish_applied, " ", maximum(abs(found[k] - truth[k]) / abs(truth[k]) for k in keys(truth)))
		"""
	command = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $script`
	polish_applied, worst_error = split(read(command, String))
	@test polish_applied == "true"
	@test parse(Float64, worst_error) < 1e-9
end
