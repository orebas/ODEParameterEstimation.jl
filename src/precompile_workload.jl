# The estimation that runs while the package is precompiled. Julia saves the
# code it compiles then, so the first `estimate` of a session does not have to
# compile the whole default path again.

# The workload always draws the same random numbers, so every precompilation
# takes the same path through the estimator.
const _WORKLOAD_SEED = 20261006

# `:skipped` until the workload has run, then `:ok` or `:failed`. The value is
# saved with the precompiled package.
const _WORKLOAD_STATUS = Ref(:skipped)

"""
	_reset_session_state!() -> Nothing

Forget what earlier estimations left in the module: the reuse bundle and the
branch-completion record of the last run, and the cached validation Jacobians.
The precompile workload calls this so that none of it is saved with the
package.
"""
function _reset_session_state!()
	lock(_NOISE_VALIDATION_CACHE_LOCK) do
		empty!(_NOISE_VALIDATION_CACHE)
	end
	_LAST_ESTIMATION_REUSE[] = nothing
	_LAST_BRANCH_COMPLETION_DEBUG[] = nothing
	return nothing
end

"""
	_precompile_workload() -> Vector{ParameterEstimationResult}

Estimate a small model the way the README does: build the problem from a
ModelingToolkit system, simulate exact data, call `estimate` with no options
and display a result.

The model has two measured states and four parameters, the size of the README
example, because part of the refinement is compiled for the number of unknowns.
It is a different model, so that the README example is a fair test of what the
workload saves on a model it has not seen. `estimate` is called without
keywords because a keyword call is compiled for its particular set of
keywords.
"""
function _precompile_workload()
	return _with_scoped_rng(_WORKLOAD_SEED) do
		t, D = ModelingToolkit.t_nounits, ModelingToolkit.D_nounits
		@parameters k1 k2 k3 k4
		@variables x1(t) x2(t) y1(t) y2(t)
		# `System` is qualified because HomotopyContinuation exports one too.
		workload = ModelingToolkit.System([
			D(x1) ~ -k1 * x1 + k2 * x2 - k3 * x1 * x2,
			D(x2) ~ k1 * x1 - k2 * x2 - k4 * x2,
		], t; name = :workload)
		problem = ParameterEstimationProblem(workload, [y1 ~ x1, y2 ~ x2];
			true_values = [k1 => 0.6, k2 => 0.3, k3 => 0.8, k4 => 0.2, x1 => 1.0, x2 => 0.5])
		sampled = sample_problem_data(problem; datasize = 101, time_interval = [0.0, 10.0])
		results = estimate(sampled)
		show(IOBuffer(), MIME"text/plain"(), first(results))
		results
	end
end

"""
	_in_registry_check() -> Bool

Whether this process is the General registry's install check, which sets
`JULIA_REGISTRYCI_AUTOMERGE` to `true`. That check precompiles the package and
all of its dependencies on two Julia versions within a time limit, and asks
only whether the package loads. The workload is most of this package's share
of that time and cannot make loading fail, so it is skipped there.
"""
_in_registry_check() = get(ENV, "JULIA_REGISTRYCI_AUTOMERGE", "false") == "true"

@compile_workload begin
	if !_in_registry_check()
		try
			# A default run prints nothing and writes no files, and the tests check
			# that. Installing the package must not do either, whatever happens.
			mktempdir() do workload_dir
				cd(workload_dir) do
					redirect_stdout(devnull) do
						redirect_stderr(devnull) do
							_precompile_workload()
						end
					end
				end
			end
			_WORKLOAD_STATUS[] = :ok
		catch err
			_rethrow_if_interrupt(err)
			_WORKLOAD_STATUS[] = :failed
			@warn "ODEParameterEstimation could not run its precompile workload. The package works, but the first `estimate` of each session will be slow. Please report this." exception = (err, catch_backtrace())
		finally
			_reset_session_state!()
		end
	end
end
