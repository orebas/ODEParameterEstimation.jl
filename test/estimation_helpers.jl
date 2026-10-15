using ODEParameterEstimation
using Logging

function quiet_call(f)
    redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            with_logger(NullLogger()) do
                return f()
            end
        end
    end
end

function run_canary(ctor, opts)
    pep = ctor()
    sampled = ODEParameterEstimation.sample_problem_data(pep, opts)
    raw_results, analysis, uq = quiet_call() do
        ODEParameterEstimation.analyze_parameter_estimation_problem(sampled, opts)
    end
    return pep, raw_results, analysis, uq
end

function best_cluster_solution(analysis)
    isempty(analysis[1]) && return nothing
    return first(analysis[1])
end

function state_has_transformed_input(result)
    any(contains(string(state), "_trfn_") for state in keys(result.states))
end

const FAST_STANDARD_OPTS = EstimationOptions(
    datasize = 21,
    noise_level = 0.0,
    shooting_points = 0,
    nooutput = true,
    diagnostics = false,
    flow = FlowStandard,
    use_si_template = true,
    use_parameter_homotopy = false,
    # An explicit pool is required: the default pool overrides the singular
    # `interpolator` option and may select a synthesized aggregate candidate.
    interpolators = [InterpolatorAAA],
    save_system = false,
    polish_solver_solutions = false,
    polish_solutions = false,
)

const FAST_DIRECT_OPTS = EstimationOptions(
    datasize = 21,
    noise_level = 0.0,
    nooutput = true,
    diagnostics = false,
    flow = FlowDirectOpt,
    opt_maxiters = 5000,
    save_system = false,
)
