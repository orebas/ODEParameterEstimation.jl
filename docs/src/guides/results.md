# Results

[`estimate`](@ref) returns a vector of [`ParameterEstimationResult`](@ref)s,
best fit first.

```@example results
using ODEParameterEstimation, ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D

@parameters a b
@variables x1(t) x2(t) y1(t) y2(t)
@named oscillator = System([D(x1) ~ -a * x2, D(x2) ~ b * x1], t)

problem = ParameterEstimationProblem(oscillator, [y1 ~ x1, y2 ~ x2];
    true_values = [a => 0.4, b => 0.9, x1 => 1.0, x2 => 0.5])
problem = sample_problem_data(problem; datasize = 41, time_interval = [0.0, 5.0])

results = estimate(problem)
```

## One result

```@example results
best = results[1]
```

The printout has the parameters, the initial conditions with the time they
belong to, and the fit error. The values can be taken out by symbol or by name:

```@example results
best[a], best[:b], best["x1"]
```

or all at once, as dictionaries keyed by symbol:

```@example results
best.parameters
```

```@example results
best.states
```

`best.err` is the fit error: the sum of squared differences between the data
and the model simulated with these values, after each series is scaled by a
power of two to be of order one (unless `auto_rescale = false`). It is what results are ranked by. Use it to
compare the results of one run with each other.

## Using the estimates

`best.states` and `best.parameters` are in the form ModelingToolkit takes, so
simulating the fitted model is one line:

```julia
using OrdinaryDiffEq

fitted = solve(ODEProblem(mtkcompile(oscillator), merge(best.states, best.parameters), (0.0, 5.0)), Tsit5())
```

[Seeing the fit](@ref) plots such a simulation against the data.

## Several results

When the vector holds more than one result, each is a different set of values
that fits the data. When a result marks some values *not identifiable*, the
data do not determine them. [More than one answer](@ref) covers both.

An empty vector means no solution was found. See [Troubleshooting](@ref).

## When the true values are known

For testing a model or the method, [`analyze_parameter_estimation_problem`](@ref)
returns the results together with a comparison against the problem's true
values:

```@example results
raw, analysis, uq = analyze_parameter_estimation_problem(problem, EstimationOptions())
analysis.best_max_error
```

`analysis.returned_results` is what `estimate` returns, except that `estimate`
leaves out the states the package adds for inputs such as `sin(c * t)`.
`best_max_error` is the largest relative error over the parameters and initial
conditions the data can determine, for the candidate solution that came
closest to the truth. That candidate is usually
the first result, but it does not have to be, so to score the first result
itself, compare its values with the truth directly.

## Where a result came from

Each result records how it was produced. `best.interpolator_source` names the
curve fit whose derivative estimates led to it, by a short internal label such
as `:agp_robust` for `InterpolatorGP`, and `best.provenance` holds the rest:
which time point it was solved at and whether it was refined afterwards.
