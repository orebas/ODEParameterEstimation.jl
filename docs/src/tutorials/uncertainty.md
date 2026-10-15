# Uncertainty

!!! warning "Experimental"
    Uncertainty estimates are experimental. They have been tested on only a
    few models, and it is not yet known how well they hold up in general. If
    you try them, please [open an issue](https://github.com/orebas/ODEParameterEstimation.jl/issues)
    and say how they did. Good or bad, that is useful to hear.

With noisy data an estimate is only good to within some margin. The package
can estimate that margin, as a standard error for each parameter and initial
condition.

```@example uq
using ODEParameterEstimation, ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D

@parameters a b
@variables x1(t) x2(t) y1(t) y2(t)
@named oscillator = System([D(x1) ~ -a * x2, D(x2) ~ b * x1], t)

problem = ParameterEstimationProblem(oscillator, [y1 ~ x1, y2 ~ x2];
    true_values = [a => 0.4, b => 0.8, x1 => 1.0, x2 => 0.5])
problem = sample_problem_data(problem;
    datasize = 101, time_interval = [0.0, 5.0], noise_level = 0.01, seed = 1)
nothing # hide
```

Two options turn it on. `compute_uncertainty = true` asks for it.
`InterpolatorGPUncertainty` is the curve fit that makes it possible: a Gaussian process
that estimates the noise in the data along with the curve. The uncertainty
comes back as the third value of [`analyze_parameter_estimation_problem`](@ref):

```@example uq
options = EstimationOptions(interpolators = [InterpolatorGPUncertainty], compute_uncertainty = true, seed = 1)
raw, analysis, uncertainty = analyze_parameter_estimation_problem(problem, options)

for (name, value, standard_error) in zip(uncertainty.param_labels, uncertainty.estimate_values, uncertainty.param_std)
    println(rpad(name, 4), round(value; sigdigits = 4), " ± ", round(standard_error; sigdigits = 2))
end
```

The data were simulated with `a = 0.4`, `b = 0.8`, `x1 = 1` and `x2 = 0.5`, and
each estimate is within a standard error or two of its true value.

`uncertainty.param_covariance` is the full covariance matrix, and
`print_uncertainty_results(uncertainty)` prints a longer report.

## What the numbers mean

The Gaussian process learns how noisy the data are. That noise is then carried
through the least-squares fit to the parameters, using a linear approximation
around the estimate. The standard errors say how far the estimate would move if
the same experiment were measured again with fresh noise.

Keep three things in mind:

- They belong to the first result. If the model has several answers, they say
  nothing about the others.
- They rest on a linear approximation. With little noise it is good. With a lot
  of noise, or a model that reacts sharply to its parameters, treat them as a
  guide to the size of the error, not as exact confidence intervals.
- They assume the model is right. If it does not describe the system, small
  standard errors do not make the estimates meaningful.

## When it cannot be computed

If the uncertainty cannot be worked out, the third value is a `UncertaintyUnavailable`
in place of a report, and its `message` says why:

```julia
if uncertainty isa UncertaintyReport
    uncertainty.param_std
else
    println(uncertainty.message)
end
```

A report also carries its own checks. `uncertainty.status` is `:ok` when they
passed, and `uncertainty.warnings` lists the assumptions behind the numbers.
