# Example models

The package comes with ready-made problems to try things on. They live in a
submodule, `ODEParameterEstimation.Examples`. Each is a function that returns a
[`ParameterEstimationProblem`](@ref) with true values and no data, so the usual
pattern is:

```julia
using ODEParameterEstimation
using ODEParameterEstimation.Examples

problem = sample_problem_data(lotka_volterra(); datasize = 101)
results = estimate(problem)
```

Every model carries a time interval that suits it, and `sample_problem_data`
uses it unless you pass `time_interval` yourself.

```@example models
using ODEParameterEstimation
using ODEParameterEstimation.Examples

function describe(group)
    println(rpad("model", 32), "states  parameters  measured  interval")
    for name in example_models(group)
        problem = example_problem(name)
        println(rpad(name, 32), rpad(length(problem.ic), 8), rpad(length(problem.p_true), 12),
            rpad(length(problem.measured_quantities), 10), problem.recommended_time_interval)
    end
end
nothing # hide
```

## Looking inside one

A problem carries its ModelingToolkit system, its measured quantities and its
true values:

```@example models
using ModelingToolkit

problem = lotka_volterra()
equations(problem.model.system)
```

```@example models
problem.measured_quantities
```

```@example models
problem.p_true
```

## First steps

Small models that are solved quickly and accurately.

```@example models
describe(:first_steps)
```

```@autodocs
Modules = [ODEParameterEstimation.Examples]
Order = [:function]
Private = false
Filter = f -> nameof(f) in ODEParameterEstimation.Examples.example_models(:first_steps)
```

## Something cannot be determined

In each of these the measured quantities leave at least one parameter or
initial condition open. See [More than one answer](@ref).

```@example models
describe(:cannot_be_determined)
```

```@autodocs
Modules = [ODEParameterEstimation.Examples]
Order = [:function]
Private = false
Filter = f -> nameof(f) in ODEParameterEstimation.Examples.example_models(:cannot_be_determined)
```

## Several answers

Two models in which several sets of values fit the data equally well, each
beside a version that is measured fully enough to have one answer.

```@example models
describe(:several_answers)
```

```@autodocs
Modules = [ODEParameterEstimation.Examples]
Order = [:function]
Private = false
Filter = f -> nameof(f) in ODEParameterEstimation.Examples.example_models(:several_answers)
```

## Larger models

Most of these are benchmarks from the identifiability literature. They take
longer, some of them minutes, and two are beyond what the package solves in
reasonable time at present. Their descriptions say which.

```@example models
describe(:larger_models)
```

```@autodocs
Modules = [ODEParameterEstimation.Examples]
Order = [:function]
Private = false
Filter = f -> nameof(f) in ODEParameterEstimation.Examples.example_models(:larger_models)
```

## Control systems

Systems driven by a known sinusoidal input. See
[Inputs that depend on time](@ref).

```@example models
describe(:control_systems)
```

```@autodocs
Modules = [ODEParameterEstimation.Examples]
Order = [:function]
Private = false
Filter = f -> nameof(f) in ODEParameterEstimation.Examples.example_models(:control_systems)
```

## A model that has to be rewritten

One tank, written as it comes and written so that the package accepts it. See
[Rewriting a model so that it is accepted](@ref).

```@example models
describe(:rewriting)
```

```@autodocs
Modules = [ODEParameterEstimation.Examples]
Order = [:function]
Private = false
Filter = f -> nameof(f) in ODEParameterEstimation.Examples.example_models(:rewriting)
```

## Listing them

```@docs
ODEParameterEstimation.Examples
ODEParameterEstimation.Examples.example_models
ODEParameterEstimation.Examples.example_groups
ODEParameterEstimation.Examples.example_problem
```
