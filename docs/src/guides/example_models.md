# Example models

The package comes with ready-made problems to try things on. Each is a function
that returns a [`ParameterEstimationProblem`](@ref) with true values and no
data, so the usual pattern is:

```julia
using ODEParameterEstimation
using ODEParameterEstimation.Examples

problem = sample_problem_data(lotka_volterra(); datasize = 51, time_interval = [0.0, 10.0])
results = estimate(problem)
```

```@example models
using ODEParameterEstimation
using ODEParameterEstimation.Examples

function describe(models...)
    println(rpad("model", 22), "states  parameters  measured")
    for model in models
        problem = model()
        println(rpad(problem.name, 22), rpad(length(problem.ic), 8),
            rpad(length(problem.p_true), 12), length(problem.measured_quantities))
    end
end
nothing # hide
```

## Good ones to start with

Small models that are solved quickly and accurately.

```@example models
describe(simple, lotka_volterra, vanderpol, harmonic, forced_decay, daisy_mamil3, repressilator)
```

## Models with something that cannot be determined

In each of these the measured quantities leave at least one parameter or
initial condition open. See [More than one answer](@ref).

```@example models
describe(trivial_unident, global_unident_test, substr_test, treatment)
```

## Harder models

Benchmarks from the identifiability literature. They take longer, and with
noisy data their estimates are less accurate.

```@example models
describe(hiv, biohydrogenation, fitzhugh_nagumo, daisy_mamil4, brusselator)
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

More models are defined than are exported, among them a set of control-systems
examples. `ODEParameterEstimation.Examples.available_models()` lists every
name, and each is called as `ODEParameterEstimation.Examples.dc_motor()`.

## Reference

```@docs
ODEParameterEstimation.Examples
```

```@autodocs
Modules = [ODEParameterEstimation.Examples]
Order = [:function]
Private = false
```
