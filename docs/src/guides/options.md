# Options

`estimate(problem)` uses defaults that are meant to work as they are. This page
covers the few options worth knowing about. [`EstimationOptions`](@ref) lists
them all, and marks the experimental ones, which may change in a minor
release.

Options are keywords of [`estimate`](@ref):

```julia
results = estimate(problem; seed = 1, progress = true)
```

To use the same options more than once, build them first:

```julia
options = EstimationOptions(seed = 1, progress = true)
results = estimate(problem, options)
```

## Watching a long run

A run prints nothing. With `progress = true` it prints a line as each stage
starts and finishes:

```
[14:02:11] ▶ Setup (identifiability)
[14:02:11] ✓ Setup (identifiability) (0.5s)
[14:02:11] ▶ SI Template (SIAN analysis)
[14:02:17] ✓ SI Template (SIAN analysis) (5.9s)
[14:02:17] ▶ Equation construction + Solving
[14:02:17] ▶ interpolator 1/9 agp_robust
```

## Going faster

The first time a model is estimated in a Julia session, Julia compiles code for
it. Nothing here changes that, and the second call shows the real speed.

After that, most of the time goes into solving the polynomial equations once
for every curve fit and every time point. Two options reduce the count.

`interpolators` chooses the curve fits. The default uses nine, and one is often
enough:

```julia
estimate(problem; interpolators = [InterpolatorAAA])   # exact or nearly exact data
estimate(problem; interpolators = [InterpolatorGP])    # noisy data
```

`shooting_points` is the number of time points the equations are solved at. The
default is 12.

```julia
estimate(problem; shooting_points = 4)
```

Both trade some reliability for speed. The best answer is chosen from all the
attempts, and with fewer attempts there is less to choose from.

## Refining estimates from noisy data

Each answer is refined by least squares against the data, starting from the
algebraic estimate. [Noisy data](@ref) shows what that buys.
`polish_solutions = false` turns it off and returns the algebraic estimates as
they are.

If you know the range the values must lie in, `opt_lb` and `opt_ub` keep the
refinement inside it. Each is a vector with the states first and the parameters
after them, in the order ModelingToolkit lists them:
`unknowns(problem.model.system)`, then `parameters(problem.model.system)`.

## Repeatable results

Several steps draw random numbers, so two runs on the same data can differ in
the last digits, and now and then in which answer comes first. A `seed` makes
runs repeat exactly:

```julia
estimate(problem; seed = 1)
```

`sample_problem_data` takes the same option, to fix the noise it adds. The seed
does not touch Julia's default random number generator, so your own `rand`
calls are unaffected.

## Standard errors

`compute_uncertainty = true` adds standard errors for the best answer. This
part of the package is experimental. See [Uncertainty](@ref).

## The ODE solver

Candidate answers are tested by simulating the model. The solver for that
belongs to the problem:

```julia
using OrdinaryDiffEq

problem = ParameterEstimationProblem(model, measured; data, solver = Rodas5P())
```

The default, `AutoVern9(Rodas5P())`, switches to a stiff method when it needs
to. `abstol` and `reltol`, both `1e-14` by default, are its tolerances.

## Seeing what the package is doing

`nooutput = false` prints the package's own account of a run: the stages, the
candidates and a summary table. `diagnostics = true` adds much more and writes
a log file. Both were built for developing the package, and neither is needed
to use it.
