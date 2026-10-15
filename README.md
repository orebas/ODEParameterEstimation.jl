# ODEParameterEstimation.jl

[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://orebas.github.io/ODEParameterEstimation.jl/dev/)
[![Build Status](https://github.com/orebas/ODEParameterEstimation.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/orebas/ODEParameterEstimation.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/orebas/ODEParameterEstimation.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/orebas/ODEParameterEstimation.jl)

ODEParameterEstimation.jl fits the parameters and initial conditions of an ODE
model to time-series data.

- **No starting guesses and no bounds.** You give the model and the data.
- **Every answer, not just one.** If two parameter sets explain the data
  equally well, you get both.
- **It tells you what the data cannot determine.** Parameters that no amount of
  this data could pin down are flagged.

It works by solving equations rather than by searching. Derivatives of the data
are estimated, the model turns them into polynomial equations for the unknowns,
and those equations are solved for all their solutions, which are then refined
against the data. Because the equations must be polynomial, the model has to be
built from polynomials and ratios of polynomials.

## Installation

Julia 1.12 or later is required. The package is not in the General registry
yet, so install it from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/orebas/ODEParameterEstimation.jl")
Pkg.add("ModelingToolkit")
```

## A first example

Models are written with [ModelingToolkit](https://docs.sciml.ai/ModelingToolkit/stable/).
This is a predator-prey model with four unknown rates.

```julia
using ODEParameterEstimation, ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D

@parameters α β γ δ
@variables prey(t) predators(t) y1(t) y2(t)
@named lotka_volterra = System([
    D(prey) ~ α * prey - β * prey * predators,
    D(predators) ~ δ * prey * predators - γ * predators,
], t)

# Both populations were counted. Here the counts are simulated from known values.
problem = ParameterEstimationProblem(lotka_volterra, [y1 ~ prey, y2 ~ predators];
    true_values = [α => 1.1, β => 0.4, γ => 0.4, δ => 0.1, prey => 1.0, predators => 0.5])
problem = sample_problem_data(problem; datasize = 101, time_interval = [0.0, 10.0])

results = estimate(problem)
results[1]
```

```
ParameterEstimationResult
  Parameters
    α = 1.1
    β = 0.4
    γ = 0.4
    δ = 0.1
  Initial conditions (t = 0)
    prey(t) = 1
    predators(t) = 0.5
  Fit error: 7.41e-26
```

With your own measurements, pass them instead of simulating:

```julia
problem = ParameterEstimationProblem(lotka_volterra, [y1 ~ prey, y2 ~ predators];
    data = (t = times, y1 = prey_counts, y2 = predator_counts))
```

The first time a model is estimated in a session, Julia compiles code for it.
For this example that takes under a minute, and after that a model of this
size takes seconds.

## Documentation

The [manual](https://orebas.github.io/ODEParameterEstimation.jl/dev/) starts
with a [walk through this example](https://orebas.github.io/ODEParameterEstimation.jl/dev/getting_started/)
and continues with:

- [using your own data](https://orebas.github.io/ODEParameterEstimation.jl/dev/tutorials/own_data/)
- [noisy data](https://orebas.github.io/ODEParameterEstimation.jl/dev/tutorials/noisy_data/)
- [models with more than one answer](https://orebas.github.io/ODEParameterEstimation.jl/dev/tutorials/identifiability/)
- [which models work](https://orebas.github.io/ODEParameterEstimation.jl/dev/guides/models/)
- [how the method works](https://orebas.github.io/ODEParameterEstimation.jl/dev/guides/how_it_works/)
- [forty ready-made example models](https://orebas.github.io/ODEParameterEstimation.jl/dev/guides/example_models/)

The [`examples`](examples) folder has scripts to run and read.

## Citing

If you use this package in your work, please cite the papers it is based on:

> O. Bassik, A. Demin, A. Ovchinnikov. *Practical algebraic parameter estimation
> for noisy data via Gaussian process regression.*
> [arXiv:2609.30451](https://arxiv.org/abs/2609.30451) (2026).

> O. Bassik, Y. Berman, S. Go, H. Hong, I. Ilmer, A. Ovchinnikov, C. Rackauckas,
> P. Soto, C. Yap. *Robust parameter estimation for rational ordinary
> differential equations.* Applied Mathematics and Computation 509 (2026).
> [doi:10.1016/j.amc.2025.129638](https://doi.org/10.1016/j.amc.2025.129638)

BibTeX entries are in [`CITATION.bib`](CITATION.bib).

## Contributing

Bug reports and pull requests are welcome. The manual's
[contributing page](https://orebas.github.io/ODEParameterEstimation.jl/dev/contributing/)
says how to run the tests and build the documentation.

## License

[GPL-3.0](LICENSE). The package includes code adapted from GaussianProcesses.jl
and SIAN-Julia, which keep their MIT notices beside the source.
