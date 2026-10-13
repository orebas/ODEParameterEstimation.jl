# A first example: fit a predator-prey model to data.
#
#     julia --startup-file=no examples/first_example.jl
#
# The manual walks through the same example:
# https://orebas.github.io/ODEParameterEstimation.jl/dev/getting_started/

using ODEParameterEstimation
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D

# The model, with four unknown rates.
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

# With your own measurements, pass them instead:
#
#     problem = ParameterEstimationProblem(lotka_volterra, [y1 ~ prey, y2 ~ predators];
#         data = (t = times, y1 = prey_counts, y2 = predator_counts))

results = estimate(problem)
display(results[1])
