# A model with a constant input: a quantity that decays while it is topped up
# at a steady rate.
#
#     julia --startup-file=no examples/constant_input.jl
#
# Think of a drug level under a steady drip, or a room with a heater that is
# always on. The input is one more unknown number: here it is estimated along
# with the decay rate. An input that changes with time is written as a function
# of `t`; the manual's page "Which models work" shows how.

using ODEParameterEstimation
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D

# x decays at the rate a and is replenished at the rate u.
@parameters a u
@variables x(t) y(t)
@named drip = System([D(x) ~ -a * x + u], t)

# x starts at zero and rises towards u / a = 2. The measurements are simulated
# from known values.
problem = ParameterEstimationProblem(drip, [y ~ x]; true_values = [a => 0.5, u => 1.0, x => 0.0])
problem = sample_problem_data(problem; datasize = 101, time_interval = [0.0, 10.0])

# Both numbers are recovered: how fast x rises shows a, and where it levels
# off shows u / a.
results = estimate(problem)
display(results[1])
