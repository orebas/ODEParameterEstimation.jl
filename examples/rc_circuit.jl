# A capacitor charging through a resistor, written two ways. One way every
# parameter can be found from the data. The other way it cannot.
#
#     julia --startup-file=no examples/rc_circuit.jl
#
#     V_in ────[R]────┬────── V_c
#                     │
#                    [C]
#                     │
#     GND ────────────┴──────
#
# The capacitor voltage follows V_c' = (V_in - V_c) / (R C), and V_c is
# measured.

using ODEParameterEstimation
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D

@variables V_c(t) y(t)

# First way: the time constant tau = R C is one parameter.
@parameters tau V_in
@named circuit_tau = System([D(V_c) ~ (V_in - V_c) / tau], t)

problem = ParameterEstimationProblem(circuit_tau, [y ~ V_c]; true_values = [tau => 1.0, V_in => 5.0, V_c => 0.0])
problem = sample_problem_data(problem; datasize = 101, time_interval = [0.0, 5.0])

# Both are recovered: tau from how fast the voltage rises, V_in from where it
# levels off.
results_tau = estimate(problem)
display(results_tau[1])

# Second way: R and C are separate parameters. Only their product enters the
# equation, so no voltage measurement can tell R = 2, C = 0.5 from R = 1, C = 1.
@parameters R C
@named circuit_separate = System([D(V_c) ~ (V_in - V_c) / (R * C)], t)

problem = ParameterEstimationProblem(circuit_separate, [y ~ V_c];
	true_values = [R => 2.0, C => 0.5, V_in => 5.0, V_c => 0.0])
problem = sample_problem_data(problem; datasize = 101, time_interval = [0.0, 5.0])

# The result says so: R and C are listed as not determined, and the values
# shown for them are arbitrary. V_in is still recovered.
results_separate = estimate(problem)
display(results_separate[1])
println("Not determined: ", join(results_separate[1].all_unidentifiable, ", "))
