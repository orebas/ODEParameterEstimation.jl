# Models that tests need and that are not examples for users.

using ODEParameterEstimation
using ModelingToolkit
using OrderedCollections: OrderedDict

function _fixture_problem(name, states, parameters, equations, measured_quantities, p_true, ic_true)
	model, mq = ODEParameterEstimation.create_ordered_ode_system(name, states, parameters, equations, measured_quantities)
	return ParameterEstimationProblem(name, model, mq, nothing, nothing, nothing,
		OrderedDict(parameters .=> p_true), OrderedDict(states .=> ic_true), 0)
end

# `beta` enters only through the product a^2 * b^2 * beta.
function substr_test()
	t, D = ModelingToolkit.t_nounits, ModelingToolkit.D_nounits
	parameters = @parameters a b beta
	states = @variables x1(t) x2(t) x3(t)
	@variables y1(t) y2(t) y3(t)
	equations = [
		D(x1) ~ -a * x2,
		D(x2) ~ b * (x1),
		D(x3) ~ a * b * beta * b * a * x3,
	]
	return _fixture_problem("substr_test", states, parameters, equations, [y1 ~ x1, y2 ~ x2, y3 ~ x3],
		[0.1, 0.2, 0.3], [2.0, 3.0, 4.0])
end

# Three states, each measured through its cube.
function threesp_cubed()
	t, D = ModelingToolkit.t_nounits, ModelingToolkit.D_nounits
	parameters = @parameters a b c
	states = @variables x1(t) x2(t) x3(t)
	@variables y1(t) y2(t) y3(t)
	equations = [
		D(x1) ~ -a * x2,
		D(x2) ~ -b * x1,
		D(x3) ~ -c * x1,
	]
	return _fixture_problem("threesp_cubed", states, parameters, equations,
		[y1 ~ x1 * x1 * x1, y2 ~ x2 * x2 * x2, y3 ~ x3 * x3 * x3], [0.1, 0.2, 0.3], [2.0, 3.0, 4.0])
end

# A cart with a pendulum on it. `sin(theta)` and `cos(theta)` of a state make
# it a model the package refuses.
function cart_pole()
	t, D = ModelingToolkit.t_nounits, ModelingToolkit.D_nounits
	parameters = @parameters M m l g F
	#                        kg kg m m/s² N
	# F is the INPUT (horizontal force on cart)

	states = @variables x(t) v(t) theta(t) omega(t)
	#                   m    m/s  rad      rad/s

	observables = @variables y1(t) y2(t)

	p_true = [
		1.0,    # M: cart mass (kg)
		0.1,    # m: pendulum mass (kg)
		0.5,    # l: pendulum length (m)
		9.81,   # g: gravity (m/s²)
		0.0,    # F: INPUT - applied force (N) - zero for free response
	]

	# Start near upright with small perturbation
	ic_true = [0.0, 0.0, 0.1, 0.0]  # Small angle from vertical

	# Full nonlinear equations of motion
	# Derived from Lagrangian: L = T - V with constraints
	# These are the standard cart-pole equations
	equations = [
		D(x) ~ v,
		D(v) ~ (F + m * sin(theta) * (l * omega^2 + g * cos(theta))) / (M + m * sin(theta)^2),
		D(theta) ~ omega,
		D(omega) ~ (-F * cos(theta) - m * l * omega^2 * cos(theta) * sin(theta) - (M + m) * g * sin(theta)) / (l * (M + m * sin(theta)^2)),
	]

	# Typically measure cart position and pendulum angle
	measured_quantities = [y1 ~ x, y2 ~ theta]

	model, mq = ODEParameterEstimation.create_ordered_ode_system("cart_pole", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"cart_pole",
		model,
		mq,
		nothing,
		[0.0, 5.0],  # Watch the pendulum fall (or be stabilized)
		nothing,
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end

# A DC motor with a sinusoidal supply voltage and the torque constant unknown,
# so that only the ratios Kt/J and b/J are determined. The example model
# `dc_motor` takes the torque constant as known.
function dc_motor_sinusoidal()
	t, D = ModelingToolkit.t_nounits, ModelingToolkit.D_nounits
    @parameters Kt J b

    # FIXED electrical parameters
    R_val = 2.0
    L_val = 0.5
    Kb_val = 0.1

    # Input parameters
    V0_val = 12.0
    Va_val = 2.0
    omega_val = 5.0

    @variables omega_m(t) i(t)
    @variables y1(t)

    p_true = [0.1, 0.01, 0.1]  # Kt, J, b
    ic_true = [0.0, 0.0]

    # Just write sin() directly!
    V_input = V0_val + Va_val * sin(omega_val * t)

    equations = [
        D(omega_m) ~ (Kt * i - b * omega_m) / J,
        D(i) ~ (V_input - R_val * i - Kb_val * omega_m) / L_val,
    ]

    measured_quantities = [y1 ~ omega_m]

    states = [omega_m, i]
    parameters = [Kt, J, b]

    model, mq = ODEParameterEstimation.create_ordered_ode_system("dc_motor_sinusoidal", states, parameters, equations, measured_quantities)

    return ParameterEstimationProblem(
        "dc_motor_sinusoidal",
        model,
        mq,
        nothing,
        [0.0, 5.0],
        nothing,
        OrderedDict(parameters .=> p_true),
        OrderedDict(states .=> ic_true),
        0,
    )
end
