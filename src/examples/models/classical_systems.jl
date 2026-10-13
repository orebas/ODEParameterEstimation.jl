
function lotka_volterra()
	parameters = @parameters k1 k2 k3
	states = @variables r(t) w(t)
	observables = @variables y1(t)
	p_true = [1.0, 0.5, 0.3]  # prey growth, predation rate, predator death
	ic_true = [2.0, 1.0]      # initial prey and predator populations

	equations = [
		D(r) ~ k1 * r - k2 * r * w,
		D(w) ~ k2 * r * w - k3 * w,
	]
	measured_quantities = [y1 ~ r]

	model, mq = create_ordered_ode_system("Lotka_Volterra", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"lotka_volterra",
		model,
		mq,
		nothing,
		[0.0, 20.0],  # recommended timescale: 20 time units to see multiple oscillations
		nothing,  # solver
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end

function vanderpol()
	parameters = @parameters a b
	states = @variables x1(t) x2(t)
	observables = @variables y1(t) y2(t)
	p_true = [1.0, 1.0]      # standard van der pol parameters
	ic_true = [2.0, 0.0]     # typical initial displacement and velocity

	equations = [
		D(x1) ~ a * x2,
		D(x2) ~ -(x1) - b * (x1^2 - 1) * (x2),
	]
	measured_quantities = [y1 ~ x1, y2 ~ x2]

	model, mq = create_ordered_ode_system("vanderpol", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"vanderpol",
		model,
		mq,
		nothing, nothing,
		nothing,
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end

function brusselator()
	parameters = @parameters a b
	states = @variables X(t) Y(t)
	observables = @variables y1(t) y2(t)
	p_true = [1.0, 3.0]     # standard parameters for oscillatory behavior
	ic_true = [1.0, 1.0]    # initial concentrations

	equations = [
		D(X) ~ 1.0 - (b + 1) * X + a * X^2 * Y,
		D(Y) ~ b * X - a * X^2 * Y,
	]
	measured_quantities = [y1 ~ X, y2 ~ Y]

	model, mq = create_ordered_ode_system("brusselator", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"brusselator",
		model,
		mq,
		nothing,
		[0.0, 20.0],  # recommended timescale: 20 time units to see oscillations
		nothing,  # solver
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end



function harmonic()

	parameters = @parameters a b
	states = @variables x1(t) x2(t)
	observables = @variables y1(t) y2(t)

	# Example parameter values and initial conditions
	p_true  = [1.0, 1.0]       # e.g., a = 1.0, b = 1.0
	ic_true = [1.0, 0.0]       # e.g., x1(0) = 1.0, x2(0) = 0.0

	equations = [
		D(x1) ~ -a * x2,
		D(x2) ~ x1 / b,
	]

	# Define measured quantities (here, we simply measure the states)
	measured_quantities = [y1 ~ x1, y2 ~ x2]

	# Create the ODE system model
	model, mq = create_ordered_ode_system("harmonic", states, parameters, equations, measured_quantities)

	# Return a ParameterEstimationProblem with a recommended timespan.
	return ParameterEstimationProblem(
		"harmonic",
		model,
		mq,
		nothing,          # no noise model specified
		[0.0, 10.0],      # recommended timescale (adjust as needed)
		nothing,          # no custom solver specified
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,                 # model ID or additional flags
	)
end

"""
    forced_decay()

Minimal 1-state model with sinusoidal forcing: dx/dt = -a*x + b*sin(0.5*t).
The sin(0.5t) term triggers the `_obs_trfn_` transcendental transformation machinery,
making this a fast test model for transcendental UQ (runs in seconds vs minutes for CSTR).
"""
function forced_decay()
	parameters = @parameters a b
	states = @variables x(t)
	observables = @variables y1(t)

	p_true = [0.5, 2.0]
	ic_true = [1.0]

	equations = [
		D(x) ~ -a * x + b * sin(0.5 * t),
	]

	measured_quantities = [y1 ~ x]

	model, mq = create_ordered_ode_system("forced_decay", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"forced_decay",
		model,
		mq,
		nothing,
		[0.0, 10.0],
		nothing,
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end
