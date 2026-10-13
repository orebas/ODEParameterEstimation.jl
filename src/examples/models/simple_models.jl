
function simple()
	parameters = @parameters a b
	states = @variables x1(t) x2(t)
	observables = @variables y1(t) y2(t)
	p_true = [0.4, 0.8]
	ic_true = [0.333, 0.667]

	equations = [
		D(x1) ~ -a * x2,
		D(x2) ~ b * x1,
	]
	measured_quantities = [y1 ~ x1, y2 ~ x2]

	model, mq = create_ordered_ode_system("simple", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"simple",
		model,
		mq,
		nothing,
		nothing,  # no specific timescale needed
		nothing,  # solver
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end

function simple_linear_combination()
	parameters = @parameters a b
	states = @variables x1(t) x2(t)
	observables = @variables y1(t) y2(t)
	p_true = [0.4, 0.8]
	ic_true = [1.0, 2.0]

	equations = [
		D(x1) ~ -a * x2,
		D(x2) ~ b * x1,
	]
	measured_quantities = [y1 ~ 2.0 * x1 + 0.5 * x2, y2 ~ 3.0 * x1 - 0.25 * x2]

	model, mq = create_ordered_ode_system("simple_linear_combination", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"simple_linear_combination",
		model,
		mq,
		nothing, nothing,
		nothing,
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end

function onesp_cubed()
	parameters = @parameters a
	states = @variables x1(t)
	observables = @variables y1(t)
	p_true = [0.1]
	ic_true = [2.0]

	equations = [
		D(x1) ~ -a * x1,
	]
	measured_quantities = [y1 ~ x1 * x1 * x1]

	model, mq = create_ordered_ode_system("onesp_cubed", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"onesp_cubed",
		model,
		mq,
		nothing, nothing,
		nothing,
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end

function threesp_cubed()
	parameters = @parameters a b c
	states = @variables x1(t) x2(t) x3(t)
	observables = @variables y1(t) y2(t) y3(t)
	p_true = [0.1, 0.2, 0.3]
	ic_true = [2.0, 3.0, 4.0]

	equations = [
		D(x1) ~ -a * x2,
		D(x2) ~ -b * x1,
		D(x3) ~ -c * x1,
	]
	measured_quantities = [y1 ~ x1 * x1 * x1, y2 ~ x2 * x2 * x2, y3 ~ x3 * x3 * x3]

	model, mq = create_ordered_ode_system("threesp_cubed", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"threesp_cubed",
		model,
		mq,
		nothing, nothing,
		nothing,
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end


function onevar_exp()
	parameters = @parameters a
	states = @variables x1(t)
	observables = @variables y1(t)
	p_true = [0.1]
	ic_true = [2.0]

	equations = [
		D(x1) ~ -a * x1,
	]
	measured_quantities = [y1 ~ x1]

	model, mq = create_ordered_ode_system("onevar_exp", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"onevar_exp",
		model,
		mq,
		nothing, nothing,
		nothing,
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end

