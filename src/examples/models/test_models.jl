
function substr_test()
	parameters = @parameters a b beta
	states = @variables x1(t) x2(t) x3(t)
	observables = @variables y1(t) y2(t) y3(t)
	p_true = [0.1, 0.2, 0.3]
	ic_true = [2.0, 3.0, 4.0]

	equations = [
		D(x1) ~ -a * x2,
		D(x2) ~ b * (x1),
		D(x3) ~ a * b * beta * b * a * x3,
	]
	measured_quantities = [y1 ~ x1, y2 ~ x2, y3 ~ x3]

	model, mq = create_ordered_ode_system("substr_test", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"substr_test",
		model,
		mq,
		nothing,
		nothing,  # no specific timescale needed for test model
		nothing,  # solver
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end

function global_unident_test()
	parameters = @parameters a b c d
	states = @variables x1(t) x2(t) x3(t)
	observables = @variables y1(t) y2(t)
	p_true = [0.1, 0.2, 0.3, 0.4]
	ic_true = [2.0, 3.0, 4.0]

	equations = [
		D(x1) ~ -a * x1,
		D(x2) ~ (b + c) * (x1),
		D(x3) ~ d * x1,
	]
	measured_quantities = [y1 ~ x1, y2 ~ x2]

	model, mq = create_ordered_ode_system("global_unident_test", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"global_unident_test",
		model,
		mq,
		nothing, nothing,
		nothing,
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		4,
	)
end

function sum_test()
	parameters = @parameters a b c
	states = @variables x1(t) x2(t) x3(t)
	observables = @variables y1(t)
	p_true = [0.1, 0.2, 0.3]
	ic_true = [2.0, 3.0, 4.0]

	equations = [
		D(x1) ~ -a * x1,
		D(x2) ~ b * (x2),
		D(x3) ~ c * (x1 + x2),
	]
	measured_quantities = [y1 ~ x3]

	model, mq = create_ordered_ode_system("sum_test", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"sum_test",
		model,
		mq,
		nothing,
		nothing,  # no specific timescale needed for test model
		nothing,  # solver
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		3,
	)
end

function trivial_unident()
	parameters = @parameters a b
	states = @variables x1(t)
	observables = @variables y1(t)
	p_true = [0.4 0.6]
	ic_true = [2.0]

	equations = [
		D(x1) ~ (a + b) * x1,
	]
	measured_quantities = [y1 ~ x1]

	model, mq = create_ordered_ode_system("trivial_unident", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"trivial_unident",
		model,
		mq,
		nothing,
		nothing,  # no specific timescale needed for test model
		nothing,  # solver
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		4,
	)
end
