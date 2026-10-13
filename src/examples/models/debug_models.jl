
function biohydrogenation_debug()
	name = "biohydrogenation_debug"
	parameters = @parameters k5 k6 k7 k8 k9 k10
	states = @variables x4(t) x5(t) x6(t) x7(t)
	observables = @variables y1(t) y2(t)
	state_equations = [
		D(x4) ~ - k5 * x4 / (k6 + x4),
		D(x5) ~ k5 * x4 / (k6 + x4) - k7 * x5/(k8 + x5 + x6),
		D(x6) ~ k7 * x5 / (k8 + x5 + x6) - k9 * x6 * (k10 - x6) / k10,
		D(x7) ~ k9 * x6 * (k10 - x6) / k10,
	]
	measured_quantities = [
		y1 ~ x4,
		y2 ~ x5,
	]
	ic = [0.45, 0.813, 0.871, 0.407]
	p_true = [0.539, 0.672, 0.582, 0.536, 0.439, 0.617]
	time_interval = [-0.5, 0.5]

	model, mq = ODEParameterEstimation.create_ordered_ode_system(
		name,
		states,
		parameters,
		state_equations,
		measured_quantities,
	)

	return ParameterEstimationProblem(
		name,
		model,
		mq,
		nothing,
		time_interval,
		nothing,
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic),
		0,
	)
end
