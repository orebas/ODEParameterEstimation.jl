

function seir()
	parameters = @parameters a b nu
	states = @variables S(t) E(t) In(t) N(t)
	observables = @variables y1(t) y2(t)
	p_true = [0.2, 0.4, 0.15]  # recovery rate, infection rate, exposed->infected rate
	ic_true = [990.0, 10.0, 0.0, 1000.0]  # S0=990, E0=10, I0=0, N0=1000

	equations = [
		D(S) ~ -b * S * In / N,
		D(E) ~ b * S * In / N - nu * E,
		D(In) ~ nu * E - a * In,
		D(N) ~ 0,
	]
	measured_quantities = [y1 ~ In, y2 ~ N]

	model, mq = create_ordered_ode_system("SEIR", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"seir",
		model,
		mq,
		nothing,
		[0.0, 60.0],  # recommended timescale: 2 months to see disease progression
		nothing,  # solver
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end

function seir_m1()
	parameters = @parameters a b nu
	states = @variables S(t) E(t) In(t) N(t)
	observables = @variables y1(t) y2(t) y3(t)
	p_true = [0.2, 0.4, 0.15]
	ic_true = [990.0, 10.0, 0.0, 1000.0]

	equations = [
		D(S) ~ -b * S * In / N,
		D(E) ~ b * S * In / N - nu * E,
		D(In) ~ nu * E - a * In,
		D(N) ~ 0,
	]
	measured_quantities = [y1 ~ In, y2 ~ N, y3 ~ E]

	model, mq = create_ordered_ode_system("SEIR_M1", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"seir_m1",
		model,
		mq,
		nothing,
		[0.0, 60.0],
		nothing,
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end

function treatment()
	parameters = @parameters a b d g nu
	states = @variables In(t) N(t) S(t) Tr(t)
	observables = @variables y1(t) y2(t)
	p_true = [0.1,    # a: recovery rate without treatment
		0.8,    # b: infection rate (increased to make disease spread faster)
		2.0,    # d: treatment effectiveness multiplier (increased impact)
		0.3,    # g: rate of starting treatment (increased)
		0.1]    # nu: treatment clearance rate (decreased to maintain treatment longer)
	ic_true = [50.0,    # In: initial infected (increased initial outbreak)
		1000.0,  # N: total population
		950.0,   # S: initial susceptible
		0.0]     # Tr: initial under treatment

	equations = [
		D(In) ~ b * S * In / N + d * b * S * Tr / N - (a + g) * In,
		D(N) ~ 0,
		D(S) ~ -b * S * In / N - d * b * S * Tr / N,
		D(Tr) ~ g * In - nu * Tr,
	]
	measured_quantities = [y1 ~ Tr, y2 ~ N]

	model, mq = create_ordered_ode_system("treatment", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"treatment",
		model,
		mq,
		nothing, nothing,
		[0.0, 40.0],  # recommended timescale: 40 days to see treatment effects
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		6,
	)
end

function biohydrogenation()
	parameters = @parameters k5 k6 k7 k8 k9 k10
	states = @variables x4(t) x5(t) x6(t) x7(t)
	observables = @variables y1(t) y2(t)
	p_true = [0.5,    # k5: max rate of first reaction
		2.0,    # k6: Michaelis constant for first reaction
		0.3,    # k7: max rate of second reaction
		1.0,    # k8: Michaelis constant for second reaction
		0.2,    # k9: max rate of third reaction
		5.0]    # k10: carrying capacity
	ic_true = [4.0,   # x4: initial substrate
		0.0,   # x5: initial intermediate 1
		0.0,   # x6: initial intermediate 2
		0.0]   # x7: initial product

	equations = [
		D(x4) ~ -k5 * x4 / (k6 + x4),
		D(x5) ~ k5 * x4 / (k6 + x4) - k7 * x5 / (k8 + x5 + x6),
		D(x6) ~ k7 * x5 / (k8 + x5 + x6) - k9 * x6 * (k10 - x6) / k10,
		D(x7) ~ k9 * x6 * (k10 - x6) / k10,
	]
	measured_quantities = [y1 ~ x4, y2 ~ x5]

	model, mq = create_ordered_ode_system("BioHydrogenation", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"biohydrogenation",
		model,
		mq,
		nothing, nothing,
		[0.0, 36.0],  # recommended timescale: 36 hours for complete reaction chain
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		1,
	)
end

function biohydrogenation_m1()
	parameters = @parameters k5 k6 k7 k8 k9 k10
	states = @variables x4(t) x5(t) x6(t) x7(t)
	observables = @variables y1(t) y2(t) y3(t)
	p_true = [0.5, 2.0, 0.3, 1.0, 0.2, 5.0]
	ic_true = [4.0, 0.0, 0.0, 0.0]

	equations = [
		D(x4) ~ -k5 * x4 / (k6 + x4),
		D(x5) ~ k5 * x4 / (k6 + x4) - k7 * x5 / (k8 + x5 + x6),
		D(x6) ~ k7 * x5 / (k8 + x5 + x6) - k9 * x6 * (k10 - x6) / k10,
		D(x7) ~ k9 * x6 * (k10 - x6) / k10,
	]
	measured_quantities = [y1 ~ x4, y2 ~ x5, y3 ~ x6]

	model, mq = create_ordered_ode_system("BioHydrogenation_M1", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"biohydrogenation_m1",
		model,
		mq,
		nothing,
		[0.0, 36.0],
		nothing,
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		1,
	)
end

function repressilator()
	parameters = @parameters beta n alpha
	states = @variables m1(t) m2(t) m3(t) p1(t) p2(t) p3(t)
	observables = @variables y1(t) y2(t) y3(t)
	p_true = [2.0,    # beta: promoter strength
		2.0,    # n: Hill coefficient (used in rational approximation)
		1.0]    # alpha: protein degradation rate
	ic_true = [0.0, 0.0, 0.0,      # initial mRNA levels
		2.0, 1.0, 3.0]      # initial protein levels

	equations = [
		D(m1) ~ -m1 + beta * (1 / (1 + p3 * n)),
		D(m2) ~ -m2 + beta * (1 / (1 + p1 * n)),
		D(m3) ~ -m3 + beta * (1 / (1 + p2 * n)),
		D(p1) ~ -alpha * (p1 - m1),
		D(p2) ~ -alpha * (p2 - m2),
		D(p3) ~ -alpha * (p3 - m3),
	]
	measured_quantities = [y1 ~ p1, y2 ~ p2, y3 ~ p3]

	model, mq = create_ordered_ode_system("repressilator", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"repressilator",
		model,
		mq,
		nothing,
		[0.0, 24.0],  # recommended timescale: 24 hours to see multiple gene expression cycles
		nothing,  # solver
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end


function hiv()
	parameters = @parameters lm d beta a k u c q b h
	states = @variables x(t) y(t) v(t) w(t) z(t)
	observables = @variables y1(t) y2(t) y3(t) y4(t)
	p_true = [1.0,     # lm: production of immune cells (per day)
		0.01,    # d: natural death rate of immune cells (per day)
		2e-5,     # beta: infection rate (per virion per day)
		0.5,     # a: death rate of infected cells (per day)
		50.0,    # k: virus production by infected cells (per day)
		3.0,     # u: viral clearance rate (per day)
		0.05,    # c: immune response rate (per day)
		0.1,     # q: probability of immune cell activation
		0.002,   # b: death rate of immune cells (per day)
		0.1]     # h: death rate of activated cells (per day)
	ic_true = [1000.0,  # x: initial uninfected CD4+ T cells (per μL)
		1.0,     # y: initial infected cells (non-zero for practical identifiability of c, q)
		1e-3,    # v: initial virus (per mL)
		1.0,     # w: initial immune response
		0.0]     # z: initial activated immune cells

	equations = [
		D(x) ~ lm - d * x - beta * x * v,
		D(y) ~ beta * x * v - a * y,
		D(v) ~ k * y - u * v,
		D(w) ~ c * z * y * w - c * q * y * w - b * w,
		D(z) ~ c * q * y * w - h * z,
	]
	measured_quantities = [y1 ~ w, y2 ~ z, y3 ~ x, y4 ~ y + v]

	model, mq = create_ordered_ode_system("hiv", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"hiv",
		model,
		mq,
		nothing,
		[0.0, 25.0],  # recommended timescale: half year (180) to see full immune response, shortened for testing
		# switch back from 25 to 180 to see full immune response
		nothing,  # solver
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end


function hiv_old_wrong()
	parameters = @parameters lm d beta a k u c q b h
	states = @variables x(t) y(t) v(t) w(t) z(t)
	observables = @variables y1(t) y2(t) y3(t) y4(t)
	p_true = [1.0,     # lm: production of immune cells (per day)
		0.01,    # d: natural death rate of immune cells (per day)
		2e-5,     # beta: infection rate (per virion per day)
		0.5,     # a: death rate of infected cells (per day)
		50.0,    # k: virus production by infected cells (per day)
		3.0,     # u: viral clearance rate (per day)
		0.05,    # c: immune response rate (per day)
		0.1,     # q: probability of immune cell activation
		0.002,   # b: death rate of immune cells (per day)
		0.1]     # h: death rate of activated cells (per day)
	ic_true = [1000.0,  # x: initial uninfected CD4+ T cells (per μL)
		0.0,     # y: initial infected cells
		1e-3,    # v: initial virus (per mL)
		1.0,     # w: initial immune response
		0.0]     # z: initial activated immune cells

	equations = [
		D(x) ~ lm - d * x - beta * x * v,
		D(y) ~ beta * x * v - a * y,
		D(v) ~ k * y - u * v,
		D(w) ~ c * x * y * w - c * q * y * w - b * w,
		D(z) ~ c * q * y * w - h * z,
	]
	measured_quantities = [y1 ~ w, y2 ~ z, y3 ~ x, y4 ~ y + v]

	model, mq = create_ordered_ode_system("hiv_old_wrong", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"hiv",
		model,
		mq,
		nothing,
		[0.0, 25.0],  # recommended timescale: half year (180) to see full immune response, shortened for testing
		# switch back from 25 to 180 to see full immune response
		nothing,  # solver
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end
