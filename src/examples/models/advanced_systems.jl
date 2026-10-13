
# Historical Crauste example adapted from ParameterEstimation.jl (GPL-3.0),
# examples/all-global/crauste.jl; a preserved source is revision
# 9e7adc33a4a11954acde0214f482cffea9551170. See docs/internal/registry_preparation.md.
function crauste()
	parameters = @parameters mu_N mu_EE mu_LE mu_LL mu_M mu_P mu_PE mu_PL delta_NE delta_EL delta_LM rho_E rho_P
	parameters = @parameters muN muEE muLE muLL muM muP muPE muPL deltaNE deltaEL deltaLM rhoE rhoP

	states = @variables n(t) e(t) s(t) m(t) p(t)
	observables = @variables y1(t) y2(t) y3(t) y4(t)
	p_true = [
		0.75,            # mu_N
		0.0000216,       # mu_EE
		0.000000036,     # mu_LE
		0.0000075,       # mu_LL
		0.0,             # mu_M
		0.055,           # mu_P
		0.00000018,      # mu_PE
		0.000018,        # mu_PL
		0.009,           # delta_NE
		0.59,            # delta_EL
		0.025,           # delta_LM
		0.64,            # rho_E
		0.15,             # rho_P
	]
	ic_true = [8090.0, 0.0, 0.0, 0.0, 1.0]
	#CORRECT
	#equations = [
	#	D(N) ~ -N * mu_N - N * P * delta_NE,
	#	D(E) ~ N * P * delta_NE + E * (rho_E * P - mu_EE * E - delta_EL),
	#	D(L) ~ delta_EL * E - L * (mu_LL * L + mu_LE * E + delta_LM),
	# same as 
	#delta_EL * E  -mu_LL * s * s - mu_LE * E * s - delta_LM * s
	#	D(M) ~ L * delta_LM - mu_M * M,
	#	D(P) ~ P * (rho_P * P - mu_PE * E - mu_PL * L - mu_P),
	#]
	#WRONG
	equations = [
		D(n) ~ -1 * n * muN - n * p * deltaNE,
		D(e) ~ n * p * deltaNE - e * e * muEE - e * deltaEL + e * p * rhoE,
		D(s) ~ s * deltaEL - s * deltaLM - s * s * muLL - e * s * muLE,
		D(m) ~ s * deltaLM - muM * m,
		D(p) ~ p * p * rhoP - p * muP - e * p * muPE - s * p * muPL,
	]


	measured_quantities = [y1 ~ n, y2 ~ e, y3 ~ s + m, y4 ~ p]

	model, mq = create_ordered_ode_system("crauste", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"crauste",
		model,
		mq,
		nothing,
		[0.0, 1.0],  # recommended timescale: 100 days for cell population dynamics
		nothing,  # solver
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end


function crauste_corrected()
	parameters = @parameters mu_N mu_EE mu_LE mu_LL mu_M mu_P mu_PE mu_PL delta_NE delta_EL delta_LM rho_E rho_P
	states = @variables N(t) E(t) L(t) M(t) P(t)
	observables = @variables y1(t) y2(t) y3(t) y4(t)
	p_true = [
		0.75,            # mu_N
		0.0000216,       # mu_EE
		0.000000036,     # mu_LE
		0.0000075,       # mu_LL
		0.0,             # mu_M
		0.055,           # mu_P
		0.00000018,      # mu_PE
		0.000018,        # mu_PL
		0.009,           # delta_NE
		0.59,            # delta_EL
		0.025,           # delta_LM
		0.64,            # rho_E
		0.15,             # rho_P
	]
	ic_true = [8090.0, 0.0, 0.0, 0.0, 1.0]

	equations = [
		D(N) ~ -N * mu_N - N * P * delta_NE,
		D(E) ~ N * P * delta_NE + E * (rho_E * P - mu_EE * E - delta_EL),
		D(L) ~ delta_EL * E - L * (mu_LL * L + mu_LE * E + delta_LM),
		D(M) ~ L * delta_LM - mu_M * M,
		D(P) ~ P * (rho_P * P - mu_PE * E - mu_PL * L - mu_P),
	]
	measured_quantities = [y1 ~ N, y2 ~ E, y3 ~ L + M, y4 ~ P]

	model, mq = create_ordered_ode_system("crauste", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"crauste_corrected",
		model,
		mq,
		nothing,
		[0.0, 25.0],  # recommended timescale: 100 days for cell population dynamics
		nothing,  # solver
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end



function crauste_revised()
	parameters = @parameters mu_N mu_EE mu_LE mu_LL mu_M mu_P mu_PE mu_PL delta_NE delta_EL delta_LM rho_E rho_P mu_EL mu_E mu_L
	states = @variables N(t) E(t) L(t) M(t) P(t)
	observables = @variables y1(t) y2(t) y3(t) y4(t)
	p_true = [
		0.75,            # mu_N
		0.0000216,       # mu_EE
		0.000000036,     # mu_LE
		0.0000075,       # mu_LL
		0.0,             # mu_M
		0.055,           # mu_P
		0.00000018,      # mu_PE
		0.000018,        # mu_PL
		0.009,           # delta_NE
		0.59,            # delta_EL
		0.025,           # delta_LM
		0.64,            # rho_E
		0.15,             # rho_P
		0.0,             # mu_EL (new)
		0.0,             # mu_E (new)
		0.0,             # mu_L (new)
	]
	ic_true = [8090.0, 0.0, 0.0, 0.0, 1.0]

	equations = [
		D(N) ~ -N * mu_N - N * P * delta_NE,
		D(E) ~ N * P * delta_NE + E * (rho_E * P - mu_EE * E - mu_EL * L - mu_E - delta_EL),
		D(L) ~ delta_EL * E - L * (mu_LL * L + mu_LE * E + mu_L + delta_LM),
		D(M) ~ L * delta_LM - mu_M * M,
		D(P) ~ P * (rho_P * P - mu_PE * E - mu_PL * L - mu_P),
	]
	measured_quantities = [y1 ~ N, y2 ~ E, y3 ~ L + M, y4 ~ P]

	model, mq = create_ordered_ode_system("crauste_revised", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"crauste_revised",
		model,
		mq,
		nothing,
		[0.0, 25.0],  # recommended timescale: 100 days for cell population dynamics
		nothing,  # solver
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end


function daisy_ex3()
	parameters = @parameters p1 p3 p4 p6 p7
	states = @variables x1(t) x2(t) x3(t) u0(t)
	observables = @variables y1(t) y2(t)
	p_true = [0.167, 0.333, 0.5, 0.667, 0.833]
	ic_true = [0.2, 0.4, 0.6, 0.8]

	equations = [
		D(x1) ~ -1.0 * p1 * x1 + x2 + u0,
		D(x2) ~ p3 * x1 - p4 * x2 + x3,
		D(x3) ~ p6 * x1 - p7 * x3,
		D(u0) ~ 1.0,
	]
	measured_quantities = [y1 ~ x1, y2 ~ u0]

	model, mq = create_ordered_ode_system("DAISY_ex3", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"daisy_ex3",
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

function daisy_mamil3()
	parameters = @parameters a12 a13 a21 a31 a01
	states = @variables x1(t) x2(t) x3(t)
	observables = @variables y1(t) y2(t)
	p_true = [0.167, 0.333, 0.5, 0.667, 0.833]
	ic_true = [0.25, 0.5, 0.75]

	equations = [
		D(x1) ~ -(a21 + a31 + a01) * x1 + a12 * x2 + a13 * x3,
		D(x2) ~ a21 * x1 - a12 * x2,
		D(x3) ~ a31 * x1 - a13 * x3,
	]
	measured_quantities = [y1 ~ x1, y2 ~ x2]

	model, mq = create_ordered_ode_system("DAISY_mamil3", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"daisy_mamil3",
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

function daisy_mamil4()
	parameters = @parameters k01 k12 k13 k14 k21 k31 k41
	states = @variables x1(t) x2(t) x3(t) x4(t)
	observables = @variables y1(t) y2(t) y3(t)
	p_true = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
	ic_true = [0.2, 0.4, 0.6, 0.8]

	equations = [
		D(x1) ~ -k01 * x1 + k12 * x2 + k13 * x3 + k14 * x4 - k21 * x1 - k31 * x1 - k41 * x1,
		D(x2) ~ -k12 * x2 + k21 * x1,
		D(x3) ~ -k13 * x3 + k31 * x1,
		D(x4) ~ -k14 * x4 + k41 * x1,
	]
	measured_quantities = [y1 ~ x1, y2 ~ x2, y3 ~ x3 + x4]

	model, mq = create_ordered_ode_system("DAISY_mamil4", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"daisy_mamil4",
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

function daisy_mamil4_m1()
	parameters = @parameters k01 k12 k13 k14 k21 k31 k41
	states = @variables x1(t) x2(t) x3(t) x4(t)
	observables = @variables y1(t) y2(t) y3(t) y4(t)
	p_true = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
	ic_true = [0.2, 0.4, 0.6, 0.8]

	equations = [
		D(x1) ~ -k01 * x1 + k12 * x2 + k13 * x3 + k14 * x4 - k21 * x1 - k31 * x1 - k41 * x1,
		D(x2) ~ -k12 * x2 + k21 * x1,
		D(x3) ~ -k13 * x3 + k31 * x1,
		D(x4) ~ -k14 * x4 + k41 * x1,
	]
	measured_quantities = [y1 ~ x1, y2 ~ x2, y3 ~ x3 + x4, y4 ~ x3]

	model, mq = create_ordered_ode_system("DAISY_mamil4_M1", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"daisy_mamil4_m1",
		model,
		mq,
		nothing,
		nothing,
		nothing,
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end

function fitzhugh_nagumo()
	parameters = @parameters g a b
	states = @variables V(t) R(t)
	observables = @variables y1(t)
	p_true = [3.0, 0.2, 0.2]  # time scale, threshold, recovery
	ic_true = [-1.0, 0.0]     # initial voltage and recovery

	equations = [
		D(V) ~ g * (V - V^3 / 3 + R),
		D(R) ~ 1 / g * (V - a + b * R),
	]
	measured_quantities = [y1 ~ V]

	model, mq = create_ordered_ode_system("fitzhugh-nagumo", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"fitzhugh_nagumo",
		model,
		mq,
		nothing,
		[0.0, 0.03],  # recommended timescale: 30ms for action potential
		nothing,  # solver
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end

function lv_periodic()
	parameters = @parameters a b c d
	states = @variables x1(t) x2(t)
	observables = @variables y1(t) y2(t)
	p_true = [1.5, 0.9, 3.0, 0.8]
	ic_true = [2.0, 0.5]

	equations = [
		D(x1) ~ a * x1 - b * x1 * x2,
		D(x2) ~ -c * x2 + d * x1 * x2,
	]
	measured_quantities = [y1 ~ x1, y2 ~ x2]

	model, mq = create_ordered_ode_system("lv_periodic", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"lv_periodic",
		model,
		mq,
		nothing,
		[0.0, 15.0],  # recommended timescale: 15 time units to see periodic behavior
		nothing,  # solver
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end

function slowfast()
	parameters = @parameters k1 k2 # eB
	states = @variables xA(t) xB(t) xC(t) eA(t) eC(t) eB(t)
	observables = @variables y1(t) y2(t) y3(t) y4(t)
	p_true = [0.25, 0.5] #, 0.75]
	ic_true = [0.166, 0.333, 0.5, 0.666, 0.833, 0.75]

	equations = [
		D(xA) ~ -k1 * xA,
		D(xB) ~ k1 * xA - k2 * xB,
		D(xC) ~ k2 * xB,
		D(eA) ~ 0,
		D(eC) ~ 0,
		D(eB) ~ 0,  # Add trivial equation for eB
	]
	measured_quantities = [y1 ~ xC, y2 ~ eA * xA + eB * xB + eC * xC, y3 ~ eA, y4 ~ eC]

	model, mq = create_ordered_ode_system("slowfast", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"slowfast",
		model,
		mq,
		nothing,
		[0.0, 10.0],  # recommended timescale: 10 time units to see separation of scales
		nothing,  # solver
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end

function slow_fast_m1()
	parameters = @parameters k1 k2
	states = @variables xA(t) xB(t) xC(t) eA(t) eC(t) eB(t)
	observables = @variables y1(t) y2(t) y3(t) y4(t) y5(t)
	p_true = [0.25, 0.5]
	ic_true = [0.166, 0.333, 0.5, 0.666, 0.833, 0.75]

	equations = [
		D(xA) ~ -k1 * xA,
		D(xB) ~ k1 * xA - k2 * xB,
		D(xC) ~ k2 * xB,
		D(eA) ~ 0,
		D(eC) ~ 0,
		D(eB) ~ 0,
	]
	measured_quantities = [y1 ~ xC, y2 ~ eA * xA + eB * xB + eC * xC, y3 ~ eA, y4 ~ eC, y5 ~ eB]

	model, mq = create_ordered_ode_system("slow_fast_m1", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"slow_fast_m1",
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

slowfast_m1() = slow_fast_m1()

function sirsforced()
	parameters = @parameters b0 b1 g M mu nu
	states = @variables i(t) r(t) s(t) x1(t) x2(t)
	observables = @variables y1(t) y2(t)
	p_true = [0.143, 0.286, 0.429, 0.571, 0.714, 0.857]
	ic_true = [0.167, 0.333, 0.5, 0.667, 0.833]

	equations = [
		D(i) ~ b0 * (1.0 + b1 * x1) * i * s - (nu + mu) * i,
		D(r) ~ nu * i - (mu + g) * r,
		D(s) ~ mu - mu * s - b0 * (1.0 + b1 * x1) * i * s + g * r,
		D(x1) ~ -M * x2,
		D(x2) ~ M * x1,
	]
	measured_quantities = [y1 ~ i, y2 ~ r]

	model, mq = create_ordered_ode_system("sirsforced", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"sirsforced",
		model,
		mq,
		nothing,
		[0.0, 30.0],  # recommended timescale for epidemiological dynamics
		nothing,  # solver
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		3,
	)
end





function allee_competition()
	parameters = @parameters r1 r2 K1 K2 A1 A2 b12 b21
	states = @variables N1(t) N2(t)
	observables = @variables y1(t) y2(t)
	p_true = [1.0,     # r1: growth rate species 1
		0.8,     # r2: growth rate species 2
		10.0,    # K1: carrying capacity 1
		8.0,     # K2: carrying capacity 2
		0.2,     # A1: Allee threshold 1
		0.1,     # A2: Allee threshold 2
		0.3,     # b12: competition effect of 2 on 1
		0.4]     # b21: competition effect of 1 on 2
	ic_true = [2.0,    # N1: initial population 1
		1.5]    # N2: initial population 2

	equations = [
		D(N1) ~ r1 * N1 * (1 - N1 / K1) * (N1 - A1) / K1 - b12 * N1 * N2,
		D(N2) ~ r2 * N2 * (1 - N2 / K2) * (N2 - A2) / K2 - b21 * N1 * N2,
	]
	measured_quantities = [y1 ~ N1, y2 ~ N2]

	model, mq = create_ordered_ode_system("allee_competition", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"allee_competition",
		model,
		mq,
		nothing,
		nothing,
		nothing,
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end

function two_compartment_pk()
	parameters = @parameters k12 k21 ke V1 V2
	states = @variables C1(t) C2(t)
	observables = @variables y1(t)
	p_true = [0.5,     # k12: transfer rate central to peripheral
		0.25,    # k21: transfer rate peripheral to central
		0.15,    # ke: elimination rate
		1.0,     # V1: volume of central compartment
		2.0]     # V2: volume of peripheral compartment
	ic_true = [10.0,   # C1: initial concentration in central compartment
		0.0]    # C2: initial concentration in peripheral compartment

	equations = [
		D(C1) ~ -k12 * C1 + k21 * C2 * V2 / V1 - ke * C1,
		D(C2) ~ k12 * C1 * V1 / V2 - k21 * C2,
	]
	measured_quantities = [y1 ~ C1]  # typically only central compartment is measured

	model, mq = create_ordered_ode_system("two_compartment_pk", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"two_compartment_pk",
		model,
		mq,
		nothing,
		[0.0, 48.0],  # recommended timescale: 48 hours for drug distribution and elimination
		nothing,  # solver
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end
