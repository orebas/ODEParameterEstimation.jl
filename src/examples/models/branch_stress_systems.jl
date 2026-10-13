
function latent_subpopulation_branch()
	parameters = @parameters a1 a2 a3 b1 b2 b3
	states = @variables S(t) I1(t) I2(t) I3(t) R(t)
	observables = @variables y1(t) y2(t) y3(t)
	p_true = [0.15, 0.35, 0.65, 0.20, 0.45, 0.90]
	ic_true = [0.82, 0.08, 0.05, 0.03, 0.02]

	equations = [
		D(S) ~ -b1 * S * I1 - b2 * S * I2 - b3 * S * I3,
		D(I1) ~ b1 * S * I1 - a1 * I1,
		D(I2) ~ b2 * S * I2 - a2 * I2,
		D(I3) ~ b3 * S * I3 - a3 * I3,
		D(R) ~ a1 * I1 + a2 * I2 + a3 * I3,
	]
	measured_quantities = [y1 ~ S, y2 ~ I1 + I2 + I3, y3 ~ R]

	model, mq = create_ordered_ode_system(
		"latent_subpopulation_branch",
		states,
		parameters,
		equations,
		measured_quantities,
	)

	return ParameterEstimationProblem(
		"latent_subpopulation_branch",
		model,
		mq,
		nothing,
		[0.0, 12.0],
		nothing,
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end

function latent_subpopulation_observed_control()
	parameters = @parameters a1 a2 a3 b1 b2 b3
	states = @variables S(t) I1(t) I2(t) I3(t) R(t)
	observables = @variables y1(t) y2(t) y3(t) y4(t) y5(t)
	p_true = [0.15, 0.35, 0.65, 0.20, 0.45, 0.90]
	ic_true = [0.82, 0.08, 0.05, 0.03, 0.02]

	equations = [
		D(S) ~ -b1 * S * I1 - b2 * S * I2 - b3 * S * I3,
		D(I1) ~ b1 * S * I1 - a1 * I1,
		D(I2) ~ b2 * S * I2 - a2 * I2,
		D(I3) ~ b3 * S * I3 - a3 * I3,
		D(R) ~ a1 * I1 + a2 * I2 + a3 * I3,
	]
	measured_quantities = [y1 ~ S, y2 ~ I1, y3 ~ I2, y4 ~ I3, y5 ~ R]

	model, mq = create_ordered_ode_system(
		"latent_subpopulation_observed_control",
		states,
		parameters,
		equations,
		measured_quantities,
	)

	return ParameterEstimationProblem(
		"latent_subpopulation_observed_control",
		model,
		mq,
		nothing,
		[0.0, 12.0],
		nothing,
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end

function receptor_subtype_binding_branch()
	parameters = @parameters R1tot R2tot kon1 kon2 koff1 koff2
	states = @variables L(t) Ca(t) Cb(t)
	observables = @variables y1(t) y2(t)
	p_true = [1.10, 0.70, 0.80, 1.30, 0.40, 0.60]
	ic_true = [2.00, 0.10, 0.20]

	equations = [
		D(L) ~ -kon1 * L * (R1tot - Ca) + koff1 * Ca - kon2 * L * (R2tot - Cb) + koff2 * Cb,
		D(Ca) ~ kon1 * L * (R1tot - Ca) - koff1 * Ca,
		D(Cb) ~ kon2 * L * (R2tot - Cb) - koff2 * Cb,
	]
	measured_quantities = [y1 ~ L, y2 ~ Ca + Cb]

	model, mq = create_ordered_ode_system(
		"receptor_subtype_binding_branch",
		states,
		parameters,
		equations,
		measured_quantities,
	)

	return ParameterEstimationProblem(
		"receptor_subtype_binding_branch",
		model,
		mq,
		nothing,
		[0.0, 8.0],
		nothing,
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end

function receptor_subtype_binding_observed_control()
	parameters = @parameters R1tot R2tot kon1 kon2 koff1 koff2
	states = @variables L(t) Ca(t) Cb(t)
	observables = @variables y1(t) y2(t) y3(t)
	p_true = [1.10, 0.70, 0.80, 1.30, 0.40, 0.60]
	ic_true = [2.00, 0.10, 0.20]

	equations = [
		D(L) ~ -kon1 * L * (R1tot - Ca) + koff1 * Ca - kon2 * L * (R2tot - Cb) + koff2 * Cb,
		D(Ca) ~ kon1 * L * (R1tot - Ca) - koff1 * Ca,
		D(Cb) ~ kon2 * L * (R2tot - Cb) - koff2 * Cb,
	]
	measured_quantities = [y1 ~ L, y2 ~ Ca, y3 ~ Cb]

	model, mq = create_ordered_ode_system(
		"receptor_subtype_binding_observed_control",
		states,
		parameters,
		equations,
		measured_quantities,
	)

	return ParameterEstimationProblem(
		"receptor_subtype_binding_observed_control",
		model,
		mq,
		nothing,
		[0.0, 8.0],
		nothing,
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end
