# Models with more than one answer, each beside a version measured fully
# enough to have one.

const _SUBPOPULATION_VALUES = (p_true = [0.15, 0.35, 0.65, 0.20, 0.45, 0.90], ic_true = [0.82, 0.08, 0.05, 0.03, 0.02])

function _subpopulation_model()
	parameters = @parameters a1 a2 a3 b1 b2 b3
	states = @variables S(t) I1(t) I2(t) I3(t) R(t)
	equations = [
		D(S) ~ -b1 * S * I1 - b2 * S * I2 - b3 * S * I3,
		D(I1) ~ b1 * S * I1 - a1 * I1,
		D(I2) ~ b2 * S * I2 - a2 * I2,
		D(I3) ~ b3 * S * I3 - a3 * I3,
		D(R) ~ a1 * I1 + a2 * I2 + a3 * I3,
	]
	return parameters, states, equations
end

"""
	hidden_subpopulations()

An epidemic in which the infected fall into three groups, each with its own
infection rate `b` and recovery rate `a`. The susceptible, the recovered and
the *total* number infected are measured, so the three groups are never seen
apart.

Nothing in the data says which group is which. There are six answers, one for
each ordering of the three groups, and all six fit equally well. Compare
[`hidden_subpopulations_measured`](@ref). It takes a long time with the
default options.
"""
function hidden_subpopulations()
	parameters, states, equations = _subpopulation_model()
	S, I1, I2, I3, R = states
	@variables y1(t) y2(t) y3(t)
	return _example_problem(:hidden_subpopulations, states, parameters, equations,
		[y1 ~ S, y2 ~ I1 + I2 + I3, y3 ~ R]; _SUBPOPULATION_VALUES..., time_interval = [0.0, 12.0])
end

"""
	hidden_subpopulations_measured()

The model of [`hidden_subpopulations`](@ref) with each of the three infected
groups measured on its own. There is one answer.
"""
function hidden_subpopulations_measured()
	parameters, states, equations = _subpopulation_model()
	S, I1, I2, I3, R = states
	@variables y1(t) y2(t) y3(t) y4(t) y5(t)
	return _example_problem(:hidden_subpopulations_measured, states, parameters, equations,
		[y1 ~ S, y2 ~ I1, y3 ~ I2, y4 ~ I3, y5 ~ R]; _SUBPOPULATION_VALUES..., time_interval = [0.0, 12.0])
end

const _RECEPTOR_VALUES = (p_true = [1.10, 0.70, 0.80, 1.30, 0.40, 0.60], ic_true = [2.00, 0.10, 0.20])

function _receptor_model()
	parameters = @parameters R1tot R2tot kon1 kon2 koff1 koff2
	states = @variables L(t) Ca(t) Cb(t)
	equations = [
		D(L) ~ -kon1 * L * (R1tot - Ca) + koff1 * Ca - kon2 * L * (R2tot - Cb) + koff2 * Cb,
		D(Ca) ~ kon1 * L * (R1tot - Ca) - koff1 * Ca,
		D(Cb) ~ kon2 * L * (R2tot - Cb) - koff2 * Cb,
	]
	return parameters, states, equations
end

"""
	receptor_subtypes()

A ligand `L` that binds two receptor subtypes, each with its own amount and
its own binding and release rates. The free ligand and the *total* amount
bound are measured.

The two subtypes can be exchanged, so there are two answers. Compare
[`receptor_subtypes_measured`](@ref).

!!! warning
    The package does not solve this model in reasonable time at present. It
    is here as a model to work on, and no test estimates it.
"""
function receptor_subtypes()
	parameters, states, equations = _receptor_model()
	L, Ca, Cb = states
	@variables y1(t) y2(t)
	return _example_problem(:receptor_subtypes, states, parameters, equations, [y1 ~ L, y2 ~ Ca + Cb];
		_RECEPTOR_VALUES..., time_interval = [0.0, 8.0])
end

"""
	receptor_subtypes_measured()

The model of [`receptor_subtypes`](@ref) with the amount bound to each subtype
measured on its own. There is one answer, and it is found quickly.
"""
function receptor_subtypes_measured()
	parameters, states, equations = _receptor_model()
	L, Ca, Cb = states
	@variables y1(t) y2(t) y3(t)
	return _example_problem(:receptor_subtypes_measured, states, parameters, equations, [y1 ~ L, y2 ~ Ca, y3 ~ Cb];
		_RECEPTOR_VALUES..., time_interval = [0.0, 8.0])
end
