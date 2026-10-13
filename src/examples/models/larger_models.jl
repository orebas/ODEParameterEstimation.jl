# Larger models, most of them benchmarks from the identifiability literature.

# Adapted from ParameterEstimation.jl (GPL-3.0), examples/all-global/crauste.jl;
# a preserved source is revision 9e7adc33a4a11954acde0214f482cffea9551170. These
# are the corrected equations, called `crauste_corrected` before 1.0. See
# docs/internal/registry_preparation.md.
"""
	crauste()

The Crauste model of the CD8 T-cell response to infection: naive cells `N`,
early and late effector cells `E` and `L`, memory cells `M` and the pathogen
`P`, with thirteen rates. `N`, `E`, `P` and the sum `L + M` are measured.

!!! warning
    The package does not solve this model in reasonable time at present. It
    is here as a model to work on, and no test estimates it.
"""
function crauste()
	parameters = @parameters mu_N mu_EE mu_LE mu_LL mu_M mu_P mu_PE mu_PL delta_NE delta_EL delta_LM rho_E rho_P
	states = @variables N(t) E(t) L(t) M(t) P(t)
	@variables y1(t) y2(t) y3(t) y4(t)
	equations = [
		D(N) ~ -N * mu_N - N * P * delta_NE,
		D(E) ~ N * P * delta_NE + E * (rho_E * P - mu_EE * E - delta_EL),
		D(L) ~ delta_EL * E - L * (mu_LL * L + mu_LE * E + delta_LM),
		D(M) ~ L * delta_LM - mu_M * M,
		D(P) ~ P * (rho_P * P - mu_PE * E - mu_PL * L - mu_P),
	]
	return _example_problem(:crauste, states, parameters, equations, [y1 ~ N, y2 ~ E, y3 ~ L + M, y4 ~ P];
		p_true = [0.75, 0.0000216, 0.000000036, 0.0000075, 0.0, 0.055, 0.00000018, 0.000018, 0.009, 0.59, 0.025, 0.64, 0.15],
		ic_true = [8090.0, 0.0, 0.0, 0.0, 1.0], time_interval = [0.0, 25.0])
end

"""
	daisy_ex3()

A linear three-state model driven by a ramp input `u0`, from the examples of
the DAISY identifiability software. `x1` and the input are measured.

There are two answers. It takes several minutes with the default options.
"""
function daisy_ex3()
	parameters = @parameters p1 p3 p4 p6 p7
	states = @variables x1(t) x2(t) x3(t) u0(t)
	@variables y1(t) y2(t)
	equations = [
		D(x1) ~ -1.0 * p1 * x1 + x2 + u0,
		D(x2) ~ p3 * x1 - p4 * x2 + x3,
		D(x3) ~ p6 * x1 - p7 * x3,
		D(u0) ~ 1.0,
	]
	return _example_problem(:daisy_ex3, states, parameters, equations, [y1 ~ x1, y2 ~ u0];
		p_true = [0.167, 0.333, 0.5, 0.667, 0.833], ic_true = [0.2, 0.4, 0.6, 0.8], time_interval = [0.0, 10.0])
end

"""
	daisy_mamil4()

A linear four-compartment model with seven exchange rates. `x1`, `x2` and the
sum `x3 + x4` are measured.

Compartments 3 and 4 are seen only together and can be exchanged, so there are
two answers.
"""
function daisy_mamil4()
	parameters = @parameters k01 k12 k13 k14 k21 k31 k41
	states = @variables x1(t) x2(t) x3(t) x4(t)
	@variables y1(t) y2(t) y3(t)
	equations = [
		D(x1) ~ -k01 * x1 + k12 * x2 + k13 * x3 + k14 * x4 - k21 * x1 - k31 * x1 - k41 * x1,
		D(x2) ~ -k12 * x2 + k21 * x1,
		D(x3) ~ -k13 * x3 + k31 * x1,
		D(x4) ~ -k14 * x4 + k41 * x1,
	]
	return _example_problem(:daisy_mamil4, states, parameters, equations, [y1 ~ x1, y2 ~ x2, y3 ~ x3 + x4];
		p_true = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875], ic_true = [0.2, 0.4, 0.6, 0.8], time_interval = [0.0, 10.0])
end

"""
	fitzhugh_nagumo()

The FitzHugh–Nagumo model of a nerve cell: a fast voltage `V` with a cubic
term and a slow recovery variable `R`. Only the voltage is measured.

The three parameters and the unmeasured `R(0)` are recovered.
"""
function fitzhugh_nagumo()
	parameters = @parameters g a b
	states = @variables V(t) R(t)
	@variables y1(t)
	equations = [
		D(V) ~ g * (V - V^3 / 3 + R),
		D(R) ~ 1 / g * (V - a + b * R),
	]
	return _example_problem(:fitzhugh_nagumo, states, parameters, equations, [y1 ~ V];
		p_true = [3.0, 0.2, 0.2], ic_true = [-1.0, 0.0], time_interval = [0.0, 10.0])
end

"""
	hiv()

A model of HIV infection and the immune response to it: uninfected cells `x`,
infected cells `y`, virus `v`, and immune cells `w` and `z`, with ten rates.
Four quantities are measured, one of them the sum `y + v`.

The values span eight orders of magnitude, from `2e-5` to `1000`. The package
rescales the problem internally (`auto_rescale`). It takes several minutes with
the default options.
"""
function hiv()
	parameters = @parameters lm d beta a k u c q b h
	states = @variables x(t) y(t) v(t) w(t) z(t)
	@variables y1(t) y2(t) y3(t) y4(t)
	equations = [
		D(x) ~ lm - d * x - beta * x * v,
		D(y) ~ beta * x * v - a * y,
		D(v) ~ k * y - u * v,
		D(w) ~ c * z * y * w - c * q * y * w - b * w,
		D(z) ~ c * q * y * w - h * z,
	]
	return _example_problem(:hiv, states, parameters, equations, [y1 ~ w, y2 ~ z, y3 ~ x, y4 ~ y + v];
		p_true = [1.0, 0.01, 2e-5, 0.5, 50.0, 3.0, 0.05, 0.1, 0.002, 0.1],
		ic_true = [1000.0, 1.0, 1e-3, 1.0, 0.0], time_interval = [0.0, 25.0])
end

"""
	biohydrogenation()

A chain of three reactions with saturating rates of the form
`k x / (K + x)`. The first two of the four concentrations are measured.

The final product `x7` affects nothing that is measured, so `x7(0)` is
reported as undetermined. There are two answers for the rest. It takes
several minutes with the default options.
"""
function biohydrogenation()
	parameters = @parameters k5 k6 k7 k8 k9 k10
	states = @variables x4(t) x5(t) x6(t) x7(t)
	@variables y1(t) y2(t)
	equations = [
		D(x4) ~ -k5 * x4 / (k6 + x4),
		D(x5) ~ k5 * x4 / (k6 + x4) - k7 * x5 / (k8 + x5 + x6),
		D(x6) ~ k7 * x5 / (k8 + x5 + x6) - k9 * x6 * (k10 - x6) / k10,
		D(x7) ~ k9 * x6 * (k10 - x6) / k10,
	]
	return _example_problem(:biohydrogenation, states, parameters, equations, [y1 ~ x4, y2 ~ x5];
		p_true = [0.5, 2.0, 0.3, 1.0, 0.2, 5.0], ic_true = [4.0, 0.0, 0.0, 0.0],
		time_interval = [0.0, 36.0], unident_count = 1)
end

"""
	seir()

The SEIR epidemic model: susceptible `S`, exposed `E`, infected `In` and the
constant population `N`. The infected and the population size are measured.

There are two answers: the data cannot tell the recovery rate `a` from the
rate `nu` at which the exposed become infected. It takes several minutes with
the default options.
"""
function seir()
	parameters = @parameters a b nu
	states = @variables S(t) E(t) In(t) N(t)
	@variables y1(t) y2(t)
	equations = [
		D(S) ~ -b * S * In / N,
		D(E) ~ b * S * In / N - nu * E,
		D(In) ~ nu * E - a * In,
		D(N) ~ 0,
	]
	return _example_problem(:seir, states, parameters, equations, [y1 ~ In, y2 ~ N];
		p_true = [0.2, 0.4, 0.15], ic_true = [990.0, 10.0, 0.0, 1000.0], time_interval = [0.0, 60.0])
end

"""
	sirsforced()

An SIRS epidemic model whose infection rate varies with the seasons. The
seasonal factor is produced by two extra states, `x1` and `x2`, that form an
oscillator of unknown frequency `M`. The infected and the recovered are
measured.

!!! warning
    The package does not solve this model in reasonable time at present. It
    is here as a model to work on, and no test estimates it.
"""
function sirsforced()
	parameters = @parameters b0 b1 g M mu nu
	states = @variables i(t) r(t) s(t) x1(t) x2(t)
	@variables y1(t) y2(t)
	equations = [
		D(i) ~ b0 * (1.0 + b1 * x1) * i * s - (nu + mu) * i,
		D(r) ~ nu * i - (mu + g) * r,
		D(s) ~ mu - mu * s - b0 * (1.0 + b1 * x1) * i * s + g * r,
		D(x1) ~ -M * x2,
		D(x2) ~ M * x1,
	]
	return _example_problem(:sirsforced, states, parameters, equations, [y1 ~ i, y2 ~ r];
		p_true = [0.143, 0.286, 0.429, 0.571, 0.714, 0.857], ic_true = [0.167, 0.333, 0.5, 0.667, 0.833],
		time_interval = [0.0, 30.0], unident_count = 3)
end

"""
	slowfast()

A chain of two reactions, A to B to C, watched through an absorbance signal
`eA xA + eB xB + eC xC`. The concentration of C, the signal and two of the
three coefficients are measured. The coefficients are constants, written as
states that do not change.

There are two answers.
"""
function slowfast()
	parameters = @parameters k1 k2
	states = @variables xA(t) xB(t) xC(t) eA(t) eC(t) eB(t)
	@variables y1(t) y2(t) y3(t) y4(t)
	equations = [
		D(xA) ~ -k1 * xA,
		D(xB) ~ k1 * xA - k2 * xB,
		D(xC) ~ k2 * xB,
		D(eA) ~ 0,
		D(eC) ~ 0,
		D(eB) ~ 0,
	]
	return _example_problem(:slowfast, states, parameters, equations,
		[y1 ~ xC, y2 ~ eA * xA + eB * xB + eC * xC, y3 ~ eA, y4 ~ eC];
		p_true = [0.25, 0.5], ic_true = [0.166, 0.333, 0.5, 0.666, 0.833, 0.75], time_interval = [0.0, 10.0])
end

"""
	allee_competition()

Two competing species, each with logistic growth and an Allee threshold below
which it declines. Both populations are measured. Eight parameters, with the
carrying capacities in denominators.

There are four answers that fit the data equally well.
"""
function allee_competition()
	parameters = @parameters r1 r2 K1 K2 A1 A2 b12 b21
	states = @variables N1(t) N2(t)
	@variables y1(t) y2(t)
	equations = [
		D(N1) ~ r1 * N1 * (1 - N1 / K1) * (N1 - A1) / K1 - b12 * N1 * N2,
		D(N2) ~ r2 * N2 * (1 - N2 / K2) * (N2 - A2) / K2 - b21 * N1 * N2,
	]
	return _example_problem(:allee_competition, states, parameters, equations, [y1 ~ N1, y2 ~ N2];
		p_true = [1.0, 0.8, 10.0, 8.0, 0.2, 0.1, 0.3, 0.4], ic_true = [2.0, 1.5], time_interval = [0.0, 10.0])
end
