# Models in which the measured quantities leave something open.

"""
	sum_of_rates()

Growth at the rate `a + b`: `x1' = (a + b) x1`, with `x1` measured. Only the
sum can be determined, so `a` and `b` are both reported as undetermined. The
smallest example of that.
"""
function sum_of_rates()
	parameters = @parameters a b
	states = @variables x1(t)
	@variables y1(t)
	return _example_problem(:sum_of_rates, states, parameters, [D(x1) ~ (a + b) * x1], [y1 ~ x1];
		p_true = [0.4, 0.6], ic_true = [2.0], time_interval = [0.0, 10.0], unident_count = 2)
end

"""
	partly_observed_chain()

Three states, two of them measured. `x2` grows at the rate `b + c`, so only
that sum is determined, and `x3` is neither measured nor felt by anything that
is, so `d` and its initial value are open.

`a` and the two measured initial conditions are recovered. `b`, `c`, `d` and
`x3(0)` are reported as undetermined.
"""
function partly_observed_chain()
	parameters = @parameters a b c d
	states = @variables x1(t) x2(t) x3(t)
	@variables y1(t) y2(t)
	equations = [
		D(x1) ~ -a * x1,
		D(x2) ~ (b + c) * (x1),
		D(x3) ~ d * x1,
	]
	return _example_problem(:partly_observed_chain, states, parameters, equations, [y1 ~ x1, y2 ~ x2];
		p_true = [0.1, 0.2, 0.3, 0.4], ic_true = [2.0, 3.0, 4.0], time_interval = [0.0, 10.0], unident_count = 4)
end

"""
	hidden_scale()

Two unmeasured states feed a third, `x3' = c (x1 + x2)`, and only `x3` is
measured. Scaling `x1` and `x2` up and `c` down by the same factor changes
nothing that is measured.

The rates `a` and `b` and `x3(0)` are determined. `c`, `x1(0)` and `x2(0)` are
reported as undetermined. Two results are returned, one for each way of
assigning the two rates to the two hidden states.
"""
function hidden_scale()
	parameters = @parameters a b c
	states = @variables x1(t) x2(t) x3(t)
	@variables y1(t)
	equations = [
		D(x1) ~ -a * x1,
		D(x2) ~ b * (x2),
		D(x3) ~ c * (x1 + x2),
	]
	return _example_problem(:hidden_scale, states, parameters, equations, [y1 ~ x3];
		p_true = [0.1, 0.2, 0.3], ic_true = [2.0, 3.0, 4.0], time_interval = [0.0, 10.0], unident_count = 3)
end

"""
	treatment()

An epidemic model with treatment: susceptible `S`, infected `In`, treated `Tr`
and the constant population `N`. Only the treated and the population size are
measured.

Most of the model is left open by those two measurements: `a`, `b`, `d`, `g`,
`In(0)` and `S(0)` are reported as undetermined, and `nu` is recovered. It
takes a long time with the default options.
"""
function treatment()
	parameters = @parameters a b d g nu
	states = @variables In(t) N(t) S(t) Tr(t)
	@variables y1(t) y2(t)
	equations = [
		D(In) ~ b * S * In / N + d * b * S * Tr / N - (a + g) * In,
		D(N) ~ 0,
		D(S) ~ -b * S * In / N - d * b * S * Tr / N,
		D(Tr) ~ g * In - nu * Tr,
	]
	return _example_problem(:treatment, states, parameters, equations, [y1 ~ Tr, y2 ~ N];
		p_true = [0.1, 0.8, 2.0, 0.3, 0.1], ic_true = [50.0, 1000.0, 950.0, 0.0],
		time_interval = [0.0, 40.0], unident_count = 6)
end

"""
	two_compartment_pk()

A drug that moves between a central and a peripheral compartment and is
eliminated from the central one. Only the central concentration is measured.

One concentration curve cannot separate the five parameters: all of them and
the peripheral initial concentration are reported as undetermined. Only the
central initial concentration is recovered.
"""
function two_compartment_pk()
	parameters = @parameters k12 k21 ke V1 V2
	states = @variables C1(t) C2(t)
	@variables y1(t)
	equations = [
		D(C1) ~ -k12 * C1 + k21 * C2 * V2 / V1 - ke * C1,
		D(C2) ~ k12 * C1 * V1 / V2 - k21 * C2,
	]
	return _example_problem(:two_compartment_pk, states, parameters, equations, [y1 ~ C1];
		p_true = [0.5, 0.25, 0.15, 1.0, 2.0], ic_true = [10.0, 0.0], time_interval = [0.0, 48.0], unident_count = 6)
end

"""
	mass_spring_damper()

A mass on a spring with a damper, pushed by a constant unknown force:
`m x'' + c x' + k x = F`. The position is measured.

Dividing the equation by `m` shows that only the ratios `c/m`, `k/m` and `F/m`
matter. All four parameters are reported as undetermined, and the initial
position and velocity are recovered.
"""
function mass_spring_damper()
	parameters = @parameters m c k F
	states = @variables x(t) v(t)
	@variables y1(t)
	equations = [
		D(x) ~ v,
		D(v) ~ (F - c * v - k * x) / m,
	]
	return _example_problem(:mass_spring_damper, states, parameters, equations, [y1 ~ x];
		p_true = [1.0, 0.5, 4.0, 1.0], ic_true = [0.5, 0.0], time_interval = [0.0, 10.0], unident_count = 4)
end
