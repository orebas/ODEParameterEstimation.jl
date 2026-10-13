# Small models that are solved quickly and accurately.

# The tests use this model as a fixture in about twenty files. Its equations
# and true values must not change.
"""
	harmonic_oscillator()

Two states that drive each other, `x1' = -a x2` and `x2' = b x1`, both
measured. The smallest model with two parameters.

Both parameters and both initial conditions are recovered.
"""
function harmonic_oscillator()
	parameters = @parameters a b
	states = @variables x1(t) x2(t)
	@variables y1(t) y2(t)
	equations = [
		D(x1) ~ -a * x2,
		D(x2) ~ b * x1,
	]
	return _example_problem(:harmonic_oscillator, states, parameters, equations, [y1 ~ x1, y2 ~ x2];
		p_true = [0.4, 0.8], ic_true = [0.333, 0.667], time_interval = [0.0, 10.0])
end

"""
	oscillator_mixed_outputs()

The equations of [`harmonic_oscillator`](@ref), with neither state measured on
its own: the two measured quantities are `2 x1 + 0.5 x2` and `3 x1 - 0.25 x2`.

Everything is still recovered.
"""
function oscillator_mixed_outputs()
	parameters = @parameters a b
	states = @variables x1(t) x2(t)
	@variables y1(t) y2(t)
	equations = [
		D(x1) ~ -a * x2,
		D(x2) ~ b * x1,
	]
	return _example_problem(:oscillator_mixed_outputs, states, parameters, equations,
		[y1 ~ 2.0 * x1 + 0.5 * x2, y2 ~ 3.0 * x1 - 0.25 * x2];
		p_true = [0.4, 0.8], ic_true = [1.0, 2.0], time_interval = [0.0, 10.0])
end

"""
	exponential_decay()

One state that decays, `x1' = -a x1`, measured directly. The simplest model
there is: one parameter and one initial condition.
"""
function exponential_decay()
	parameters = @parameters a
	states = @variables x1(t)
	@variables y1(t)
	return _example_problem(:exponential_decay, states, parameters, [D(x1) ~ -a * x1], [y1 ~ x1];
		p_true = [0.1], ic_true = [2.0], time_interval = [0.0, 10.0])
end

"""
	decay_cubed_output()

The decay of [`exponential_decay`](@ref), measured through its cube: the
measured quantity is `x1^3`. A measured quantity may be any rational function
of the states and parameters.
"""
function decay_cubed_output()
	parameters = @parameters a
	states = @variables x1(t)
	@variables y1(t)
	return _example_problem(:decay_cubed_output, states, parameters, [D(x1) ~ -a * x1], [y1 ~ x1 * x1 * x1];
		p_true = [0.1], ic_true = [2.0], time_interval = [0.0, 10.0])
end

"""
	forced_decay()

A decaying state pushed by a known sinusoid, `x' = -a x + b sin(0.5 t)`,
measured directly. The smallest model with an input that depends on time. See
[Inputs that depend on time](@ref).
"""
function forced_decay()
	parameters = @parameters a b
	states = @variables x(t)
	@variables y1(t)
	return _example_problem(:forced_decay, states, parameters, [D(x) ~ -a * x + b * sin(0.5 * t)], [y1 ~ x];
		p_true = [0.5, 2.0], ic_true = [1.0], time_interval = [0.0, 10.0])
end

"""
	lotka_volterra()

The Lotka–Volterra predator-prey model with three rates. Only the prey `r` is
measured.

The three rates and the unmeasured initial number of predators are recovered.
"""
function lotka_volterra()
	parameters = @parameters k1 k2 k3
	states = @variables r(t) w(t)
	@variables y1(t)
	equations = [
		D(r) ~ k1 * r - k2 * r * w,
		D(w) ~ k2 * r * w - k3 * w,
	]
	return _example_problem(:lotka_volterra, states, parameters, equations, [y1 ~ r];
		p_true = [1.0, 0.5, 0.3], ic_true = [2.0, 1.0], time_interval = [0.0, 20.0])
end

"""
	vanderpol()

The van der Pol oscillator, `x1' = a x2` and `x2' = -x1 - b (x1^2 - 1) x2`,
with both states measured.
"""
function vanderpol()
	parameters = @parameters a b
	states = @variables x1(t) x2(t)
	@variables y1(t) y2(t)
	equations = [
		D(x1) ~ a * x2,
		D(x2) ~ -(x1) - b * (x1^2 - 1) * (x2),
	]
	return _example_problem(:vanderpol, states, parameters, equations, [y1 ~ x1, y2 ~ x2];
		p_true = [1.0, 1.0], ic_true = [2.0, 0.0], time_interval = [0.0, 10.0])
end

"""
	brusselator()

The Brusselator, a two-species chemical oscillator with a cubic term, with
both concentrations measured.
"""
function brusselator()
	parameters = @parameters a b
	states = @variables X(t) Y(t)
	@variables y1(t) y2(t)
	equations = [
		D(X) ~ 1.0 - (b + 1) * X + a * X^2 * Y,
		D(Y) ~ b * X - a * X^2 * Y,
	]
	return _example_problem(:brusselator, states, parameters, equations, [y1 ~ X, y2 ~ Y];
		p_true = [1.0, 3.0], ic_true = [1.0, 1.0], time_interval = [0.0, 20.0])
end

"""
	daisy_mamil3()

A linear three-compartment model with five exchange rates, a standard
identifiability benchmark. Two of the three compartments are measured.

All five rates and the unmeasured initial amount are recovered.
"""
function daisy_mamil3()
	parameters = @parameters a12 a13 a21 a31 a01
	states = @variables x1(t) x2(t) x3(t)
	@variables y1(t) y2(t)
	equations = [
		D(x1) ~ -(a21 + a31 + a01) * x1 + a12 * x2 + a13 * x3,
		D(x2) ~ a21 * x1 - a12 * x2,
		D(x3) ~ a31 * x1 - a13 * x3,
	]
	return _example_problem(:daisy_mamil3, states, parameters, equations, [y1 ~ x1, y2 ~ x2];
		p_true = [0.167, 0.333, 0.5, 0.667, 0.833], ic_true = [0.25, 0.5, 0.75], time_interval = [0.0, 10.0])
end

"""
	repressilator()

A ring of three genes, each repressing the next, with saturating repression
terms of the form `beta / (1 + n p)`. The three proteins are measured and the
three mRNA levels are not.

A model with parameters in denominators. It takes several minutes with the
default options.
"""
function repressilator()
	parameters = @parameters beta n alpha
	states = @variables m1(t) m2(t) m3(t) p1(t) p2(t) p3(t)
	@variables y1(t) y2(t) y3(t)
	equations = [
		D(m1) ~ -m1 + beta * (1 / (1 + p3 * n)),
		D(m2) ~ -m2 + beta * (1 / (1 + p1 * n)),
		D(m3) ~ -m3 + beta * (1 / (1 + p2 * n)),
		D(p1) ~ -alpha * (p1 - m1),
		D(p2) ~ -alpha * (p2 - m2),
		D(p3) ~ -alpha * (p3 - m3),
	]
	return _example_problem(:repressilator, states, parameters, equations, [y1 ~ p1, y2 ~ p2, y3 ~ p3];
		p_true = [2.0, 2.0, 1.0], ic_true = [0.0, 0.0, 0.0, 2.0, 1.0, 3.0], time_interval = [0.0, 24.0])
end
