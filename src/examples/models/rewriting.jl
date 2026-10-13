# One model written twice: as it comes, which the package refuses, and
# rewritten so that it is accepted. See "Rewriting a model so that it is
# accepted" in the manual.

"""
	tank_level()

A tank that drains through a hole in its floor. The level `h` falls at a rate
proportional to `sqrt(h)`, and the inflow `0.4 + 0.15 sin(0.5 t)` is known.
The level is measured.

`sqrt(h)` is not a rational function of the state, so `estimate` refuses this
model with an [`UnsupportedModelClassError`](@ref).
[`tank_level_rewritten`](@ref) is the same tank in a form that is accepted.
"""
function tank_level()
	parameters = @parameters A k
	states = @variables h(t)
	@variables y1(t)
	Qin = 0.4 + 0.15 * sin(0.5 * t)
	return _example_problem(:tank_level, states, parameters, [D(h) ~ (Qin - k * sqrt(h)) / A], [y1 ~ h];
		p_true = [1.0, 0.3], ic_true = [1.0], time_interval = [0.0, 30.0])
end

"""
	tank_level_rewritten()

The tank of [`tank_level`](@ref) with the state `z = sqrt(h)` in place of the
level. Then `z' = (Qin - k z) / (2 A z)`, which is rational, and the measured
level is `z^2`. The data are the same.

The area `A` and the outflow coefficient `k` are recovered. A second result
has `z` and `k` both negated: it gives the same level, and is the price of
measuring `z` only through its square.
"""
function tank_level_rewritten()
	parameters = @parameters A k
	states = @variables z(t)
	@variables y1(t)
	Qin = 0.4 + 0.15 * sin(0.5 * t)
	return _example_problem(:tank_level_rewritten, states, parameters, [D(z) ~ (Qin - k * z) / (2 * A * z)], [y1 ~ z^2];
		p_true = [1.0, 0.3], ic_true = [1.0], time_interval = [0.0, 30.0])
end
