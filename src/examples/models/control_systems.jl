# Control systems driven by a known sinusoidal input. The input is written as
# it is, with `sin`: see "Inputs that depend on time" in the manual. Quantities
# that would be known from a data sheet or a separate measurement are numbers,
# not parameters.

"""
	dc_motor()

A DC motor: the armature current `i` produces a torque that turns the rotor
at speed `omega_m`. The supply voltage is `12 + 2 sin(5 t)` and the speed is
measured. The electrical constants and the torque constant are taken as
known.

The rotor inertia `J` and the friction coefficient `b` are recovered. With the
torque constant unknown as well, only the ratios `Kt/J` and `b/J` could be
determined.
"""
function dc_motor()
	parameters = @parameters J b
	states = @variables omega_m(t) i(t)
	@variables y1(t)
	R, L, Kb, Kt = 2.0, 0.5, 0.1, 0.1
	V = 12.0 + 2.0 * sin(5.0 * t)
	equations = [
		D(omega_m) ~ (Kt * i - b * omega_m) / J,
		D(i) ~ (V - R * i - Kb * omega_m) / L,
	]
	return _example_problem(:dc_motor, states, parameters, equations, [y1 ~ omega_m];
		p_true = [0.01, 0.1], ic_true = [0.0, 0.0], time_interval = [0.0, 5.0])
end

"""
	quadrotor_altitude()

The altitude `z` of a quadrotor with linear drag, driven by a net vertical
force `2 sin(t)`. The altitude is measured.

The mass `m` and the drag coefficient `d` are recovered.
"""
function quadrotor_altitude()
	parameters = @parameters m d
	states = @variables z(t) w(t)
	@variables y1(t)
	equations = [
		D(z) ~ w,
		D(w) ~ (2.0 * sin(1.0 * t) - d * w) / m,
	]
	return _example_problem(:quadrotor_altitude, states, parameters, equations, [y1 ~ z];
		p_true = [1.0, 0.1], ic_true = [5.0, 0.0], time_interval = [0.0, 10.0])
end

"""
	forced_lotka_volterra()

A predator-prey model whose prey is harvested and restocked through the
seasons, at the rate `0.3 sin(2 t)`. Both populations are measured.

All four rates are recovered.
"""
function forced_lotka_volterra()
	parameters = @parameters alpha beta delta gamma
	states = @variables x(t) y(t)
	@variables y1(t) y2(t)
	equations = [
		D(x) ~ alpha * x - beta * x * y - 0.3 * sin(2.0 * t),
		D(y) ~ delta * x * y - gamma * y,
	]
	return _example_problem(:forced_lotka_volterra, states, parameters, equations, [y1 ~ x, y2 ~ y];
		p_true = [1.5, 1.0, 0.5, 3.0], ic_true = [1.0, 1.0], time_interval = [0.0, 5.0])
end

"""
	magnetic_levitation()

A ball held by an electromagnet, linearised about its resting point: position
`x`, velocity `v` and coil current `i`. The coil voltage is `5 + sin(5 t)` and
only the position is measured. The electrical constants are taken as known.

The effective mass, stiffness and damping are recovered.
"""
function magnetic_levitation()
	parameters = @parameters m_lin k_lin b_lin
	states = @variables x(t) v(t) i(t)
	@variables y1(t)
	R_coil, L, ki, V0 = 2.0, 0.05, 10.0, 5.0
	V = V0 + 1.0 * sin(5.0 * t)
	equations = [
		D(x) ~ v,
		D(v) ~ (ki * (i - V0 / R_coil) - k_lin * x - b_lin * v) / m_lin,
		D(i) ~ (V - R_coil * i) / L,
	]
	return _example_problem(:magnetic_levitation, states, parameters, equations, [y1 ~ x];
		p_true = [0.1, 50.0, 2.0], ic_true = [0.0, 0.0, V0 / R_coil], time_interval = [0.0, 5.0])
end

"""
	aircraft_pitch()

The short-period pitch dynamics of an aircraft: pitch angle `theta`, pitch
rate `q` and angle of attack `alpha`. The elevator moves as `0.05 sin(2 t)`
and the pitch rate is measured.

The four aerodynamic coefficients are recovered. The pitch angle itself
affects nothing that is measured, so `theta(0)` is reported as undetermined.
"""
function aircraft_pitch()
	parameters = @parameters M_alpha M_q M_delta_e Z_alpha
	states = @variables theta(t) q(t) alpha(t)
	@variables y1(t)
	V_air = 50.0
	delta_e = 0.0 + 0.05 * sin(2.0 * t)
	equations = [
		D(theta) ~ q,
		D(q) ~ M_alpha * alpha + M_q * q + M_delta_e * delta_e,
		D(alpha) ~ Z_alpha * alpha / V_air + q,
	]
	return _example_problem(:aircraft_pitch, states, parameters, equations, [y1 ~ q];
		p_true = [-5.0, -2.0, -10.0, -0.5], ic_true = [0.0, 0.0, 0.05], time_interval = [0.0, 10.0], unident_count = 1)
end

"""
	bicycle_model()

The lateral dynamics of a car as a single-track ("bicycle") model: lateral
velocity `vy` and yaw rate `r`, with the steering angle `0.05 sin(0.5 t)`.
Both states are measured. The geometry and the speed are taken as known.

The tyre stiffnesses `Cf` and `Cr` enter only divided by the mass or by the
yaw inertia, so only those ratios are determined. All four parameters are
reported as undetermined.
"""
function bicycle_model()
	parameters = @parameters Cf Cr m_veh Iz
	states = @variables vy(t) r(t)
	@variables y1(t) y2(t)
	lf, lr, Vx = 1.2, 1.4, 20.0
	delta = 0.05 * sin(0.5 * t)
	equations = [
		D(vy) ~ (Cf * (delta - (vy + lf * r) / Vx) + Cr * (-(vy - lr * r) / Vx)) / m_veh - Vx * r,
		D(r) ~ (lf * Cf * (delta - (vy + lf * r) / Vx) - lr * Cr * (-(vy - lr * r) / Vx)) / Iz,
	]
	return _example_problem(:bicycle_model, states, parameters, equations, [y1 ~ r, y2 ~ vy];
		p_true = [80000.0, 80000.0, 1500.0, 2500.0], ic_true = [0.0, 0.0], time_interval = [0.0, 20.0], unident_count = 4)
end

"""
	boost_converter()

The averaged model of a boost converter: inductor current `iL` and capacitor
voltage `vC`, with the duty cycle varying as `0.5 + 0.1 sin(100 t)`. Both
states are measured and the input voltage is taken as known.

The inductance, capacitance and load resistance are recovered.
"""
function boost_converter()
	parameters = @parameters L C_cap R_load
	states = @variables iL(t) vC(t)
	@variables y1(t) y2(t)
	Vin = 12.0
	d_complement = (1.0 - 0.5) - 0.1 * sin(100.0 * t)
	equations = [
		D(iL) ~ (Vin - d_complement * vC) / L,
		D(vC) ~ (d_complement * iL - vC / R_load) / C_cap,
	]
	return _example_problem(:boost_converter, states, parameters, equations, [y1 ~ vC, y2 ~ iL];
		p_true = [0.001, 0.001, 10.0], ic_true = [1.0, 24.0], time_interval = [0.0, 0.5])
end

"""
	bilinear_system()

A two-state system in which the input `u = 1 + 0.5 sin(2 t)` both adds to the
states and multiplies them. Both states are measured.

All eight coefficients are recovered.
"""
function bilinear_system()
	parameters = @parameters a11 a12 a21 a22 b1 b2 n1 n2
	states = @variables x1(t) x2(t)
	@variables y1(t) y2(t)
	u = 1.0 + 0.5 * sin(2.0 * t)
	equations = [
		D(x1) ~ a11 * x1 + a12 * x2 + b1 * u + n1 * x1 * u,
		D(x2) ~ a21 * x1 + a22 * x2 + b2 * u + n2 * x2 * u,
	]
	return _example_problem(:bilinear_system, states, parameters, equations, [y1 ~ x1, y2 ~ x2];
		p_true = [-0.5, 0.2, 0.1, -0.3, 1.0, 0.5, 0.2, 0.1], ic_true = [1.0, 0.5], time_interval = [0.0, 15.0])
end
