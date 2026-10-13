# =============================================================================
# Control Systems Examples for Parameter Estimation

# Designed for IEEE Transactions on Automatic Control audience.
# These examples represent canonical control systems studied across
# mechanical, electrical, chemical, and aerospace engineering.

# CONVENTION:
# - Parameters marked "# INPUT" represent control inputs
# - In parameter estimation, we treat inputs as known constants
# - For trajectory-based methods, inputs would be known time signals
# - States are the dynamic variables of the system
# - Observables are what can be measured (sensors)

# Organization:
# - TIER 1: Must include (classic benchmarks every control engineer knows)
# - TIER 2: Highly recommended (important application domains)
# - TIER 3: Good to include (specialized but valuable)
# - TIER 4: Additional systems (bilinear, ecological, etc.)
# - LINEARIZED: Linear versions of nonlinear systems for comparison
# ============================================================================= #


# =============================================================================
#                            TIER 1: MUST INCLUDE
#             Classic control benchmarks known to every control engineer
# ============================================================================= #

"""
    dc_motor()

Armature-controlled DC motor - the most fundamental electromechanical system.

Physics:
- Electrical: V = Ri + L(di/dt) + Kb*ω  (Kirchhoff's voltage law)
- Mechanical: J(dω/dt) = Kt*i - b*ω     (Newton's second law for rotation)

States: angular velocity ω, armature current i
Input: armature voltage V (constant)
Parameters: R (resistance), L (inductance), Kb (back-EMF constant),
           Kt (torque constant), J (rotor inertia), b (viscous friction)
Observable: angular velocity (measured by tachometer or encoder)

Control relevance: Foundation of electric drives, servo systems, robotics.
Every textbook on control systems includes this example.
"""
function dc_motor()
	# System parameters
	parameters = @parameters R L Kb Kt J b V
	#                        Ω  H  V·s/rad  N·m/A  kg·m²  N·m·s/rad  V
	# V is the INPUT (armature voltage)

	states = @variables omega(t) i(t)
	#                   rad/s    A

	observables = @variables y1(t)

	# Typical values for a small DC motor
	p_true = [
		2.0,    # R: armature resistance (Ohms)
		0.5,    # L: armature inductance (Henries)
		0.1,    # Kb: back-EMF constant (V·s/rad)
		0.1,    # Kt: torque constant (N·m/A) - often Kt ≈ Kb
		0.01,   # J: rotor inertia (kg·m²)
		0.1,    # b: viscous friction (N·m·s/rad)
		12.0,   # V: INPUT - applied voltage (V)
	]

	ic_true = [0.0, 0.0]  # Start from rest, no initial current

	# State equations derived from electrical and mechanical subsystems
	equations = [
		D(omega) ~ (Kt * i - b * omega) / J,           # Mechanical dynamics
		D(i) ~ (V - R * i - Kb * omega) / L,           # Electrical dynamics
	]

	# Typically measure angular velocity (tachometer/encoder)
	measured_quantities = [y1 ~ omega]

	model, mq = create_ordered_ode_system("dc_motor", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"dc_motor",
		model,
		mq,
		nothing,
		[0.0, 2.0],  # 2 seconds to reach steady state
		nothing,
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end

#-------------------------------------------------------------------------------
# Polynomialized / Identifiable Control Systems
# Copied from tier_a_identifiable.jl, tier_b_sqrt.jl, tier_c_reparametrized.jl
#-------------------------------------------------------------------------------

"""
    dc_motor_identifiable()

DC motor with electrical parameters fixed (from measurement/datasheet).
"""
function dc_motor_identifiable()
    # Parameters to ESTIMATE
    @parameters Kt J b

    # FIXED electrical parameters (known from measurement/datasheet)
    R_val = 2.0     # armature resistance (Ohms) - measured
    L_val = 0.5     # armature inductance (H) - measured
    Kb_val = 0.1    # back-EMF constant (V·s/rad) - from datasheet

    # Input parameters - FIXED
    V0_val = 12.0   # DC voltage offset
    Va_val = 2.0    # AC voltage amplitude
    omega_val = 5.0 # Input frequency (rad/s)

    # States
    @variables omega_m(t) i(t) u_sin(t) u_cos(t)
    @variables y1(t) y2(t) y3(t)

    # True values for parameters to estimate
    p_true = [
        0.1,    # Kt: torque constant
        0.01,   # J: rotor inertia
        0.1,    # b: viscous friction
    ]

    ic_true = [0.0, 0.0, 0.0, 1.0]  # [omega_m, i, u_sin, u_cos]

    V_input = V0_val + Va_val * u_sin

    equations = [
        D(omega_m) ~ (Kt * i - b * omega_m) / J,
        D(i) ~ (V_input - R_val * i - Kb_val * omega_m) / L_val,
        D(u_sin) ~ omega_val * u_cos,
        D(u_cos) ~ -omega_val * u_sin,
    ]

    measured_quantities = [y1 ~ omega_m, y2 ~ u_sin, y3 ~ u_cos]

    states = [omega_m, i, u_sin, u_cos]
    parameters = [Kt, J, b]

    model, mq = create_ordered_ode_system("dc_motor_identifiable", states, parameters, equations, measured_quantities)

    return ParameterEstimationProblem(
        "dc_motor_identifiable",
        model,
        mq,
        nothing,
        [0.0, 5.0],
        nothing,
        OrderedDict(parameters .=> p_true),
        OrderedDict(states .=> ic_true),
        0,
    )
end

"""
    quadrotor_altitude_identifiable()

Quadrotor altitude with gravity fixed.
"""
function quadrotor_altitude_identifiable()
    @parameters m d

    # FIXED physical constant
    g_val = 9.81    # gravity (m/s²)

    # Input parameters - FIXED
    Ta_val = 2.0    # thrust oscillation amplitude
    omega_val = 1.0 # frequency

    @variables z(t) w(t) u_sin(t) u_cos(t)
    @variables y1(t) y2(t) y3(t)

    p_true = [
        1.0,     # m: mass
        0.1,     # d: drag
    ]

    ic_true = [5.0, 0.0, 0.0, 1.0]

    equations = [
        D(z) ~ w,
        D(w) ~ (Ta_val * u_sin - d * w) / m,
        D(u_sin) ~ omega_val * u_cos,
        D(u_cos) ~ -omega_val * u_sin,
    ]

    measured_quantities = [y1 ~ z, y2 ~ u_sin, y3 ~ u_cos]

    states = [z, w, u_sin, u_cos]
    parameters = [m, d]

    model, mq = create_ordered_ode_system("quadrotor_altitude_identifiable", states, parameters, equations, measured_quantities)

    return ParameterEstimationProblem(
        "quadrotor_altitude_identifiable",
        model,
        mq,
        nothing,
        [0.0, 20.0],
        nothing,
        OrderedDict(parameters .=> p_true),
        OrderedDict(states .=> ic_true),
        0,
    )
end

"""
    magnetic_levitation_identifiable()

Magnetic levitation with electrical parameters fixed.
"""
function magnetic_levitation_identifiable()
    @parameters m_lin k_lin b_lin

    # FIXED electrical parameters (measured)
    R_coil_val = 2.0   # coil resistance (Ohms)
    L_val = 0.05       # inductance (H)
    ki_val = 10.0      # current-to-force gain

    # Input parameters - FIXED
    V0_val = 5.0
    Va_val = 1.0
    omega_val = 5.0

    @variables x(t) v(t) i(t) u_sin(t) u_cos(t)
    @variables y1(t) y2(t) y3(t)

    p_true = [
        0.1,    # m_lin: effective mass
        50.0,   # k_lin: linearized stiffness
        2.0,    # b_lin: damping
    ]

    i_eq = V0_val / R_coil_val
    ic_true = [0.0, 0.0, i_eq, 0.0, 1.0]

    V_input = V0_val + Va_val * u_sin

    equations = [
        D(x) ~ v,
        D(v) ~ (ki_val * (i - V0_val / R_coil_val) - k_lin * x - b_lin * v) / m_lin,
        D(i) ~ (V_input - R_coil_val * i) / L_val,
        D(u_sin) ~ omega_val * u_cos,
        D(u_cos) ~ -omega_val * u_sin,
    ]

    measured_quantities = [y1 ~ x, y2 ~ u_sin, y3 ~ u_cos]

    states = [x, v, i, u_sin, u_cos]
    parameters = [m_lin, k_lin, b_lin]

    model, mq = create_ordered_ode_system("magnetic_levitation_identifiable", states, parameters, equations, measured_quantities)

    return ParameterEstimationProblem(
        "magnetic_levitation_identifiable",
        model,
        mq,
        nothing,
        [0.0, 5.0],
        nothing,
        OrderedDict(parameters .=> p_true),
        OrderedDict(states .=> ic_true),
        0,
    )
end

"""
    aircraft_pitch_identifiable()

Aircraft pitch dynamics with airspeed fixed.
"""
function aircraft_pitch_identifiable()
    @parameters M_alpha M_q M_delta_e Z_alpha

    # FIXED from flight conditions
    V_air_val = 50.0  # true airspeed (m/s) - from pitot tube

    # Input parameters - FIXED
    delta_e0_val = 0.0
    delta_ea_val = 0.05
    omega_val = 2.0

    @variables theta(t) q(t) alpha(t) u_sin(t) u_cos(t)
    @variables y1(t) y2(t) y3(t)

    p_true = [
        -5.0,     # M_alpha
        -2.0,     # M_q
        -10.0,    # M_delta_e
        -0.5,     # Z_alpha
    ]

    ic_true = [0.0, 0.0, 0.05, 0.0, 1.0]

    delta_e_input = delta_e0_val + delta_ea_val * u_sin

    equations = [
        D(theta) ~ q,
        D(q) ~ M_alpha * alpha + M_q * q + M_delta_e * delta_e_input,
        D(alpha) ~ Z_alpha * alpha / V_air_val + q,
        D(u_sin) ~ omega_val * u_cos,
        D(u_cos) ~ -omega_val * u_sin,
    ]

    measured_quantities = [y1 ~ q, y2 ~ u_sin, y3 ~ u_cos]

    states = [theta, q, alpha, u_sin, u_cos]
    parameters = [M_alpha, M_q, M_delta_e, Z_alpha]

    model, mq = create_ordered_ode_system("aircraft_pitch_identifiable", states, parameters, equations, measured_quantities)

    return ParameterEstimationProblem(
        "aircraft_pitch_identifiable",
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

"""
    bicycle_model_identifiable()

Vehicle lateral dynamics with speed and geometry fixed.
"""
function bicycle_model_identifiable()
    @parameters Cf Cr m_veh Iz

    # FIXED from vehicle specs and test conditions
    lf_val = 1.2    # front axle to CG (m)
    lr_val = 1.4    # rear axle to CG (m)
    Vx_val = 20.0   # test speed (m/s)

    # Input parameters - FIXED
    delta_a_val = 0.05
    omega_val = 0.5

    @variables vy(t) r(t) u_sin(t) u_cos(t)
    @variables y1(t) y2(t) y3(t) y4(t)

    p_true = [
        80000.0,   # Cf
        80000.0,   # Cr
        1500.0,    # m_veh
        2500.0,    # Iz
    ]

    ic_true = [0.0, 0.0, 0.0, 1.0]

    delta_input = delta_a_val * u_sin

    equations = [
        D(vy) ~ (Cf * (delta_input - (vy + lf_val * r) / Vx_val) + Cr * (-(vy - lr_val * r) / Vx_val)) / m_veh - Vx_val * r,
        D(r) ~ (lf_val * Cf * (delta_input - (vy + lf_val * r) / Vx_val) - lr_val * Cr * (-(vy - lr_val * r) / Vx_val)) / Iz,
        D(u_sin) ~ omega_val * u_cos,
        D(u_cos) ~ -omega_val * u_sin,
    ]

    measured_quantities = [y1 ~ r, y2 ~ vy, y3 ~ u_sin, y4 ~ u_cos]

    states = [vy, r, u_sin, u_cos]
    parameters = [Cf, Cr, m_veh, Iz]

    model, mq = create_ordered_ode_system("bicycle_model_identifiable", states, parameters, equations, measured_quantities)

    return ParameterEstimationProblem(
        "bicycle_model_identifiable",
        model,
        mq,
        nothing,
        [0.0, 20.0],
        nothing,
        OrderedDict(parameters .=> p_true),
        OrderedDict(states .=> ic_true),
        0,
    )
end

"""
    boost_converter_identifiable()

Boost converter with input voltage fixed and inductor current observed.
"""
function boost_converter_identifiable()
    @parameters L C_cap R_load

    # FIXED from measurement
    Vin_val = 12.0  # input voltage (V)

    # Input parameters - FIXED
    d0_val = 0.5
    da_val = 0.1
    omega_val = 100.0

    @variables iL(t) vC(t) u_sin(t) u_cos(t)
    @variables y1(t) y2(t) y3(t) y4(t)  # Added y4 for iL observation

    p_true = [
        0.001,   # L
        0.001,   # C_cap
        10.0,    # R_load
    ]

    ic_true = [1.0, 24.0, 0.0, 1.0]

    d_complement = (1.0 - d0_val) - da_val * u_sin

    equations = [
        D(iL) ~ (Vin_val - d_complement * vC) / L,
        D(vC) ~ (d_complement * iL - vC / R_load) / C_cap,
        D(u_sin) ~ omega_val * u_cos,
        D(u_cos) ~ -omega_val * u_sin,
    ]

    # Observe both vC and iL (current sensor added)
    measured_quantities = [y1 ~ vC, y2 ~ iL, y3 ~ u_sin, y4 ~ u_cos]

    states = [iL, vC, u_sin, u_cos]
    parameters = [L, C_cap, R_load]

    model, mq = create_ordered_ode_system("boost_converter_identifiable", states, parameters, equations, measured_quantities)

    return ParameterEstimationProblem(
        "boost_converter_identifiable",
        model,
        mq,
        nothing,
        [0.0, 0.5],
        nothing,
        OrderedDict(parameters .=> p_true),
        OrderedDict(states .=> ic_true),
        0,
    )
end

"""
    bilinear_system_identifiable()

Bilinear system with both states observed.
"""
function bilinear_system_identifiable()
    @parameters a11 a12 a21 a22 b1 b2 n1 n2

    # Input parameters - FIXED
    u0_val = 1.0
    ua_val = 0.5
    omega_val = 2.0

    @variables x1(t) x2(t) u_sin(t) u_cos(t)
    @variables y1(t) y2(t) y3(t) y4(t)  # y1=x1, y2=x2, y3,y4=input

    p_true = [
        -0.5,    # a11
        0.2,     # a12
        0.1,     # a21
        -0.3,    # a22
        1.0,     # b1
        0.5,     # b2
        0.2,     # n1
        0.1,     # n2
    ]

    ic_true = [1.0, 0.5, 0.0, 1.0]

    u_input = u0_val + ua_val * u_sin

    equations = [
        D(x1) ~ a11 * x1 + a12 * x2 + b1 * u_input + n1 * x1 * u_input,
        D(x2) ~ a21 * x1 + a22 * x2 + b2 * u_input + n2 * x2 * u_input,
        D(u_sin) ~ omega_val * u_cos,
        D(u_cos) ~ -omega_val * u_sin,
    ]

    # Observe BOTH x1 and x2 (dual-sensor setup)
    measured_quantities = [y1 ~ x1, y2 ~ x2, y3 ~ u_sin, y4 ~ u_cos]

    states = [x1, x2, u_sin, u_cos]
    parameters = [a11, a12, a21, a22, b1, b2, n1, n2]

    model, mq = create_ordered_ode_system("bilinear_system_identifiable", states, parameters, equations, measured_quantities)

    return ParameterEstimationProblem(
        "bilinear_system_identifiable",
        model,
        mq,
        nothing,
        [0.0, 15.0],
        nothing,
        OrderedDict(parameters .=> p_true),
        OrderedDict(states .=> ic_true),
        0,
    )
end

"""
    forced_lotka_volterra_identifiable()

Lotka-Volterra with both populations observed.
"""
function forced_lotka_volterra_identifiable()
    @parameters a b c d_pred

    # Input parameters - FIXED
    u0_val = 0.2
    ua_rel_val = 0.5
    omega_val = 0.5

    @variables prey(t) predator(t) u_sin(t) u_cos(t)
    @variables y1(t) y2(t) y3(t) y4(t)

    p_true = [
        1.0,     # a: prey growth
        0.5,     # b: predation
        0.5,     # c: predator death
        0.25,    # d_pred: predator efficiency
    ]

    ic_true = [2.0, 1.0, 0.0, 1.0]

    u_input = u0_val + u0_val * ua_rel_val * u_sin

    equations = [
        D(prey) ~ a * prey - b * prey * predator + u_input,
        D(predator) ~ -c * predator + d_pred * prey * predator,
        D(u_sin) ~ omega_val * u_cos,
        D(u_cos) ~ -omega_val * u_sin,
    ]

    # Observe BOTH prey and predator
    measured_quantities = [y1 ~ prey, y2 ~ predator, y3 ~ u_sin, y4 ~ u_cos]

    states = [prey, predator, u_sin, u_cos]
    parameters = [a, b, c, d_pred]

    model, mq = create_ordered_ode_system("forced_lotka_volterra_identifiable", states, parameters, equations, measured_quantities)

    return ParameterEstimationProblem(
        "forced_lotka_volterra_identifiable",
        model,
        mq,
        nothing,
        [0.0, 50.0],
        nothing,
        OrderedDict(parameters .=> p_true),
        OrderedDict(states .=> ic_true),
        0,
    )
end

# =============================================================================
# Tier B: Polynomialized Control Systems - sqrt(state) Dynamics
# ============================================================================= #

"""
    tank_level_poly()

Tank level with polynomialized sqrt dynamics.
"""
function tank_level_poly()
    @parameters A k_out

    # Input parameters - FIXED
    Q0_val = 0.4     # mean inlet flow
    Qa_val = 0.15    # flow oscillation amplitude
    omega_val = 0.5  # frequency

    # z = sqrt(h) is the transformed state
    @variables z(t) u_sin(t) u_cos(t)
    @variables y1(t) y2(t) y3(t)

    p_true = [
        1.0,    # A: tank area
        0.3,    # k_out: outflow coefficient
    ]

    # Original h(0) = 1.0, so z(0) = sqrt(1.0) = 1.0
    ic_true = [1.0, 0.0, 1.0]  # [z, u_sin, u_cos]

    # Qin(t) = Q0 + Qa*u_sin
    Qin = Q0_val + Qa_val * u_sin

    # D(z) = (Qin - k_out*z) / (2*A*z)
    equations = [
        D(z) ~ (Qin - k_out * z) / (2 * A * z),
        D(u_sin) ~ omega_val * u_cos,
        D(u_cos) ~ -omega_val * u_sin,
    ]

    # We observe z (which represents sqrt(h))
    measured_quantities = [y1 ~ z, y2 ~ u_sin, y3 ~ u_cos]

    states = [z, u_sin, u_cos]
    parameters = [A, k_out]

    model, mq = create_ordered_ode_system("tank_level_poly", states, parameters, equations, measured_quantities)

    return ParameterEstimationProblem(
        "tank_level_poly",
        model,
        mq,
        nothing,
        [0.0, 30.0],
        nothing,
        OrderedDict(parameters .=> p_true),
        OrderedDict(states .=> ic_true),
        0,
    )
end

"""
    two_tank_poly()

Two-tank system with polynomialized sqrt dynamics.
"""
function two_tank_poly()
    @parameters A1 A2 k1 k2 k12

    # Input parameters - FIXED
    Q0_val = 0.5     # mean flow
    Qa_val = 0.2     # oscillation
    omega_val = 0.3  # frequency
    eps_val = 0.01   # small constant for sqrt safety

    # Transformed states: z1 = sqrt(h1), z2 = sqrt(h2), z12 = sqrt(h1-h2+eps)
    @variables z1(t) z2(t) z12(t) u_sin(t) u_cos(t)
    @variables y1(t) y2(t) y3(t) y4(t)

    p_true = [
        1.0,     # A1
        1.0,     # A2
        0.3,     # k1
        0.3,     # k2
        0.2,     # k12
    ]

    # Original: h1(0) = 1.0, h2(0) = 0.5
    # So: z1(0) = 1.0, z2(0) = sqrt(0.5) ≈ 0.707
    # z12(0) = sqrt(1.0 - 0.5 + 0.01) = sqrt(0.51) ≈ 0.714
    ic_true = [1.0, sqrt(0.5), sqrt(0.51), 0.0, 1.0]

    # Qin = Q0 + Qa*u_sin
    Qin = Q0_val + Qa_val * u_sin

    # D(h1) = (Qin - k1*z1 - k12*z12) / A1
    Dh1 = (Qin - k1 * z1 - k12 * z12) / A1
    # D(h2) = (k12*z12 - k2*z2) / A2
    Dh2 = (k12 * z12 - k2 * z2) / A2

    equations = [
        # D(z1) = D(h1) / (2*z1)
        D(z1) ~ Dh1 / (2 * z1),
        # D(z2) = D(h2) / (2*z2)
        D(z2) ~ Dh2 / (2 * z2),
        # D(z12) = (D(h1) - D(h2)) / (2*z12)
        D(z12) ~ (Dh1 - Dh2) / (2 * z12),
        # Input oscillator
        D(u_sin) ~ omega_val * u_cos,
        D(u_cos) ~ -omega_val * u_sin,
    ]

    # NOTE: We observe z1 and z2. Physical h1 = z1^2, h2 = z2^2
    measured_quantities = [y1 ~ z1, y2 ~ z2, y3 ~ u_sin, y4 ~ u_cos]

    states = [z1, z2, z12, u_sin, u_cos]
    parameters = [A1, A2, k1, k2, k12]

    model, mq = create_ordered_ode_system("two_tank_poly", states, parameters, equations, measured_quantities)

    return ParameterEstimationProblem(
        "two_tank_poly",
        model,
        mq,
        nothing,
        [0.0, 50.0],
        nothing,
        OrderedDict(parameters .=> p_true),
        OrderedDict(states .=> ic_true),
        0,
    )
end

# =============================================================================
# Tier C: Reparametrized CSTR
# ============================================================================= #

"""
    cstr_reparametrized()

CSTR with k0*exp(-E_R/T) as a single state variable.
"""
function cstr_reparametrized()
    # Reparametrized parameters - k0 is eliminated
    @parameters E_R tau Tin Cin dH_rhoCP UA_VrhoCP

    # Input parameters - FIXED
    Tc0_val = 300.0   # mean coolant temp (K)
    Tca_val = 10.0    # coolant oscillation amplitude (K)
    omega_val = 0.5   # frequency

    # States: C (concentration), T (temperature), r_eff (effective rate constant), plus input oscillator
    @variables C(t) T(t) r_eff(t) u_sin(t) u_cos(t)
    @variables y1(t) y2(t) y3(t)

    # True parameter values (6 params instead of 7)
    p_true = [
        8750.0,   # E_R: activation energy / gas constant (K)
        1.0,      # tau: residence time (s)
        350.0,    # Tin: inlet temperature (K)
        1.0,      # Cin: inlet concentration (mol/L)
        5.0,      # dH_rhoCP: heat release parameter
        1.0,      # UA_VrhoCP: heat transfer parameter
    ]

    # Initial conditions
    C0 = 0.5    # Initial concentration
    T0 = 350.0  # Initial temperature

    # r_eff(0) = k0 * exp(-E_R / T0)
    # With k0 = 7.2e10 and E_R = 8750, T0 = 350:
    # r_eff(0) = 7.2e10 * exp(-8750/350) ≈ 1.0
    k0_original = 7.2e10  # This is now encoded in the IC, not a parameter
    r_eff0 = k0_original * exp(-p_true[1] / T0)

    ic_true = [C0, T0, r_eff0, 0.0, 1.0]  # [C, T, r_eff, u_sin, u_cos]

    # Coolant temperature: Tc(t) = Tc0 + Tca*u_sin
    Tc = Tc0_val + Tca_val * u_sin

    # Reaction rate = r_eff * C (instead of k0 * exp(-E_R/T) * C)
    reaction_rate = r_eff * C

    equations = [
        # Concentration dynamics: D(C) = (Cin - C)/tau - r_eff*C
        D(C) ~ (Cin - C) / tau - reaction_rate,

        # Temperature dynamics
        D(T) ~ (Tin - T) / tau + dH_rhoCP * reaction_rate - UA_VrhoCP * (T - Tc),

        # Effective rate constant dynamics
        D(r_eff) ~ r_eff * (E_R / T^2) * ((Tin - T) / tau + dH_rhoCP * reaction_rate - UA_VrhoCP * (T - Tc)),

        # Input oscillator
        D(u_sin) ~ omega_val * u_cos,
        D(u_cos) ~ -omega_val * u_sin,
    ]

    # We observe temperature (common measurement) and input
    measured_quantities = [y1 ~ T, y2 ~ u_sin, y3 ~ u_cos]

    states = [C, T, r_eff, u_sin, u_cos]
    parameters = [E_R, tau, Tin, Cin, dH_rhoCP, UA_VrhoCP]

    model, mq = create_ordered_ode_system("cstr_reparametrized", states, parameters, equations, measured_quantities)

    return ParameterEstimationProblem(
        "cstr_reparametrized",
        model,
        mq,
        nothing,
        [0.0, 20.0],
        nothing,
        OrderedDict(parameters .=> p_true),
        OrderedDict(states .=> ic_true),
        0,
    )
end

"""
    cstr_fixed_activation()

CSTR with fixed activation energy E_R (known from chemistry tables).
"""
function cstr_fixed_activation()
    # Parameters to estimate (5 instead of 7)
    @parameters tau Tin Cin dH_rhoCP UA_VrhoCP

    # FIXED activation energy - known from chemistry
    E_R_val = 8750.0  # E/R in Kelvin

    # Input parameters - FIXED
    Tc0_val = 300.0
    Tca_val = 10.0
    omega_val = 0.5

    @variables C(t) T(t) r_eff(t) u_sin(t) u_cos(t)
    @variables y1(t) y2(t) y3(t)

    # True parameter values (5 params)
    p_true = [
        1.0,      # tau
        350.0,    # Tin
        1.0,      # Cin
        5.0,      # dH_rhoCP
        1.0,      # UA_VrhoCP
    ]

    C0 = 0.5
    T0 = 350.0
    k0_original = 7.2e10
    r_eff0 = k0_original * exp(-E_R_val / T0)

    ic_true = [C0, T0, r_eff0, 0.0, 1.0]

    Tc = Tc0_val + Tca_val * u_sin
    reaction_rate = r_eff * C

    equations = [
        D(C) ~ (Cin - C) / tau - reaction_rate,
        D(T) ~ (Tin - T) / tau + dH_rhoCP * reaction_rate - UA_VrhoCP * (T - Tc),
        # E_R is now a fixed value, not a parameter
        D(r_eff) ~ r_eff * (E_R_val / T^2) * ((Tin - T) / tau + dH_rhoCP * reaction_rate - UA_VrhoCP * (T - Tc)),
        D(u_sin) ~ omega_val * u_cos,
        D(u_cos) ~ -omega_val * u_sin,
    ]

    measured_quantities = [y1 ~ T, y2 ~ u_sin, y3 ~ u_cos]

    states = [C, T, r_eff, u_sin, u_cos]
    parameters = [tau, Tin, Cin, dH_rhoCP, UA_VrhoCP]

    model, mq = create_ordered_ode_system("cstr_fixed_activation", states, parameters, equations, measured_quantities)

    return ParameterEstimationProblem(
        "cstr_fixed_activation",
        model,
        mq,
        nothing,
        [0.0, 20.0],
        nothing,
        OrderedDict(parameters .=> p_true),
        OrderedDict(states .=> ic_true),
        0,
    )
end

"""
    mass_spring_damper()

Canonical second-order mechanical system.

Physics: m*ẍ + c*ẋ + k*x = F  (Newton's second law)
Converted to first-order: ẋ = v, v̇ = (F - c*v - k*x)/m

States: position x, velocity v
Input: applied force F
Parameters: m (mass), c (damping coefficient), k (spring constant)
Observable: position (displacement sensor)

Control relevance: Fundamental to vibration control, suspension systems,
structural dynamics. The standard second-order system in every textbook.
"""
function mass_spring_damper()
	parameters = @parameters m c k F
	#                        kg  N·s/m  N/m  N
	# F is the INPUT (applied force)

	states = @variables x(t) v(t)
	#                   m    m/s

	observables = @variables y1(t)

	p_true = [
		1.0,    # m: mass (kg)
		0.5,    # c: damping coefficient (N·s/m)
		4.0,    # k: spring constant (N/m)
		1.0,    # F: INPUT - applied force (N)
	]

	ic_true = [0.5, 0.0]  # Initial displacement, at rest

	equations = [
		D(x) ~ v,
		D(v) ~ (F - c * v - k * x) / m,
	]

	measured_quantities = [y1 ~ x]

	model, mq = create_ordered_ode_system("mass_spring_damper", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"mass_spring_damper",
		model,
		mq,
		nothing,
		[0.0, 10.0],  # Several oscillation periods
		nothing,
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end

"""
    cart_pole()

Inverted pendulum on a cart - THE benchmark for nonlinear control.

Physics: Coupled cart-pendulum dynamics derived from Lagrangian mechanics.
The system is underactuated (1 input, 2 DOF) and unstable at the upright position.

States: cart position x, cart velocity v, pendulum angle θ (from vertical), angular velocity ω
Input: horizontal force F on cart
Parameters: M (cart mass), m (pendulum mass), l (pendulum length), g (gravity)
Observable: cart position, pendulum angle

Control relevance: Classic benchmark for:
- Nonlinear control (stabilization around unstable equilibrium)
- Swing-up control
- Model predictive control demonstrations
- Reinforcement learning benchmarks
"""
function cart_pole()
	parameters = @parameters M m l g F
	#                        kg kg m m/s² N
	# F is the INPUT (horizontal force on cart)

	states = @variables x(t) v(t) theta(t) omega(t)
	#                   m    m/s  rad      rad/s

	observables = @variables y1(t) y2(t)

	p_true = [
		1.0,    # M: cart mass (kg)
		0.1,    # m: pendulum mass (kg)
		0.5,    # l: pendulum length (m)
		9.81,   # g: gravity (m/s²)
		0.0,    # F: INPUT - applied force (N) - zero for free response
	]

	# Start near upright with small perturbation
	ic_true = [0.0, 0.0, 0.1, 0.0]  # Small angle from vertical

	# Full nonlinear equations of motion
	# Derived from Lagrangian: L = T - V with constraints
	# These are the standard cart-pole equations
	equations = [
		D(x) ~ v,
		D(v) ~ (F + m * sin(theta) * (l * omega^2 + g * cos(theta))) / (M + m * sin(theta)^2),
		D(theta) ~ omega,
		D(omega) ~ (-F * cos(theta) - m * l * omega^2 * cos(theta) * sin(theta) - (M + m) * g * sin(theta)) / (l * (M + m * sin(theta)^2)),
	]

	# Typically measure cart position and pendulum angle
	measured_quantities = [y1 ~ x, y2 ~ theta]

	model, mq = create_ordered_ode_system("cart_pole", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"cart_pole",
		model,
		mq,
		nothing,
		[0.0, 5.0],  # Watch the pendulum fall (or be stabilized)
		nothing,
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end

"""
    tank_level()

Single tank liquid level control - classic process control example.

Physics: Mass balance with nonlinear outflow
- dh/dt = (Qin - Qout) / A
- Qout = k * sqrt(h)  (Torricelli's law for gravity-driven flow)

States: liquid level h
Input: inlet flow rate Qin
Parameters: A (tank cross-section), k (outflow coefficient)
Observable: liquid level (measured by pressure sensor or float)

Control relevance: Fundamental process control. Nonlinear (square root).
Appears in chemical plants, water treatment, food processing.
"""
function tank_level()
	parameters = @parameters A k Qin
	#                        m² m^2.5/s m³/s
	# Qin is the INPUT (inlet volumetric flow rate)

	states = @variables h(t)
	#                   m

	observables = @variables y1(t)

	p_true = [
		1.0,    # A: tank cross-sectional area (m²)
		0.5,    # k: outflow coefficient (m^2.5/s)
		0.3,    # Qin: INPUT - inlet flow rate (m³/s)
	]

	ic_true = [1.0]  # Initial level 1 meter

	# Mass balance: rate of change = inflow - outflow
	equations = [
		D(h) ~ (Qin - k * sqrt(h)) / A,
	]

	measured_quantities = [y1 ~ h]

	model, mq = create_ordered_ode_system("tank_level", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"tank_level",
		model,
		mq,
		nothing,
		[0.0, 20.0],  # Time to approach steady state
		nothing,
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end

# =============================================================================
#                         TIER 2: HIGHLY RECOMMENDED
#                    Important application domains in control
# ============================================================================= #

"""
    cstr()

Continuous Stirred Tank Reactor - chemical process control benchmark.

Physics: Energy and mass balance for exothermic first-order reaction A → B
- Concentration: dC/dt = (Cin - C)/τ - k(T)*C
- Temperature: dT/dt = (Tin - T)/τ + (-ΔH/ρCp)*k(T)*C - UA/(V*ρCp)*(T - Tc)
- Arrhenius: k(T) = k0 * exp(-E/(R*T))

States: concentration C, temperature T
Input: coolant temperature Tc
Parameters: k0 (pre-exponential), E_R (E/R), τ (residence time), etc.
Observable: temperature (thermocouple), sometimes concentration

Control relevance: Multiple steady states, thermal runaway risk.
Classic nonlinear process control benchmark.
"""
function cstr()
	parameters = @parameters k0 E_R tau Tin Cin dH_rhoCP UA_VrhoCP Tc
	# k0: pre-exponential factor (1/s)
	# E_R: activation energy / gas constant (K)
	# tau: residence time (s)
	# dH_rhoCP: (-ΔH)/(ρ*Cp) (K·L/mol)
	# UA_VrhoCP: UA/(V*ρ*Cp) (1/s)
	# Tc is the INPUT (coolant temperature)

	states = @variables C(t) T(t)
	#                   mol/L  K

	observables = @variables y1(t)

	p_true = [
		7.2e10,   # k0: pre-exponential factor (1/s)
		8750.0,   # E_R: E/R (K) - typical for many reactions
		1.0,      # tau: residence time (s)
		350.0,    # Tin: inlet temperature (K)
		1.0,      # Cin: inlet concentration (mol/L)
		5.0,      # dH_rhoCP: heat release parameter (K·L/mol)
		1.0,      # UA_VrhoCP: heat transfer parameter (1/s)
		300.0,    # Tc: INPUT - coolant temperature (K)
	]

	ic_true = [0.5, 350.0]  # Initial concentration and temperature

	# Reaction rate via Arrhenius equation
	# k(T) = k0 * exp(-E_R/T)
	equations = [
		D(C) ~ (Cin - C) / tau - k0 * exp(-E_R / T) * C,
		D(T) ~ (Tin - T) / tau + dH_rhoCP * k0 * exp(-E_R / T) * C - UA_VrhoCP * (T - Tc),
	]

	# Temperature is most commonly measured
	measured_quantities = [y1 ~ T]

	model, mq = create_ordered_ode_system("cstr", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"cstr",
		model,
		mq,
		nothing,
		[0.0, 10.0],  # Several residence times
		nothing,
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end

"""
    quadrotor_altitude()

Quadrotor vertical/altitude dynamics - modern robotics and aerospace.

Physics: Vertical motion with thrust and drag
- m*z̈ = T - m*g - d*ż  (Newton's second law)

States: altitude z, vertical velocity w
Input: thrust T
Parameters: m (mass), g (gravity), d (drag coefficient)
Observable: altitude (altimeter, barometer, or GPS)

Control relevance: Topical application in UAVs, drones, aerial robotics.
Simplified 1D model useful for altitude hold controllers.
"""
function quadrotor_altitude()
	parameters = @parameters m g d T
	#                        kg m/s² N·s/m N
	# T is the INPUT (total thrust from rotors)

	states = @variables z(t) w(t)
	#                   m    m/s (vertical)

	observables = @variables y1(t)

	p_true = [
		1.0,     # m: mass (kg)
		9.81,    # g: gravity (m/s²)
		0.1,     # d: drag coefficient (N·s/m)
		10.0,    # T: INPUT - thrust (N) - slightly above hover
	]

	ic_true = [0.0, 0.0]  # Start at ground level, at rest

	equations = [
		D(z) ~ w,
		D(w) ~ (T - m * g - d * w) / m,
	]

	measured_quantities = [y1 ~ z]

	model, mq = create_ordered_ode_system("quadrotor_altitude", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"quadrotor_altitude",
		model,
		mq,
		nothing,
		[0.0, 10.0],  # Watch altitude response
		nothing,
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end

"""
    thermal_system()

Room/building temperature control - HVAC and energy systems.

Physics: Lumped thermal model
- C*dT/dt = Q - (T - Ta)/R  (thermal energy balance)

States: room temperature T
Input: heater power Q
Parameters: C (thermal capacitance), R (thermal resistance), Ta (ambient)
Observable: temperature (thermostat)

Control relevance: HVAC control, building energy management.
Simple first-order system but ubiquitous practical application.
"""
function thermal_system()
	parameters = @parameters C R Ta Q
	#                        J/K K/W K  W
	# Q is the INPUT (heating power)

	states = @variables T(t)
	#                   K (or °C)

	observables = @variables y1(t)

	p_true = [
		1000.0,   # C: thermal capacitance (J/K) - room-scale
		0.01,     # R: thermal resistance (K/W) - insulation quality
		293.0,    # Ta: ambient temperature (K) ≈ 20°C
		500.0,    # Q: INPUT - heater power (W)
	]

	ic_true = [288.0]  # Start at 15°C

	equations = [
		D(T) ~ (Q - (T - Ta) / R) / C,
	]

	measured_quantities = [y1 ~ T]

	model, mq = create_ordered_ode_system("thermal_system", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"thermal_system",
		model,
		mq,
		nothing,
		[0.0, 1000.0],  # Long time to see thermal dynamics
		nothing,
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end

"""
    ball_beam()

Ball and beam system - classic underactuated control benchmark.

Physics: Ball rolling on tilted beam, beam angle controlled
- Ball: (J_b/R² + m)*r̈ = m*g*sin(θ) - m*r*θ̇²
- Simplified (small angles): r̈ ≈ (m*g*θ)/(J_b/R² + m)
- Beam: J_beam * θ̈ = τ - m*g*r*cos(θ)

States: ball position r, ball velocity v, beam angle θ, angular velocity ω
Input: beam torque τ
Parameters: m (ball mass), R (ball radius), J_b (ball inertia), g (gravity)
Observable: ball position, beam angle

Control relevance: Classic underactuated system used in control labs.
Requires careful linearization and feedback design.
"""
function ball_beam()
	parameters = @parameters m R J_beam g tau
	# m: ball mass, R: ball radius
	# J_beam: beam moment of inertia about pivot
	# tau is the INPUT (torque applied to beam)

	states = @variables r(t) rdot(t) theta(t) omega(t)
	#                   m    m/s     rad      rad/s

	observables = @variables y1(t) y2(t)

	# Ball moment of inertia: J_b = (2/5)*m*R² for solid sphere
	# Effective inertia: J_eff = J_b/R² + m = (2/5)*m + m = (7/5)*m
	p_true = [
		0.1,     # m: ball mass (kg)
		0.02,    # R: ball radius (m)
		0.5,     # J_beam: beam inertia (kg·m²)
		9.81,    # g: gravity (m/s²)
		0.0,     # tau: INPUT - beam torque (N·m)
	]

	ic_true = [0.1, 0.0, 0.0, 0.0]  # Ball displaced, system at rest

	# Simplified equations (small angle approximation for sin/cos)
	# Full nonlinear version would use sin(theta), cos(theta)
	# J_eff = (7/5)*m for solid sphere rolling without slipping
	equations = [
		D(r) ~ rdot,
		D(rdot) ~ (5.0 / 7.0) * g * sin(theta) - r * omega^2,  # Ball dynamics
		D(theta) ~ omega,
		D(omega) ~ (tau - m * g * r * cos(theta)) / J_beam,    # Beam dynamics
	]

	measured_quantities = [y1 ~ r, y2 ~ theta]

	model, mq = create_ordered_ode_system("ball_beam", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"ball_beam",
		model,
		mq,
		nothing,
		[0.0, 5.0],  # Watch system evolve
		nothing,
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end

# =============================================================================
#                          TIER 3: GOOD TO INCLUDE
#                       Specialized but valuable domains
# ============================================================================= #

"""
    bicycle_model()

Vehicle lateral dynamics using the bicycle (single-track) model.

Physics: Lateral force balance and yaw moment balance
- m*(v̇y + Vx*r) = Fyf + Fyr  (lateral)
- Iz*ṙ = lf*Fyf - lr*Fyr      (yaw)
- Cornering forces: Fyf = Cf*αf, Fyr = Cr*αr (linear tire model)
- Slip angles depend on velocities and steering

States: lateral velocity vy, yaw rate r
Input: steering angle δ
Parameters: Cf, Cr (cornering stiffness), m, Iz, lf, lr, Vx
Observable: yaw rate (gyroscope), lateral acceleration

Control relevance: Autonomous vehicles, lane keeping, electronic stability.
Standard model in automotive control literature.
"""
function bicycle_model()
	parameters = @parameters Cf Cr m Iz lf lr Vx delta
	# Cf, Cr: front/rear cornering stiffness (N/rad)
	# m: vehicle mass (kg)
	# Iz: yaw moment of inertia (kg·m²)
	# lf, lr: distance from CG to front/rear axle (m)
	# Vx: longitudinal velocity (m/s) - assumed constant
	# delta is the INPUT (front wheel steering angle)

	states = @variables vy(t) r(t)
	#                   m/s   rad/s

	observables = @variables y1(t) y2(t)

	p_true = [
		80000.0,   # Cf: front cornering stiffness (N/rad)
		80000.0,   # Cr: rear cornering stiffness (N/rad)
		1500.0,    # m: vehicle mass (kg)
		2500.0,    # Iz: yaw inertia (kg·m²)
		1.2,       # lf: CG to front axle (m)
		1.4,       # lr: CG to rear axle (m)
		20.0,      # Vx: forward speed (m/s) ≈ 72 km/h
		0.02,      # delta: INPUT - steering angle (rad) ≈ 1 degree
	]

	ic_true = [0.0, 0.0]  # Initially going straight

	# Linear single-track model equations
	# Front slip angle: αf ≈ δ - (vy + lf*r)/Vx
	# Rear slip angle: αr ≈ -(vy - lr*r)/Vx
	# Lateral forces: Fyf = Cf*αf, Fyr = Cr*αr
	equations = [
		D(vy) ~ (Cf * (delta - (vy + lf * r) / Vx) + Cr * (-(vy - lr * r) / Vx)) / m - Vx * r,
		D(r) ~ (lf * Cf * (delta - (vy + lf * r) / Vx) - lr * Cr * (-(vy - lr * r) / Vx)) / Iz,
	]

	# Yaw rate is commonly measured, lateral acceleration can be computed
	measured_quantities = [y1 ~ r, y2 ~ vy]

	model, mq = create_ordered_ode_system("bicycle_model", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"bicycle_model",
		model,
		mq,
		nothing,
		[0.0, 5.0],  # Response to steering input
		nothing,
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end

"""
    swing_equation()

Power system generator swing equation - grid stability analysis.

Physics: Rotor dynamics of synchronous generator
- 2H/ωs * d²δ/dt² = Pm - Pe - D*(dδ/dt)
- Pe = Pmax * sin(δ)  (power-angle relationship)

States: rotor angle δ, frequency deviation Δω
Input: mechanical power Pm
Parameters: H (inertia constant), D (damping), Pmax (sync. power)
Observable: frequency deviation (system frequency)

Control relevance: Power grid stability, frequency regulation.
Fundamental model for transient stability analysis.
"""
function swing_equation()
	parameters = @parameters H D_damp Pmax omega_s Pm
	# H: inertia constant (s)
	# D_damp: damping coefficient (pu)
	# Pmax: maximum synchronizing power (pu)
	# omega_s: synchronous speed (rad/s)
	# Pm is the INPUT (mechanical power input)

	states = @variables delta(t) Delta_omega(t)
	#                   rad      rad/s (deviation from synchronous)

	observables = @variables y1(t)

	p_true = [
		5.0,      # H: inertia constant (s) - typical for large generator
		1.0,      # D_damp: damping coefficient
		1.0,      # Pmax: maximum sync power (pu)
		377.0,    # omega_s: 60 Hz → 2π*60 rad/s
		0.8,      # Pm: INPUT - mechanical power (pu)
	]

	# Initial condition: steady state at angle where Pm = Pe
	# sin(δ0) = Pm/Pmax = 0.8 → δ0 ≈ 0.927 rad
	ic_true = [0.927, 0.0]  # At equilibrium, no frequency deviation

	# Electrical power: Pe = Pmax * sin(delta)
	equations = [
		D(delta) ~ Delta_omega,
		D(Delta_omega) ~ (omega_s / (2 * H)) * (Pm - Pmax * sin(delta) - D_damp * Delta_omega),
	]

	# Frequency is the primary measured quantity
	measured_quantities = [y1 ~ Delta_omega]

	model, mq = create_ordered_ode_system("swing_equation", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"swing_equation",
		model,
		mq,
		nothing,
		[0.0, 10.0],  # Observe transient response
		nothing,
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end

"""
    magnetic_levitation()

Magnetic levitation system - nonlinear, unstable electromagnetic system.

Physics:
- Mechanical: m*z̈ = m*g - F_mag(i,z)
- Electrical: V = R*i + L(z)*di/dt + (∂L/∂z)*i*ż
- Magnetic force: F_mag = c*i²/z²  (approximation)

States: ball position z (down positive), velocity v, coil current i
Input: coil voltage V
Parameters: m (mass), R (resistance), c (force constant), g (gravity)
Observable: ball position (optical or inductive sensor)

Control relevance: Inherently unstable, requires feedback.
Used in maglev trains, active magnetic bearings, contactless conveyors.
"""
function magnetic_levitation()
	parameters = @parameters m R c g L0 V
	# m: ball mass
	# R: coil resistance
	# c: magnetic force constant
	# L0: nominal inductance
	# V is the INPUT (coil voltage)

	states = @variables z(t) v(t) i(t)
	#                   m    m/s  A

	observables = @variables y1(t)

	p_true = [
		0.01,    # m: ball mass (kg)
		1.0,     # R: coil resistance (Ω)
		0.0001,  # c: force constant (N·m²/A²)
		9.81,    # g: gravity (m/s²)
		0.01,    # L0: nominal inductance (H)
		1.0,     # V: INPUT - coil voltage (V), matched to the nominal 1 A operating current
	]

	# Use a physically consistent near-equilibrium operating point.
	# For i = 1 A, lift balance requires z_eq = sqrt(c / (m * g)) ≈ 3.19 cm.
	# Start slightly below that height to generate motion without immediately destabilizing the sampling solve.
	ic_true = [0.033, 0.0, 1.0]  # z≈3.3 cm, at rest, 1 A current

	# Simplified model with constant inductance
	# Magnetic force: F = c*i²/z²
	equations = [
		D(z) ~ v,
		D(v) ~ g - (c * i^2) / (m * z^2),    # Gravity minus magnetic lift
		D(i) ~ (V - R * i) / L0,              # RL circuit dynamics
	]

	measured_quantities = [y1 ~ z]

	model, mq = create_ordered_ode_system("magnetic_levitation", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"magnetic_levitation",
		model,
		mq,
		nothing,
		[0.0, 0.1],  # Fast dynamics; keep the raw open-loop model in a short, well-sampled horizon
		nothing,
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end

"""
    aircraft_pitch()

Aircraft longitudinal (pitch) dynamics - aerospace control.

Physics: Short-period approximation of longitudinal dynamics
- Pitch rate: q̇ = Mα*α + Mq*q + Mδe*δe
- Angle of attack: α̇ = Zα*α/V + q
- (where M and Z are stability/control derivatives)

States: pitch angle θ, pitch rate q, angle of attack α
Input: elevator deflection δe
Parameters: Mα, Mq, Mδe, Zα (stability derivatives)
Observable: pitch rate (rate gyro)

Control relevance: Fundamental aerospace control problem.
Autopilot design, stability augmentation systems.
"""
function aircraft_pitch()
	parameters = @parameters M_alpha M_q M_delta_e Z_alpha V_air delta_e
	# Stability derivatives (1/s and 1/s² depending on term)
	# V_air: airspeed (m/s)
	# delta_e is the INPUT (elevator deflection)

	states = @variables theta(t) q(t) alpha(t)
	#                   rad      rad/s  rad

	observables = @variables y1(t)

	# Typical values for a light aircraft
	p_true = [
		-5.0,     # M_alpha: pitch stiffness derivative (1/s²)
		-2.0,     # M_q: pitch damping derivative (1/s)
		-10.0,    # M_delta_e: elevator effectiveness (1/s²)
		-0.5,     # Z_alpha: lift curve slope effect (1/s)
		50.0,     # V_air: airspeed (m/s)
		0.05,     # delta_e: INPUT - elevator deflection (rad)
	]

	ic_true = [0.0, 0.0, 0.05]  # Level flight, small angle of attack

	# Short period approximation
	equations = [
		D(theta) ~ q,
		D(q) ~ M_alpha * alpha + M_q * q + M_delta_e * delta_e,
		D(alpha) ~ Z_alpha * alpha / V_air + q,
	]

	# Pitch rate is commonly measured by rate gyros
	measured_quantities = [y1 ~ q]

	model, mq = create_ordered_ode_system("aircraft_pitch", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"aircraft_pitch",
		model,
		mq,
		nothing,
		[0.0, 5.0],  # Short-period response
		nothing,
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end

# =============================================================================
#                         TIER 4: ADDITIONAL SYSTEMS
#             Including bilinear, ecological, and specialized systems
# ============================================================================= #

"""
    two_tank()

Coupled two-tank system - multivariable process control.

Physics: Two interconnected tanks with gravity-driven flow
- Tank 1: A1*dh1/dt = Qin - k1*sqrt(h1) - k12*sign(h1-h2)*sqrt(|h1-h2|)
- Tank 2: A2*dh2/dt = k12*sign(h1-h2)*sqrt(|h1-h2|) - k2*sqrt(h2)

States: levels h1, h2
Input: inlet flow Q1
Parameters: A1, A2 (areas), k1, k2, k12 (flow coefficients)
Observable: both levels

Control relevance: Multivariable control, interacting loops.
Classic process control laboratory experiment.
"""
function two_tank()
	parameters = @parameters A1 A2 k1 k2 k12 Qin
	# A1, A2: tank cross-sections
	# k1, k2: outlet coefficients
	# k12: interconnection coefficient
	# Qin is the INPUT

	states = @variables h1(t) h2(t)

	observables = @variables y1(t) y2(t)

	p_true = [
		1.0,     # A1: tank 1 area (m²)
		1.0,     # A2: tank 2 area (m²)
		0.3,     # k1: tank 1 outlet coefficient
		0.3,     # k2: tank 2 outlet coefficient
		0.2,     # k12: interconnection coefficient
		0.5,     # Qin: INPUT - inlet flow (m³/s)
	]

	ic_true = [1.0, 0.5]  # Initial levels

	# Simplified model (assuming h1 > h2 always for the interconnection)
	equations = [
		D(h1) ~ (Qin - k1 * sqrt(h1) - k12 * sqrt(h1 - h2 + 0.01)) / A1,
		D(h2) ~ (k12 * sqrt(h1 - h2 + 0.01) - k2 * sqrt(h2)) / A2,
	]

	measured_quantities = [y1 ~ h1, y2 ~ h2]

	model, mq = create_ordered_ode_system("two_tank", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"two_tank",
		model,
		mq,
		nothing,
		[0.0, 30.0],
		nothing,
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end

"""
    boost_converter()

DC-DC boost converter (averaged model) - power electronics.

Physics: State-space averaged model
- L*diL/dt = Vin - (1-d)*vC
- C*dvC/dt = (1-d)*iL - vC/R

States: inductor current iL, capacitor voltage vC
Input: duty cycle d (0 < d < 1)
Parameters: L (inductance), C (capacitance), R (load), Vin (input voltage)
Observable: output voltage vC

Control relevance: Power electronics, renewable energy systems.
Nonlinear due to multiplication by duty cycle.
"""
function boost_converter()
	parameters = @parameters L C R Vin d
	# L: inductance (H)
	# C: capacitance (F)
	# R: load resistance (Ω)
	# Vin: input voltage (V)
	# d is the INPUT (duty cycle)

	states = @variables iL(t) vC(t)
	#                   A     V

	observables = @variables y1(t)

	p_true = [
		0.001,   # L: inductance (1 mH)
		0.001,   # C: capacitance (1 mF)
		10.0,    # R: load resistance (Ω)
		12.0,    # Vin: input voltage (V)
		0.5,     # d: INPUT - duty cycle (50%)
	]

	# Boost ratio: Vout/Vin = 1/(1-d) = 2 for d=0.5
	ic_true = [1.0, 24.0]  # Near steady state

	# d' = 1 - d (complement of duty cycle)
	equations = [
		D(iL) ~ (Vin - (1 - d) * vC) / L,
		D(vC) ~ ((1 - d) * iL - vC / R) / C,
	]

	# Output voltage is the primary measurement
	measured_quantities = [y1 ~ vC]

	model, mq = create_ordered_ode_system("boost_converter", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"boost_converter",
		model,
		mq,
		nothing,
		[0.0, 0.1],  # Fast electrical dynamics
		nothing,
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end

"""
    flexible_arm()

Flexible robot arm - vibration control and non-collocated control.

Physics: Motor-driven link with flexibility (2-DOF model)
- Motor: Jm*θ̈m + bm*θ̇m + k*(θm - θt) = τ
- Tip: Jt*θ̈t + bt*θ̇t + k*(θt - θm) = 0

States: motor angle θm, motor velocity ωm, tip angle θt, tip velocity ωt
Input: motor torque τ
Parameters: Jm, Jt (inertias), bm, bt (damping), k (stiffness)
Observable: motor angle, tip angle (or just motor)

Control relevance: Flexible manipulators, vibration suppression.
Non-collocated control (actuator and sensor at different locations).
"""
function flexible_arm()
	parameters = @parameters Jm Jt bm bt k tau
	# Jm, Jt: motor and tip inertias
	# bm, bt: damping coefficients
	# k: joint stiffness
	# tau is the INPUT (motor torque)

	states = @variables theta_m(t) omega_m(t) theta_t(t) omega_t(t)

	observables = @variables y1(t) y2(t)

	p_true = [
		0.1,     # Jm: motor inertia (kg·m²)
		0.05,    # Jt: tip inertia (kg·m²)
		0.1,     # bm: motor damping (N·m·s/rad)
		0.05,    # bt: tip damping (N·m·s/rad)
		10.0,    # k: stiffness (N·m/rad)
		0.5,     # tau: INPUT - motor torque (N·m)
	]

	ic_true = [0.0, 0.0, 0.0, 0.0]  # Start at rest

	equations = [
		D(theta_m) ~ omega_m,
		D(omega_m) ~ (tau - bm * omega_m - k * (theta_m - theta_t)) / Jm,
		D(theta_t) ~ omega_t,
		D(omega_t) ~ (-bt * omega_t - k * (theta_t - theta_m)) / Jt,
	]

	# Often only motor angle is measured (encoder)
	measured_quantities = [y1 ~ theta_m, y2 ~ theta_t]

	model, mq = create_ordered_ode_system("flexible_arm", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"flexible_arm",
		model,
		mq,
		nothing,
		[0.0, 10.0],  # See vibration response
		nothing,
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end

"""
    bilinear_system()

Generic bilinear system - important class between linear and nonlinear.

Physics: ẋ = Ax + Bu + N*x*u (state-input product terms)
Bilinear systems arise in:
- Heat exchangers (temperature × flow rate)
- DC motors (current × angular velocity)
- Mixing processes (concentration × flow)
- Population dynamics with harvesting

States: x1, x2
Input: u
Parameters: A matrix (a11,a12,a21,a22), B vector (b1,b2), N vector (n1,n2)
Observable: x1

Control relevance: Bridge between linear and nonlinear systems.
Admits special analysis techniques (Lie algebra, Carleman linearization).
"""
function bilinear_system()
	parameters = @parameters a11 a12 a21 a22 b1 b2 n1 n2 u
	# A = [a11 a12; a21 a22]: linear state matrix
	# B = [b1; b2]: input matrix
	# N terms: x1*u and x2*u coefficients
	# u is the INPUT

	states = @variables x1(t) x2(t)

	observables = @variables y1(t)

	p_true = [
		-0.5,    # a11
		0.2,     # a12
		0.1,     # a21
		-0.3,    # a22
		1.0,     # b1
		0.5,     # b2
		0.2,     # n1: coefficient for x1*u
		0.1,     # n2: coefficient for x2*u
		1.0,     # u: INPUT
	]

	ic_true = [1.0, 0.5]

	# Bilinear dynamics: ẋ = Ax + Bu + diag(N)*x*u
	equations = [
		D(x1) ~ a11 * x1 + a12 * x2 + b1 * u + n1 * x1 * u,
		D(x2) ~ a21 * x1 + a22 * x2 + b2 * u + n2 * x2 * u,
	]

	measured_quantities = [y1 ~ x1]

	model, mq = create_ordered_ode_system("bilinear_system", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"bilinear_system",
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

"""
    forced_lotka_volterra()

Lotka-Volterra predator-prey with control input - ecological control.

Physics: Standard predator-prey with harvesting/stocking
- Prey: dx/dt = ax - bxy + u  (u: stocking rate or harvest if negative)
- Predator: dy/dt = -cy + dxy

States: prey population x, predator population y
Input: prey input rate u (stocking if positive, harvesting if negative)
Parameters: a (prey growth), b (predation), c (predator death), d (conversion)
Observable: prey population

Control relevance: Fisheries management, pest control, conservation.
Classic example of how control input affects equilibrium.
"""
function forced_lotka_volterra()
	parameters = @parameters a b c d u
	# a: prey growth rate
	# b: predation rate
	# c: predator death rate
	# d: predator efficiency
	# u is the INPUT (prey stocking/harvesting rate)

	states = @variables x(t) y(t)
	#                   prey predator

	observables = @variables y1(t)

	p_true = [
		1.0,     # a: prey growth rate
		0.5,     # b: predation rate
		0.5,     # c: predator death rate
		0.25,    # d: predator efficiency
		0.1,     # u: INPUT - small stocking rate
	]

	ic_true = [2.0, 1.0]  # Initial populations

	equations = [
		D(x) ~ a * x - b * x * y + u,
		D(y) ~ -c * y + d * x * y,
	]

	# Prey population is typically easier to measure
	measured_quantities = [y1 ~ x]

	model, mq = create_ordered_ode_system("forced_lotka_volterra", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"forced_lotka_volterra",
		model,
		mq,
		nothing,
		[0.0, 30.0],  # Several oscillation periods
		nothing,
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end

# =============================================================================
#                      LINEARIZED VERSIONS FOR COMPARISON
#           Linear models for comparing nonlinear vs linear identifiability
# ============================================================================= #

"""
    cart_pole_linear()

Linearized cart-pole around upright equilibrium.

For small angles θ ≈ 0: sin(θ) ≈ θ, cos(θ) ≈ 1
This gives a linear state-space model suitable for LQR design.

States: cart position x, cart velocity v, pendulum angle θ, angular velocity ω
Input: force F
Observable: cart position, pendulum angle

This version shows what happens when the nonlinear system is linearized -
useful for comparing identifiability of the linear vs nonlinear models.
"""
function cart_pole_linear()
	parameters = @parameters M m l g F
	# Same parameters as nonlinear cart-pole

	states = @variables x(t) v(t) theta(t) omega(t)

	observables = @variables y1(t) y2(t)

	p_true = [
		1.0,    # M: cart mass (kg)
		0.1,    # m: pendulum mass (kg)
		0.5,    # l: pendulum length (m)
		9.81,   # g: gravity (m/s²)
		0.0,    # F: INPUT - applied force (N)
	]

	ic_true = [0.0, 0.0, 0.1, 0.0]  # Small angle

	# Linearized equations around θ=0, ω=0
	# sin(θ) ≈ θ, cos(θ) ≈ 1, higher order terms dropped
	equations = [
		D(x) ~ v,
		D(v) ~ (F + m * g * theta) / M,  # Linearized: sin(θ)→θ, dropped θ² terms
		D(theta) ~ omega,
		D(omega) ~ (-F - (M + m) * g * theta) / (l * M),  # Linearized
	]

	measured_quantities = [y1 ~ x, y2 ~ theta]

	model, mq = create_ordered_ode_system("cart_pole_linear", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"cart_pole_linear",
		model,
		mq,
		nothing,
		[0.0, 5.0],
		nothing,
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end

"""
    maglev_linear()

Linearized magnetic levitation around nominal operating point.

Linearized around equilibrium (z0, i0) where m*g = c*i0²/z0²
Perturbation model: δz, δv, δi from equilibrium

States: position perturbation δz, velocity perturbation δv, current perturbation δi
Input: voltage perturbation δV
Observable: position perturbation

Demonstrates how linearization affects identifiability analysis.
"""
function maglev_linear()
	# Linearization constants (derived from equilibrium conditions)
	parameters = @parameters Kz Ki tau R delta_V
	# Kz: position gain (linearized force w.r.t. position)
	# Ki: current gain (linearized force w.r.t. current)
	# tau: electrical time constant
	# R: resistance
	# delta_V is the INPUT (voltage perturbation)

	states = @variables dz(t) dv(t) di(t)
	#                   m     m/s   A (perturbations)

	observables = @variables y1(t)

	# These are linearization coefficients, not the original physical parameters
	p_true = [
		1000.0,   # Kz: position sensitivity (N/m) - unstable!
		0.1,      # Ki: current sensitivity (N/A)
		0.01,     # tau: electrical time constant (s)
		1.0,      # R: resistance (Ω)
		0.0,      # delta_V: INPUT - voltage perturbation (V)
	]

	ic_true = [0.001, 0.0, 0.0]  # Small perturbation

	# Linearized model (perturbation dynamics)
	# Note: Kz > 0 means unstable (gravity wins if ball moves down)
	equations = [
		D(dz) ~ dv,
		D(dv) ~ Kz * dz - Ki * di,  # Linearized force balance
		D(di) ~ (delta_V - R * di) / tau,  # Electrical dynamics (linearized)
	]

	measured_quantities = [y1 ~ dz]

	model, mq = create_ordered_ode_system("maglev_linear", states, parameters, equations, measured_quantities)

	return ParameterEstimationProblem(
		"maglev_linear",
		model,
		mq,
		nothing,
		[0.0, 0.1],  # Fast dynamics
		nothing,
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		0,
	)
end

# =============================================================================
#                    NATURAL SINUSOIDAL INPUT MODELS
#   These use sin(ω*t) directly instead of manual oscillator polynomialization.
#   The transcendental handling layer automatically transforms them.
# ============================================================================= #

"""
    dc_motor_sinusoidal()

DC motor with sinusoidal voltage input, written naturally using sin().
This is the natural version of `dc_motor_identifiable()` — the transcendental
handling layer automatically converts sin(5.0*t) into polynomial form.

Compared to `dc_motor_identifiable()`:
- 2 fewer states (no u_sin, u_cos)
- 2 fewer ODEs (no oscillator equations)
- 2 fewer observables (no oscillator observations)
- 2 fewer initial conditions
"""
function dc_motor_sinusoidal()
    @parameters Kt J b

    # FIXED electrical parameters
    R_val = 2.0
    L_val = 0.5
    Kb_val = 0.1

    # Input parameters
    V0_val = 12.0
    Va_val = 2.0
    omega_val = 5.0

    @variables omega_m(t) i(t)
    @variables y1(t)

    p_true = [0.1, 0.01, 0.1]  # Kt, J, b
    ic_true = [0.0, 0.0]

    # Just write sin() directly!
    V_input = V0_val + Va_val * sin(omega_val * t)

    equations = [
        D(omega_m) ~ (Kt * i - b * omega_m) / J,
        D(i) ~ (V_input - R_val * i - Kb_val * omega_m) / L_val,
    ]

    measured_quantities = [y1 ~ omega_m]

    states = [omega_m, i]
    parameters = [Kt, J, b]

    model, mq = create_ordered_ode_system("dc_motor_sinusoidal", states, parameters, equations, measured_quantities)

    return ParameterEstimationProblem(
        "dc_motor_sinusoidal",
        model,
        mq,
        nothing,
        [0.0, 5.0],
        nothing,
        OrderedDict(parameters .=> p_true),
        OrderedDict(states .=> ic_true),
        0,
    )
end

"""
    quadrotor_sinusoidal()

Quadrotor altitude with sinusoidal thrust, written naturally using sin().
This is the natural version of `quadrotor_altitude_identifiable()`.
"""
function quadrotor_sinusoidal()
    @parameters m d

    g_val = 9.81
    Ta_val = 2.0
    omega_val = 1.0

    @variables z(t) w(t)
    @variables y1(t)

    p_true = [1.0, 0.1]
    ic_true = [5.0, 0.0]

    equations = [
        D(z) ~ w,
        D(w) ~ (Ta_val * sin(omega_val * t) - d * w) / m,
    ]

    measured_quantities = [y1 ~ z]

    states = [z, w]
    parameters = [m, d]

    model, mq = create_ordered_ode_system("quadrotor_sinusoidal", states, parameters, equations, measured_quantities)

    return ParameterEstimationProblem(
        "quadrotor_sinusoidal",
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

"""
    forced_lv_sinusoidal()

Forced Lotka-Volterra with sinusoidal harvesting, written naturally.
This is the natural version of `forced_lotka_volterra_identifiable()`.
"""
function forced_lv_sinusoidal()
    @parameters alpha beta delta gamma

    h_val = 0.3
    omega_val = 2.0

    @variables x(t) y(t)
    @variables y1(t) y2(t)

    p_true = [1.5, 1.0, 0.5, 3.0]
    ic_true = [1.0, 1.0]

    equations = [
        D(x) ~ alpha * x - beta * x * y - h_val * sin(omega_val * t),
        D(y) ~ delta * x * y - gamma * y,
    ]

    measured_quantities = [y1 ~ x, y2 ~ y]

    states = [x, y]
    parameters = [alpha, beta, delta, gamma]

    model, mq = create_ordered_ode_system("forced_lv_sinusoidal", states, parameters, equations, measured_quantities)

    return ParameterEstimationProblem(
        "forced_lv_sinusoidal",
        model,
        mq,
        nothing,
        [0.0, 5.0],
        nothing,
        OrderedDict(parameters .=> p_true),
        OrderedDict(states .=> ic_true),
        0,
    )
end

"""
    magnetic_levitation_sinusoidal()

Magnetic levitation with sinusoidal voltage, written naturally using sin().
This is the natural version of `magnetic_levitation_identifiable()`.
"""
function magnetic_levitation_sinusoidal()
    @parameters m_lin k_lin b_lin

    # FIXED electrical parameters (measured)
    R_coil_val = 2.0   # coil resistance (Ohms)
    L_val = 0.05       # inductance (H)
    ki_val = 10.0      # current-to-force gain

    # Input parameters - FIXED
    V0_val = 5.0
    Va_val = 1.0
    omega_val = 5.0

    @variables x(t) v(t) i(t)
    @variables y1(t)

    p_true = [
        0.1,    # m_lin: effective mass
        50.0,   # k_lin: linearized stiffness
        2.0,    # b_lin: damping
    ]

    i_eq = V0_val / R_coil_val
    ic_true = [0.0, 0.0, i_eq]

    V_input = V0_val + Va_val * sin(omega_val * t)

    equations = [
        D(x) ~ v,
        D(v) ~ (ki_val * (i - V0_val / R_coil_val) - k_lin * x - b_lin * v) / m_lin,
        D(i) ~ (V_input - R_coil_val * i) / L_val,
    ]

    measured_quantities = [y1 ~ x]

    states = [x, v, i]
    parameters = [m_lin, k_lin, b_lin]

    model, mq = create_ordered_ode_system("magnetic_levitation_sinusoidal", states, parameters, equations, measured_quantities)

    return ParameterEstimationProblem(
        "magnetic_levitation_sinusoidal",
        model,
        mq,
        nothing,
        [0.0, 5.0],
        nothing,
        OrderedDict(parameters .=> p_true),
        OrderedDict(states .=> ic_true),
        0,
    )
end

"""
    aircraft_pitch_sinusoidal()

Aircraft pitch dynamics with sinusoidal elevator input, written naturally using sin().
This is the natural version of `aircraft_pitch_identifiable()`.
"""
function aircraft_pitch_sinusoidal()
    @parameters M_alpha M_q M_delta_e Z_alpha

    # FIXED from flight conditions
    V_air_val = 50.0  # true airspeed (m/s)

    # Input parameters - FIXED
    delta_e0_val = 0.0
    delta_ea_val = 0.05
    omega_val = 2.0

    @variables theta(t) q(t) alpha(t)
    @variables y1(t)

    p_true = [
        -5.0,     # M_alpha
        -2.0,     # M_q
        -10.0,    # M_delta_e
        -0.5,     # Z_alpha
    ]

    ic_true = [0.0, 0.0, 0.05]

    delta_e_input = delta_e0_val + delta_ea_val * sin(omega_val * t)

    equations = [
        D(theta) ~ q,
        D(q) ~ M_alpha * alpha + M_q * q + M_delta_e * delta_e_input,
        D(alpha) ~ Z_alpha * alpha / V_air_val + q,
    ]

    measured_quantities = [y1 ~ q]

    states = [theta, q, alpha]
    parameters = [M_alpha, M_q, M_delta_e, Z_alpha]

    model, mq = create_ordered_ode_system("aircraft_pitch_sinusoidal", states, parameters, equations, measured_quantities)

    return ParameterEstimationProblem(
        "aircraft_pitch_sinusoidal",
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

"""
    bicycle_model_sinusoidal()

Vehicle lateral dynamics with sinusoidal steering input, written naturally using sin().
This is the natural version of `bicycle_model_identifiable()`.
"""
function bicycle_model_sinusoidal()
    @parameters Cf Cr m_veh Iz

    # FIXED from vehicle specs and test conditions
    lf_val = 1.2    # front axle to CG (m)
    lr_val = 1.4    # rear axle to CG (m)
    Vx_val = 20.0   # test speed (m/s)

    # Input parameters - FIXED
    delta_a_val = 0.05
    omega_val = 0.5

    @variables vy(t) r(t)
    @variables y1(t) y2(t)

    p_true = [
        80000.0,   # Cf
        80000.0,   # Cr
        1500.0,    # m_veh
        2500.0,    # Iz
    ]

    ic_true = [0.0, 0.0]

    delta_input = delta_a_val * sin(omega_val * t)

    equations = [
        D(vy) ~ (Cf * (delta_input - (vy + lf_val * r) / Vx_val) + Cr * (-(vy - lr_val * r) / Vx_val)) / m_veh - Vx_val * r,
        D(r) ~ (lf_val * Cf * (delta_input - (vy + lf_val * r) / Vx_val) - lr_val * Cr * (-(vy - lr_val * r) / Vx_val)) / Iz,
    ]

    measured_quantities = [y1 ~ r, y2 ~ vy]

    states = [vy, r]
    parameters = [Cf, Cr, m_veh, Iz]

    model, mq = create_ordered_ode_system("bicycle_model_sinusoidal", states, parameters, equations, measured_quantities)

    return ParameterEstimationProblem(
        "bicycle_model_sinusoidal",
        model,
        mq,
        nothing,
        [0.0, 20.0],
        nothing,
        OrderedDict(parameters .=> p_true),
        OrderedDict(states .=> ic_true),
        0,
    )
end

"""
    boost_converter_sinusoidal()

Boost converter with sinusoidal duty cycle, written naturally using sin().
This is the natural version of `boost_converter_identifiable()`.
"""
function boost_converter_sinusoidal()
    @parameters L C_cap R_load

    # FIXED from measurement
    Vin_val = 12.0  # input voltage (V)

    # Input parameters - FIXED
    d0_val = 0.5
    da_val = 0.1
    omega_val = 100.0

    @variables iL(t) vC(t)
    @variables y1(t) y2(t)

    p_true = [
        0.001,   # L
        0.001,   # C_cap
        10.0,    # R_load
    ]

    ic_true = [1.0, 24.0]

    d_complement = (1.0 - d0_val) - da_val * sin(omega_val * t)

    equations = [
        D(iL) ~ (Vin_val - d_complement * vC) / L,
        D(vC) ~ (d_complement * iL - vC / R_load) / C_cap,
    ]

    measured_quantities = [y1 ~ vC, y2 ~ iL]

    states = [iL, vC]
    parameters = [L, C_cap, R_load]

    model, mq = create_ordered_ode_system("boost_converter_sinusoidal", states, parameters, equations, measured_quantities)

    return ParameterEstimationProblem(
        "boost_converter_sinusoidal",
        model,
        mq,
        nothing,
        [0.0, 0.5],
        nothing,
        OrderedDict(parameters .=> p_true),
        OrderedDict(states .=> ic_true),
        0,
    )
end

"""
    bilinear_system_sinusoidal()

Bilinear system with sinusoidal input, written naturally using sin().
This is the natural version of `bilinear_system_identifiable()`.
"""
function bilinear_system_sinusoidal()
    @parameters a11 a12 a21 a22 b1 b2 n1 n2

    # Input parameters - FIXED
    u0_val = 1.0
    ua_val = 0.5
    omega_val = 2.0

    @variables x1(t) x2(t)
    @variables y1(t) y2(t)

    p_true = [
        -0.5,    # a11
        0.2,     # a12
        0.1,     # a21
        -0.3,    # a22
        1.0,     # b1
        0.5,     # b2
        0.2,     # n1
        0.1,     # n2
    ]

    ic_true = [1.0, 0.5]

    u_input = u0_val + ua_val * sin(omega_val * t)

    equations = [
        D(x1) ~ a11 * x1 + a12 * x2 + b1 * u_input + n1 * x1 * u_input,
        D(x2) ~ a21 * x1 + a22 * x2 + b2 * u_input + n2 * x2 * u_input,
    ]

    measured_quantities = [y1 ~ x1, y2 ~ x2]

    states = [x1, x2]
    parameters = [a11, a12, a21, a22, b1, b2, n1, n2]

    model, mq = create_ordered_ode_system("bilinear_system_sinusoidal", states, parameters, equations, measured_quantities)

    return ParameterEstimationProblem(
        "bilinear_system_sinusoidal",
        model,
        mq,
        nothing,
        [0.0, 15.0],
        nothing,
        OrderedDict(parameters .=> p_true),
        OrderedDict(states .=> ic_true),
        0,
    )
end
