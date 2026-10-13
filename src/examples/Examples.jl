"""
	ODEParameterEstimation.Examples

Ready-made problems to try the package on. Each model is a function that
returns a [`ParameterEstimationProblem`](@ref) with true values and no data:

```julia
using ODEParameterEstimation
using ODEParameterEstimation.Examples

problem = sample_problem_data(lotka_volterra(); datasize = 101, time_interval = [0.0, 20.0])
results = estimate(problem)
```
"""
module Examples

using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrderedCollections: OrderedDict
import ..ODEParameterEstimation
using ..ODEParameterEstimation: ParameterEstimationProblem, create_ordered_ode_system,
	sample_problem_data, estimate, UnsupportedModelClassError

include("models/advanced_systems.jl")
include("models/biological_systems.jl")
include("models/branch_stress_systems.jl")
include("models/classical_systems.jl")
include("models/simple_models.jl")
include("models/test_models.jl")
include("models/debug_models.jl")
include("models/control_systems.jl")

#=============================================================================
                    SINGLE SOURCE OF TRUTH: MODEL REGISTRIES

All available models are registered here.
=============================================================================#

"""Models that currently run cleanly and are reasonable to present as straightforward examples."""
const GREEN_MODELS = Dict(
	# Simple models
	:simple => simple,
	:simple_linear_combination => simple_linear_combination,
	:onesp_cubed => onesp_cubed,
	:threesp_cubed => threesp_cubed,
	:onevar_exp => onevar_exp,

	# Classical systems
	:lotka_volterra => lotka_volterra,
	:vanderpol => vanderpol,
	:harmonic => harmonic,
	:forced_decay => forced_decay,

	:sum_test => sum_test,

	# DAISY models
	:daisy_mamil3 => daisy_mamil3,

	# Specialized models
	:slowfast => slowfast,
	:repressilator => repressilator,

	# Control systems and transformed variants that currently estimate well
	:dc_motor => dc_motor,
	:quadrotor_altitude => quadrotor_altitude,
	:thermal_system => thermal_system,
	:bicycle_model => bicycle_model,
	:flexible_arm => flexible_arm,
	:bilinear_system => bilinear_system,
	:maglev_linear => maglev_linear,

	:quadrotor_altitude_identifiable => quadrotor_altitude_identifiable,
	:aircraft_pitch_identifiable => aircraft_pitch_identifiable,
	:tank_level_poly => tank_level_poly,
	:two_tank_poly => two_tank_poly,
	:bicycle_model_identifiable => bicycle_model_identifiable,

	# Natural sinusoidal input models (auto-polynomialized via transcendental handling)
	:quadrotor_sinusoidal => quadrotor_sinusoidal,
	:forced_lv_sinusoidal => forced_lv_sinusoidal,
	:aircraft_pitch_sinusoidal => aircraft_pitch_sinusoidal,
)

"""Models that are intentionally useful as structural-unidentifiability demonstrations."""
const STRUCTURAL_UNIDENTIFIABILITY_MODELS = Dict(
	:substr_test => substr_test,
	:global_unident_test => global_unident_test,
	:trivial_unident => trivial_unident,
	:aircraft_pitch => aircraft_pitch,
	:dc_motor_identifiable => dc_motor_identifiable,
	:mass_spring_damper => mass_spring_damper,
	:two_compartment_pk => two_compartment_pk,
	:treatment => treatment,
)

"""Models that run, but are harder, less accurate, or more experimental than the green set."""
const HARD_MODELS = Dict(
	:hiv => hiv,
	:hiv_old_wrong => hiv_old_wrong,
	:crauste_corrected => crauste_corrected,
	:allee_competition => allee_competition,
	:biohydrogenation => biohydrogenation,
	:fitzhugh_nagumo => fitzhugh_nagumo,
	:boost_converter_identifiable => boost_converter_identifiable,
	:boost_converter_sinusoidal => boost_converter_sinusoidal,
	:daisy_ex3 => daisy_ex3,
	:daisy_mamil4 => daisy_mamil4,
	:brusselator => brusselator,
	:lv_periodic => lv_periodic,
	:forced_lotka_volterra => forced_lotka_volterra,
	:forced_lotka_volterra_identifiable => forced_lotka_volterra_identifiable,
	:bilinear_system_identifiable => bilinear_system_identifiable,
	:bilinear_system_sinusoidal => bilinear_system_sinusoidal,
	:cart_pole_linear => cart_pole_linear,
	:magnetic_levitation => magnetic_levitation,
	:magnetic_levitation_identifiable => magnetic_levitation_identifiable,
	:magnetic_levitation_sinusoidal => magnetic_levitation_sinusoidal,
	:bicycle_model_sinusoidal => bicycle_model_sinusoidal,
	:dc_motor_sinusoidal => dc_motor_sinusoidal,
	:cstr_fixed_activation => cstr_fixed_activation,
	:cstr_reparametrized => cstr_reparametrized,
	:sirsforced => sirsforced,
)

"""Constructed models for validating branch-aware output behavior."""
const BRANCH_STRESS_MODELS = Dict(
	:latent_subpopulation_branch => latent_subpopulation_branch,
	:latent_subpopulation_observed_control => latent_subpopulation_observed_control,
	:receptor_subtype_binding_branch => receptor_subtype_binding_branch,
	:receptor_subtype_binding_observed_control => receptor_subtype_binding_observed_control,
)

"""Minimally modified variants for bounded-physical M=1 benchmark runs."""
const M1_BENCHMARK_MODELS = Dict(
	:biohydrogenation_m1 => biohydrogenation_m1,
	:daisy_mamil4_m1 => daisy_mamil4_m1,
	:seir_m1 => seir_m1,
	:slow_fast_m1 => slow_fast_m1,
)

"""Models retained in-package but currently best treated as limitations or active failure cases."""
const LIMITATION_MODELS = Dict(
	:seir => seir,
	:cart_pole => cart_pole,
	:tank_level => tank_level,
	:cstr => cstr,
	:ball_beam => ball_beam,
	:swing_equation => swing_equation,
	:two_tank => two_tank,
	:boost_converter => boost_converter,
	:crauste => crauste,
	:crauste_revised => crauste_revised,
)

"""Default runnable set: green examples plus the explicit structural-unidentifiability demos."""
const STANDARD_MODELS = merge(GREEN_MODELS, STRUCTURAL_UNIDENTIFIABILITY_MODELS)

"""All available models (standard + hard)."""
const ALL_MODELS = merge(STANDARD_MODELS, HARD_MODELS, BRANCH_STRESS_MODELS, M1_BENCHMARK_MODELS, LIMITATION_MODELS)

"""Get list of all available model names."""
available_models() = sort(collect(keys(ALL_MODELS)))

"""Get a dictionary of model registries by category name."""
function available_model_categories()
	return Dict(
		:green => GREEN_MODELS,
		:structural_unidentifiability => STRUCTURAL_UNIDENTIFIABILITY_MODELS,
		:standard => STANDARD_MODELS,
		:hard => HARD_MODELS,
		:branch_stress => BRANCH_STRESS_MODELS,
		:m1_benchmark => M1_BENCHMARK_MODELS,
		:limitations => LIMITATION_MODELS,
		:all => ALL_MODELS,
	)
end

export simple, simple_linear_combination, onesp_cubed, threesp_cubed
export lotka_volterra, vanderpol, brusselator, harmonic, fitzhugh_nagumo, forced_decay
export seir, seir_m1, treatment, biohydrogenation, biohydrogenation_m1, repressilator
export crauste, daisy_mamil3, daisy_mamil4, daisy_mamil4_m1, hiv
export latent_subpopulation_branch, latent_subpopulation_observed_control
export receptor_subtype_binding_branch, receptor_subtype_binding_observed_control
export slow_fast_m1, slowfast_m1
export substr_test, global_unident_test, sum_test, trivial_unident

end # module Examples
