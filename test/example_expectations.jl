# What each example model is expected to do when it is estimated with the
# default options from 201 exact data points over its own time interval.
# test_example_models.jl and example_estimation.jl both read this table.
#
# - `tier`: `:quick` models are estimated on every push, `:slow` ones in the
#   slow group, `:refused` ones must be refused, and `:none` ones are not
#   estimated because the package does not solve them in reasonable time.
# - `undetermined`: what the results must report as not determined.
# - `answers`: how many results are expected.
# - `tolerance`: the largest relative error allowed, over everything that is
#   determined, in the best of the results.

function _expectation(tier::Symbol; undetermined = String[], answers::Int = 1, tolerance::Float64 = 1e-6,
	unident_count::Int = length(undetermined))
	return (; tier, undetermined = Set{String}(undetermined), answers, tolerance, unident_count)
end

const EXAMPLE_EXPECTATIONS = Dict{Symbol, NamedTuple}(
	# First steps
	:harmonic_oscillator => _expectation(:quick),
	:oscillator_mixed_outputs => _expectation(:quick),
	:exponential_decay => _expectation(:quick),
	:decay_cubed_output => _expectation(:quick),
	:forced_decay => _expectation(:quick),
	:lotka_volterra => _expectation(:quick),
	:vanderpol => _expectation(:quick),
	# Good to 1e-5 before the final refinement and 1e-14 after it, so this
	# tolerance fails if the refinement stops being applied.
	:brusselator => _expectation(:quick; tolerance = 1e-9),
	:daisy_mamil3 => _expectation(:quick),
	:repressilator => _expectation(:slow),
	# Something cannot be determined
	:sum_of_rates => _expectation(:quick; undetermined = ["a", "b"]),
	:partly_observed_chain => _expectation(:quick; undetermined = ["b", "c", "d", "x3(t)"]),
	:hidden_scale => _expectation(:quick; undetermined = ["c", "x1(t)", "x2(t)"], answers = 2),
	:treatment => _expectation(:slow; undetermined = ["In(t)", "S(t)", "a", "b", "d", "g"], answers = 2),
	:two_compartment_pk => _expectation(:quick; undetermined = ["C2(t)", "V1", "V2", "k12", "k21", "ke"], answers = 2),
	:mass_spring_damper => _expectation(:quick; undetermined = ["F", "c", "k", "m"]),
	# More than one answer
	:hidden_subpopulations => _expectation(:slow; answers = 6),
	:hidden_subpopulations_measured => _expectation(:quick),
	:receptor_subtypes => _expectation(:none),
	:receptor_subtypes_measured => _expectation(:quick),
	# Larger models
	:crauste => _expectation(:none),
	:daisy_ex3 => _expectation(:slow; answers = 2),
	:daisy_mamil4 => _expectation(:slow; answers = 2),
	:fitzhugh_nagumo => _expectation(:quick),
	:hiv => _expectation(:slow),
	:biohydrogenation => _expectation(:slow; undetermined = ["x7(t)"], answers = 2),
	:seir => _expectation(:slow; answers = 2),
	:sirsforced => _expectation(:none; unident_count = 3),
	:slowfast => _expectation(:quick; answers = 2),
	:allee_competition => _expectation(:slow; answers = 4),
	# Control systems
	:dc_motor => _expectation(:quick),
	:quadrotor_altitude => _expectation(:quick),
	:forced_lotka_volterra => _expectation(:quick),
	:magnetic_levitation => _expectation(:quick),
	:aircraft_pitch => _expectation(:quick; undetermined = ["theta(t)"]),
	:bicycle_model => _expectation(:quick; undetermined = ["Cf", "Cr", "Iz", "m_veh"]),
	:boost_converter => _expectation(:quick; tolerance = 1e-5),
	:bilinear_system => _expectation(:quick),
	# A model that has to be rewritten
	:tank_level => _expectation(:refused),
	:tank_level_rewritten => _expectation(:quick; answers = 2),
)

# The few models that every push estimates on CI. Together they cover a plain
# recovery, a model whose answer before the final refinement is good to five
# digits only, an undetermined set with two answers, a time-dependent input and
# a refusal. The whole quick tier takes about an hour there, so it runs weekly.
const EXAMPLE_CI_MODELS = [:lotka_volterra, :brusselator, :hidden_scale, :dc_motor, :tank_level]
