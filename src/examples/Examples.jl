"""
	ODEParameterEstimation.Examples

Ready-made problems to try the package on. Each model is a function that
returns a [`ParameterEstimationProblem`](@ref) with true values and no data:

```julia
using ODEParameterEstimation
using ODEParameterEstimation.Examples

problem = sample_problem_data(lotka_volterra(); datasize = 101)
results = estimate(problem)
```

[`example_models`](@ref) lists them and [`example_groups`](@ref) lists the
groups they are sorted into.
"""
module Examples

using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrderedCollections: OrderedDict
using ..ODEParameterEstimation: ParameterEstimationProblem, create_ordered_ode_system,
	sample_problem_data, estimate, UnsupportedModelClassError

# Every model is built here, so that its name, its true values and its time
# interval cannot land in the wrong argument.
function _example_problem(name::Symbol, states, parameters, equations, measured;
	p_true::Vector{Float64}, ic_true::Vector{Float64}, time_interval::Vector{Float64}, unident_count::Int = 0)
	length(p_true) == length(parameters) || throw(ArgumentError("$name: $(length(parameters)) parameters, $(length(p_true)) true values"))
	length(ic_true) == length(states) || throw(ArgumentError("$name: $(length(states)) states, $(length(ic_true)) true values"))
	model, mq = create_ordered_ode_system(String(name), states, parameters, equations, measured)
	return ParameterEstimationProblem(
		String(name),
		model,
		mq,
		nothing,
		time_interval,
		nothing,
		OrderedDict(parameters .=> p_true),
		OrderedDict(states .=> ic_true),
		unident_count,
	)
end

# The one list of models. The exports, `example_models` and the tests all
# read it. A group says what a model is for, not how well it is estimated.
const _CATALOGUE = OrderedDict{Symbol, Vector{Symbol}}(
	:first_steps => [:harmonic_oscillator, :oscillator_mixed_outputs, :exponential_decay, :decay_cubed_output,
		:forced_decay, :lotka_volterra, :vanderpol, :brusselator, :daisy_mamil3, :repressilator],
	:cannot_be_determined => [:sum_of_rates, :partly_observed_chain, :hidden_scale, :treatment,
		:two_compartment_pk, :mass_spring_damper],
	:several_answers => [:hidden_subpopulations, :hidden_subpopulations_measured,
		:receptor_subtypes, :receptor_subtypes_measured],
	:larger_models => [:crauste, :daisy_ex3, :daisy_mamil4, :fitzhugh_nagumo, :hiv, :biohydrogenation,
		:seir, :sirsforced, :slowfast, :allee_competition],
	:control_systems => [:dc_motor, :quadrotor_altitude, :forced_lotka_volterra, :magnetic_levitation,
		:aircraft_pitch, :bicycle_model, :boost_converter, :bilinear_system],
	:rewriting => [:tank_level, :tank_level_rewritten],
)

include("models/first_steps.jl")
include("models/cannot_be_determined.jl")
include("models/several_answers.jl")
include("models/larger_models.jl")
include("models/control_systems.jl")
include("models/rewriting.jl")

"""
	example_groups() -> Vector{Symbol}

The groups the example models are sorted into: `:first_steps`,
`:cannot_be_determined`, `:several_answers`, `:larger_models`,
`:control_systems` and `:rewriting`.
"""
example_groups() = collect(keys(_CATALOGUE))

"""
	example_models() -> Vector{Symbol}
	example_models(group::Symbol) -> Vector{Symbol}

The names of the example models, all of them or those of one group.

# Example
```julia
example_models(:first_steps)
```
"""
example_models() = reduce(vcat, values(_CATALOGUE))
function example_models(group::Symbol)
	haskey(_CATALOGUE, group) || throw(ArgumentError(
		"`$group` is not a group of example models. The groups are $(join(("`:$g`" for g in keys(_CATALOGUE)), ", "))."))
	return copy(_CATALOGUE[group])
end

"""
	example_problem(name::Symbol) -> ParameterEstimationProblem

The example model called `name`: `example_problem(:lotka_volterra)` is
`lotka_volterra()`.
"""
function example_problem(name::Symbol)
	name in example_models() || throw(ArgumentError(
		"`$name` is not an example model. `example_models()` lists them."))
	return getfield(Examples, name)()
end

export example_groups, example_models, example_problem
for name in example_models()
	@eval export $name
end

end # module Examples
