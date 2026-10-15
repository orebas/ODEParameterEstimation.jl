# The short path through the package: build a problem from a ModelingToolkit
# system and your data, call `estimate`, read the results.

# The names a variable goes by outside the symbolic world: "y(t)" and "y".
function _variable_names(variable)
	full = string(variable)
	return (full, first(split(full, '(')))
end

_is_symbolic(value) = value isa Num || value isa SymbolicUtils.BasicSymbolic
_is_plain_number(value) = value isa Real && !_is_symbolic(value)

# `(key, values)` pairs of a dictionary, or of anything with named columns
# (a NamedTuple, a DataFrame).
_data_entries(data::AbstractDict) = collect(pairs(data))
_data_entries(data) = [name => getproperty(data, name) for name in propertynames(data)]

function _find_entry(matches, entries)
	index = findfirst(entry -> matches(first(entry)), entries)
	return isnothing(index) ? nothing : last(entries[index])
end

"""
	_observation_dictionary(system, measured_quantities, data)

Convert user data into the form the estimator reads: times under `"t"` and one
series per measured quantity, keyed by its right-hand side. `data` may be a
dictionary or anything with named columns. Each measured quantity is found by
its symbol, its name (as a `String` or `Symbol`), or its right-hand side.
"""
function _observation_dictionary(system, measured_quantities, data)
	entries = _data_entries(data)
	found = join((string(first(entry)) for entry in entries), ", ")
	time_variable = ModelingToolkit.get_iv(system)
	time_names = (string(time_variable), "t")

	times = _find_entry(entries) do key
		_is_symbolic(key) ? isequal(Num(key), Num(time_variable)) : string(key) in time_names
	end
	isnothing(times) &&
		throw(ArgumentError("The data has no time points. Add them under the key `t`. The data has: $found."))
	times = collect(Float64, times)
	all(isfinite, times) || throw(ArgumentError("The time points contain NaN or Inf."))
	order = issorted(times) ? nothing : sortperm(times)

	observations = OrderedDict{Union{String, Num}, Vector{Float64}}("t" => isnothing(order) ? times : times[order])
	for equation in measured_quantities
		names = _variable_names(equation.lhs)
		series = _find_entry(entries) do key
			if _is_symbolic(key)
				isequal(Num(key), Num(equation.lhs)) || isequal(Num(key), Num(equation.rhs))
			else
				string(key) in names
			end
		end
		isnothing(series) &&
			throw(ArgumentError("The data has no entry for the measured quantity $(names[2]). The data has: $found."))
		any(ismissing, series) && throw(ArgumentError(
			"The data for $(names[2]) has missing values. Remove those rows, or pass an `ObservationData` if the series were measured at different times."))
		series = collect(Float64, series)
		length(series) == length(times) ||
			throw(ArgumentError("$(names[2]) has $(length(series)) values, but there are $(length(times)) time points."))
		all(isfinite, series) || throw(ArgumentError("The data for $(names[2]) contains NaN or Inf."))
		observations[Num(equation.rhs)] = isnothing(order) ? series : series[order]
	end
	return observations
end

"""
	_true_value_pairs(supplied)

The `true_values` argument as `symbol => value` pairs.
"""
function _true_value_pairs(supplied)
	isnothing(supplied) && return Pair{Num, Float64}[]
	entries = supplied isa AbstractDict ? collect(pairs(supplied)) : supplied isa Pair ? [supplied] : collect(supplied)
	all(entry -> entry isa Pair && _is_symbolic(first(entry)) && _is_plain_number(last(entry)), entries) || throw(ArgumentError(
		"`true_values` takes pairs of a model symbol and a number, such as `[a => 0.4, x => 1.0]`."))
	return Pair{Num, Float64}[Num(first(entry)) => Float64(last(entry)) for entry in entries]
end

"""
	_known_values(system, variables, supplied)

The value of each variable: from the `supplied` pairs if it is there, otherwise
the number the system carries for it (from `@parameters a = 0.4` or the system's
`initial_conditions`), otherwise `NaN`, meaning unknown.
"""
function _known_values(system, variables::Vector{Num}, supplied::Vector{Pair{Num, Float64}})
	carried = ModelingToolkit.initial_conditions(system)
	values = OrderedDict{Num, Float64}()
	for variable in variables
		symbol = Symbolics.unwrap(variable)
		default = if haskey(carried, symbol)
			carried[symbol]
		elseif ModelingToolkit.hasdefault(symbol)
			ModelingToolkit.getdefault(symbol)
		else
			NaN
		end
		default = Symbolics.value(default)   # a number, unless it is an expression
		values[variable] = _is_plain_number(default) ? Float64(default) : NaN
	end
	for (variable, value) in supplied
		haskey(values, variable) && (values[variable] = value)
	end
	return values
end

"""
	ParameterEstimationProblem(system, measured_quantities; data, true_values, name, solver)

Set up an estimation problem from a ModelingToolkit `System` and the quantities
that were measured.

`measured_quantities` is a vector of equations such as `[y1 ~ x1, y2 ~ x1 + x2]`.
Each left-hand side names one measured series; each right-hand side says what it
measures in terms of the model's states and parameters.

# Keywords
- `data`: the measurements. A `NamedTuple`, a dictionary, or anything with named
  columns such as a `DataFrame`, with the time points under `t` and one vector
  per measured quantity under its name, for example
  `(t = times, y1 = first_series, y2 = second_series)`. All series share the
  time points. For series measured at different times, pass an
  [`ObservationData`](@ref). Leave it out to simulate data with
  [`sample_problem_data`](@ref).
- `true_values`: the parameter values and initial conditions that generated the
  data, as pairs such as `[a => 0.4, x1 => 1.0]`, when they are known. They are
  needed only to simulate data and to report errors against the truth. Numbers
  the system already carries, as in `@parameters a = 0.4`, are used for
  anything not listed.
- `name`: a label for the problem. Defaults to the system's name.
- `solver`: the ODE solver used to simulate the model while estimating.
  Defaults to `AutoVern9(Rodas5P())`. [`sample_problem_data`](@ref) replaces
  it with its `ode_solver` option.

# Example
```julia
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D

@parameters a b
@variables x1(t) x2(t) y1(t) y2(t)
@named model = System([D(x1) ~ -a * x2, D(x2) ~ b * x1], t)

problem = ParameterEstimationProblem(model, [y1 ~ x1, y2 ~ x2];
    data = (t = times, y1 = first_series, y2 = second_series))
results = estimate(problem)
```
"""
function ParameterEstimationProblem(
	system::ModelingToolkit.AbstractSystem,
	measured_quantities::AbstractVector{<:ModelingToolkit.Equation};
	data = nothing,
	true_values = nothing,
	name::AbstractString = string(nameof(system)),
	solver = package_wide_default_ode_solver,
)
	system = ModelingToolkit.iscomplete(system) ? system : complete(system)
	states = Num.(ModelingToolkit.unknowns(system))
	parameters = Num.(ModelingToolkit.parameters(system))
	measured = collect(ModelingToolkit.Equation, measured_quantities)
	isempty(measured) && throw(ArgumentError("`measured_quantities` is empty. List at least one, such as `[y ~ x]`."))
	known = _true_value_pairs(true_values)
	for (variable, _) in known
		any(isequal(variable), states) || any(isequal(variable), parameters) || throw(ArgumentError(
			"`true_values` names $(variable), which is not a state or parameter of the model."))
	end

	observations = if isnothing(data) || data isa ObservationData
		data
	else
		_observation_dictionary(system, measured, data)
	end
	return ParameterEstimationProblem(
		String(name),
		OrderedODESystem(system, parameters, states),
		measured,
		observations,
		nothing,
		solver,
		_known_values(system, parameters, known),
		_known_values(system, states, known),
		0,
	)
end

"""
	estimate(problem; options...) -> Vector{ParameterEstimationResult}
	estimate(problem, options::EstimationOptions)

Estimate the parameters and initial conditions of `problem` from its data.

Returns the solutions found, best fit first. There is usually one. There are
several when the data cannot tell them apart, and each then fits equally well.
The vector is empty if no solution was found.

Keyword arguments are the fields of [`EstimationOptions`](@ref), for example
`estimate(problem; seed = 1, progress = true)`.

These are the `returned_results` of
[`analyze_parameter_estimation_problem`](@ref), without the states the package
adds for inputs such as `sin(c * t)`. Call that function instead when
you also want the error statistics against known true values or the
uncertainty report, which is experimental.

# Example
```julia
results = estimate(problem)
best = first(results)
best.parameters      # OrderedDict of parameter => value
best.states          # initial conditions
best[a]              # one value, by symbol or by name (best[:a])
```
"""
function estimate(problem::ParameterEstimationProblem, options::EstimationOptions)
	isnothing(problem.data_sample) && throw(ArgumentError(
		"The problem has no data. Pass `data = ...` to `ParameterEstimationProblem`, or simulate some with `sample_problem_data`."))
	_, analysis, _ = analyze_parameter_estimation_problem(problem, options)
	return ParameterEstimationResult[_without_auxiliary_states!(result, problem) for result in analysis.returned_results]
end

# An input such as `sin(0.5t)` is handled by adding states to the model. A
# result lists only the states the problem was posed with.
function _without_auxiliary_states!(result::ParameterEstimationResult, problem::ParameterEstimationProblem)
	own = problem.model.original_states
	for state in collect(keys(result.states))
		any(isequal(state), own) || delete!(result.states, state)
	end
	return result
end
estimate(problem::ParameterEstimationProblem; options...) = estimate(problem, _options_from_keywords(options))

# ── Reading a result ──────────────────────────────────────────────────────────

_display_number(value::Real) = @sprintf("%.6g", value)
_display_number(value) = string(value)

"""
	result[variable]

The estimated value of a parameter, or of a state's initial condition. `variable`
is the symbol, or its name: `result[a]`, `result[:a]` and `result["a"]` are the
same.
"""
function Base.getindex(result::ParameterEstimationResult, variable)
	wanted(key) = _is_symbolic(variable) ? isequal(Num(variable), key) : string(variable) in _variable_names(key)
	for estimates in (result.parameters, result.states), (key, value) in estimates
		wanted(key) && return value
	end
	throw(KeyError(variable))
end

function Base.show(io::IO, result::ParameterEstimationResult)
	print(io, "ParameterEstimationResult(")
	join(io, ("$(key) = $(_display_number(value))" for (key, value) in result.parameters), ", ")
	isempty(result.parameters) || isempty(result.states) || print(io, "; ")
	join(io, ("$(key) = $(_display_number(value))" for (key, value) in result.states), ", ")
	print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", result::ParameterEstimationResult)
	function list(values)
		for (key, value) in values
			note = key in result.all_unidentifiable ? "  (not identifiable)" : ""
			println(io, "    ", key, " = ", _display_number(value), note)
		end
	end
	println(io, "ParameterEstimationResult")
	isempty(result.parameters) || println(io, "  Parameters")
	list(result.parameters)
	print(io, "  Initial conditions")
	isnothing(result.report_time) || print(io, " (t = ", _display_number(result.report_time), ")")
	println(io)
	list(result.states)
	print(io, "  Fit error: ", isnothing(result.err) ? "not computed" : @sprintf("%.3g", result.err))
end
