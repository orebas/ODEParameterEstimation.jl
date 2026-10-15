

"""
	add_relative_noise(data::OrderedDict, noise_level::Float64)

Add relative Gaussian noise to data values while preserving time points.

# Arguments
- `data`: OrderedDict containing time series data
- `noise_level`: Standard deviation of the relative noise to add

# Returns
- New OrderedDict with noisy data
"""
function add_relative_noise(data::OrderedDict, noise_level::Float64)
	noisy_data = OrderedDict{Union{String, Num}, Vector{Float64}}()

	# Copy time points unchanged
	noisy_data["t"] = data["t"]

	# Add noise to each measurement
	for (key, values) in data
		if key != "t"  # Skip time points
			noise = 1.0 .+ noise_level .* randn(length(values))
			noisy_data[key] = values .* noise
		end
	end

	return noisy_data
end

"""
	add_additive_noise(data::OrderedDict, noise_level::Float64)

Add homoskedastic Gaussian noise to each observed series while preserving time
points. For each observable, the noise standard deviation is fixed over time:
`noise_level * mean(abs.(clean_values))`.
"""
function add_additive_noise(data::OrderedDict, noise_level::Float64)
	noisy_data = OrderedDict{Union{String, Num}, Vector{Float64}}()

	# Copy time points unchanged
	noisy_data["t"] = data["t"]

	# Add noise to each measurement
	for (key, values) in data
		if key != "t"  # Skip time points
			scale = mean(abs.(values))
			noise = scale .* noise_level .* randn(length(values))
			noisy_data[key] = values .+ noise
		end
	end

	return noisy_data
end

function add_synthetic_noise(data::OrderedDict, noise_level::Float64, noise_model::Symbol)
	noise_level <= 0 && return data
	if noise_model == :relative
		return add_relative_noise(data, noise_level)
	elseif noise_model == :additive
		return add_additive_noise(data, noise_level)
	elseif noise_model == :none
		return data
	end
	throw(ArgumentError("Unknown synthetic noise model: $noise_model"))
end

"""
	SamplingFailureError

Thrown by [`sample_problem_data`](@ref) when the model cannot be simulated over
the whole time interval, for example because the solution blows up.
"""
struct SamplingFailureError <: Exception
	model_name::String
	retcode
	requested_points::Int
	returned_points::OrderedDict{String, Int}
end

function Base.showerror(io::IO, err::SamplingFailureError)
	print(io, "Could not simulate data for $(err.model_name): the ODE solver stopped with $(err.retcode) before the end of the time interval.")
	print(io, " $(err.requested_points) time points were asked for; the series have $(Dict(err.returned_points)).")
	print(io, " Check the true values and the time interval, or try another `ode_solver`.")
end

function validate_sampled_trajectory!(
	model::ModelingToolkit.AbstractSystem,
	measured_data::Vector{ModelingToolkit.Equation},
	solution_true,
	requested_points::Int,
)
	returned_points = OrderedDict{String, Int}()
	for measurement in measured_data
		symbol = Num(measurement.rhs)
		returned_points[string(symbol)] = length(solution_true[symbol])
	end

	if !SciMLBase.successful_retcode(solution_true) || any(values(returned_points) .!= requested_points)
		model_name = string(ModelingToolkit.getname(model))
		throw(SamplingFailureError(model_name, solution_true.retcode, requested_points, returned_points))
	end

	return nothing
end


#This is a utility function which fills in observed data by solving an ODE.

function sample_data(model::ModelingToolkit.AbstractSystem,
	measured_data::Vector{ModelingToolkit.Equation},
	time_interval::Vector{T},
	p_true,
	u0,
	num_points::Int;
	uneven_sampling_times = Vector{T}(),
	solver = package_wide_default_ode_solver, inject_noise = false, mean_noise = 0,
	stddev_noise = 1, abstol = 1e-14, reltol = 1e-14) where {T <: Number}
	if !isempty(uneven_sampling_times)
		if length(uneven_sampling_times) != num_points
			error("Uneven sampling times must be of length num_points")
		end
		sampling_times = uneven_sampling_times
		# The initial conditions hold at the first sample, as the estimator reads them.
		time_span = [first(sampling_times), last(sampling_times)]
	else
		sampling_times = range(time_interval[1], time_interval[2], length = num_points)
		time_span = time_interval
	end
	# Get parameters in the correct order from the model
	ordered_params = [p_true[p] for p in ModelingToolkit.parameters(model)]
	ordered_u0 = [u0[s] for s in ModelingToolkit.unknowns(model)]

	sys = ModelingToolkit.complete(model)

	problem = ODEProblem(sys, merge(if isempty(ordered_u0)
				Dict()
			else
				Dict(ModelingToolkit.unknowns(sys) .=> ordered_u0)
			end, if isempty(ordered_params)
				Dict()
			else
				Dict(ModelingToolkit.parameters(sys) .=> ordered_params)
			end), time_span)
	solution_true = ModelingToolkit.solve(problem, solver,
		saveat = sampling_times;
		abstol, reltol)
	validate_sampled_trajectory!(model, measured_data, solution_true, num_points)

	#if false # Plot state variables
	#	states = ModelingToolkit.unknowns(model)
	#	for state in states
	#		plot(solution_true.t, solution_true[state],
	#			label = string(state),
	#			xlabel = "Time",
	#			ylabel = "Value")
	#		savefig("state_$(state)_plot.png")
	#	end
	#end

	data_sample = OrderedDict{Union{String, Num}, Vector{T}}(Num(v.rhs) => solution_true[Num(v.rhs)]
											  for v in measured_data)
	if inject_noise
		for (key, sample) in data_sample
			data_sample[key] = sample + randn(num_points) .* stddev_noise .+ mean_noise
		end
	end
	data_sample["t"] = sampling_times
	return data_sample
end


"""
	sample_problem_data(problem; options...) -> ParameterEstimationProblem
	sample_problem_data(problem, options::EstimationOptions)

Simulate measurements for `problem` from its true parameter values and initial
conditions, and return a copy of the problem that carries them.

```julia
problem = sample_problem_data(problem; datasize = 51, time_interval = [0.0, 10.0], noise_level = 0.01)
```

The options that matter here are:
- `datasize`: the number of time points.
- `time_interval`: `[start, stop]`; the points are evenly spaced over it. An
  example model from `ODEParameterEstimation.Examples` has an interval of its
  own, which the keyword form uses when you give none.
- `uneven_sampling_times` (experimental): if not empty, use these time points
  instead of `time_interval`. There must be `datasize` of them, in increasing
  order. The initial conditions are the values at the first of them.
- `noise_level` and `noise_model`: add Gaussian noise. `:additive` uses a
  standard deviation of `noise_level` times the mean absolute value of each
  series; `:relative` scales each value by `1 + noise_level * randn()`;
  `:none` adds none.
- `ode_solver`, `abstol`, `reltol`: the ODE solver and tolerances for the
  simulation. The returned problem keeps `ode_solver` as its solver.
- `seed`: with an integer, the noise is the same on every call and the caller's
  default RNG is left untouched. With `nothing` (the default), noise is drawn
  from the caller's default RNG, so `Random.seed!` controls it.
"""
function sample_problem_data(problem::ParameterEstimationProblem; options...)
	# An example model carries the interval it is meant to be watched over.
	interval = problem.recommended_time_interval
	if !haskey(options, :time_interval) && interval isa AbstractVector{<:Real} && length(interval) == 2
		return sample_problem_data(problem, _options_from_keywords((; options..., time_interval = collect(Float64, interval))))
	end
	return sample_problem_data(problem, _options_from_keywords(options))
end

function sample_problem_data(problem::ParameterEstimationProblem, opts::EstimationOptions)
	validate_options(opts) || throw(ArgumentError("Invalid EstimationOptions; fix the reported configuration errors before sampling data."))
	# The whole call is scoped, not only the noise draw: building and solving
	# the model forks tasks, which advances the caller's RNG fork state.
	return _with_quiet_logging(opts) do
		_with_noise_seed(() -> _sample_problem_data(problem, opts), opts.seed)
	end
end

function _sample_problem_data(problem::ParameterEstimationProblem, opts::EstimationOptions)
	unknown = [string(variable) for (variable, value) in merge(problem.p_true, problem.ic) if isnan(value)]
	isempty(unknown) || throw(ArgumentError(
		"Simulating data needs a true value for every parameter and initial condition. Missing: $(join(unknown, ", ")). " *
		"Pass them as `true_values` when constructing the problem."))
	# Create new OrderedODESystem with completed system
	ordered_system = OrderedODESystem(
		complete(problem.model.system),
		problem.model.original_parameters,
		problem.model.original_states,
	)

	# Generate clean data
	clean_data = ODEParameterEstimation.sample_data(
		ordered_system.system,
		problem.measured_quantities,
		opts.time_interval,
		problem.p_true,
		problem.ic,
		opts.datasize,
		solver = opts.ode_solver,
		uneven_sampling_times = opts.uneven_sampling_times,
		abstol = opts.abstol,
		reltol = opts.reltol)

	# Add noise if requested
	data = add_synthetic_noise(clean_data, opts.noise_level, opts.noise_model)

	return ParameterEstimationProblem(
		problem.name,
		ordered_system,
		problem.measured_quantities,
		data,
		problem.recommended_time_interval,
		opts.ode_solver,
		problem.p_true,
		problem.ic,
		problem.unident_count,
	)
end
