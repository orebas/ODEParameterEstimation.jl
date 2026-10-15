# ParameterEstimationProblem(system, measured_quantities; ...):
# the short constructor takes a ModelingToolkit system and data under the names
# of the measured quantities, and stores both in the form the estimator reads.
# Also how a result prints. Nothing here runs an estimation.

using ODEParameterEstimation
using ODEParameterEstimation: package_wide_default_ode_solver
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrderedCollections
using Test

@testset "Problem from a ModelingToolkit system" begin
	@parameters a b
	@variables x1(t) x2(t) y1(t) y2(t)
	@named oscillator = System([D(x1) ~ -a * x2, D(x2) ~ b * x1], t)
	measured = [y1 ~ x1, y2 ~ x1 + x2]
	times = [0.0, 0.5, 1.0]
	first_series, second_series = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]

	@testset "Model" begin
		problem = ParameterEstimationProblem(oscillator, measured)
		@test problem.name == "oscillator"
		@test ParameterEstimationProblem(oscillator, measured; name = "run 3").name == "run 3"
		@test ModelingToolkit.iscomplete(problem.model.system)
		@test isequal(problem.model.original_parameters, Num.(ModelingToolkit.parameters(problem.model.system)))
		@test isequal(problem.model.original_states, Num.(ModelingToolkit.unknowns(problem.model.system)))
		@test isequal(Set(problem.model.original_parameters), Set([a, b]))
		@test isequal(Set(problem.model.original_states), Set([x1, x2]))
		@test isequal(problem.measured_quantities, measured)
		@test problem.data_sample === nothing
		@test problem.solver === package_wide_default_ode_solver
		@test ParameterEstimationProblem(oscillator, measured; solver = nothing).solver === nothing

		# A system that is already completed or compiled is used as it is.
		for prepared in (complete(oscillator), mtkcompile(oscillator))
			again = ParameterEstimationProblem(prepared, measured)
			@test again.model.system === prepared
			@test isequal(Set(again.model.original_states), Set([x1, x2]))
		end
		@test_throws "`measured_quantities` is empty" ParameterEstimationProblem(oscillator, Equation[])
	end

	@testset "Data" begin
		forms = (
			(t = times, y1 = first_series, y2 = second_series),
			Dict("t" => times, "y1" => first_series, "y2" => second_series),
			Dict(:t => times, :y1 => first_series, :y2 => second_series),
			Dict("t" => times, "y1(t)" => first_series, "y2(t)" => second_series),
			Dict(t => times, y1 => first_series, y2 => second_series),
			OrderedDict{Any, Any}("t" => times, x1 => first_series, x1 + x2 => second_series),
			(t = 0:0.5:1, y1 = 1:3, y2 = (4, 5, 6), note = "other columns are ignored"),
		)
		for data in forms
			problem = ParameterEstimationProblem(oscillator, measured; data)
			@test problem.data_sample isa OrderedDict{Union{String, Num}, Vector{Float64}}
			@test isequal(collect(keys(problem.data_sample)), Any["t", Num(x1), Num(x1 + x2)])
			@test collect(values(problem.data_sample)) == [times, first_series, second_series]
		end

		# The problem owns its data, in time order.
		owned = ParameterEstimationProblem(oscillator, measured; data = first(forms))
		@test owned.data_sample["t"] !== times
		@test owned.data_sample[Num(x1)] !== first_series
		shuffled = ParameterEstimationProblem(oscillator, measured;
			data = (t = [1.0, 0.0, 0.5], y1 = [3.0, 1.0, 2.0], y2 = [6.0, 4.0, 5.0]))
		@test collect(values(shuffled.data_sample)) == [times, first_series, second_series]

		# Series measured at different times come in as ObservationData, untouched.
		separate = ObservationData([
			ObservationSeries("y1", "experiment", x1, [0.0, 0.5, 1.0], first_series),
			ObservationSeries("y2", "experiment", x1 + x2, [0.0, 0.25, 1.0], second_series),
		])
		@test ParameterEstimationProblem(oscillator, measured; data = separate).data_sample === separate

		broken(data) = ParameterEstimationProblem(oscillator, measured; data)
		@test_throws "no entry for the measured quantity y2. The data has: t, y1." broken((t = times, y1 = first_series))
		@test_throws "no time points" broken((y1 = first_series, y2 = second_series))
		@test_throws "y1 has 2 values, but there are 3 time points" broken((t = times, y1 = [1.0, 2.0], y2 = second_series))
		@test_throws "data for y2 contains NaN or Inf" broken((t = times, y1 = first_series, y2 = [4.0, NaN, 6.0]))
		@test_throws "data for y1 has missing values" broken((t = times, y1 = [1.0, missing, 3.0], y2 = second_series))
		@test_throws "time points contain NaN or Inf" broken((t = [0.0, Inf, 1.0], y1 = first_series, y2 = second_series))
	end

	@testset "True values" begin
		unknown = ParameterEstimationProblem(oscillator, measured)
		@test isequal(collect(keys(unknown.p_true)), unknown.model.original_parameters)
		@test isequal(collect(keys(unknown.ic)), unknown.model.original_states)
		@test all(isnan, values(unknown.p_true))
		@test all(isnan, values(unknown.ic))
		@test unknown.unident_count == 0

		known = ParameterEstimationProblem(oscillator, measured; true_values = [a => 0.4, x1 => 1])
		@test known.p_true[a] == 0.4
		@test isnan(known.p_true[b])
		@test known.ic[x1] == 1.0
		@test isnan(known.ic[x2])
		@test ParameterEstimationProblem(oscillator, measured; true_values = Dict(b => 2.0)).p_true[b] == 2.0
		@test ParameterEstimationProblem(oscillator, measured; true_values = b => 2.0).p_true[b] == 2.0

		# Numbers the system already carries are used, and can be overridden.
		@parameters c = 0.4 d
		@variables z1(t) = 1.0 z2(t)
		@named carrying = System([D(z1) ~ -c * z2, D(z2) ~ d * z1], t; initial_conditions = Dict(z2 => 0.5))
		carried = ParameterEstimationProblem(carrying, [y1 ~ z1])
		@test carried.p_true[c] == 0.4
		@test isnan(carried.p_true[d])
		@test carried.ic[z1] == 1.0
		@test carried.ic[z2] == 0.5
		overridden = ParameterEstimationProblem(carrying, [y1 ~ z1]; true_values = [c => 0.7, d => 2.0])
		@test overridden.p_true[c] == 0.7
		@test overridden.p_true[d] == 2.0
		@test overridden.ic[z1] == 1.0

		@parameters elsewhere
		@test_throws "`true_values` names elsewhere" ParameterEstimationProblem(oscillator, measured; true_values = [elsewhere => 1.0])
		@test_throws "pairs of a model symbol and a number" ParameterEstimationProblem(oscillator, measured; true_values = (a = 0.4,))
		@test_throws "pairs of a model symbol and a number" ParameterEstimationProblem(oscillator, measured; true_values = ["a" => 0.4])
		@test_throws "pairs of a model symbol and a number" ParameterEstimationProblem(oscillator, measured; true_values = [a => b])
	end

	@testset "What a problem cannot do yet says so" begin
		without_truth = ParameterEstimationProblem(oscillator, measured)
		@test_throws "Missing: a, b, x1(t), x2(t)" sample_problem_data(without_truth, EstimationOptions())
		@test_throws "The problem has no data" estimate(without_truth)
		with_data = ParameterEstimationProblem(oscillator, measured; data = (t = times, y1 = first_series, y2 = second_series))
		@test_throws "`datasiz` is not an estimation option" estimate(with_data; datasiz = 3)
		@test_throws "`datasiz` and `sed` are not estimation options" estimate(with_data; datasiz = 3, sed = 1)
		@test_throws "`datasiz` is not an estimation option" sample_problem_data(with_data; datasiz = 3)
		@test_throws "as in `interpolators = [InterpolatorAAA]`" estimate(with_data; interpolator = InterpolatorAAA)

		# A model that is not rational is refused in the terms it was written in.
		@parameters g
		@variables θ(t) ω(t)
		@named pendulum = System([D(θ) ~ ω, D(ω) ~ -g * sin(θ)], t)
		swinging = ParameterEstimationProblem(pendulum, [y1 ~ θ]; data = (t = times, y1 = first_series))
		@test_throws "not rational: it contains sin(θ(t))" estimate(swinging)
	end
end

@testset "The nine-field constructor takes a plain Dict of data" begin
	# Older scripts pass a Dict keyed by the measured right-hand sides. A
	# Union-typed field once refused it.
	@parameters k
	@variables x(t) y(t)
	model, mq = create_ordered_ode_system("decay", [x], [k], [D(x) ~ -k * x], [y ~ x])
	times = collect(range(0.0, 1.0, length = 5))
	data = Dict(mq[1].rhs => 2.0 .* exp.(-0.7 .* times), "t" => times)
	pep = ParameterEstimationProblem("decay", model, mq, data, [0.0, 1.0], nothing,
		OrderedDict(k => 0.7), OrderedDict(x => 2.0), 0)
	@test pep.data_sample isa OrderedDict{Union{String, Num}, Vector{Float64}}
	@test first(keys(pep.data_sample)) == "t"
	@test pep.data_sample["t"] == times
	@test pep.data_sample[mq[1].rhs] ≈ 2.0 .* exp.(-0.7 .* times)
end

@testset "Simulated data can be sampled at uneven times" begin
	@parameters k
	@variables x(t) y(t)
	@named decay = System([D(x) ~ -k * x], t)
	problem = ParameterEstimationProblem(decay, [y ~ x]; true_values = [k => 0.7, x => 2.0])
	# Outside the default time interval, and not starting at zero: the initial
	# conditions are the values at the first sample.
	times = [0.5, 0.6, 1.0, 2.5]
	sampled = sample_problem_data(problem; datasize = 4, uneven_sampling_times = times)
	@test sampled.data_sample["t"] == times
	@test sampled.data_sample[Num(x)] ≈ 2.0 .* exp.(-0.7 .* (times .- 0.5)) rtol = 1e-8
end

@testset "State names need not be ASCII" begin
	# These helpers cut "(t)" off by byte offset, which is not a character
	# boundary after θ and raised StringIndexError.
	@parameters g
	@variables θ(t) ω(t)
	states = ModelingToolkit.unwrap.([θ, ω])
	@test ODEParameterEstimation._state_base_name_set(states) == Set(["θ", "ω"])
	@test isequal(ODEParameterEstimation._model_symbol_from_name("ω", states, [ModelingToolkit.unwrap(g)]), states[2])
end

@testset "Reading a result" begin
	@parameters a b
	@variables x1(t) x2(t)
	result = ODEParameterEstimation.ParameterEstimationResult(
		OrderedDict(a => 0.4, b => 1 / 3), OrderedDict(x1 => 1.0, x2 => 12345.678),
		0.5, 2.5e-9, :Success, 21, 0.0, nothing, Set{Num}(), nothing,
	)
	@test result[a] == 0.4
	@test result[x2] == 12345.678
	# By name, for when the symbols are not at hand.
	@test result[:b] == 1 / 3
	@test result["x1"] == 1.0
	@test result["x2(t)"] == 12345.678
	@parameters elsewhere
	@test_throws KeyError result[elsewhere]
	@test_throws KeyError result[:x]

	@test sprint(show, result) == "ParameterEstimationResult(a = 0.4, b = 0.333333; x1(t) = 1, x2(t) = 12345.7)"
	@test sprint(show, MIME"text/plain"(), result) == """
		ParameterEstimationResult
		  Parameters
		    a = 0.4
		    b = 0.333333
		  Initial conditions (t = 0)
		    x1(t) = 1
		    x2(t) = 12345.7
		  Fit error: 2.5e-09"""
	# A vector of results prints one per line.
	@test occursin("\n ParameterEstimationResult(a = 0.4,", sprint(show, MIME"text/plain"(), [result, result]))

	partial = ODEParameterEstimation.ParameterEstimationResult(
		OrderedDict(a => 0.4, b => 2.0), OrderedDict(x1 => 1.0, x2 => 0.0),
		0.0, nothing, nothing, 21, nothing, OrderedDict(b => 2.0), Set{Num}([b, x2]), nothing,
	)
	@test sprint(show, MIME"text/plain"(), partial) == """
		ParameterEstimationResult
		  Parameters
		    a = 0.4
		    b = 2  (not identifiable)
		  Initial conditions
		    x1(t) = 1
		    x2(t) = 0  (not identifiable)
		  Fit error: not computed"""
end
