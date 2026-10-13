# Every example model builds, is described, and can be simulated.
# Estimating them is the job of example_estimation.jl.

using ODEParameterEstimation
using ODEParameterEstimation.Examples
using Test

include("example_expectations.jl")

@testset "Example models" begin
	Examples = ODEParameterEstimation.Examples
	names_in_catalogue = example_models()

	@testset "one list" begin
		@test allunique(names_in_catalogue)
		@test Set(names_in_catalogue) == Set(keys(EXAMPLE_EXPECTATIONS))
		exported = Set(names(Examples))
		@test exported == union(Set(names_in_catalogue), Set([:Examples, :example_groups, :example_models, :example_problem]))
		@test reduce(vcat, example_models.(example_groups())) == names_in_catalogue
		@test_throws ArgumentError example_models(:no_such_group)
		@test_throws ArgumentError example_problem(:no_such_model)
		# Nothing from the library is exported by the package itself.
		@test isempty(intersect(Set(names(ODEParameterEstimation)), setdiff(exported, [:Examples])))
	end

	@testset "$name" for name in names_in_catalogue
		constructor = getfield(Examples, name)
		problem = constructor()
		expected = EXAMPLE_EXPECTATIONS[name]

		@test problem isa ParameterEstimationProblem
		@test problem.name == String(name)
		@test String(nameof(problem.model.system)) == String(name)
		@test example_problem(name).name == problem.name
		@test Base.Docs.hasdoc(Examples, name)

		@test !isempty(problem.p_true) && !isempty(problem.ic)
		@test all(isfinite, values(problem.p_true))
		@test all(isfinite, values(problem.ic))
		@test isnothing(problem.data_sample)

		interval = problem.recommended_time_interval
		@test interval isa Vector{Float64} && length(interval) == 2 && interval[1] < interval[2]
		@test problem.unident_count == expected.unident_count

		sampled = sample_problem_data(problem; datasize = 21)
		times = sampled.data_sample["t"]
		@test length(times) == 21
		@test times[1] == interval[1] && times[end] == interval[2]
		@test all(all(isfinite, series) for series in values(sampled.data_sample))
		@test length(sampled.data_sample) == length(problem.measured_quantities) + 1
	end
end
