# EstimationOptions.seed (2026-10-05): seeded sampling and estimation repeat
# exactly and leave the caller's default RNG where it was; with no seed the
# package draws from the caller's RNG, as it always has.

using ODEParameterEstimation
using ODEParameterEstimation.Examples
using Logging
using Random
using Test

const _SEED_OPTS = (;
	datasize = 31, interpolators = [InterpolatorAAAD], use_parameter_homotopy = false,
	nooutput = true, diagnostics = false, save_system = false, polish_solutions = false,
)

# Order-independent record of every candidate a run produced.
_candidate_pool(solutions) = sort!([
	repr((collect(values(r.states)), collect(values(r.parameters)), r.err)) for r in solutions
])
_observations(pep) = [v for (k, v) in pep.data_sample if k != "t"]

@testset "Seeded synthetic noise" begin
	rng = Random.default_rng()
	model = simple()   # built first: constructing a model forks tasks in the caller
	seeded = EstimationOptions(; _SEED_OPTS..., noise_level = 1e-3, seed = 7)
	Random.seed!(rng, 31)
	before = copy(rng)
	first_draw = sample_problem_data(model, seeded)
	@test copy(rng) == before
	@test _observations(sample_problem_data(model, seeded)) == _observations(first_draw)
	other_seed = EstimationOptions(; _SEED_OPTS..., noise_level = 1e-3, seed = 8)
	@test _observations(sample_problem_data(model, other_seed)) != _observations(first_draw)
	@test copy(rng) == before

	# No seed: the caller's RNG decides the noise and is advanced by it.
	unseeded = EstimationOptions(; _SEED_OPTS..., noise_level = 1e-3)
	@test unseeded.seed === nothing
	ambient = sample_problem_data(model, unseeded)
	@test rand(copy(rng)) != rand(copy(before))
	Random.seed!(rng, 31)
	@test _observations(sample_problem_data(model, unseeded)) == _observations(ambient)
	@test _observations(ambient) != _observations(first_draw)
end

@testset "Seeded estimation" begin
	rng = Random.default_rng()
	seeded = EstimationOptions(; _SEED_OPTS..., seed = 7)
	pep = sample_problem_data(simple(), seeded)

	Random.seed!(rng, 41)
	before = copy(rng)
	raw, analysis, _ = analyze_parameter_estimation_problem(pep, seeded)
	@test copy(rng) == before
	raw_again, analysis_again, _ = analyze_parameter_estimation_problem(pep, seeded)
	@test copy(rng) == before
	@test !isempty(raw[1])
	@test _candidate_pool(raw_again[1]) == _candidate_pool(raw[1])
	best, best_again = first(analysis.returned_results), first(analysis_again.returned_results)
	@test collect(values(best_again.parameters)) == collect(values(best.parameters))
	@test collect(values(best_again.states)) == collect(values(best.states))
	@test analysis.best_max_error < 1e-8

	# The exported lower-level entry point seeds itself when called directly.
	direct, _, _, _ = optimized_multishot_parameter_estimation(pep, seeded)
	direct_again, _, _, _ = optimized_multishot_parameter_estimation(pep, seeded)
	@test _candidate_pool(direct_again) == _candidate_pool(direct)
	@test copy(rng) == before

	# No seed: the run follows the caller's RNG, so seeding it first still
	# reproduces a run, and the run advances it.
	unseeded = EstimationOptions(; _SEED_OPTS...)
	Random.seed!(rng, 41)
	ambient, _, _ = analyze_parameter_estimation_problem(pep, unseeded)
	@test rand(copy(rng)) != rand(copy(before))
	Random.seed!(rng, 41)
	ambient_again, _, _ = analyze_parameter_estimation_problem(pep, unseeded)
	@test _candidate_pool(ambient_again[1]) == _candidate_pool(ambient[1])
end

@testset "Seed option validation" begin
	@test validate_options(EstimationOptions(seed = 7))
	@test_logs (:warn, r"gamma_seed") validate_options(EstimationOptions(seed = 7, gamma_seed = -1))
	@test_throws Exception EstimationOptions(seed = 1.5)
	@test occursin("seed: 7", sprint(show, EstimationOptions(seed = 7)))
	@test !occursin("seed", sprint(show, EstimationOptions()))
end
