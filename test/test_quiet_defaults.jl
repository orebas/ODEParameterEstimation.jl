# A run with default options prints nothing, logs nothing and writes no files
# (2026-10-06). `progress = true` adds phase lines to a quiet run;
# `nooutput = false` and `diagnostics = true` bring the development output back.

using ODEParameterEstimation
using ODEParameterEstimation.Examples
using Logging
using Test

include(joinpath(@__DIR__, "support", "observe_run.jl"))

const _QUICK_OPTS = (;
	datasize = 31, interpolators = [InterpolatorAAAD], use_parameter_homotopy = false, polish_solutions = false,
)

@testset "Default options are quiet" begin
	defaults = EstimationOptions()
	@test defaults.nooutput
	@test !defaults.diagnostics
	@test !defaults.save_system
	@test !defaults.progress

	sampling = _observe_run(() -> sample_problem_data(harmonic_oscillator(), defaults))
	@test isempty(sampling.printed)
	@test isempty(sampling.logs)
	@test isempty(sampling.files)

	run = _observe_run(() -> analyze_parameter_estimation_problem(sampling.value, defaults))
	@test isempty(run.printed)
	@test isempty(run.logs)
	@test isempty(run.files)
	_, analysis, _ = run.value
	@test analysis.best_max_error < 1e-6
end

@testset "Counting solutions draws no progress meter" begin
	# HomotopyContinuation.mixed_volume has no switch for its progress meter,
	# which showed on stderr for models slow enough to trigger it (hiv). The
	# package counts through MixedSubdivisions instead; the count must agree.
	HC = ODEParameterEstimation.HomotopyContinuation
	x, y, z, p, q = HC.Variable.((:x, :y, :z, :p, :q))
	systems = (
		HC.System([x^2 + y^2 - 1, x * y - 0.5]; variables = [x, y]),
		HC.System([x^3 * y + 2x - 1, y^2 + x * y - 3]; variables = [x, y]),
		HC.System([p * x^2 + y - 1, x * y + q]; variables = [x, y], parameters = [p, q]),
		HC.System([x * y * z - 1, x^2 + y^2 + z^2 - 4, x + y + z - 0.5]; variables = [x, y, z]),
		HC.System([x^2 - y * z, y^2 - x * z, x * y - z^2 + x^2]; variables = [x, y, z]),   # homogeneous
	)
	for system in systems
		@test ODEParameterEstimation._mixed_volume_without_progress(system) == HC.mixed_volume(system)
	end
	@test ODEParameterEstimation._mixed_volume_without_progress(first(systems)) == 4
end

@testset "Output on request" begin
	problem = sample_problem_data(harmonic_oscillator(), EstimationOptions(; _QUICK_OPTS...))
	estimate_with(; options...) =
		_observe_run(() -> analyze_parameter_estimation_problem(problem, EstimationOptions(; _QUICK_OPTS..., options...)))

	# Phase lines only, one per line, and still no log records or files.
	progress = estimate_with(progress = true)
	lines = split(strip(progress.printed), '\n')
	@test length(lines) >= 4
	@test all(line -> occursin(r"^\[\d\d:\d\d:\d\d\] [▶✓•] \S", line), lines)
	@test isempty(progress.logs)
	@test isempty(progress.files)

	verbose = estimate_with(nooutput = false)
	@test occursin("Starting model: harmonic_oscillator", verbose.printed)
	@test isempty(verbose.files)

	diagnostics = estimate_with(diagnostics = true)
	@test !isempty(diagnostics.logs)
	@test "artifacts" in diagnostics.files

	# The three runs differ only in what they report.
	best(run) = first(run.value[2].returned_results)
	@test collect(values(best(progress).parameters)) ≈ collect(values(best(verbose).parameters))
end

@testset "Configuration problems are still reported" begin
	problem = sample_problem_data(harmonic_oscillator(), EstimationOptions(; _QUICK_OPTS...))
	conflicting = EstimationOptions(; _QUICK_OPTS..., seed = 1, gamma_seed = -1)
	@test conflicting.nooutput
	@test_logs (:warn, r"gamma_seed") match_mode = :any analyze_parameter_estimation_problem(problem, conflicting)
	@test_logs (:error, r"noise_model") match_mode = :any begin
		@test_throws ArgumentError analyze_parameter_estimation_problem(problem, EstimationOptions(; _QUICK_OPTS..., noise_model = :unknown))
	end
end
