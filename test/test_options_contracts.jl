# EstimationOptions surface contracts (2026-08-12 dead-options cleanup):
# deleted fields error LOUDLY at construction (the rtol-vs-reltol typo class),
# and the newly wired fields actually reach their consumers.

using ODEParameterEstimation
using ODEParameterEstimation: InterpolatorMethod, PolishMethod, ResultProvenance,
    get_ad_backend, get_polish_optimizer, print_options, provenance_metadata_dict,
    validate_options
using Test
using OrderedCollections

@testset "EstimationOptions surface contracts" begin
	# Deleted fields: setting one is a construction error, not a silent no-op.
	@test_throws MethodError EstimationOptions(rtol = 1e-8)
	@test_throws MethodError EstimationOptions(max_deriv_level = 12)
	@test_throws MethodError EstimationOptions(use_monodromy = true)
	# The single-interpolator options, which had no effect beside the default
	# `interpolators` list (removed 2026-10-06).
	@test_throws MethodError EstimationOptions(interpolator = InterpolatorAAAD)
	@test_throws MethodError EstimationOptions(custom_interpolator = identity)

	# Wired fields exist with their behavior-neutral defaults.
	o = EstimationOptions()
	# A mistyped option name sends people to `?EstimationOptions`, so it names every field.
	documentation = string(@doc EstimationOptions)
	@test all(name -> occursin("`$name`", documentation), fieldnames(EstimationOptions))

	# Solutions are refined against the data unless that is turned off (default
	# since 2026-10-06).
	@test o.polish_solutions

	# Output is opt-in: a default run prints nothing and writes no files.
	@test o.nooutput
	@test !o.diagnostics
	@test !o.save_system
	@test !o.progress
	@test o.clustering_threshold == 1.0e-5   # == the old CLUSTERING_THRESHOLD constant
	@test o.si_probability == 0.99           # == SIAN's old hardcoded p
	@test o.si_fix_strategy == :local_basis
	@test validate_options(EstimationOptions(si_fix_strategy=:local_basis))
	@test !validate_options(EstimationOptions(si_fix_strategy=:unknown))
	@test o.save_filepath == ""              # empty = legacy "saved_systems/" base
	@test o.hc_threading === true
	@test o.hc_compile_mode === :all
	@test o.uq_noise_source === :learned_gp_homoscedastic
	@test o.gp_derivative_lengthscale_factor == 1.0
	@test o.multipoint_pair_strategy == :spread
	@test o.polish_solver_jacobian == :forwarddiff
	@test o.polish_solver_chunk_size == 1
	@test !validate_options(EstimationOptions(polish_solver_jacobian = :unknown))
	@test !validate_options(EstimationOptions(polish_solver_chunk_size = -1))
	@test !validate_options(EstimationOptions(gp_derivative_lengthscale_factor = 0.0))
	@test !validate_options(EstimationOptions(gp_derivative_lengthscale_factor = Inf))
	@test !validate_options(EstimationOptions(multipoint_pair_strategy = :unknown))

	# Removed AD backend: rejected up front instead of failing inside each polish.
	@test !validate_options(EstimationOptions(opt_ad_backend = :enzyme))
	@test_throws ErrorException get_ad_backend(:enzyme)

	# Every polish method maps to something its solve path can run: a tagged
	# residual solver, or an Optim.jl optimizer for `Optimization.solve`.
	for method in instances(PolishMethod)
		optimizer = get_polish_optimizer(method)
		if ODEParameterEstimation.is_residual_polish_method(method)
			@test optimizer isa Tuple{Symbol, Any}
		else
			@test optimizer() isa ODEParameterEstimation.Optim.AbstractOptimizer
		end
	end
	# The two methods that never satisfied this are gone, not silently remapped.
	@test !isdefined(ODEParameterEstimation, :PolishLevenberg)
	@test !isdefined(ODEParameterEstimation, :PolishGaussNewton)

	undersmoothed = EstimationOptions(
		interpolators = InterpolatorMethod[InterpolatorAGPUQ],
		gp_derivative_lengthscale_factor = 0.75,
	)
	method, interpolator = only(ODEParameterEstimation.resolve_interpolator_list(undersmoothed))
	@test method == InterpolatorAGPUQ
	@test interpolator isa Function
	xs = collect(range(-0.5, 0.5; length = 15))
	fit = interpolator(xs, sin.(xs))
	@test fit.lengthscale_factor == 0.75
	@test fit.lengthscale ≈ 0.75 * fit.fitted_lengthscale

	# clustering_threshold genuinely controls full-space clustering: with a huge
	# threshold everything merges into one cluster; with a tiny one nothing does.
	fake(v) = (states = OrderedDict(:x => v), parameters = OrderedDict(:a => v))
	rows = [fake(1.0), fake(1.5), fake(2.0)]
	@test length(ODEParameterEstimation.cluster_solutions(rows; threshold = 10.0)) == 1
	@test length(ODEParameterEstimation.cluster_solutions(rows; threshold = 1e-12)) == 3

	# gp_s3_refinement shim survives (deprecation warning path, not deleted).
	@test EstimationOptions(gp_s3_refinement = false).gp_s3_refinement === false

	# The ordinary DISPLAY paths must never reference deleted fields
	# (2026-08-13 regression: show() FieldError'd on a stale print group that
	# the full gate never exercised).
	@test !isempty(sprint(show, EstimationOptions()))
	@test !isempty(sprint(io -> print_options(io, EstimationOptions())))
	@test !isempty(sprint(io -> print_options(io, EstimationOptions(); compact = true)))
end

@testset "provenance_metadata_dict (single-source metadata block)" begin
	prov = ResultProvenance(primary_method = :algebraic, source_type = :single_point,
		template_status = :determined, notes = [:terminal_fallback])
	d = provenance_metadata_dict(prov)
	expected = sort(["aggregation_source_indices", "aggregation_strategy", "equations_dropped_by_rank_trimming",
		"estimator_identity",
		"interpolator_source", "multipoint_combo_index", "multipoint_time_indices", "notes",
		"polish_applied", "polish_source_hc_idx", "post_polish_error", "pre_polish_error",
		"practical_identifiability_status", "primary_method", "representative_assignments",
		"rescue_path", "source_candidate_index", "source_shooting_index", "source_type",
		"structural_fix_set", "template_status", "was_terminal_fallback"])
	@test sort(collect(keys(d))) == expected
	@test d["primary_method"] == "algebraic"
	@test d["template_status"] == "determined"
	@test d["was_terminal_fallback"] === true      # via the :terminal_fallback note
	@test d["interpolator_source"] === nothing
	@test d["estimator_identity"]["estimator_kind"] == "unknown"
	@test provenance_metadata_dict(ResultProvenance())["was_terminal_fallback"] === false
end
