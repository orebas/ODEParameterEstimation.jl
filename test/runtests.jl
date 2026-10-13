using ODEParameterEstimation
using Random
using Test
using LinearAlgebra

# Record only numerical-runtime metadata, never the environment or secrets.
# BLAS CPU dispatch can affect nearly noiseless GP optimization paths.
println((; julia=string(VERSION), cpu=Sys.CPU_NAME,
    julia_threads=Threads.nthreads(), blas=string(BLAS.get_config()),
    blas_threads=BLAS.get_num_threads(),
    openblas_coretype=get(ENV, "OPENBLAS_CORETYPE", "automatic"),
    coverage=Base.JLOptions().code_coverage))
flush(stdout)

# Keep each file in its own testset so a load error is reported without
# preventing the remaining files from running.
const TEST_FILES = [
    "package_contracts.jl",
    "dependency_compat.jl",
    "test_internal_backends.jl",
    "fast_core.jl",
    "refactor_safety_net.jl",
    "test_label_parsers.jl",
    "test_example_models.jl", # every example model builds, is documented and simulates
    "example_canaries.jl",
    "examples_smoke.jl",
    "identifiability_regressions.jl",
    "test_si_local_basis.jl",
    "test_si_multiplicity_fixing.jl",
    "result_processing_helpers.jl",
    "feature_regressions.jl",
    "test_rescaling.jl",
    "test_core_types.jl",     # re-homed 2026-07-21 (was orphaned + red; now green)
    "test_solution_distance.jl",
    "test_math_utils.jl",
    "test_derivative_utils.jl",
    "test_observation_data.jl",
    "test_model_utils.jl",    # ordered model construction; also in the unit group
    "column_scaling.jl",      # re-homed 2026-07-21 (was orphaned; green)
    "test_noise_rank_matrix.jl", # Jacobian values and rank under bounded AD compilation
    "test_robust_system.jl", # reusable polynomial kernels, Jacobians, and root polishing
    "test_deferred_derivatives.jl", # rational rank support and on-demand polynomial derivatives
    "test_interrupt_propagation.jl",  # Ctrl-C class fix 2026-07-24 (_rethrow_if_interrupt)
    "test_run_context.jl",            # scoped RunContext contracts 2026-07-24
    "test_seed_reproducibility.jl",   # EstimationOptions.seed: repeatable runs, caller RNG untouched 2026-10-05
    "test_quiet_defaults.jl",         # default runs print, log and write nothing; output is opt-in 2026-10-06
    "test_problem_constructor.jl",    # problem from an MTK system + named data; result display 2026-10-06
    "test_estimate.jl",               # estimate(): measured data in, ranked results out 2026-10-06
    "test_precompile_workload.jl",    # the precompile workload is a quiet default run and leaves no state 2026-10-06
    "test_gp_kernel_optimization.jl", # recovered 2026-08-12 (gitignore-trap survivor, Feb 2026)
    "test_cross_observable_covariance.jl",  # recovered 2026-08-12 (joint-GP covariance, Mar 2026)
    "test_polish_maxtime.jl",         # recovered 2026-08-12 (polish_maxtime enforcement, May 2026)
    "test_hc_sanitize.jl",            # HC name injectivity + cross-list dedup 2026-08-12
    "test_options_contracts.jl",      # dead-options cleanup contracts 2026-08-12
    "test_taylor_propagation.jl",     # estimate-conditioned Taylor cores 2026-08-13
    "test_estimate_conditioned_uq.jl", # estimate-conditioned S = default UQ path 2026-08-13
    "test_uq_coverage_smoke.jl",      # N=20 two_exp coverage tripwires (test/support helper) 2026-08-13
    "test_exact_index_matching.jl",   # per-point index lists + exact obs-name match 2026-08-13
    "test_stacked_jet_covariance.jl", # cross-time W-stack Σ_d core (MC-validated) 2026-08-13
    "test_ift_solve.jl",              # factorized IFT, loud degradation 2026-08-13
    "test_multipoint_sensitivity.jl", # multipoint estimate-conditioned S + FD validation 2026-08-14
    "test_multipoint_pipeline.jl", # actual adaptive + fixed-pair MP production routes 2026-08-16
    "test_estimator_aware_uq.jl", # exact rank-one target + typed outcome/lineage contract 2026-08-14
    "test_polish_uq_pipeline.jl", # actual tiny-ODE polish/direct report + perturb/refit influence 2026-08-16
    "test_branch_uq_pipeline.jl", # actual retained parent->jet->sibling covariance composition 2026-08-16
    "test_gp_factorization_consistency.jl", # one SE recipe + scale-relative jitter telemetry 2026-08-15
]

test_files = if isempty(ARGS) || ARGS == ["all"]
    TEST_FILES
elseif ARGS == ["unit"]
    ["fast_unit.jl"]
elseif ARGS == ["benchmark"]
    ["benchmark_smoke.jl"]
elseif ARGS == ["examples"]
    ["example_estimation.jl"]
elseif ARGS == ["examples_slow"]
    ["example_estimation_slow.jl"]
elseif length(ARGS) == 1 && only(ARGS) in TEST_FILES
    copy(ARGS)
else
    throw(ArgumentError("test arguments must be all, unit, benchmark, examples, examples_slow, or a file in TEST_FILES"))
end

@testset "ODEParameterEstimation suite" begin
    for test_file in test_files
        @info "Running test file" test_file
        flush(stderr)
        @testset "$test_file" begin
            Random.seed!(12345)
            # Keep diagnostic sidecars out of the checkout and prevent one
            # file's artifacts from satisfying another file's expectations.
            mktempdir() do test_dir
                cd(test_dir) do
                    # Example scripts define model names. A separate module
                    # prevents those definitions and test helpers leaking into
                    # later files and changing which implementation they test.
                    test_module = Module(gensym(:ODEPETest))
                    Core.eval(test_module, :(include(path) = Base.include($test_module, path)))
                    Base.include(test_module, joinpath(@__DIR__, test_file))
                end
            end
        end
        # Pkg.test can buffer stdout separately from the progress messages on
        # stderr. Surface assertion failures before the next long test file.
        flush(stdout)
        flush(stderr)
    end
end
