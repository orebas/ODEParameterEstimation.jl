# test/fast_unit.jl
#
# Fast, quiet unit gate. Pure/near-pure CONTRACT tests only — no
# `analyze_parameter_estimation_problem`, no SIAN / HomotopyContinuation, no
# consensus or benchmark rendering. The one exception to "no ODE solves" is
# test_example_models.jl, which simulates every example model once. Use this
# for rapid iteration while refactoring.
#
# This is NOT a substitute for the full FAST gate. Per CLAUDE.md, any
# estimation-touching change must still be verified with:
#     julia --startup-file=no -e 'using Pkg; Pkg.test("ODEParameterEstimation")'
#
# Composition: standalone unit contracts, including the restored math and
# derivative utility regressions.
#
#     julia --startup-file=no -e 'using Pkg; Pkg.test("ODEParameterEstimation"; test_args=["unit"])'

using ODEParameterEstimation
using Test

@testset "fast_unit (quiet contract gate)" begin
    include("package_contracts.jl")
	include("test_nested_ad.jl")          # derivatives of an ODE solve nest in a fixed order
	include("test_example_models.jl")     # every example model builds, is documented and simulates
	include("test_core_types.jl")        # Result/PEP/DerivativeData type contracts + constants
	include("test_solution_distance.jl") # clustering compares symbolic keys, not dictionary positions
	include("test_math_utils.jl")        # symbolic arithmetic and time-series summaries
	include("test_derivative_utils.jl")  # analytic derivative identities and input ownership
	include("column_scaling.jl")         # compute_column_scales / scale_hc_system / order_mag
	include("test_noise_rank_matrix.jl") # exact Jacobian and full/deficient rank contracts
	include("test_deferred_derivatives.jl") # independent rational/product-rule derivative contracts
	include("test_model_utils.jl")       # ordered model construction helpers
	include("test_problem_constructor.jl") # problem from an MTK system + named data; result display
	include("test_label_parsers.jl")     # SIAN/Symbolics derivative-name parsing (name round-tripping)
	include("test_run_context.jl")       # scoped RunContext contracts (auto-M hand-off, sinks, isolation)
	include("test_hc_sanitize.jl")       # HC name injectivity + cross-list dedup (silent-collision class)
	include("test_options_contracts.jl") # deleted-field loud errors + wired-field defaults (dead-options cleanup)
	include("test_taylor_propagation.jl") # estimate-conditioned Taylor cores (jet recursion + GP-jet view)
	include("test_exact_index_matching.jl") # per-point index lists + exact obs-name match (silent-mismatch fixes)
	include("test_ift_solve.jl")          # factorized IFT + loud degradation (no silent pinv)
end
