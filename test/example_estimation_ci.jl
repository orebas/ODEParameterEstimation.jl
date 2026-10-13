# A handful of example models, estimated on every push:
#
#   julia --startup-file=no test/current.jl examples_ci
#
# example_estimation.jl estimates the whole quick tier and
# example_estimation_slow.jl the slow one. Both run weekly.

include("support/example_estimation_core.jl")

@testset "Example models are estimated (every push)" begin
	@test all(EXAMPLE_EXPECTATIONS[name].tier in (:quick, :refused) for name in EXAMPLE_CI_MODELS)
	test_example_models(EXAMPLE_CI_MODELS)
end
