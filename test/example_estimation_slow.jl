# The example models that take minutes each. Run weekly and before a release:
#
#   julia --startup-file=no test/current.jl examples_slow

#
# Each model is estimated in a Julia process of its own, with a time limit.

include("support/example_estimation_core.jl")

@testset "Example models are estimated (slow)" begin
	test_example_tier_isolated(:slow)
end
