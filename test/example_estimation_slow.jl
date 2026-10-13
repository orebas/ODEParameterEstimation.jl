# The example models that take minutes each. Run weekly and before a release:
#
#   julia --startup-file=no test/current.jl examples_slow

include("support/example_estimation_core.jl")

@testset "Example models are estimated (slow)" begin
	test_example_tier(:slow)
end
