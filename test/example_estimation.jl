# The example models that are estimated in about a minute each, and the one
# that must be refused. Run on every push:
#
#   julia --startup-file=no test/current.jl examples
#
# The slow ones are in example_estimation_slow.jl.

include("support/example_estimation_core.jl")

@testset "Example models are estimated (quick)" begin
	test_example_tier(:quick, :refused)
end
