# Test the versions and development checkouts in the active environment.
# Run from the global Julia environment, after developing this checkout:
#   julia --startup-file=no test/current.jl [all|unit|benchmark|examples|examples_slow]
using Pkg

group = isempty(ARGS) ? "all" : only(ARGS)
(group in ("all", "unit", "benchmark", "examples", "examples_slow") ||
 (basename(group) == group && isfile(joinpath(@__DIR__, group)))) ||
    throw(ArgumentError("test group must be all, unit, benchmark, examples, examples_slow, or an active test filename"))

dependencies = collect(values(Pkg.dependencies()))
package_index = findfirst(dependency -> dependency.name == "ODEParameterEstimation", dependencies)
isnothing(package_index) && error("Develop this checkout in the active environment before testing it.")
package = dependencies[package_index]
realpath(package.source) == realpath(dirname(@__DIR__)) ||
    error("The active environment points to a different ODEParameterEstimation checkout: $(package.source)")

println("Julia: ", VERSION)
println("Active project: ", Base.active_project())
for dependency in sort!(dependencies; by=dependency -> dependency.name)
    if dependency.is_direct_dep || dependency.is_tracking_path || dependency.is_tracking_repo
        println((; name=dependency.name, version=dependency.version,
                 path=dependency.is_tracking_path, repo=dependency.is_tracking_repo,
                 source=dependency.source))
    end
end
flush(stdout)

# A test-only dependency conflict must fail visibly rather than select a
# different dependency stack from the one the caller intends to validate.
Pkg.test("ODEParameterEstimation"; allow_reresolve=false, test_args=[group])
