# Resolve this checkout in a fresh environment, without inheriting development
# overrides from the caller's environment. Run from the repo root with:
#   julia --startup-file=no test/registered.jl
# Arguments: all|unit|benchmark|examples|examples_ci|examples_slow, followed optionally by
# default|modern.
using Pkg
using TOML

length(ARGS) <= 2 || error("Usage: test/registered.jl [all|unit|benchmark|examples|examples_ci|examples_slow] [default|modern]")
group = isempty(ARGS) ? "all" : first(ARGS)
group in ("all", "unit", "benchmark", "examples", "examples_ci", "examples_slow") || error("Unknown test group: $group")
profile = length(ARGS) == 2 ? ARGS[2] : "default"
profile in ("default", "modern") || error("Unknown dependency profile: $profile")

mktempdir() do environment_dir
    if profile == "modern"
        project = Dict(
            "deps" => Dict(
                "Optim" => "429524aa-4258-5aef-a3af-852621145aeb",
                "OrdinaryDiffEq" => "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed",
                "SciMLBase" => "0bca4576-84f4-4d90-8ffe-ffa030f20462",
                "OrderedCollections" => "bac558e1-5e72-5ebc-8fee-abe8a469f55d",
                "Nemo" => "2edaba10-b0f1-5616-af89-8c11ac63239a"),
            "compat" => Dict("Optim"=>"2", "OrdinaryDiffEq"=>"7", "SciMLBase"=>"3",
                "OrderedCollections"=>"2", "Nemo"=>"0.56"))
        open(joinpath(environment_dir,"Project.toml"),"w") do io
            TOML.print(io,project)
        end
    end
    Pkg.activate(environment_dir)
    Pkg.develop(path=dirname(@__DIR__))

    overrides = sort!([
        dependency.name for dependency in values(Pkg.dependencies())
        if dependency.name != "ODEParameterEstimation" &&
           (dependency.is_tracking_path || dependency.is_tracking_repo)
    ])
    isempty(overrides) || error("Expected registered dependencies; found overrides: $(join(overrides, ", "))")
    removed = filter(d -> d.name in ("GaussianProcesses", "SIAN"), collect(values(Pkg.dependencies())))
    isempty(removed) || error("Internalized backends still appear in the dependency graph: $(getproperty.(removed,:name))")
    Pkg.status(;mode=Pkg.PKGMODE_MANIFEST)

    # Preserve the exact resolver output before the temporary environment is
    # removed, including when tests fail. CI publishes this as a run artifact.
    evidence_dir = get(ENV, "ODEPE_VALIDATION_DIR", "")
    if !isempty(evidence_dir)
        mkpath(evidence_dir)
        for filename in ("Project.toml", "Manifest.toml")
            cp(joinpath(environment_dir, filename), joinpath(evidence_dir, filename); force=true)
        end
        open(joinpath(evidence_dir, "validation.toml"), "w") do io
            TOML.print(io, Dict("julia"=>string(VERSION), "profile"=>profile,
                "group"=>group, "revision"=>get(ENV, "GITHUB_SHA", "local-worktree"),
                "coverage"=>get(ENV, "ODEPE_TEST_COVERAGE", "false") == "true"))
        end
    end

    Pkg.test("ODEParameterEstimation"; allow_reresolve=false, test_args=[group],
             coverage=get(ENV, "ODEPE_TEST_COVERAGE", "false") == "true")
end
