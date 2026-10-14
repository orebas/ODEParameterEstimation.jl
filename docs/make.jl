using ODEParameterEstimation
using Documenter

# The examples draw their figures without a display.
ENV["GKSwstype"] = "100"

DocMeta.setdocmeta!(ODEParameterEstimation, :DocTestSetup, :(using ODEParameterEstimation); recursive=true)

makedocs(;
    modules=[ODEParameterEstimation],
    checkdocs=:public,
    # These replace external dependencies behind private adapters. Their
    # contracts and attribution live beside the source, outside the public API.
    checkdocs_ignored_modules=[ODEParameterEstimation.GPBackend, ODEParameterEstimation.SIANBackend],
    authors="Oren Bassik <orebas@yahoo.com> and contributors",
    sitename="ODEParameterEstimation.jl",
    format=Documenter.HTML(;
        canonical="https://orebas.github.io/ODEParameterEstimation.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "Tutorials" => [
            "Your own data" => "tutorials/own_data.md",
            "Noisy data" => "tutorials/noisy_data.md",
            "More than one answer" => "tutorials/identifiability.md",
            "Uncertainty (experimental)" => "tutorials/uncertainty.md",
        ],
        "Guides" => [
            "Which models work" => "guides/models.md",
            "Options" => "guides/options.md",
            "Results" => "guides/results.md",
            "How it works" => "guides/how_it_works.md",
            "Troubleshooting" => "guides/troubleshooting.md",
            "Example models" => "guides/example_models.md",
        ],
        "Reference" => [
            "Main functions and types" => "reference/api.md",
            "Other types" => "reference/types.md",
            "Other functions" => "reference/functions.md",
        ],
        "Contributing" => "contributing.md",
    ],
)

deploydocs(; repo="github.com/orebas/ODEParameterEstimation.jl.git", devbranch="main")
