```@meta
CurrentModule = ODEParameterEstimation
```

# Other functions

The remaining supported functions, as they are documented in the source. The
functions meant for everyday use are on [Main functions and types](@ref). A
name that is not exported is written with the module name, as in
`ODEParameterEstimation.merge_options`.

```@autodocs
Modules = [ODEParameterEstimation]
Order = [:function]
Private = false
Filter = t -> !(t in (sample_problem_data, observation_times, estimate, analyze_parameter_estimation_problem))
```
