```@meta
CurrentModule = ODEParameterEstimation
```

# Other types

The option types and the types that come with a result, as they are
documented in the source. The names meant for everyday use are on
[the previous page](@ref "Main functions and types"). A name that is not
exported is written with the module name, as in
`ODEParameterEstimation.InterpolatorMethod`.

The types to do with uncertainty, `UncertaintyReport` and those around it,
belong to an experimental part of the package. See [Uncertainty](@ref).

```@autodocs
Modules = [ODEParameterEstimation]
Order = [:type]
Private = false
Filter = t -> !(t in (ParameterEstimationProblem, ObservationData, ObservationSeries, EstimationOptions, ParameterEstimationResult, UnsupportedModelClassError, UnsupportedDerivativeOrderError, SamplingFailureError))
```
