# Examples

Scripts to run and read. Each is self-contained:

```
julia --startup-file=no examples/first_example.jl
```

| Script | What it shows |
|---|---|
| [first_example.jl](first_example.jl) | Fitting a predator-prey model to data. The manual's first example. |
| [constant_input.jl](constant_input.jl) | A model with a constant input, estimated along with its rate. |
| [rc_circuit.jl](rc_circuit.jl) | One circuit written two ways: once so that every parameter can be found, once so that two of them cannot. |

The package also ships a library of ready-made models,
`ODEParameterEstimation.Examples`. The
[manual](https://orebas.github.io/ODEParameterEstimation.jl/dev/guides/example_models/)
lists them.
