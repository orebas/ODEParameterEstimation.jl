# Provenance and attribution

The package is under the [GPL-3.0 license](../../LICENSE). It includes code
adapted from the packages below.

| Material | Source and retained attribution |
| --- | --- |
| Internal Gaussian process module (`src/internal/gp`) | Adapted from STOR-i/GaussianProcesses.jl at revision `3e896e9dbd0c41341c723ab16dcf0c261fc7b95a`. Jamie Fairbrother and Christopher Nemeth's MIT notice is kept in [the module's directory](../../src/internal/gp/LICENSE); its [README](../../src/internal/gp/README.md) lists the scope and source files. |
| Internal SIAN module (`src/internal/sian`) | Adapted from [SIAN-Julia](https://github.com/alexeyovchinnikov/SIAN-Julia), as preserved at [`2f78ca8`](https://github.com/orebas/SIAN-Julia/tree/2f78ca8a0cc93f99eb2f800f08c1dbd20f8b28d9). Ilia Ilmer, Alexey Ovchinnikov and Gleb Pogudin's MIT notice is kept in [the module's directory](../../src/internal/sian/LICENSE). Its [README](../../src/internal/sian/README.md) keeps the upstream note that some helpers were adapted from an unnamed earlier source. |
| Barycentric and rational interpolation helpers (`src/core/derivatives.jl`) | Adapted from [ParameterEstimation.jl at `99f4bd5`](https://github.com/iliailmer/ParameterEstimation.jl/blob/99f4bd59d9c6cedcfeb644672168aaa3f0088984/src/rational_interpolation/bary_derivs.jl), by the ParameterEstimation.jl authors, under GPL-3.0. The code has been changed since it was imported. |
| Identifiability preprocessing and some example models | Source comments mark the parts adapted from [ParameterEstimation.jl](https://github.com/iliailmer/ParameterEstimation.jl), whose contributors are listed in its [Project.toml](https://github.com/iliailmer/ParameterEstimation.jl/blob/9e7adc33a4a11954acde0214f482cffea9551170/Project.toml). The [Crauste model](https://github.com/iliailmer/ParameterEstimation.jl/blob/9e7adc33a4a11954acde0214f482cffea9551170/examples/all-global/crauste.jl) comes from there, under GPL-3.0. |
| Test fixtures for the internal modules and recovery checks | Synthetic data and exact comparisons with the upstream packages. Source revisions, environments and how to regenerate them are in [test/reference](../../test/reference/README.md). |
