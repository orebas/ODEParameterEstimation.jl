# Internal dense SE Gaussian process

This private module implements the zero-mean, one-dimensional dense SE fit
used by `aaad_gpr_pivot`. It imports only LinearAlgebra, Statistics, Optim and
LineSearches; it does not depend on GaussianProcesses, PDMats or ODEPE.

Adapted from <https://github.com/STOR-i/GaussianProcesses.jl> and the locally
validated revision `3e896e9dbd0c41341c723ab16dcf0c261fc7b95a`. Relevant source
is `GP.jl`, `GPE.jl`, `optimize.jl`, `kernels/se_iso.jl`, and the isotropic
kernel/distance helpers. The MIT notice is retained in `LICENSE`.

## Preserved fitting policy

- Parameter order: log noise SD, log length scale, log signal SD.
- The wrapper standardizes observations; initial parameters are -2,
  log(std(times)/8), and 0. Noise variance therefore starts at exp(-4).
- Dense SE covariance, analytic marginal-likelihood gradients and unconstrained
  LBFGS with BackTracking, using Optim's default stopping options.
- The fitted noise variance is the diagonal nugget. There is no adaptive
  jitter escalation. Invalid Cholesky trials return Inf; unexpected errors and
  interrupts propagate. The final covariance must factor successfully.
- Prediction uses elementary kernel arithmetic and supports TaylorDiff and
  ForwardDiff queries. Only the fitted mean is needed by the public wrapper.

The extraction uses a private typed posterior and Julia's Cholesky directly.
It omits upstream likelihood/sampler/kernel hierarchies, covariance prediction,
sparse/elastic storage, plotting, priors, and external-type method extensions.

## Storage ownership

Each `fit_se` call owns an `SEWorkspace`: covariance, Cholesky storage,
inverse-score matrix, solve vector and gradient. Optimizer trials refill these
buffers, including after a failed factorization. `evaluate_se!` returns a
borrowed state valid only until the next evaluation; `evaluate_se` allocates
independent storage for callers that need to retain the state.

The final covariance, factor and solve vector transfer to the posterior.
Later fits use separate workspaces, and prediction does not mutate the fit.
The arithmetic and gradient accumulation order match the original extraction.
Workspace reset, posterior independence and a warmed allocation budget are
covered by the internal-backend tests.

## Replacement boundary

`aaad_gpr_pivot` is the sole production adapter: `fit_se` accepts normalized
observations and `predict_mean` returns a normalized scalar mean. Replace those
two calls and remove the parent include to excise the module. `GPRapprox` and
`InterpolatorAAADGPR` are internal; the default list of curve fits must keep
behaving the same.
AGP/AGPUQ keep their own fitting policies and uncertainty contracts.

Tests contain upstream fixed-parameter fixtures, independent SE derivative
identities and analytic-gradient checks. Optional upstream comparisons live in
`test/reference`, outside the ordinary dependency graph.
