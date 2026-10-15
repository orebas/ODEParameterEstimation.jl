# Engineering notes

Notes for people working on the package itself. They are not user
documentation: to learn how to use the package, read the
[manual](https://orebas.github.io/ODEParameterEstimation.jl/dev/).

- [polynomial_solve_mechanics.md](polynomial_solve_mechanics.md) explains how
  the polynomial systems are built, trimmed and solved, and where the solve is
  fragile.
- [2026-10-06_first_call_latency.md](2026-10-06_first_call_latency.md) explains
  why column scales are passed to HomotopyContinuation as parameters and what
  the precompile workload does. Read it before changing either.
- [2026-10-04_internal_backends.md](2026-10-04_internal_backends.md) records how
  the Gaussian process and SIAN code became internal modules, and how they were
  checked against the packages they came from.
- [provenance.md](provenance.md) lists the code adapted from other packages and
  where their licence notices are kept.

The dated notes record measurements made on that date. Later changes may have
moved the numbers.
