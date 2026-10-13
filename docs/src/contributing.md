# Contributing

Bug reports and pull requests are welcome on
[GitHub](https://github.com/orebas/ODEParameterEstimation.jl).

## Running the tests

Add your checkout with `Pkg.develop`, then run these from Julia's default
environment:

```bash
julia --startup-file=no test/current.jl unit        # quick tests of the parts
julia --startup-file=no test/current.jl             # everything
julia --startup-file=no test/current.jl benchmark   # recovery from noisy data on full-size models
julia --startup-file=no test/current.jl examples    # estimates the example models
```

Run the full suite for any change that touches estimation. `unit` is for quick
iteration and does not replace it.

`examples` estimates each model in `ODEParameterEstimation.Examples` with
default options and checks the answer against `test/example_expectations.jl`.
A few models take many minutes each; those are in a group of their own,
`examples_slow`, which runs weekly. A new example model needs an entry in that
file.

To check that the package installs and passes with dependencies taken from the
registry only, with none of your local development versions:

```bash
julia --startup-file=no test/registered.jl
```

## Precompiling

While the package is precompiled it estimates a small model, so that most of
what `estimate` needs is compiled before anyone calls it. With that step,
precompiling takes about six minutes every time `src` changes. Without it, it
takes half a minute. To skip it while you work, put this in a
`LocalPreferences.toml` beside the `Project.toml` of the environment you
develop in:

```toml
[ODEParameterEstimation]
precompile_workload = false
```

Julia also reads that file from your default environment when another
environment is active, so a setting left there reaches every environment that
does not set its own.

Take it out again before you time anything or run the tests for the last
time. While it is there, they run on a package precompiled without that step.

## Building this manual

```bash
julia --startup-file=no --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=docs docs/make.jl
```

The pages are in `docs/src`. Their examples run during the build, so a page
that builds is a page whose code works. Open `docs/build/index.html` to read
the result.

## Engineering notes

[`docs/internal`](https://github.com/orebas/ODEParameterEstimation.jl/tree/main/docs/internal)
holds the notes written while the package was developed: investigations, design
discussions and release checklists. They are a record, not documentation, and
many describe code that has since changed. Its `review_map.md` says where
things are in the source.
