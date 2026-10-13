# ODEParameterEstimation.jl registry preparation

This checklist tracks the first General registration candidate, **1.0.0**,
prepared on 2026-10-05. The installable core is on `main`; ongoing studies and
their evidence remain on `research`. External SIAN and GaussianProcesses are
no longer runtime or test dependencies. The repository has been renamed to
[`orebas/ODEParameterEstimation.jl`](https://github.com/orebas/ODEParameterEstimation.jl).

This work stops at a validated, pushed candidate. **No Registrator request,
release tag, or GitHub release is submitted.** The maintainer selected 1.0.0
after the initial 1.1.0 candidate exposed AutoMerge's initial-version rule.
The selected 1.0.0 satisfies that rule; no version-related manual exception is
needed. Registry review and registration remain separate actions.
The dated [production-readiness record](2026-09-10_production_readiness.md)
retains earlier dependency and test history.

## Priority order and first milestone

1. Preserve the current tracked research tree on a `research` branch in the
   namesake repository, record its starting commit, and publish selected
   provenance-checked local results there. Keep `main` as the installable core.
2. Remove research and benchmark APIs, the PEtab pilot, and the unfinished
   RS/RUR extension from `main`. Keep the standard estimator, maintained
   examples, diagnostics, and opt-in UQ. Retain tests needed for that surface.
3. Compare the GaussianProcesses.jl default interpolator with AGP/AGPUQ on
   frozen, paired inputs, including any reproducible case where GP.jl wins.
4. Map SIAN's equation-construction calls and decide whether an internal
   prototype is justified. Keep StructuralIdentifiability.jl's classification
   role separate.
5. Apply the resulting dependency decisions and complete release checks.
   The follow-up now implements private SIAN and GP backends; see the
   [internalization record](2026-10-04_internal_backends.md). No registry
   submission is part of this work.

The research line receives selected core fixes by cherry-pick. Merging the
`main` cleanup commit into `research` would remove the files that branch exists
to preserve. Ignored generated output remains local unless deliberately
promoted with provenance and a hash; the branch alone cannot preserve it.

General requires an OSI-approved license and registered dependencies. Its
AutoMerge checks also cover package and repository naming, a release version
without prerelease data, bounded compatibility entries, installation, and
loading. See the [General registration guide](https://github.com/JuliaRegistries/General/blob/master/README.md#registering-a-package-in-general)
and [RegistryCI AutoMerge guidelines](https://juliaregistries.github.io/RegistryCI.jl/stable/guidelines/).

## Registration checks

| Check | Candidate evidence | Release gate |
| --- | --- | --- |
| License | Root GPL-3.0 and retained MIT notices; see [attribution](#provenance-and-attribution). | Preserve these notices in the registered tree. |
| Package identity | `ODEParameterEstimation`, UUID `482fc905-5656-4c69-b8fe-7a66cd0f77b3`. | AutoMerge 1.1.0 name and similarity checks pass; dependency names/UUIDs match General. |
| Release version | `1.0.0`, selected by the maintainer; no prerelease/build suffix. | Standard initial-version check passes. |
| Repository URL | Canonical target `https://github.com/orebas/ODEParameterEstimation.jl.git`. | Rename verified: repository ID and branch heads unchanged; former URL returns HTTP 301. |
| Compatibility | Every non-stdlib dependency has bounded `[compat]`; no unregistered weak dependencies remain. | Resolve both registered profiles with no development overrides. |
| Installation and loading | Required CI covers Julia 1.12 and 1.13; modern dependencies also receive the recovery benchmark. | Require success at the exact candidate SHA and test a fresh URL install. |

The October 5 metadata checks used AutoMerge 1.1.0 on Julia 1.13.1 and General
tree `d0d9933332d876a1617ccc36bf88401c4d459c98`. Exact-name, similarity,
identifier, ASCII, minimum-length, no-prerelease/build, GPL license detection,
and registered non-stdlib dependency/compat checks passed. The initial-version
check also passes after selecting 1.0.0.

The rename retained GitHub repository ID `796574275`, `main` at
`ef35226c644c101aa3e27b17f120e9bfe1f98e11` before the release edits, and
`research` at `aab64fc5dce9e4304a7bee3e7f52b564414cfcce`. The old web URL
returns HTTP 301 to the new name, the old API route resolves to the same ID,
and the local remote uses the new URL. The local checkout path is unchanged.

The local General snapshot did not contain `RS` or
`RationalUnivariateRepresentation` on 2026-10-02. General
[does not accept unregistered dependencies](https://github.com/JuliaRegistries/General/blob/master/README.md#can-my-package-in-this-registry-depend-on-unregistered-packages).
The [RS/RUR extension on `research`](https://github.com/orebas/ODEParameterEstimation.jl/tree/research/ext/ODEParameterEstimationRSExt)
also imports a missing core helper and has no active integration gate, according
to the [readiness record](2026-09-10_production_readiness.md#optional-integrations).

## Supported release surface

The documented core is the `FlowStandard` SI-template estimator for supported
polynomial and rational-style ODE models, with structural representative fixes,
algebraic candidate solving, trajectory-fit ranking, and explicit early errors
for unsupported raw model classes. Use the [quickstart](2026-03-17_user_quickstart.md),
[result contract](2026-03-17_results_and_api.md), and
[model limits](2026-03-17_supported_models_and_limitations.md) as the initial
user-facing contract. Confirm that examples and default options still match it
at the release commit.

The first-milestone release-surface decisions are:

- **PEtab:** The [restricted pilot](https://github.com/orebas/ODEParameterEstimation.jl/tree/research/ext/petab)
  stays on `research` and is absent from the first core release surface.
- **RS/RUR:** The unfinished extension, `SolverRS`, and its unregistered weak
  dependencies stay on `research`.
- **Uncertainty quantification:** Production code exists, but the default user
  path leaves it off. The audited single-point calibration result does not
  establish nonlinear multipoint or polished-estimator coverage. State this
  limit wherever UQ is exposed; use the [UQ contract](2026-08-14_estimator_aware_uq.md)
  and [audited campaign](2026-08-16_audited_repeated_uq_campaign.md) before
  making release claims.
- **Research API:** Consensus, model-assisted correction, and SHADE+LM APIs
  stay on `research`. Core timing support remains on `main` because the
  production estimator uses it.

## Publishable main and continuing research

The namesake GitHub repository's default `main` branch should be the package
people install and the branch used for registration. Its README, CI, and source
tree must describe the same supported release. The active investigations and
their evidence must remain available for continued work; reducing the published
tree must not discard them.

The [research branch](https://github.com/orebas/ODEParameterEstimation.jl/tree/research)
was created and pushed from commit `01b816f` before the `main` cleanup. Its
selected previously ignored raw results and provenance manifest are in commit
`c4358bd`. The frozen GP comparison is in `dd27080`. All 4,498 files removed
from `main` were checked against `research` and retain identical Git blobs.
The resulting core tree is about 3.5 MB before Git history. Existing Git
history remains part of a full Git clone even when the release tree becomes
smaller. Local research outputs remain in place and are ignored on `main`.
Links from retained documentation to moved evidence now point at the preserved
research commit; the PEtab guide explicitly applies to `research`.

- [x] Inventory `src/research/`, `repro/`, `artifacts/`, `experiments/`,
  `deprecated/`, and example scripts. For each item, identify whether package
  loading, active tests, documentation, or an ongoing study requires it.
- [x] Inventory ignored and untracked outputs separately. A branch or fork
  preserves tracked files, but cannot preserve local-only generated PEtab models
  and other ignored evidence without an explicit retention decision.
- [x] Choose the branch, fork, or separate-repository arrangement. Preserve an
  immutable reference to the current research tree before slimming `main`, keep
  reproducibility instructions and source attribution intact, and document how
  future core changes reach ongoing studies. Cherry-pick selected core commits
  into `research`; do not merge the `main` deletion commit into it.
- [x] Check the resulting `main` with the normal package gates and a fresh
  registered environment. Retain the research source, provenance, and
  reproduction instructions against identified commits. The research branch's
  full experimental suite was not rerun as part of the core split validation.
  Do not infer a smaller Git clone from a smaller release tree: retained
  history still contributes to clone size.

## Study whether to bring dependency functionality into ODEPE

The initial dependency study is preserved in
[the decision record](2026-10-04_dependency_decisions.md). The user subsequently
chose to own the required functionality. Private `SIANBackend` and `GPBackend`
modules now replace the external packages; GP's fitting policy remains distinct
from AGP. See [implementation and validation](2026-10-04_internal_backends.md).
The release gates apply to these replacements, including fresh registered-only
installation and recovery checks.

- [x] **SIAN scope and first decision:** Map the calls in
  [SI equation construction](../../src/core/si_equation_builder.jl), including
  jet equations, generic sampling, variable ordering, rank selection, and
  polynomial template construction. Keep this distinct from
  `StructuralIdentifiability.jl`'s identifiability classification. Compare a
  narrow internal implementation, upstream repair, and a maintained fork. The
  initial preference for upstream repair was superseded by internalization.
  The internal backend must agree on
  representative models' equations, variable roles, structural fixes,
  multiplicity, and candidate recovery, with acceptable construction time.
- [x] **GaussianProcesses.jl scope and first decision:** Map the
  [GP-pivoted interpolator](../../src/core/derivatives.jl) and the related PDMats
  compatibility bridge in [the main module](../../src/ODEParameterEstimation.jl).
  Compare retaining the registered dependency, upstream repair, a narrow
  internal GP fit, and improvements to the existing AGP path. The old
  [interpolator comparison script on `research`](https://github.com/orebas/ODEParameterEstimation.jl/blob/research/src/examples/compare_interpolators.jl)
  is exploratory. The new paired checkpoint found a small GP.jl win on a
  clean smooth curve; internalization preserves that fitting policy.
- [x] Validate the internal GP against its original implementation at fixed
  parameters, after optimization, and on frozen ODE recovery data. Preserve
  higher derivatives and GP-only/default-pool recovery. This replacement does
  not consolidate AGP policies or change the UQ fitting path.
- [x] Write a [first decision record](2026-10-04_dependency_decisions.md) with
  measured tradeoffs and the APIs to retain. If code is later adapted from
  upstream, review its license and preserve required attribution. Remove a
  dependency or compatibility bridge only after the replacement passes the
  registered-dependency full suite, recovery benchmark, and relevant focused
  contracts.

## Validation required for a release commit

These gates apply anew to each candidate. The final handoff identifies the
candidate SHA, its required GitHub checks, and the clean-install result; a
passing run at an earlier revision does not satisfy them.

- Resolve the checkout with only registered dependency versions and no
  development overrides. Run `julia --startup-file=no test/registered.jl` from
  the repository root; record the Julia version, resolved dependency versions,
  commit, and assertion results. This script develops the checkout in a fresh
  temporary environment and then invokes `Pkg.test`.
- Run the full suite on each currently released Julia version claimed by
  the release documentation and CI. Currently `julia = "1.12"` admits Julia
  1.12 and later 1.x releases, while CI has required jobs for 1.12 and 1.13. A
  [September 11 CI follow-up](2026-09-10_production_readiness.md#september-11-follow-up)
  records a Julia 1.12.7 UQ test failure after earlier green runs. The October
  internalization worktree passes the full Julia 1.12.7 suite after the test
  portability and compilation-budget corrections documented below. Recheck
  the eventual release commit. Julia nightly is advisory in the current
  [CI matrix](../../.github/workflows/CI.yml); its recorded GPUCompiler
  precompilation crash is a separate dependency issue.
- Run the seeded recovery benchmark with registered dependencies on the
  selected release stack. The full and benchmark commands are documented in
  [CLAUDE.md](../../CLAUDE.md). A passing unit group alone is insufficient for an
  estimation change; recovery accuracy does not certify UQ coverage.
- If SIAN or GaussianProcesses.jl functionality moves inside ODEPE, run
  paired behavior and performance comparisons before and after the change on
  frozen inputs. Keep both winning and losing cases in the decision record;
  preserve the existing GP route until a replacement earns its removal.
- Verify a clean install, `import ODEParameterEstimation`, the README's
  minimal example, and the published documentation links from the release
  commit. Test the repository URL that Registrator will use.

An attempted fresh registered-dependency unit run on 2026-10-02 did not reach
package tests: the workspace's Julia depot was read-only, and a retry with a
temporary writable depot could not resolve network hosts needed to download a
registered dependency. This is an environment limitation, not evidence that the
package passed or failed the gate. Earlier results in the readiness record are
historical and must not be presented as validation of a later release commit.

### October 4 implementation validation

Package loading passed on Julia 1.13.1 using a writable temporary depot.
The local unit gate passed **419/419** assertions. This environment uses the
GP and SIAN development checkouts identified in the dependency decision record.
The first full run reached 2,054 passing assertions but failed four checks in
the DC-motor example canary: its singular `interpolator` option was overridden
by the default pool, whose selected synthesized aggregate did not carry the
single-point template provenance assumed by the test. A paired probe confirmed
that an explicit AAA pool returns the expected provenance. The shared fast
canary options now select AAA explicitly. Production estimation was unchanged
by this test-configuration correction.

The local full rerun passed **2,058/2,058** assertions in 13m30.3s. The
recovery benchmark passed **10/10** in 10m12.2s, including the noisy LV/simple,
clean LV, and rescaled HIV cases. These durations exclude dependency
precompilation. The canary fix is commit `b2a7e2d` and has also been cherry-picked
onto `research` as `e39230e`.

The fresh registered-dependency full gate also passed **2,058/2,058** in
11m22.1s after precompilation. Its resolver selected GP 0.12.6, SIAN 1.8.1,
SI 0.5.25, Nemo 0.54.2, Optim 1.13.3, and OrderedCollections 1.8.2; this is
distinct from the modern local development stack.

The validated core code is commit `02ff196`. Exact dependency versions,
commands, compressed logs, and hashes are in the
[validation record on `research`](https://github.com/orebas/ODEParameterEstimation.jl/tree/aab64fc/repro/registry_dependency_study_2026_10_04).
These results establish the split's Julia 1.13.1 baseline. Supported-Julia CI,
the registered-stack recovery benchmark, the final release version, repository
naming, and attribution review remain release work.

### October 4 internalization follow-up

The [internalization record](2026-10-04_internal_backends.md) supersedes the
GP/SIAN dependency graph and patched CI setup above. Both used backends now
live in private modules with MIT notices and documented replacement boundaries.
The public GP candidate remains in the pool. Normal installation and testing
use neither external package, and CI now resolves both dependency profiles
entirely from registered packages.

The Julia 1.13.1 registered modern full suite and Julia 1.12.7 registered full
suite each passed **2,176/2,176**. The Julia 1.13.1 registered recovery benchmark
passed **10/10**. The local unit/full/benchmark
gates passed **419/419**, **2,176/2,176** and **10/10**, respectively. The first
registered full run exposed a compilation-sensitive wall-clock limit in the
direct-UQ convergence canary; a forced-timeout reproduction and the test-only
budget correction are documented in the implementation record.
The initial Julia 1.12 run also exposed five new fixture portability failures;
explicit symbolic input ordering and a measured high-derivative tolerance
resolved them. Independent upstream/internal comparisons match exactly on
both Julia versions. Production numerical code was unchanged by these fixes.

All eight frozen GP-only/default-pool recovery comparisons preserve the measured
errors and candidate counts exactly, including the unsuccessful biohydrogenation
parameter-recovery cases. SIAN fixture costs are comparable. The initial GP
extraction allocated more and took about 25% more time on the measured
201-point workload. The October 5 workspace follow-up reduced its allocations
from 50.6 MB to 0.99 MB per fit and median time from 34.9 ms to 32.0 ms in a
new paired run. The separate upstream comparison now measures about 9% more
time with 86% fewer allocated bytes. Both performance panels and their limits
are retained in the [implementation record](2026-10-04_internal_backends.md).
The buffered version passed the local Julia 1.13.1 full suite **2,188/2,188**
and recovery benchmark **10/10**; paired GP fits still match upstream exactly
on Julia 1.12 and 1.13.
These worktree results still require validation of the eventual release commit.

### October 5 dependency cleanup

An audit of the 1.0.0 candidate found eight declared packages that the loaded
source did not use. `Plots`, `Zygote`, `BlackBoxOptim` and
`MultivariatePolynomials` were never imported. `DynamicPolynomials` and
`PolynomialRoots` were imported without a call site. `Enzyme` and
`SciMLSensitivity` served only `opt_ad_backend = :enzyme`, which did not
work: Enzyme 0.13.210 raised `IllegalTypeAnalysisException` while compiling
the trajectory loss on the `simple` model, and the batch polisher kept the
unpolished candidate. The eight dependencies and that option value are
removed; see the [changelog](../../CHANGELOG.md).

Every global referenced by the package's 2,280 methods resolves to the same
binding before and after the change, and forward-mode polish results on
`simple` are identical to the last digit. A fresh registered resolve on Julia
1.13.1 selects 308 packages instead of 434, adds none, and changes one
indirect version (UnsafeAtomics 0.3.2 to 0.3.3). GPUCompiler, where the
advisory nightly job fails, is no longer in the graph.

On the cleanup worktree, the local Julia 1.13.1 unit, full and benchmark
gates passed **421/421**, **2,190/2,190** and **10/10**. The fresh
registered-dependency full suite, on the 308-package resolve, also passed
**2,190/2,190**. The two added assertions cover rejection of the removed
option. These edits change `Project.toml`, so the earlier candidate's CI and
fresh-install results do not carry over: repeat the required CI jobs and the
fresh URL install at the new commit before registering.

That cleanup landed as `0702d50`. CI run 37373805729 passed all four required
jobs at that commit: **2,190/2,190** on Julia 1.12, on Julia 1.13 and on the
Julia 1.13 modern profile, and **10/10** on the recovery benchmark. A fresh
URL install of that commit ran the README example. The advisory nightly job
no longer fails in GPUCompiler precompilation and now runs the suite. It
passed 2,182 of 2,190 assertions on Julia 1.14.0-DEV; the eight failures are
the `substr_test` accuracy checks in `identifiability_regressions.jl`.

### October 5 name-clash fix and seed option

`LevenbergMarquardt` is exported by both NonlinearSolve and LeastSquaresOptim,
so the unqualified name had been undefined in the package since May. Four
places used it. `solve_with_robust` with `:algorithm => :levenberg` returned
no solution, `solve_multipoint_overdetermined` silently skipped its
refinement, `PolishLevenberg` raised `UndefVarError`, and a legacy keyword
mapping named it. The two solver calls are now qualified. The overdetermined
refinement also declares its residual size, without which it failed on every
genuinely overdetermined system. `PolishLevenberg` and `PolishGaussNewton` are
removed: NonlinearSolve algorithms cannot run on the scalar
`Optimization.solve` polish path, and the residual-mode methods already
provide Levenberg–Marquardt.

`EstimationOptions(seed = ...)` is new and off by default. With an integer,
sampling and estimation run on their own random streams and restore Julia's
default RNG, so repeated runs are identical and the caller's stream is
untouched. Fresh processes with one and four threads returned identical
candidate pools for `simple` and Lotka–Volterra; without a seed every pool
differed. See the [changelog](../../CHANGELOG.md) and the
[reproducibility note](2026-03-17_results_and_api.md#reproducibility).

On that worktree the local Julia 1.13.1 unit, full and benchmark gates passed
**449/449**, **2,250/2,250** and **10/10**, and the fresh registered-dependency
full suite passed **2,250/2,250**. The 60 added assertions cover the repaired
solver paths, the polish-method contract, a check that no package method
references a name two imports both export, and the seed behavior. That fresh
resolve selected 310 packages, not 308: UnsafeAtomics 0.4.0, released the same
day, depends on LLVM.jl. GPUCompiler and Enzyme remain absent. Repeat the
required CI jobs and the fresh URL install at the new commit before
registering.

### October 6 short path, quiet defaults and manual

The package gained a short way in for users, and its documentation was
rewritten around it. See the [changelog](../../CHANGELOG.md) for the details.

- `ParameterEstimationProblem(system, measured_quantities; data, true_values)`
  takes a ModelingToolkit `System` and named data, and
  `estimate(problem; options...)` returns the solutions best fit first. The
  nine-argument constructor and `analyze_parameter_estimation_problem` are
  unchanged.
- A run with default options prints nothing, logs only errors and writes no
  files. The defaults are now `nooutput = true`, `diagnostics = false` and
  `save_system = false`. With a seed, every combination of the three gave the
  same candidate pool on `simple` and Lotka–Volterra.
- The `interpolator` and `custom_interpolator` options are removed. Beside
  the default `interpolators` list they had no effect, and about 25 tests and
  examples set them anyway.
- Two defects found while running the manual's examples are fixed. A
  `sin(c*t)` input returned its coefficient doubled when rescaling halved the
  helper state for the input (`forced_decay` over `[-0.5, 0.5]`), and state
  names such as `θ(t)` raised `StringIndexError` in models with a quantity
  that cannot be determined.
- The README and the Documenter site are new. The site's examples run at build
  time, which takes about 10 minutes locally with a warm cache. The
  `Documentation` workflow publishes it to the `gh-pages` branch. GitHub Pages
  has to be pointed at that branch once, after the first deployment. TagBot
  uses `GITHUB_TOKEN`, whose tag pushes do not start workflows, so versioned
  pages for a release need a deploy key or a manual run of the workflow.
- The README says the package is not yet in the General registry. Change that
  sentence, and the install command, when it is.

On that worktree the local Julia 1.13.1 unit, full and benchmark gates passed
**548/548**, **2,414/2,414** and **10/10**, and the fresh registered-dependency
full suite passed **2,414/2,414** with 310 packages resolved. The 164 added
assertions cover the quiet defaults, the constructor's handling of data and
true values, `estimate`, the result display, the option documentation, the
removed options, and the two fixes. The documentation build ran every example.
Repeat the required CI jobs and the fresh URL install at the new commit before
registering.

A follow-up on the same day made refinement the default
(`polish_solutions = true`). On Lotka–Volterra with 101 points and 2% noise the
worst parameter error fell from 30% to 2%, and on exact data the estimates of
six models, `hiv` among them, stayed exact. Uncertainty then describes the
refined estimate, whose coverage is not yet established (see the UQ notes).
The unit, full and benchmark gates passed **549/549**, **2,420/2,420** and
**10/10**, and the fresh registered-dependency full suite passed
**2,420/2,420**. CI at `58e9038`, before this follow-up, passed its four
required jobs; the advisory nightly job failed the same eight `substr_test`
assertions it failed at `0702d50`. CI at `82ba667`, with the follow-up, passed
all five jobs. The manual and the option documentation then marked uncertainty
estimation as experimental and asked for feedback, which changes no behavior.

## Repository and documentation review

- [x] Review the first-release source tree. Before the split, Git tracked
  4,735 files totaling about 95 MB; `repro/` accounts for 3,641 files and about
  67 MB. It holds investigation scripts, frozen inputs, logs, and evidence for
  PEtab, Sneyd, UQ, HC threading, multiplicity, scaling, and polishing. It is
  outside the normal runtime path, but two active UQ tests loaded harness files
  from it. The UQ smoke helper has moved to `test/support`; campaign-only tests
  remain on `research`. Preserve research
  evidence while deciding which tracked artifacts need to accompany the
  installable package. Ignored local output is additional disk use, not part of
  those tracked-size figures.
- [x] Review the bundled examples, datasets, and third-party material for
  provenance and license notices. Keep the GPL-3.0 package license visible.
- [x] Update the README's repository URL, status wording, support limits, and
  example if the release choices change. Keep the release guide short and link
  to the detailed technical records rather than copying their results.
- [x] Keep the existing [TagBot](../../.github/workflows/TagBot.yml) and
  [CompatHelper](../../.github/workflows/CompatHelper.yml) workflows aligned with
  the final repository URL and supported dependency versions.

## Provenance and attribution

The package remains under the root [GPL-3.0 license](../../LICENSE). This review
covers source and fixtures retained on `main`; preserved experimental trees
continue to carry their original notices. The following third-party origins
are recorded explicitly:

| Material | Source and retained attribution |
| --- | --- |
| Private GP implementation | Adapted from STOR-i/GaussianProcesses.jl and validated revision `3e896e9dbd0c41341c723ab16dcf0c261fc7b95a`; Jamie Fairbrother and Christopher Nemeth's MIT notice is retained in [the GP directory](../../src/internal/gp/LICENSE). Scope and source files are listed in its [README](../../src/internal/gp/README.md). |
| Private SIAN helpers | [SIAN source at `2f78ca8`](https://github.com/orebas/SIAN-Julia/tree/2f78ca8a0cc93f99eb2f800f08c1dbd20f8b28d9); Ilia Ilmer, Alexey Ovchinnikov and Gleb Pogudin's MIT notice is retained in [the SIAN directory](../../src/internal/sian/LICENSE). The [README](../../src/internal/sian/README.md) preserves the upstream utility file's unspecified earlier-adaptation note; this review does not invent an earlier source. |
| Older barycentric/rational interpolation helpers | [ParameterEstimation.jl source at `99f4bd5`](https://github.com/iliailmer/ParameterEstimation.jl/blob/99f4bd59d9c6cedcfeb644672168aaa3f0088984/src/rational_interpolation/bary_derivs.jl), contributed by the ParameterEstimation.jl authors. ODEPE's initial import is `dfe1893`; current code has subsequent adaptations. The upstream GPL-3.0 text matches the package's retained GPL license. |
| SI preprocessing and historical model examples | The source comments identify adaptations from [ParameterEstimation.jl](https://github.com/iliailmer/ParameterEstimation.jl), whose contributors are credited in its [Project.toml](https://github.com/iliailmer/ParameterEstimation.jl/blob/9e7adc33a4a11954acde0214f482cffea9551170/Project.toml). The historical [Crauste example](https://github.com/iliailmer/ParameterEstimation.jl/blob/9e7adc33a4a11954acde0214f482cffea9551170/examples/all-global/crauste.jl) is GPL-3.0. The example equations were not changed in this attribution pass. |
| Biohydrogenation CSV | Synthetic fixture introduced by Oren Bassik in ODEPE commit `69e3a29d060fd32bbfaf76bdaaa53781a1685c65`, with the package's GPL license. It was not measured experimental data. It left the repository with the rest of `src/examples/biohydrogenation` before 1.0.0 was registered; `fadccc6` is the last commit that has it, with a README naming the generating model, parameters and initial time −1. |
| Backend and recovery fixtures | Synthetic observations and exact upstream comparisons, with pinned source, environment and regeneration instructions in [test/reference](../../test/reference/README.md). Frozen observations are reused across compared implementations. |

The trimmed tree has no bundled figure/PDF asset collection. Historical
technical notes and logs are evidence records, not promises that their older
APIs or numerical claims apply to this release. User-facing guides identify
the supported estimator and the opt-in UQ limits.

## Candidate validation and release automation

The [portability follow-up](2026-10-04_internal_backends.md#october-5-release-candidate-portability-follow-up)
explains the Julia 1.12 clean-GP test correction. Required CI jobs are:

- Julia 1.12, registered dependencies, full suite.
- Julia 1.13, registered dependencies, full suite.
- Julia 1.13, modern registered profile, full suite.
- Julia 1.13, modern registered profile, seeded recovery benchmark.

Nightly remains advisory. Each job uploads its resolved Project/Manifest and
validation metadata, and test output records CPU/BLAS/coverage details. Local
full and benchmark gates use `test/current.jl` without dependency re-resolution.
The Documenter 1.19.0 build passed on Julia 1.13.1 after splitting the exported
API reference into pages. The functions page emits a 146 KiB size advisory
(below the 200 KiB limit); no docstring or rendering errors remain. Private
GP/SIAN modules are excluded from the public manual. Build locally with:

```sh
julia --startup-file=no --project=docs -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate(); include("docs/make.jl")'
```

The three workflows pass actionlint 1.7.12. The first candidate's CI syntax
failure was corrected by moving `runner.temp` to step scope before actual
matrix testing. CI's checkout step was later updated to `actions/checkout@v6`
in a separate workflow-only commit; the workflows still pass actionlint.
A fresh temporary environment must install the final SHA from the renamed
repository and run the README example. Inspect checks for that SHA, not merely
the most recent workflow listed for the branch.

TagBot uses GitHub's automatic token. Workflow edits are committed separately
from the release version bump: the token cannot tag a commit that modifies
workflow files. No docs deployment or SSH secret is required by this setup.
CompatHelper runs Julia 1.13 with startup files disabled and explicit contents
and pull-request permissions. No deploy key is configured, so CI must be
dispatched manually on CompatHelper PR branches before merging: its automatic
token cannot trigger another workflow. This is the behavior documented by
[CompatHelper](https://juliaregistries.github.io/CompatHelper.jl/stable/#Creating-SSH-Key).
Neither automation is manually triggered here.

## Registration sequence

1. Finish required CI and the clean install at the exact candidate SHA. Keep
   `research` and the former repository URL redirect intact.
2. Inspect the final diff, version, license, supported API and checklist.
   Confirm that the initial-version and other metadata checks still pass.
3. When the maintainer elects to register, post the request below on the
   validated commit in the renamed repository. Registration is a separate
   action; a prepared comment is not a submitted request.
4. Inspect the resulting General PR and address actual review findings. TagBot
   can create the tag/release after registration is merged.

### Prepared Registrator request (not submitted)

```text
@JuliaRegistrator register

Release notes:
First General registration candidate for ODEParameterEstimation.jl, v1.0.0.
The supported package is on main; experimental integrations and research
remain on the research branch. SIAN and GaussianProcesses functionality used
by the package is now internal, with retained upstream notices and regression
fixtures. Julia 1.12+ is supported; UQ remains opt-in with documented limits.

```

Do not create a tag ahead of registration. Preserve the final commit URL,
required check results and fresh-install evidence with the release decision.
