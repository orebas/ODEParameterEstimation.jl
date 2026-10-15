# Polynomial-solve mechanics: equation ordering, mixed volume, and parameter homotopy

A system-agnostic reference for *how* ODEPE turns an ODE + observations into a polynomial system, *how*
that system is trimmed and solved, and *where* the solve is fragile. The mechanics are general; the
receptor model (`ODEParameterEstimation.Examples.receptor_subtypes()`) is the worked example because it
stresses every one of them.

Three things to understand, in order: (§2) the equation ordering + rank-trim decide *which* square
system you solve; (§3) that choice sets the mixed volume (= how many paths you track); (§4) the
parameter homotopy reuses solutions across shooting points, and fails in a specific, diagnosable way.

---

## 1. The SIAN polynomial system

For an ODE `ẋ = f(x,θ)` with observations `y = h(x)`, StructuralIdentifiability/SIAN builds a square
polynomial system `F(z; p) = 0` by **prolongation** — repeatedly differentiating the output equations
and substituting state derivatives via the ODE.

- **Unknowns `z` (what we solve for):** the parameters `θ`, plus **state jets** `X_k` = the k-th time
  derivative of state `X` at a chosen "shooting" time point. `X_0` is the state value itself.
- **Data `p` (the HC *parameters*, i.e. coefficients):** **observed-derivative jets** `y_k = dᵏy/dtᵏ`,
  obtained by differentiating an interpolant of the sampled data at the shooting time.
- **Two kinds of equation:**
  - **Data-pins (degree 1, linear):** `y_k = (k-th derivative of h, written in jets)`. E.g. for an
    observable `y1 = L`: `L_k = y1_k`. For `y2 = Ca + Cb`: `Ca_k + Cb_k = y2_k`. These tie jets directly
    to data.
  - **Dynamics (degree = the ODE's nonlinearity; 3 for bilinear mass-action):** the prolonged state
    ODEs, `X_{k+1} = (Leibniz/Faà-di-Bruno expansion of f)`.

Prolongation produces **more** equations than unknowns (over-determined). The job of the trim (§2) is to
pick a **square** subset whose Jacobian is full-rank — a locally well-posed system with isolated roots.

> Receptor example: 32 unknowns (6 params + jets `L_0..L_7`, `Ca_0..Ca_8`, `Cb_0..Cb_8`), 40 prolonged
> equations, trimmed to 32. Data-pins: `L_k=y1_k`, `Ca_k+Cb_k=y2_k`. Dynamics:
> `Ca_1 = kon1·L·(R1tot−Ca) − koff1·Ca`, etc.

---

## 2. Equation ordering + the rank-trim (the decisive, under-appreciated step)

`algebraic_independence` (`src/core/si_equation_builder.jl`) trims the over-determined system to a
square one:

1. Build the Jacobian `J = ∂F/∂z`, **evaluated at a generic (random rational) sample point** — so it's
   a numeric matrix and rank is exact-at-a-generic-point (= symbolic rank w.p. 1).
2. **Greedily walk the equations in order**: start with eq 1; for each subsequent equation keep it
   **iff its Jacobian row is linearly independent of the rows already kept** (raises the rank). Stop at
   rank = #unknowns.

**Why the Jacobian, not the polynomials?** Two equations are *functionally* (algebraically) redundant —
one adds no new constraint given the other — exactly when their gradient rows are linearly dependent at
a generic point (constant-rank / implicit-function theorem). So "linearly independent Jacobian rows" =
"functionally independent equations," and a full-rank square Jacobian = a locally well-posed system.

**This is the greedy algorithm for a basis of a linear matroid.** The Jacobian rows are vectors;
"independent set" = linearly independent rows; the greedy builds a **basis** (maximal independent set)
of the row space by walking candidates in order.

### Does ordering matter? Decisively — yes.

A matroid basis is **not unique**. The matroid framing gives the exact answer:

- **Order-INVARIANT:** the *number* of equations kept (= rank of the full Jacobian) and the *validity*
  (every order yields a full-rank square system). Every order gives a basis; all bases have the same size.
- **Order-DEPENDENT:** *which* equations are kept. When a set of equations forms a "circuit" (a minimal
  dependent set — e.g. a data-pin `X_k = y_k` and the dynamics equation that also determines `X_k`), the
  greedy keeps **whichever appears first** and drops the rest as redundant.

So ordering is the **free knob** that picks among equivalent bases. And here is the part that makes it
matter enormously:

> **Different bases are equivalent as *linearized* systems** (same row space, same rank, same
> solutions) **but wildly different as *polynomial* systems.** The greedy only sees the linearization
> (gradient rows at one point); it is **blind to degree**. So it will happily drop a **degree-1
> data-pin** and keep a **degree-3 nonlinear** equation that points the same local direction — and the
> mixed volume + conditioning depend on the *actual* equations (their Newton polytopes; §3), not their
> linearization. Same row space, same roots, but one basis is cheap and well-conditioned and another is
> 20× more expensive and 500× worse conditioned.

### SIAN's emitted order, and the "bias we want"

The greedy inherits **SIAN's emitted order** of `Et`: roughly **ascending derivative order** for the
dynamics + y2-style pins, with the directly-observed-state pins (`L_k = y1_k`) appended as a **trailing
block** (e.g. positions 33–40). Because they're trailing, the rank saturates on the dynamics first and
those linear pins are dropped — *even the low-order ones* lose to high-order dynamics.

To get a deliberate **"prefer low-order / prefer linear data-pins" bias** (the conditioning-and-speed
win), you'd run a **weighted** matroid greedy: sort `Et` by that preference *before* the rank walk
(standard result: greedy on a sorted candidate list yields the min/max-weight basis). This changes
*which* basis, not its size or validity.

**The crucial caveat — why dropping the high-order pins can be correct.** A pin `X_k = y_k` injects the
**observed k-th derivative** of the data. High-order observable derivatives from an interpolant are
**unreliable** (interpolation/noise error amplifies catastrophically with order). So a trim that drops
high-order pins is making an **accuracy** choice (don't trust `y_8`), at the cost of determining those
jets indirectly through ill-conditioned high-order dynamics (§4 shows how that backfires for the
homotopy). This is a genuine **accuracy-vs-conditioning trade-off**, not a bug.

> Receptor: keeping the linear pins vs the production trim → BKK paths **6402 → 297**, condition
> **~500× better**, 16 fewer spurious roots — but it would pin `Ca_8+Cb_8 = y2_8` to an unreliable
> 8th-derivative datum. The production trim drops it; that's right for accuracy, wrong for conditioning.

---

## 3. Mixed volume / BKK — what the number means

When HC.jl prints `mixed_volume: N`, **N is the number of paths the polyhedral homotopy tracks**, i.e.
the **BKK bound** — the number of isolated solutions in the algebraic torus `(ℂ*)ⁿ`. Two things people
get wrong:

- **It is not the number of solutions you want.** Many paths diverge to infinity. (Receptor: 6402 paths
  → **18** finite solutions; ~6384 go to infinity.) The finite/affine count is much smaller and is what
  matters; the mixed volume just bounds the *work*.
- **It is set by the Newton polytopes** (the monomial *support* of each equation), hence by the equation
  **selection** (§2). Linear equations have tiny polytopes (a standard simplex); high-degree equations
  have big ones. Mixed volume of a system is the mixed volume of its polytopes — so swapping a linear
  pin for a degree-3 dynamics equation **inflates** it. That's the entire 6402-vs-297 story.

**Column/variable scaling does not change it.** Rescaling `z = D·ẑ` (D diagonal) is monomial-preserving:
it changes coefficients (and conditioning, and the tracker's *reach*), but **not the Newton polytopes**,
so **mixed_volume is invariant** under scaling. Use scaling to help the tracker *reach* extreme-coordinate
roots; use the trim/ordering to change the path *count*.

---

## 4. Parameter homotopy — how it reuses solutions, and how it fails

To solve `F(z;p)=0` at several shooting points cheaply, ODEPE solves once and tracks:

- **Start-system homotopy (the "fresh solve"):** a total-degree or polyhedral start system → the target,
  **with the γ-trick** — `γ = cis(2π·rand())`, a random complex unit (`src/.../total_degree.jl`). The
  random complex γ makes the homotopy path *generic*, so it misses the (measure-zero, complex-codim-1)
  discriminant with probability 1. **Robust.**
- **Parameter homotopy:** track existing solutions as the data `p` moves `p_start → p_target`. HC.jl uses
  a **straight line** `H(z,t) = F(z; t·p_start + (1−t)·p_target)`, **with NO γ**
  (`src/.../homotopies/parameter_homotopy.jl`). You can't insert a γ here — the endpoint must be the
  *exact* real data `p_target`. So this is a **straight *real* segment** between two real data points,
  and it **can cross the real discriminant** (the locus of parameter values with a singular solution).

### The collapse mechanism (precise)

When the segment passes through/near a parameter value with a **singular solution**, a tracked path
reaches a point where `∂F/∂z` is **rank-deficient**. There the implicit function theorem fails — there
is no locally-unique branch to continue — Newton's corrector has no invertible Jacobian, the predictor
step collapses to zero, and the tracker exhausts its budget → **`terminated_max_steps`**. Paths are
lost. This is a **geometric** obstruction (a fold/collision on the variety), not numerical slowness.

### Diagnosis recipe (do these, in order)

1. **Count landed vs lost** per segment; lost paths return `terminated_max_steps`.
2. **Rule out the easy causes:** raise `max_steps` (HC default 10000) ×10–50, toggle `extended_precision`,
   try conservative tracker params. **If none recover the lost paths, it's geometric, not budget/precision.**
3. **Localize:** print `pr.t` for each lost path (HC tracks `t: 1→0`; 1 = start, 0 = target). Interior
   `t` = a mid-path crossing; `t≈0` = a singularity *at the target*; `t≈1` = a bad/clustered start.
4. **Dissect the system at the stall:** take `z = pr.solution` (the last point), `p(t) = t·p_start +
   (1−t)·p_target`, compute `J = ∂F/∂z`, and SVD it. **`‖F(z;p)‖ ≈ 0` with σ_min ≈ 0 / huge cond /
   rank < n = a genuine singular solution.** The **null vector** (smallest right singular vector) is the
   under-determined direction — read off which variables dominate it.
5. **Check physicality** of the stalled solutions — if they're spurious (out-of-physical-range), the
   singularity may live on spurious branches admitted by the trim (§2), not on the wanted roots.

### Fix menu

- **γ-trick yourself (complex detour):** track `p_start → p_mid → p_target` with a random *complex*
  `p_mid`. Dodges interior measure-0 crossings. **Cannot** fix a singularity *at the target* (you must
  land there) — and only partially helps if multiple obstructions (interior + endpoint) coexist.
- **Generic-complex seed:** fresh-solve once at a generic *complex* parameter (full solution cover,
  completed by monodromy if needed), then complex-detour to each real target. Avoids weak/nongeneric
  real seeds and incomplete solution counts.
- **Monodromy:** complex loops in parameter space — naturally avoids the real discriminant; good as a
  completion/fallback.
- **Lower the derivative order needed (multipoint):** if the singular/under-determined direction is a
  *high-order nuisance jet* (the system can't pin the top jets), spreading information across multiple
  shooting points lowers the max order per point → those jets are better-determined → the singularity
  shrinks. The principled structural fix.
- **NOT a fix:** re-adding high-order data-pins to "remove" the singularity — it injects the unreliable
  high-order derivative (§2 caveat). You'd trade a conditioning problem for a bad-data problem.

> Receptor: parameter homotopy lands 4 of 16 (point1→point2). `max_steps×50`, precision, conservative —
> recover **zero** → geometric. Stalls: `‖F‖≈1e−10`, **σ_min≈1e−9, cond≈1e16, rank 28–29 of 32** →
> genuine singular solutions; null direction = the order-8 jets `Ca_8+Cb_8` (the dropped `y2_8` pin),
> ~98% decoupled from the parameters; stalled solutions are spurious (negative occupancy). So the
> homotopy collapse is the trim's correct refusal-to-trust-`y2_8`, paid on a nuisance direction —
> recovery (parameters) is unaffected; the speed fix is multipoint, not pin-keeping.

---

## 5. The two trade-offs to keep in mind

- **The trim (§2):** *accuracy* (drop unreliable high-order data-pins) vs *conditioning/speed* (keeping
  them shrinks mixed volume and improves conditioning, but injects bad high-order derivatives). ODEPE's
  trim chooses accuracy.
- **The parameter homotopy (§4):** *speed* (reuse solutions across shooting points via one solve + cheap
  tracking) vs *robustness* (the straight real segment can cross the discriminant; the γ'd fresh solve
  is robust but pays a full polyhedral solve each time). When tracking collapses, ODEPE currently falls
  back to fresh γ-solves — robust, slower.

---

## Code map
- `src/core/si_equation_builder.jl` — `algebraic_independence` (the rank-trim / matroid greedy).
- `src/core/si_equation_builder.jl` — `get_polynomial_system_from_sian` (builds `Et`, returns
  `full_polynomial_system` + `selected_equation_indices`).
- `HomotopyContinuation/*/src/homotopies/parameter_homotopy.jl` — the straight-line parameter homotopy
  (no γ).
- `HomotopyContinuation/*/src/total_degree.jl` — the start-system homotopy (`γ = cis(2π·rand())`).
- `src/core/homotopy_continuation.jl` — `solve_with_hc_parameterized` (the per-point fresh + track +
  fresh-fallback loop), `compute_column_scales` / `scale_hc_system` (the scaling that leaves mixed
  volume invariant).
