"""
Multi-Point Template System for ODEParameterEstimation

Extends the single-point SI-template pattern to N-point systems:
- Build once: duplicate template N times, rename per-point vars, rank-aware strip to square
- Evaluate cheaply: substitute interpolated data at any time point set
- Solve with HC parameter homotopy: track solutions between time point pairs
"""

using OrderedCollections
using LinearAlgebra
using ForwardDiff
using Symbolics
using NonlinearSolve
using Printf: @sprintf

# ═══════════════════════════════════════════════════════════════════════════════
# Helper: parse derivative order, stripping _ptK suffix first
# ═══════════════════════════════════════════════════════════════════════════════

function _multipoint_deriv_order(name_str)
    # Deliberately SIAN-form only (NOT parse_jet_label): multipoint template
    # variables are always SIAN-style jet names, and the characterization tests
    # lock Differential-form inputs to order 0 here. Strips the _ptK multipoint
    # suffix first; failure contract is order 0.
    clean = replace(string(name_str), r"_pt\d+$" => "")
    parsed = parse_derivative_variable_name(clean)
    return isnothing(parsed) ? 0 : parsed[2]
end

# ═══════════════════════════════════════════════════════════════════════════════
# Variable renaming for multi-point
# ═══════════════════════════════════════════════════════════════════════════════

"""
    _rename_per_point_variables(equations, variables, shared_names, point_index)

Rename non-shared variables by appending `_ptK` suffix.
Point 1 is left unchanged. Returns `(renamed_equations, rename_dict)`.
"""
function _rename_per_point_variables(equations::Vector, variables::Vector,
        shared_names::Set{String}, point_index::Int)
    if point_index == 1
        return equations, Dict{Any, Any}()
    end
    rd = Dict{Any, Any}()
    for v in variables
        string(v) in shared_names && continue
        new_name = replace(string(v), r"\(.*\)$" => "") * "_pt$(point_index)"
        rd[v] = Symbolics.variable(Symbol(new_name))
    end
    renamed_eqs = [Symbolics.substitute(eq, rd) for eq in equations]
    return renamed_eqs, rd
end

"""
    _per_point_data_indices(data_vars, n_points) → Vector{Vector{Int}}

Exact per-point index lists into `data_vars`, classified by the `_ptK` suffix
(point 1 = unsuffixed). Returns lists, never ranges: indices are only
point-major-contiguous as an accident of construction order, and collapsing to
`first:last` silently mis-assigns data when they interleave.

The lists always partition `1:length(data_vars)` (every var has exactly one
point), asserted here so a future classification change fails loudly.
"""
function _per_point_data_indices(data_vars, n_points::Int)
    per_point = [Int[] for _ in 1:n_points]
    for (i, dv) in enumerate(data_vars)
        name = string(dv)
        assigned = false
        for pt in 2:n_points
            if endswith(name, "_pt$pt")
                push!(per_point[pt], i)
                assigned = true
                break
            end
        end
        assigned || push!(per_point[1], i)
    end
    @assert sum(length, per_point) == length(data_vars) "per-point index lists must partition data_vars"
    return per_point
end

"""
    _build_data_var_meta(data_vars, n_points, mq) → Vector{DataVarMeta}

Construction-time classification of every data variable into
(clean_name, obs_idx, order, point, kind), aligned 1:1 with `data_vars`.
Mirrors the resolution logic of `evaluate_multipoint_template` (trfn name
pattern, SIAN-name parse, exact observable-name map, `y<N>` positional
fallback) so UQ Σ_d assembly never re-derives it from display labels.
"""
function _build_data_var_meta(data_vars, n_points::Int, mq)
    obs_name_to_idx = Dict{String, Int}()
    for (idx, mq_eq) in enumerate(mq)
        obs_name_to_idx[replace(string(mq_eq.lhs), r"\(.*\)$" => "")] = idx
    end

    per_point = _per_point_data_indices(data_vars, n_points)
    point_of = Dict{Int, Int}()
    for (pt, idxs) in enumerate(per_point), i in idxs
        point_of[i] = pt
    end

    meta = Vector{DataVarMeta}(undef, length(data_vars))
    for (i, dv) in enumerate(data_vars)
        clean = replace(string(dv), r"_pt\d+$" => "")
        pt = point_of[i]

        if contains(clean, "_trfn_")
            meta[i] = DataVarMeta(clean, nothing, 0, pt, :transcendental)
            continue
        end

        parsed = parse_derivative_variable_name(clean)
        if isnothing(parsed)
            meta[i] = DataVarMeta(clean, nothing, 0, pt, :unresolved)
            continue
        end
        base_name, order = parsed

        obs_idx = get(obs_name_to_idx, string(base_name), nothing)
        if isnothing(obs_idx)
            # Positional fallback mirrored from evaluate_multipoint_template
            m = match(r"^y(\d+)$", string(base_name))
            if !isnothing(m)
                cand = parse(Int, m.captures[1])
                obs_idx = 1 <= cand <= length(mq) ? cand : nothing
            end
        end

        meta[i] = DataVarMeta(clean, obs_idx, order, pt,
            isnothing(obs_idx) ? :unresolved : :observable_jet)
    end
    return meta
end

# ═══════════════════════════════════════════════════════════════════════════════
# Variable classification for multi-point
# ═══════════════════════════════════════════════════════════════════════════════

"""
    _classify_solve_vs_data(all_symb_vars, inst_var_names_set, pep_data)

Partition symbolic template variables into solve_vars and data_vars by comparing
the symbolic variable set against the instantiated (post-substitution) variable set.

Data variables are those present in the symbolic equations but absent from the
instantiated equations — they got substituted with numerical values. This approach
is naming-convention-agnostic and works regardless of whether SIAN uses y-names
or x-names for observables.
"""
function _classify_solve_vs_data(all_symb_vars::Vector, inst_var_names_set::Set{String},
        pep_data::ParameterEstimationProblem)

    # Name-based classification for parameter identification
    roles = _classify_polynomial_variables(
        [replace(string(v), r"_pt\d+$" => "") for v in all_symb_vars], pep_data)

    solve_vars = Any[]
    data_vars = Any[]
    param_indices = Int[]

    for (i, v) in enumerate(all_symb_vars)
        v_name = string(v)
        clean_name = replace(v_name, r"_pt\d+$" => "")

        # A variable is "data" if its clean name is NOT in the instantiated variable set
        # (it disappeared after numerical data substitution) OR if it's a transcendental
        is_data = !(clean_name in inst_var_names_set) ||
                  contains(clean_name, "_trfn_")

        if is_data
            push!(data_vars, v)
        else
            push!(solve_vars, v)
            role = get(roles, clean_name, :state_derivative)
            if role == :parameter
                push!(param_indices, length(solve_vars))
            end
        end
    end

    return solve_vars, data_vars, param_indices
end

# ═══════════════════════════════════════════════════════════════════════════════
# Rank-aware top-down stripping
# ═══════════════════════════════════════════════════════════════════════════════

"""
    _rank_aware_topdown_strip(equations, variables, eq_metadata; rank_atol, diagnostics)

Remove highest-order structural equations from an overdetermined system,
checking Jacobian rank at each step. Returns a `BitVector` mask of kept equations.

Algorithm:
1. Sort structural equations by descending derivative order
2. For each: tentatively remove + cascade orphaned data equations
3. Check rank of remaining Jacobian rows
4. If rank dropped below remaining var count → restore and skip
5. If square and full rank → done
6. Fallback: greedy row selection (data-first, low-order preference)
"""
function _rank_aware_topdown_strip(equations::Vector, variables::Vector,
        eq_metadata::Vector; rank_atol::Float64 = 1e-8, diagnostics::Bool = false,
        n_rank_probes::Int = 3)

    n_eq = length(equations)
    n_var = length(variables)

    # Pre-compute Jacobian at multiple random points, take max rank
    f_combined = _compile_system_function(equations, variables)
    J_full = nothing
    best_full_rank = 0
    # Seeded rank probes: equation-stripping decisions
    # must be deterministic; the noise-frontier twin already seeds identically.
    probe_rng = MersenneTwister(0x9f4d0329)
    for _ in 1:n_rank_probes
        rand_point = _probe_point(probe_rng, variables, 10.0)
        J_probe = ForwardDiff.jacobian(f_combined, rand_point)
        r = rank(J_probe; atol = rank_atol)
        if r > best_full_rank
            best_full_rank = r
            J_full = J_probe
        end
    end

    if diagnostics
        println("  [MPT] Full Jacobian: $(size(J_full)), rank=$best_full_rank / $n_var")
    end

    struct_indices = [i for (i, m) in enumerate(eq_metadata) if !m.is_data]
    data_indices = [i for (i, m) in enumerate(eq_metadata) if m.is_data]
    struct_by_order = sort(struct_indices; by = i -> -eq_metadata[i].order)

    kept = trues(n_eq)

    for idx in struct_by_order
        saved_kept = copy(kept)
        kept[idx] = false
        m = eq_metadata[idx]

        # Cascade: remove data equations for orphaned variables
        struct_vars = Set{Any}()
        for i in 1:n_eq
            kept[i] && !eq_metadata[i].is_data && union!(struct_vars, Symbolics.get_variables(equations[i]))
        end
        cascaded = Int[]
        for di in data_indices
            !kept[di] && continue
            data_var = first(Symbolics.get_variables(equations[di]))
            if !(data_var in struct_vars)
                kept[di] = false
                push!(cascaded, di)
            end
        end

        # Compute remaining variable count
        remaining_var_set = OrderedCollections.OrderedSet{Any}()
        for i in 1:n_eq
            kept[i] && union!(remaining_var_set, Symbolics.get_variables(equations[i]))
        end
        remaining_var_count = length(remaining_var_set)
        remaining_eq_count = count(kept)

        # Rank check
        kept_rows = findall(kept)
        new_rank = isempty(kept_rows) ? 0 : rank(J_full[kept_rows, :]; atol = rank_atol)

        if new_rank < remaining_var_count
            # Removal broke independence — restore
            kept .= saved_kept
            if diagnostics
                println("    [MPT] -eq$idx (pt$(m.point) ord=$(m.order)) → RANK DROP ($new_rank < $remaining_var_count). KEPT.")
            end
        elseif remaining_eq_count == remaining_var_count && new_rank == remaining_var_count
            # Square and full rank — success
            if diagnostics
                println("    [MPT] -eq$idx (pt$(m.point) ord=$(m.order)) → $remaining_eq_count eqs, $remaining_var_count vars → SQUARE!")
            end
            break
        elseif remaining_eq_count < remaining_var_count
            # Underdetermined — restore
            kept .= saved_kept
            if diagnostics
                println("    [MPT] -eq$idx (pt$(m.point) ord=$(m.order)) → UNDERDETERMINED. KEPT.")
            end
            break
        else
            # Still overdetermined, removal was safe
            if diagnostics
                println("    [MPT] -eq$idx (pt$(m.point) ord=$(m.order)) → $remaining_eq_count eqs, $remaining_var_count vars (Δ=$(remaining_eq_count - remaining_var_count))")
            end
        end
    end

    # Check if we reached square
    remaining_eq_count = count(kept)
    remaining_var_set = OrderedCollections.OrderedSet{Any}()
    for i in 1:n_eq
        kept[i] && union!(remaining_var_set, Symbolics.get_variables(equations[i]))
    end
    remaining_var_count = length(remaining_var_set)

    if remaining_eq_count > remaining_var_count
        # Still overdetermined — greedy row selection fallback
        if diagnostics
            println("  [MPT] Greedy fallback: $remaining_eq_count eqs → $remaining_var_count vars")
        end
        kept_rows = findall(kept)
        sorted_kept = sort(kept_rows; by = i -> (eq_metadata[i].is_data ? -1 : 0, eq_metadata[i].order, eq_metadata[i].point))

        final_sel = Int[]
        cur_rows = zeros(eltype(J_full), 0, size(J_full, 2))
        cur_rank = 0
        for idx in sorted_kept
            test = vcat(cur_rows, J_full[idx:idx, :])
            r = rank(test; atol = rank_atol)
            if r > cur_rank
                push!(final_sel, idx)
                cur_rows = test
                cur_rank = r
            end
            cur_rank == remaining_var_count && break
        end
        kept .= false
        for idx in final_sel
            kept[idx] = true
        end
        if diagnostics
            println("  [MPT] Greedy selected $(length(final_sel)) / $remaining_var_count equations")
        end
    end

    return kept
end


# ═══════════════════════════════════════════════════════════════════════════════
# Build: construct multi-point template
# ═══════════════════════════════════════════════════════════════════════════════

"""
    build_multipoint_template(pep_data, setup, si_template; n_points=2, diagnostics=false)

Build a multi-point polynomial template from the single-point SI template.

# Algorithm
1. Evaluate SI template at N well-separated probe points
2. Classify variables (shared params vs per-point state/data)
3. Rename per-point variables with `_ptK` suffix
4. Combine and run rank-aware top-down stripping
5. Build the combined SYMBOLIC system (same renaming, no data substitution)
6. Apply stripping mask to symbolic system
7. Classify stripped vars into solve_vars vs data_vars
"""
function build_multipoint_template(
        pep_data::ParameterEstimationProblem,
        setup::NamedTuple,
        si_template;
        n_points::Int = 2,
        diagnostics::Bool = false,
        rank_atol::Float64 = 1e-8,
)
    model = pep_data.model.system
    mq = pep_data.measured_quantities
    data_sample = pep_data.data_sample
    t_vec = data_sample["t"]
    n_t = length(t_vec)

    # Template DD for observable mapping
    template_DD = hasproperty(si_template, :template_DD) ? si_template.template_DD : setup.good_DD

    # Choose well-separated probe points
    fracs = n_points == 1 ? [0.5] :
            n_points == 2 ? [0.25, 0.75] :
            collect(range(0.15, 0.85; length = n_points))
    probe_indices = [max(2, min(n_t - 1, round(Int, n_t * f))) for f in fracs]

    if diagnostics
        println("[MPT] Building $(n_points)-point template, probes at t=$(round.(t_vec[probe_indices]; digits=3))")
    end

    # ── Step 1: Evaluate at probe points ──────────────────────────────
    all_inst_eqs = Vector{Vector}()  # instantiated equations per point
    all_inst_vars = Vector{Vector}()
    all_symb_eqs = Vector{Vector}()  # symbolic equations per point (from template)

    for t_idx in probe_indices
        inst_eqs, inst_vars = construct_equation_system_from_si_template(
            model, mq, data_sample, setup.good_deriv_level,
            setup.good_udict, setup.good_varlist, setup.good_DD;
            time_index_set = [t_idx],
            precomputed_interpolants = setup.interpolants,
            si_template = si_template)
        push!(all_inst_eqs, inst_eqs)
        push!(all_inst_vars, inst_vars)
    end

    # Get the symbolic template equations (before data substitution)
    symb_template_eqs = si_template.equations

    # ── Step 2: Classify variables ────────────────────────────────────
    roles = _classify_polynomial_variables(string.(all_inst_vars[1]), pep_data)
    param_set = Set(vn for (vn, r) in roles if r == :parameter)

    if diagnostics
        println("[MPT] Shared parameters: ", param_set)
        println("[MPT] Per-point instantiated: $(length(all_inst_eqs[1])) eqs, $(length(all_inst_vars[1])) vars")
    end

    # ── Step 3: Build name mapping for symbolic template variables ────
    # The symbolic template has variables with complex Symbolics names
    # (e.g. "Differential(t)(y1(t))"). We need to map each to a clean
    # SIAN-style name for proper renaming.
    #
    # Solve vars: already have clean names (they're the same objects as inst_vars)
    # Data vars: identified from DD.obs_lhs, mapped to clean names via (obs_idx, order)

    symb_template_vars_set = OrderedCollections.OrderedSet{Any}()
    for eq in symb_template_eqs
        union!(symb_template_vars_set, Symbolics.get_variables(eq))
    end
    symb_template_vars = collect(symb_template_vars_set)

    # Build clean name mapping for ALL template variables
    inst_var_name_set = Set(string(v) for v in all_inst_vars[1])
    symb_to_clean = Dict{Any, String}()
    for v in symb_template_vars
        sname = string(v)
        if sname in inst_var_name_set
            # Solve variable — already has a clean name
            symb_to_clean[v] = sname
        else
            # Data variable — look up in DD.obs_lhs
            found = false
            for (level_idx, level_vars) in enumerate(template_DD.obs_lhs)
                for (obs_idx, lhs_v) in enumerate(level_vars)
                    if lhs_v === v || isequal(lhs_v, v)
                        deriv_order = level_idx - 1
                        obs_name = replace(string(mq[obs_idx].lhs), r"\(.*\)$" => "")
                        symb_to_clean[v] = "$(obs_name)_$(deriv_order)"
                        found = true
                        break
                    end
                end
                found && break
            end
            if !found
                # Fallback: use sanitized string name
                clean = replace(sname, r"\(.*?\)" => "")
                clean = replace(clean, r"[^a-zA-Z0-9_]" => "_")
                symb_to_clean[v] = clean
            end
        end
    end

    if diagnostics
        n_data_mapped = count(v -> !(string(v) in inst_var_name_set), symb_template_vars)
        println("[MPT] Template vars: $(length(symb_template_vars)) total, $n_data_mapped data vars mapped")
    end

    # ── Step 4: Rename + combine both instantiated and symbolic equations ─
    combined_inst_eqs = Num[]
    combined_symb_eqs = Num[]
    eq_meta = @NamedTuple{point::Int, is_data::Bool, order::Int}[]

    for pt in 1:n_points
        # Rename instantiated equations (clean names, proven approach from exp09)
        inst_renamed, _ = _rename_per_point_variables(
            all_inst_eqs[pt], all_inst_vars[pt], param_set, pt)

        # Rename symbolic equations using the clean name mapping
        # For point 1: rename data vars to clean names (solve vars already have clean names)
        # For point k > 1: rename ALL non-shared vars to clean_name_ptK
        rd = Dict{Any, Any}()
        for v in symb_template_vars
            sname = string(v)
            clean = get(symb_to_clean, v, sname)
            if pt == 1
                # Only rename if the variable has a complex name (data var)
                if sname != clean && !(clean in param_set)
                    rd[v] = Symbolics.variable(Symbol(clean))
                end
            else
                clean in param_set && continue
                rd[v] = Symbolics.variable(Symbol(clean * "_pt$(pt)"))
            end
        end
        symb_renamed = isempty(rd) ? collect(symb_template_eqs) :
                       [Symbolics.substitute(eq, rd) for eq in symb_template_eqs]

        # The legacy builder has no retained source-index metadata here, so a
        # count mismatch means it cannot safely pair instantiated and symbolic
        # equations.  The top-level multipoint caller catches this and continues
        # with single-point candidates.
        n_inst = length(inst_renamed)
        n_symb = length(symb_renamed)
        if n_inst == n_symb
            # Perfect 1:1 correspondence
            for (i, eq) in enumerate(inst_renamed)
                push!(combined_inst_eqs, eq)
                nv = length(Symbolics.get_variables(eq))
                mo = maximum(_multipoint_deriv_order(v) for v in Symbolics.get_variables(eq))
                push!(eq_meta, (point = pt, is_data = nv == 1, order = mo))
            end
            append!(combined_symb_eqs, symb_renamed)
        else
            error("[MPT] Legacy multipoint cannot safely map instantiated equations to symbolic equations at point $pt ($n_inst instantiated vs $n_symb symbolic). Use system_construction_policy=:noise_frontier for source-index mapping.")
        end
    end

    inst_cvs = OrderedCollections.OrderedSet{Any}()
    for eq in combined_inst_eqs
        union!(inst_cvs, Symbolics.get_variables(eq))
    end
    combined_inst_vars = collect(inst_cvs)

    if diagnostics
        println("[MPT] Combined instantiated: $(length(combined_inst_eqs)) eqs, $(length(combined_inst_vars)) vars")
    end

    # ── Step 5: Strip (using instantiated system for Jacobian) ────────
    if diagnostics
        println("[MPT] Running rank-aware top-down stripping...")
    end
    kept = _rank_aware_topdown_strip(combined_inst_eqs, combined_inst_vars, eq_meta;
        rank_atol = rank_atol, diagnostics = diagnostics)

    n_kept = count(kept)
    kept_equation_indices = findall(kept)
    dropped_equation_indices = findall(.!kept)
    if diagnostics
        println("[MPT] Stripping result: $n_kept / $(length(combined_inst_eqs)) equations kept")
    end

    # ── Step 6: Apply mask to symbolic equations ──────────────────────
    stripped_symb_eqs = combined_symb_eqs[kept]
    stripped_meta = eq_meta[kept]

    # Collect all variables in stripped symbolic system
    stripped_vars_set = OrderedCollections.OrderedSet{Any}()
    for eq in stripped_symb_eqs
        union!(stripped_vars_set, Symbolics.get_variables(eq))
    end
    stripped_all_vars = collect(stripped_vars_set)

    # ── Step 7: Classify into solve_vars vs data_vars ─────────────────
    # Data variables are those present in symbolic equations but absent from
    # the instantiated combined variable set (they disappeared after data subst)
    combined_inst_var_names = Set(string(v) for v in combined_inst_vars)
    solve_vars, data_vars, param_indices = _classify_solve_vs_data(
        stripped_all_vars, combined_inst_var_names, pep_data)

    # Compute clean parameter names
    param_names = String[]
    for i in param_indices
        clean = replace(string(solve_vars[i]), r"_0$" => "")
        clean = replace(clean, r"\(.*\)$" => "")
        push!(param_names, clean)
    end

    # Exact per-point data variable index lists (no contiguity assumption)
    per_point_data_indices = _per_point_data_indices(data_vars, n_points)

    is_square = length(stripped_symb_eqs) == length(solve_vars)
    if diagnostics
        println("[MPT] Final: $(length(stripped_symb_eqs)) eqs, $(length(solve_vars)) solve_vars, $(length(data_vars)) data_vars")
        println("[MPT] Square: $is_square")
        println("[MPT] Params ($(length(param_indices))): ", param_names)
    end

    if !is_square
        @warn "[MPT] Multi-point template is NOT square: $(length(stripped_symb_eqs)) eqs vs $(length(solve_vars)) solve_vars"
    end

    return MultiPointTemplate(
        n_points,
        si_template,
        combined_symb_eqs,
        stripped_symb_eqs,
        solve_vars,
        data_vars,
        param_indices,
        param_names,
        stripped_meta,
        length(combined_inst_eqs),
        Int[kept_equation_indices...],
        Int[dropped_equation_indices...],
        per_point_data_indices,
        template_DD,
        mq,
        _build_data_var_meta(data_vars, n_points, mq),
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# Evaluate: substitute data at specific time points
# ═══════════════════════════════════════════════════════════════════════════════

"""
    evaluate_multipoint_template(mpt, time_indices, interpolants, data_sample; diagnostics)

Evaluate the pre-computed template at specific time point indices.
Returns a `MultiPointEvaluation` with data values ready for HC.jl.
"""
function evaluate_multipoint_template(
        mpt::MultiPointTemplate,
        time_indices::Vector{Int},
        interpolants::Dict,
        data_sample;
        diagnostics::Bool = false,
)
    eval_t0 = time()
    stage_seconds = OrderedDict{Symbol, Float64}()
    nth_deriv_calls = 0
    nth_deriv_seconds = 0.0
    trfn_eval_seconds = 0.0
    parse_lookup_seconds = 0.0
    max_deriv_order = 0
    @assert length(time_indices) == mpt.n_points "Expected $(mpt.n_points) time indices, got $(length(time_indices))"

    t_vec = data_sample["t"]
    t_values = Float64[t_vec[i] for i in time_indices]
    # Filled by exact data_vars index so data_values[i] ↔ mpt.data_vars[i] holds
    # under ANY per-point index ordering (NaN = not evaluated → rejected by the
    # downstream all(isfinite, …) guards).
    data_values = fill(NaN, length(mpt.data_vars))

    DD = mpt.template_DD
    mq = mpt.measured_quantities

    # Build observable name → index mapping from measured_quantities
    obs_name_to_idx = _timed_detail_stage!(stage_seconds, :observable_name_map) do
        mapping = Dict{String, Int}()
        for (idx, mq_eq) in enumerate(mq)
            obs_name = replace(string(mq_eq.lhs), r"\(.*\)$" => "")
            mapping[obs_name] = idx
        end
        mapping
    end

    for (pt, t_idx) in enumerate(time_indices)
        t_point = t_values[pt]

        for dv_idx in mpt.per_point_data_var_indices[pt]
            dv = mpt.data_vars[dv_idx]
            dv_name = string(dv)

            # Strip _ptK suffix for lookup
            clean_name = replace(dv_name, r"_pt\d+$" => "")

            # Try _trfn_ evaluation first
            t_trfn = time()
            trfn_val = evaluate_trfn_template_variable(clean_name, t_point)
            if isnothing(trfn_val)
                trfn_val = evaluate_obs_trfn_template_variable(clean_name, t_point)
            end
            trfn_eval_seconds += time() - t_trfn
            if !isnothing(trfn_val)
                data_values[dv_idx] = Float64(trfn_val)
                continue
            end

            # Parse the clean SIAN-style name (e.g., "y1_0" → base="y1", order=0)
            t_parse = time()
            parsed = parse_derivative_variable_name(clean_name)
            if isnothing(parsed)
                parse_lookup_seconds += time() - t_parse
                # Fail fast: NaN (not 0.0) — the downstream all(isfinite, …)
                # guards reject this combo instead of solving a wrong system.
                @warn "[MPT-EVAL] Cannot parse data variable: $dv_name (clean=$clean_name)"
                continue
            end
            base_name, deriv_order = parsed
            max_deriv_order = max(max_deriv_order, deriv_order)

            # Find observable index by matching base name against measured quantities
            obs_idx = get(obs_name_to_idx, string(base_name), nothing)
            if isnothing(obs_idx)
                # Fallback: try y1/y2/etc. pattern
                m = match(r"^y(\d+)$", string(base_name))
                if !isnothing(m)
                    obs_idx = parse(Int, m.captures[1])
                end
            end
            parse_lookup_seconds += time() - t_parse

            val = nothing
            if !isnothing(obs_idx) && obs_idx <= length(mq)
                obs_rhs = Symbolics.diff2term(mq[obs_idx].rhs)
                if haskey(interpolants, obs_rhs)
                    t_nth = time()
                    val = _estimation_derivative(
                        interpolants[obs_rhs], deriv_order, t_point,
                    )
                    nth_deriv_seconds += time() - t_nth
                    nth_deriv_calls += 1
                else
                    obs_lhs_wrapped = Symbolics.wrap(mq[obs_idx].lhs)
                    if haskey(interpolants, obs_lhs_wrapped)
                        t_nth = time()
                        val = _estimation_derivative(
                            interpolants[obs_lhs_wrapped], deriv_order, t_point,
                        )
                        nth_deriv_seconds += time() - t_nth
                        nth_deriv_calls += 1
                    end
                end
            end

            if isnothing(val)
                # Fail fast: NaN (not 0.0) — rejected by the all(isfinite, …) guards.
                @warn "[MPT-EVAL] No interpolant for $dv_name (base=$base_name, order=$deriv_order, obs_idx=$obs_idx)"
            else
                data_values[dv_idx] = Float64(val)
            end
        end
    end

    stage_seconds[:transcendental_lookup] = trfn_eval_seconds
    stage_seconds[:parse_and_lookup] = parse_lookup_seconds
    stage_seconds[:nth_deriv] = nth_deriv_seconds
    total_seconds = time() - eval_t0
    accounted = sum(values(stage_seconds))
    stage_seconds[:other] = max(0.0, total_seconds - accounted)
    _record_detailed_timing!((
        category = :multipoint_template_evaluate,
        context = _current_detailed_timing_context(),
        total_seconds = total_seconds,
        stage_seconds = copy(stage_seconds),
        time_indices = Tuple(time_indices),
        point_count = length(time_indices),
        data_var_count = length(mpt.data_vars),
        data_value_count = length(data_values),
        nth_deriv_calls = nth_deriv_calls,
        max_deriv_order = max_deriv_order,
    ))
    return MultiPointEvaluation(mpt, time_indices, t_values, data_values)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Solve: direct and parameter homotopy
# ═══════════════════════════════════════════════════════════════════════════════

"""
    solve_multipoint_direct(eval; options)

Direct solve: substitute data values into symbolic equations, then solve with HC.jl.
Returns a vector of real solution vectors (each in solve_vars order).
"""
function solve_multipoint_direct(eval::MultiPointEvaluation; options::Dict = Dict())
    mpt = eval.template

    # Build substitution dictionary: data_var -> value
    subst_dict = Dict{Any, Any}()
    for (i, dv) in enumerate(mpt.data_vars)
        if i <= length(eval.data_values)
            subst_dict[dv] = eval.data_values[i]
        end
    end

    # Substitute into stripped equations
    instantiated_eqs = [Symbolics.substitute(eq, subst_dict) for eq in mpt.stripped_equations]

    # Solve with existing HC solver
    solutions, _, _, _ = solve_with_hc(instantiated_eqs, mpt.solve_vars; options = options)
    return solutions
end

"""
    solve_multipoint_parameterized(mpt, evaluations; options)

Parameter homotopy solve: data_vars are HC parameters, solve_vars are HC variables.
Solves at first evaluation using fresh polyhedral solve, then tracks solutions
to subsequent evaluations.

Returns one vector of solutions per evaluation.
"""
function solve_multipoint_parameterized(
        mpt::MultiPointTemplate,
        evaluations::Vector{MultiPointEvaluation};
        options::Dict = Dict(),
        precomputed_generic_solutions = nothing,
        precomputed_generic_params = nothing,
)
    if isempty(evaluations)
        return Vector{Vector{Vector{Float64}}}()
    end

    param_values_list = [e.data_values for e in evaluations]

    return solve_with_hc_parameterized(
        mpt.stripped_equations,
        mpt.solve_vars,
        mpt.data_vars,
        param_values_list;
        options = options,
        precomputed_generic_solutions = precomputed_generic_solutions,
        precomputed_generic_params = precomputed_generic_params,
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# Overdetermined solve: HC core + GN/LM refinement
# ═══════════════════════════════════════════════════════════════════════════════

"""
    solve_multipoint_overdetermined(mpt, eval_result, all_equations, all_vars;
        diagnostics=false)

Solve using HC on the stripped (square) system, then refine each solution against
the full overdetermined (pre-strip) system using Levenberg-Marquardt.

Returns refined solution vectors in `mpt.solve_vars` order.
"""
function solve_multipoint_overdetermined(
    mpt::MultiPointTemplate,
    eval_result::MultiPointEvaluation,
    all_equations::Vector,
    all_vars::Vector;
    diagnostics::Bool = false,
)
    # Step 1: solve the square stripped system via HC
    hc_solutions = solve_multipoint_direct(eval_result)
    if isempty(hc_solutions)
        diagnostics && println("  [MP-OD] HC found 0 solutions on square core")
        return Vector{Float64}[]
    end
    diagnostics && println("  [MP-OD] HC found $(length(hc_solutions)) solutions on square core")

    # Step 2: build the overdetermined residual function
    # Substitute data values into all_equations (not just stripped)
    data_subst = Dict{Any, Float64}()
    for (j, dv) in enumerate(mpt.data_vars)
        if j <= length(eval_result.data_values)
            data_subst[dv] = eval_result.data_values[j]
        end
    end

    # Substitute data into all equations
    inst_all_eqs = [Symbolics.substitute(eq, data_subst) for eq in all_equations]

    # Collect remaining variables (should be solve_vars)
    remaining_vars = mpt.solve_vars

    # Compile overdetermined residual
    f_od = try
        _compile_system_function(inst_all_eqs, remaining_vars)
    catch e
        _rethrow_if_interrupt(e)
        diagnostics && println("  [MP-OD] Failed to compile overdetermined system: $e")
        return hc_solutions  # fall back to HC solutions
    end

    n_eqs = length(inst_all_eqs)
    n_vars = length(remaining_vars)
    diagnostics && println("  [MP-OD] Overdetermined system: $n_eqs eqs × $n_vars vars")

    # Step 3: refine each HC solution using LM
    refined = Vector{Float64}[]
    for (si, sol) in enumerate(hc_solutions)
        x0 = Float64.(sol)
        try
            # Use NonlinearSolve.jl with LevenbergMarquardt
            residual! = (du, u, p) -> begin
                du .= f_od(u)
            end
            nf = NonlinearFunction(residual!; resid_prototype = zeros(n_eqs))
            prob = NonlinearLeastSquaresProblem(nf, x0)
            lm_sol = NonlinearSolve.solve(prob, NonlinearSolve.LevenbergMarquardt();
                abstol = 1e-12, reltol = 1e-12, maxiters = 100)

            refined_x = Float64.(lm_sol.u)
            # Check improvement
            res_hc = norm(f_od(x0))
            res_lm = norm(f_od(refined_x))
            if res_lm <= res_hc
                push!(refined, refined_x)
                diagnostics && println("  [MP-OD] Solution $si: HC residual=$(@sprintf("%.2e", res_hc)) → LM residual=$(@sprintf("%.2e", res_lm))")
            else
                push!(refined, x0)  # LM diverged, keep HC
                diagnostics && println("  [MP-OD] Solution $si: LM diverged ($(@sprintf("%.2e", res_lm)) > $(@sprintf("%.2e", res_hc))), keeping HC")
            end
        catch e
            _rethrow_if_interrupt(e)
            push!(refined, x0)  # LM failed, keep HC
            diagnostics && println("  [MP-OD] Solution $si: LM failed ($e), keeping HC")
        end
    end

    return refined
end

# ═══════════════════════════════════════════════════════════════════════════════
# Combinations
# ═══════════════════════════════════════════════════════════════════════════════


"""
    _generate_combinations!(result, items, k)

Generate all C(n,k) combinations of `items` taken `k` at a time. Appends to `result`.
"""
function _generate_combinations!(result::Vector{Vector{Int}}, items::Vector{Int}, k::Int)
    n = length(items)
    k > n && return
    indices = collect(1:k)
    while true
        push!(result, items[indices])
        # Find rightmost index that can be incremented
        i = k
        while i > 0 && indices[i] == n - k + i
            i -= 1
        end
        i == 0 && break
        indices[i] += 1
        for j in (i + 1):k
            indices[j] = indices[j - 1] + 1
        end
    end
end
