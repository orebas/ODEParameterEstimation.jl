# Split 2026-06-10 from the 5,443-line src/core/diagnostics.jl (Phase F2;
# pure content move, functions byte-identical). Part: diagnose()/comprehensive orchestrators, terminal equation printing.
# ─── Default interpolators for multi-interpolator sweep ────────────────

const _DIAGNOSTIC_DEFAULT_INTERPOLATORS = [
    InterpolatorAAADGPR,     # production default (rational + GP)
    InterpolatorAAAD,        # pure rational (best for stiff/boundary)
    InterpolatorAGPRobust,   # robust GP (good for smooth data)
    InterpolatorFHD,         # Floater-Hormann finite differences (baseline)
]

# ─── Top-level orchestrator ───────────────────────────────────────────

"""
    diagnose(pep; kwargs...) → DiagnosticReport | ComprehensiveDiagnosticReport

Run the full diagnostic pipeline on a `ParameterEstimationProblem`.

## Single-point mode (default, backward compatible)
    diagnose(pep)
    diagnose(pep; interpolator = aaad_gpr_pivot)

Returns a `DiagnosticReport`.

## Multi-point / multi-interpolator mode
    diagnose(pep; t_eval_points = [0.0, 5.0, 10.0])
    diagnose(pep; interpolators = [InterpolatorAAAD, InterpolatorAGPRobust])
    diagnose(pep; t_eval_points = [...], interpolators = [...])

Returns a `ComprehensiveDiagnosticReport` with a derivative accuracy grid
across all (interpolator, evaluation point) combinations.  The full 3-stage
pipeline (polynomial feasibility + sensitivity) runs for the best combination.

## Keyword arguments
- `interpolator`: Single interpolator function (default: `aaad_gpr_pivot`).
- `interpolators`: Vector of `InterpolatorMethod` enums for multi-interpolator sweep.
- `t_eval_points`: Vector of evaluation times.  Empty → production shooting points.
- `full_analysis`: Controls how many points get the full 3-stage pipeline.
  - `:best` (default) — only the best (interpolator, point) combination
  - `:top3` — top 3 best-derivative-accuracy points
  - `:all` — every shooting point
  - `Int` — top N points
  - `Vector{Float64}` — specific time points
- `save_to_disk`: Write text/CSV/HTML to `artifacts/diagnostics/`.
- `html_report`: Generate collapsible-section HTML report (default: `true`).
"""
function diagnose(
    pep::ParameterEstimationProblem;
    interpolator = aaad_gpr_pivot,
    interpolators::Vector{InterpolatorMethod} = InterpolatorMethod[],
    t_eval_points::Vector{Float64} = Float64[],
    full_analysis::Union{Symbol, Int, Vector{Float64}} = :best,
    multipoint_selection::Symbol = :best_solved_combo,
    multipoint_compare_policy::Symbol = :gate_invalid,
    save_to_disk = true,
    html_report = true,
    estimation_report::Union{Nothing, EstimationResultsReport} = nothing,
    data_config::Union{Nothing, NamedTuple} = nothing,
    kwargs...,
)
    # Auto-handle transcendental forcings (sin/cos/exp). Mirrors what
    # analyze_parameter_estimation_problem already does.
    # transform_pep_for_estimation is idempotent: short-circuits to (pep, nothing)
    # when no transcendentals are detected, so this is a no-op for polynomial models.
    t_var_for_trfn = ModelingToolkit.get_iv(pep.model.system)
    pep, _tr_info = try
        transform_pep_for_estimation(pep, t_var_for_trfn)
    catch e
        @warn "[DIAGNOSE] Transcendental transform failed (may not be needed): $e"
        (pep, nothing)
    end
    if !isnothing(_tr_info)
        @info "[DIAGNOSE] Transformed $(length(_tr_info.entries)) transcendental(s) before analysis"
    end

    multi_mode = !isempty(interpolators) || !isempty(t_eval_points)

    if multi_mode
        return _diagnose_comprehensive(pep;
            interpolator = interpolator,
            interpolators = interpolators,
            t_eval_points = t_eval_points,
            full_analysis = full_analysis,
            multipoint_selection = multipoint_selection,
            multipoint_compare_policy = multipoint_compare_policy,
            save_to_disk = save_to_disk,
            html_report = html_report,
            estimation_report = estimation_report,
            data_config = data_config,
            kwargs...)
    end

    # ── Single-point mode (backward compatible) ────────────────────────
    @info "[DIAGNOSE] Starting diagnostic for model: $(pep.name)"

    setup_data = setup_parameter_estimation(pep; interpolator = interpolator, nooutput = true)

    t_vec = pep.data_sample["t"]
    t_eval = t_vec[setup_data.time_index_set[1]]
    max_order = isempty(setup_data.good_deriv_level) ? 2 : maximum(values(setup_data.good_deriv_level))

    # Derive interpolator name from function
    _interp_name = try
        s = string(interpolator)
        # Strip module prefix if present (e.g. "ODEParameterEstimation.agp_gpr_robust" → "agp_gpr_robust")
        String(last(split(s, '.')))
    catch
        "unknown"
    end

    @info "[DIAGNOSE] Stage 1: Derivative accuracy analysis..."
    deriv_report = diagnose_derivative_accuracy(pep;
        setup_data = setup_data, t_eval = t_eval, max_order = max_order,
        interpolator_name = _interp_name, kwargs...)

    @info "[DIAGNOSE] Stage 2: Polynomial feasibility analysis..."
    poly_report = diagnose_polynomial_system(pep;
        setup_data = setup_data, t_eval = t_eval, max_order = max_order, kwargs...)

    @info "[DIAGNOSE] Stage 3: Sensitivity analysis..."
    sens_report = diagnose_sensitivity(pep;
        setup_data = setup_data, poly_report = poly_report,
        t_eval = t_eval, max_order = max_order, kwargs...)

    difficulty, bottleneck = _classify_difficulty(deriv_report, poly_report, sens_report)

    # Error budget: combine sensitivity with derivative errors
    error_budget = try
        compute_error_budget(sens_report, deriv_report, poly_report)
    catch e
        @warn "[DIAGNOSE] Error budget computation failed: $e"
        nothing
    end

    report = DiagnosticReport(
        pep.name, deriv_report, poly_report, sens_report,
        difficulty, bottleneck, Dates.now(), error_budget,
    )

    _print_diagnostic_summary(report)
    if save_to_disk
        _save_diagnostic_report(report)
        if html_report
            _save_diagnostic_html(report; pep = pep)
        end
    end

    return report
end

# ─── Comprehensive multi-point / multi-interpolator orchestrator ──────

function _diagnose_comprehensive(
    pep::ParameterEstimationProblem;
    interpolator = aaad_gpr_pivot,
    interpolators::Vector{InterpolatorMethod} = InterpolatorMethod[],
    t_eval_points::Vector{Float64} = Float64[],
    full_analysis::Union{Symbol, Int, Vector{Float64}} = :best,
    multipoint_selection::Symbol = :best_solved_combo,
    multipoint_compare_policy::Symbol = :gate_invalid,
    save_to_disk = true,
    html_report = true,
    estimation_report::Union{Nothing, EstimationResultsReport} = nothing,
    data_config::Union{Nothing, NamedTuple} = nothing,
    kwargs...,
)
    @info "[DIAGNOSE] Comprehensive diagnostic for model: $(pep.name)"
    multipoint_selection in (:best_solved_combo, :best_derivative_combo, :fixed_quartiles) ||
        throw(ArgumentError("invalid multipoint_selection=$(multipoint_selection); expected :best_solved_combo, :best_derivative_combo, or :fixed_quartiles"))
    multipoint_compare_policy in (:gate_invalid, :warn_only, :always_show) ||
        throw(ArgumentError("invalid multipoint_compare_policy=$(multipoint_compare_policy); expected :gate_invalid, :warn_only, or :always_show"))

    # Run SIAN once (structural, interpolator-independent)
    ident_data = setup_identifiability(pep; nooutput = true)
    t_vec = pep.data_sample["t"]
    max_order = isempty(ident_data.good_deriv_level) ? 2 : maximum(values(ident_data.good_deriv_level))

    # Resolve interpolator list
    if isempty(interpolators)
        interpolators = _DIAGNOSTIC_DEFAULT_INTERPOLATORS
    end

    interp_names = String[]
    interp_funcs = Function[]
    for im in interpolators
        push!(interp_names, string(interpolator_method_to_symbol(im)))
        try
            push!(interp_funcs, get_interpolator_function(im))
        catch e
            @warn "[DIAGNOSE] Skipping interpolator $im: $e"
            pop!(interp_names)
        end
    end

    # Resolve evaluation points — use production shooting points by default
    if isempty(t_eval_points)
        n_total = length(t_vec)
        shoot_indices = compute_shooting_indices(12, n_total; warp = true, beta = 3.0)
        # Avoid exact first point (boundary) — shift index 1 to index 2 if present
        if !isempty(shoot_indices) && shoot_indices[1] == 1 && n_total > 2
            shoot_indices[1] = 2
        end
        t_eval_points = unique(sort([t_vec[i] for i in shoot_indices if i >= 1 && i <= n_total]))
        @info "[DIAGNOSE] Using $(length(t_eval_points)) production shooting points (exponential warp)"
    end

    # ── Grid sweep: derivative accuracy for each (interpolator, t_eval) ─
    @info "[DIAGNOSE] Sweeping $(length(interp_names)) interpolators × $(length(t_eval_points)) points..."
    all_deriv_reports = DerivativeAccuracyReport[]
    # Track best per-point across all interpolators
    best_worst_err = Inf
    best_interp_idx = 1
    best_point_idx = 1
    # Track ranking: (worst_error, interp_idx, point_idx)
    ranking = Tuple{Float64, Int, Int}[]

    for (ii, ifunc) in enumerate(interp_funcs)
        interpolants = create_interpolants(pep.measured_quantities, pep.data_sample, t_vec, ifunc)

        # Build a setup_data-like tuple for this interpolator
        time_idx_set = pick_points(t_vec, ident_data.good_num_points, interpolants, 0.5)
        sd = (
            states = ident_data.states,
            params = ident_data.params,
            t_vector = t_vec,
            interpolants = interpolants,
            good_num_points = ident_data.good_num_points,
            good_deriv_level = ident_data.good_deriv_level,
            good_udict = ident_data.good_udict,
            good_varlist = ident_data.good_varlist,
            good_DD = ident_data.good_DD,
            time_index_set = time_idx_set,
            all_unidentifiable = Set{Num}(),
            numerical_advisory = ident_data.numerical_advisory,
        )

        for (pi, te) in enumerate(t_eval_points)
            dr = try
                diagnose_derivative_accuracy(pep;
                    setup_data = sd, t_eval = te, max_order = max_order,
                    interpolator_name = interp_names[ii], kwargs...)
            catch e
                @warn "[DIAGNOSE] derivative accuracy failed for $(interp_names[ii]) at t=$te: $e"
                continue
            end
            push!(all_deriv_reports, dr)
            push!(ranking, (dr.worst_rel_error, ii, pi))

            if dr.worst_rel_error < best_worst_err
                best_worst_err = dr.worst_rel_error
                best_interp_idx = ii
                best_point_idx = pi
            end
        end
    end

    # ── Determine which points get full 3-stage analysis ──────────────
    sort!(ranking; by = first)  # best (lowest error) first

    # Build list of (interp_idx, point_idx) for full analysis
    full_analysis_set = _resolve_full_analysis_points(
        full_analysis, ranking, t_eval_points, interp_funcs)

    # Always ensure the best is first
    if !isempty(full_analysis_set) && full_analysis_set[1] != (best_interp_idx, best_point_idx)
        filter!(x -> x != (best_interp_idx, best_point_idx), full_analysis_set)
        pushfirst!(full_analysis_set, (best_interp_idx, best_point_idx))
    elseif isempty(full_analysis_set)
        full_analysis_set = [(best_interp_idx, best_point_idx)]
    end

    # ── Full 3-stage pipeline for selected points ─────────────────────
    full_reports = DiagnosticReport[]

    for (k, (ii, pi)) in enumerate(full_analysis_set)
        ifunc = interp_funcs[ii]
        te = t_eval_points[pi]
        @info "[DIAGNOSE] Full analysis $k/$(length(full_analysis_set)): $(interp_names[ii]) at t=$(round(te; digits=4))"

        setup = setup_parameter_estimation(pep; interpolator = ifunc, nooutput = true)

        deriv = diagnose_derivative_accuracy(pep;
            setup_data = setup, t_eval = te, max_order = max_order,
            interpolator_name = interp_names[ii], kwargs...)
        poly = diagnose_polynomial_system(pep;
            setup_data = setup, t_eval = te, max_order = max_order, kwargs...)
        sens = diagnose_sensitivity(pep;
            setup_data = setup, poly_report = poly,
            t_eval = te, max_order = max_order, kwargs...)

        difficulty, bottleneck = _classify_difficulty(deriv, poly, sens)
        eb = try
            compute_error_budget(sens, deriv, poly)
        catch e
            @warn "[DIAGNOSE] Error budget failed for $(interp_names[ii]) at t=$te: $e"
            nothing
        end
        push!(full_reports, DiagnosticReport(
            pep.name, deriv, poly, sens,
            difficulty, bottleneck, Dates.now(), eb,
        ))
    end

    # Multipoint error budget (using best interpolator's setup)
    @info "[DIAGNOSE] Computing multipoint error budget..."
    mp_eb = nothing
    mp_das = DerivativeAccuracyReport[]
    mp_analysis = nothing
    try
        best_setup = setup_parameter_estimation(pep; interpolator = interp_funcs[best_interp_idx], nooutput = true)
        mp_result = _try_multipoint_error_budget(
            pep,
            best_setup,
            interp_funcs[best_interp_idx],
            interp_names[best_interp_idx];
            t_eval_points = t_eval_points,
            sp_error_budget = full_reports[1].error_budget,
            selection_policy = multipoint_selection,
            compare_policy = multipoint_compare_policy,
        )
        if !isnothing(mp_result)
            mp_eb = mp_result.error_budget
            mp_das = mp_result.derivative_reports
            mp_analysis = mp_result.analysis
        end
    catch e
        @warn "[DIAGNOSE] Multipoint error budget failed: $e"
    end

    comp = ComprehensiveDiagnosticReport(
        pep.name, full_reports, all_deriv_reports,
        interp_names, t_eval_points,
        interp_names[best_interp_idx], t_eval_points[best_point_idx],
        mp_eb, mp_das, mp_analysis,
    )

    _print_diagnostic_summary(comp.best)
    _print_grid_summary(comp)

    if save_to_disk
        _save_diagnostic_report(comp.best)
        if html_report
            _save_comprehensive_html(comp; pep = pep, estimation_report = estimation_report,
                data_config = data_config)
        end
    end

    return comp
end

"""
Resolve the `full_analysis` kwarg into a list of (interp_idx, point_idx) tuples.
"""
function _resolve_full_analysis_points(
    full_analysis::Union{Symbol, Int, Vector{Float64}},
    ranking::Vector{Tuple{Float64, Int, Int}},
    t_eval_points::Vector{Float64},
    interp_funcs)

    if full_analysis isa Symbol
        if full_analysis == :best
            n = 1
        elseif full_analysis == :top3
            n = 3
        elseif full_analysis == :all
            n = length(ranking)
        else
            n = 1
        end
        # Take top N unique (interp, point) pairs from ranking
        seen = Set{Tuple{Int, Int}}()
        result = Tuple{Int, Int}[]
        for (_, ii, pi) in ranking
            key = (ii, pi)
            if key ∉ seen
                push!(seen, key)
                push!(result, key)
                length(result) >= n && break
            end
        end
        return result
    elseif full_analysis isa Int
        n = full_analysis
        seen = Set{Tuple{Int, Int}}()
        result = Tuple{Int, Int}[]
        for (_, ii, pi) in ranking
            key = (ii, pi)
            if key ∉ seen
                push!(seen, key)
                push!(result, key)
                length(result) >= n && break
            end
        end
        return result
    elseif full_analysis isa Vector{Float64}
        # Full analysis at specific time points (using best interpolator for each)
        result = Tuple{Int, Int}[]
        for t_target in full_analysis
            # Find closest eval point
            dists = abs.(t_eval_points .- t_target)
            pi = argmin(dists)
            # Find best interpolator for this point from ranking
            best_ii = 1
            best_err = Inf
            for (err, ii, pidx) in ranking
                if pidx == pi && err < best_err
                    best_err = err
                    best_ii = ii
                end
            end
            push!(result, (best_ii, pi))
        end
        return unique(result)
    end

    return Tuple{Int, Int}[]
end

function _print_grid_summary(comp::ComprehensiveDiagnosticReport)
    println("  Interpolator × Point Grid (worst relative error)")
    println("  " * "-" ^ 68)
    @printf("  %-20s", "Interpolator")
    for te in comp.eval_points
        @printf(" %12s", @sprintf("t=%.2f", te))
    end
    println()
    println("  " * "-" ^ 68)

    n_points = length(comp.eval_points)
    for (ii, iname) in enumerate(comp.interpolator_names)
        @printf("  %-20s", iname)
        for pi in 1:n_points
            # Find matching report in grid
            grid_idx = (ii - 1) * n_points + pi
            if grid_idx <= length(comp.derivative_grid)
                dr = comp.derivative_grid[grid_idx]
                err = dr.worst_rel_error
                marker = (iname == comp.best_interpolator && comp.eval_points[pi] == comp.best_eval_point) ? "*" : " "
                @printf(" %11.2e%s", err, marker)
            else
                @printf(" %12s", "—")
            end
        end
        println()
    end
    println("  (* = best combination)")
    println("=" ^ 72)
    println()
end

function _classify_difficulty(deriv, poly, sens)
    # Infeasible: 0 solutions with production data
    if poly.n_solutions_production == 0 && poly.is_square
        return :infeasible, "No algebraic solutions found with production interpolants"
    end

    worst_err = deriv.worst_rel_error
    cond = sens.jacobian_cond

    if worst_err < 0.01 && (isnan(cond) || cond < 1e6)
        return :easy, "All derivatives accurate (<1%), well-conditioned Jacobian"
    elseif worst_err < 0.10 && (isnan(cond) || cond < 1e12)
        bottleneck = if worst_err >= 0.01
            @sprintf("Derivative accuracy bottleneck: %s order %d (%.1f%% error)", deriv.worst_obs, deriv.worst_order, worst_err * 100)
        else
            @sprintf("Jacobian conditioning: %.2e", cond)
        end
        return :moderate, bottleneck
    else
        # Distinguish conditioning-only hard from derivative+conditioning hard
        if worst_err < 0.01
            bottleneck = @sprintf("Jacobian cond %.2e (derivatives accurate, conditioning is the bottleneck)", cond)
        else
            bottleneck = @sprintf("Derivative error %.1f%% at %s order %d; Jacobian cond %.2e",
                worst_err * 100, deriv.worst_obs, deriv.worst_order, cond)
        end
        return :hard, bottleneck
    end
end

# ─── Color-coded equation printing (terminal) ─────────────────────────

const _ROLE_COLORS = Dict{Symbol, Symbol}(
    :parameter => :blue,
    :state_ic => :green,
    :state_derivative => :yellow,
    :data_derivative => :cyan,
    :transcendental => :magenta,
)

const _ROLE_LABELS = Dict{Symbol, String}(
    :parameter => "param",
    :state_ic => "state IC",
    :state_at_eval => "state at t_eval",
    :state_derivative => "state deriv",
    :data_derivative => "data deriv",
    :transcendental => "transcendental",
)

"""Print a compact legend for variable role colors."""
function _print_color_coded_legend()
    print("    Legend: ")
    for (role, label) in sort(collect(_ROLE_LABELS); by = first)
        color = get(_ROLE_COLORS, role, :normal)
        printstyled(label; color = color, bold = true)
        print("  ")
    end
    println()
end

"""
Print an equation string with variables colored by their role.
Tokenizes on word boundaries and looks up each token in the role map.
"""
function _print_color_coded_equation(eq_str::String, var_roles::Dict{String, Symbol})
    # Tokenize: split into variable-name tokens and other characters
    tokens = _tokenize_equation(eq_str)
    for (token, is_var) in tokens
        if is_var && haskey(var_roles, token)
            color = get(_ROLE_COLORS, var_roles[token], :normal)
            printstyled(token; color = color)
        else
            print(token)
        end
    end
end

"""
Tokenize an equation string into (text, is_variable_name) pairs.
Variable names are sequences of [a-zA-Z_][a-zA-Z0-9_]* that might appear in var_roles.
"""
function _tokenize_equation(eq_str::String)
    tokens = Tuple{String, Bool}[]
    i = 1
    n = length(eq_str)
    while i <= n
        c = eq_str[i]
        if isletter(c) || c == '_'
            # Read a full identifier
            j = i + 1
            while j <= n && ((isletter(eq_str[j]) || isdigit(eq_str[j])) || eq_str[j] == '_')
                j += 1
            end
            push!(tokens, (eq_str[i:j-1], true))
            i = j
        else
            # Non-identifier character(s)
            j = i + 1
            while j <= n && !isletter(eq_str[j]) && eq_str[j] != '_'
                j += 1
            end
            push!(tokens, (eq_str[i:j-1], false))
            i = j
        end
    end
    return tokens
end

function _print_diagnostic_summary(report::DiagnosticReport)
    println()
    println("=" ^ 72)
    println("  DIAGNOSTIC REPORT: $(report.model_name)")
    println("  Difficulty: $(report.difficulty) | $(report.bottleneck)")
    println("=" ^ 72)

    # Derivative accuracy table
    da = report.derivative_accuracy
    println("\n  Derivative Accuracy (t = $(@sprintf("%.4f", da.t_eval)))")
    println("  " * "-" ^ 68)
    @printf("  %-15s %5s %15s %15s %12s\n", "Observable", "Order", "True", "Interpolant", "Rel Error")
    println("  " * "-" ^ 68)
    for e in da.entries
        marker = e.rel_error > 0.10 ? " ←" : ""
        @printf("  %-15s %5d %15.6e %15.6e %11.2e%s\n",
            e.obs, e.order, e.true_val, e.interp_val, e.rel_error, marker)
    end

    # Polynomial feasibility
    pf = report.polynomial_feasibility
    println("\n  Polynomial System: $(pf.n_equations) eqs × $(pf.n_variables) vars ($(pf.is_square ? "square" : "NOT square"))")
    @printf("  Solutions — perfect: %d, production: %d\n", pf.n_solutions_perfect, pf.n_solutions_production)
    @printf("  True residual — perfect: %.2e, production: %.2e\n", pf.true_residual_perfect, pf.true_residual_production)
    @printf("  Closest distance — perfect: %.2e, production: %.2e\n", pf.closest_distance_perfect, pf.closest_distance_production)

    # Color-coded equations (first 5 + count of remaining)
    if !isempty(pf.equation_strings)
        println("\n  Equations (color-coded by variable role):")
        _print_color_coded_legend()
        n_show = min(5, length(pf.equation_strings))
        for i in 1:n_show
            print("    ")
            _print_color_coded_equation(pf.equation_strings[i], pf.variable_roles)
            println()
        end
        remaining = length(pf.equation_strings) - n_show
        if remaining > 0
            println("    ... and $remaining more equations")
        end
    end

    # Sensitivity
    sr = report.sensitivity
    @printf("\n  Jacobian cond: %.2e | Effective rank: %d / %d\n",
        sr.jacobian_cond, sr.effective_rank, length(sr.singular_values))
    if !isnan(sr.root_sensitivity)
        @printf("  Root sensitivity: %.2e\n", sr.root_sensitivity)
    end

    println("=" ^ 72)
    println()
end

function _save_diagnostic_report(report::DiagnosticReport)
    dir = joinpath("artifacts", "diagnostics", report.model_name)
    mkpath(dir)

    # Summary text
    open(joinpath(dir, "summary.txt"), "w") do io
        println(io, "Diagnostic Report: $(report.model_name)")
        println(io, "Timestamp: $(report.timestamp)")
        println(io, "Difficulty: $(report.difficulty)")
        println(io, "Bottleneck: $(report.bottleneck)")
        println(io)

        da = report.derivative_accuracy
        println(io, "Derivative Accuracy (t = $(da.t_eval)):")
        @printf(io, "%-15s %5s %15s %15s %12s\n", "Observable", "Order", "True", "Interpolant", "Rel Error")
        for e in da.entries
            @printf(io, "%-15s %5d %15.6e %15.6e %11.2e\n",
                e.obs, e.order, e.true_val, e.interp_val, e.rel_error)
        end
        println(io)

        pf = report.polynomial_feasibility
        println(io, "Polynomial System: $(pf.n_equations) eqs × $(pf.n_variables) vars")
        @printf(io, "Solutions: perfect=%d, production=%d\n", pf.n_solutions_perfect, pf.n_solutions_production)
        @printf(io, "True residual: perfect=%.2e, production=%.2e\n", pf.true_residual_perfect, pf.true_residual_production)
        @printf(io, "Closest distance: perfect=%.2e, production=%.2e\n", pf.closest_distance_perfect, pf.closest_distance_production)
        println(io)

        sr = report.sensitivity
        @printf(io, "Jacobian cond: %.2e\n", sr.jacobian_cond)
        @printf(io, "Effective rank: %d / %d\n", sr.effective_rank, length(sr.singular_values))
    end

    # Derivative accuracy CSV
    open(joinpath(dir, "derivative_accuracy.csv"), "w") do io
        println(io, "observable,order,true_val,interp_val,rel_error")
        for e in report.derivative_accuracy.entries
            @printf(io, "%s,%d,%.15e,%.15e,%.15e\n",
                e.obs, e.order, e.true_val, e.interp_val, e.rel_error)
        end
    end

    # Sensitivity CSV
    open(joinpath(dir, "sensitivity.csv"), "w") do io
        println(io, "index,singular_value")
        for (i, sv) in enumerate(report.sensitivity.singular_values)
            @printf(io, "%d,%.15e\n", i, sv)
        end
    end

    @info "[DIAGNOSE] Reports saved to $dir"
end
