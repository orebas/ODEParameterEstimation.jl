# Split 2026-06-10 from the 5,443-line src/core/diagnostics.jl (Phase F2;
# pure content move, functions byte-identical). Part: diagnose_uncertainty, UQ HTML, estimation/backsolve reports.
# ─── Uncertainty Quantification ────────────────────────────────────────

"""
    _parse_data_label(label) → (base_name, deriv_order)

Parse a data variable label that may be in SIAN style ("y1_0", "y1_2") or
Symbolics style ("y1(t)", "Differential(t, 1)(y1(t))").

Returns `("", 0)` if parsing fails.
"""
function _parse_data_label(label::String)
    # Thin adapter over the single shared parser (Phase D1); this caller's
    # failure contract is ("", 0).
    parsed = parse_jet_label(label)
    return isnothing(parsed) ? ("", 0) : parsed
end

"""
    _match_obs_name(base_name, obs_name_to_idx) → Union{Int, Nothing}

Find the observable index matching a base name. Exact match only: the previous
`startswith` prefix fallback could match "y1" to "y10" depending on dict
iteration order (same silent-mismatch class as the removed `lookup_value`
prefix fallback). Callers warn and skip the label when this returns `nothing`.
"""
function _match_obs_name(base_name::AbstractString, obs_name_to_idx::Dict{String, Int})
    return get(obs_name_to_idx, base_name, nothing)
end

const _UQ_IA_PHYSICAL_ROLES = Set([:parameter, :state_ic])

"""
    compute_practical_identifiability_index(param_covariance, labels, roles, values; scale_floor=1e-8)

Compute the normalized paper index `I_A` from propagated parameter covariance.
Only physical recovered quantities (parameters and state initial conditions) are
included in the default projection; helper derivatives and data variables are
excluded.
"""
function compute_practical_identifiability_index(
    param_covariance::AbstractMatrix{<:Real},
    param_labels::Vector{String},
    param_roles::Dict{String, Symbol},
    param_values::Vector{Float64};
    scale_floor::Float64 = 1e-8,
)::PracticalIdentifiabilityIndex
    physical_idx = [
        i for i in eachindex(param_labels)
        if get(param_roles, param_labels[i], :unknown) in _UQ_IA_PHYSICAL_ROLES
    ]
    warnings = String[]

    if isempty(physical_idx)
        push!(warnings, "No parameter or state-IC labels were available for I_A projection.")
        return PracticalIdentifiabilityIndex(
            String[], Symbol[], Float64[],
            zeros(0, 0), zeros(0, 0),
            NaN, NaN, :no_physical_unknowns, warnings,
        )
    end

    labels = param_labels[physical_idx]
    roles = [get(param_roles, label, :unknown) for label in labels]
    scales = Float64[]
    for idx in physical_idx
        v = idx <= length(param_values) ? param_values[idx] : NaN
        if isfinite(v)
            push!(scales, max(abs(v), scale_floor))
        else
            push!(scales, 1.0)
            push!(warnings, "No finite scale value for '$(param_labels[idx])'; I_A used scale 1.0.")
        end
    end

    Σ_phys = Matrix{Float64}(param_covariance)[physical_idx, physical_idx]
    Dinv = Diagonal(1.0 ./ scales)
    C_rel = _psd_symmetric_matrix(Dinv * Σ_phys * Dinv)
    λ = isempty(C_rel) ? NaN : max(maximum(eigvals(Symmetric(C_rel))), 0.0)
    ia = isfinite(λ) ? sqrt(λ) : NaN
    status = !isfinite(ia) ? :degenerate : ia < 0.5 ? :ok : ia < 2.0 ? :wide_ci : :degenerate

    return PracticalIdentifiabilityIndex(
        labels, roles, scales,
        Σ_phys, C_rel,
        λ, ia, status, warnings,
    )
end

"""
    diagnose_uncertainty(pep, setup_data, t_eval, sensitivity_report; kwargs...) → UncertaintyReport

Propagate derivative-estimator sampling covariance through the parameter-data
sensitivity matrix to compute parameter uncertainty: Σ_x = S · Σ_d · S'.

The default Σ_d is `W Sigma_y W'`, where W is the fixed-hyperparameter GP
posterior-mean jet influence matrix and `Sigma_y` uses the GP-learned
observation noise in raw observation units.

Returns `nothing` if the sensitivity matrix is empty or UQ computation fails.
"""
function diagnose_uncertainty(
    pep::ParameterEstimationProblem,
    setup_data,
    t_eval::Float64,
    sensitivity_report::SensitivityReport;
    kwargs...,
)
    S = sensitivity_report.data_sensitivity_matrix
    data_labels = sensitivity_report.data_sensitivity_data_labels

    if isempty(S) || isempty(data_labels)
        return nothing
    end

    n_unknowns, n_data = size(S)

    # Validate dimension consistency between S and unknown labels
    expected_labels = length(sensitivity_report.data_sensitivity_unknown_labels)
    if expected_labels > 0 && expected_labels != n_unknowns
        @warn "[UQ] Label count mismatch: S has $n_unknowns rows but $expected_labels unknown labels — aborting UQ"
        return nothing
    end

    # Step 1: Fit UQ GPs for each non-trfn observable
    t_data = pep.data_sample["t"]
    uq_interps = Dict{String, AGPInterpolatorUQ}()
    obs_names = String[]
    obs_posterior_mean = Vector{Float64}[]
    obs_posterior_std = Vector{Float64}[]
    obs_cov_blocks = Dict{String, Matrix{Float64}}()
    warnings = String[]

    # Determine max derivative order needed from data labels
    # Data labels may use SIAN style ("y1_0", "y1_1") or Symbolics style
    # ("y1(t)", "Differential(t, 1)(y1(t))")
    max_deriv_needed = 0
    for dl in data_labels
        _, order = _parse_data_label(dl)
        max_deriv_needed = max(max_deriv_needed, order)
    end
    # SE kernel now supports arbitrary order via Hermite recurrence.
    # Cap at a reasonable maximum to prevent runaway computation.
    max_deriv_needed = min(max_deriv_needed, 8)

    for mq in pep.measured_quantities
        obs_name = replace(string(mq.lhs), r"\(.*\)" => "")
        startswith(obs_name, "_obs_trfn_") && continue

        obs_rhs = Symbolics.diff2term(mq.rhs)
        y_data = _get_observable_data(pep, obs_rhs)
        isnothing(y_data) && continue

        try
            # Reuse estimation interpolant if it's already an AGPInterpolatorUQ
            if haskey(setup_data.interpolants, obs_rhs) &&
               setup_data.interpolants[obs_rhs] isa AGPInterpolatorUQ
                interp_uq = setup_data.interpolants[obs_rhs]
            else
                interp_uq = agp_gpr_uq(Float64.(t_data), Float64.(y_data))
            end
            uq_interps[obs_name] = interp_uq

            jet = joint_derivative_estimator_covariance(
                interp_uq, t_eval, max_deriv_needed;
                observable_name = obs_name,
                noise_source = :learned_gp_homoscedastic,
            )
            μ = jet.mean
            Σ = jet.jet_covariance
            for w in jet.warnings
                push!(warnings, "Observable '$obs_name': $w")
            end
            # Warn on negative variance before clipping
            neg_diag = findall(d -> d < -1e-10, diag(Σ))
            if !isempty(neg_diag)
                @warn "[UQ] Negative estimator variance for '$obs_name' at indices $neg_diag (values: $(diag(Σ)[neg_diag])) — clipping to zero"
            end
            σ = sqrt.(max.(diag(Σ), 0.0))

            push!(obs_names, obs_name)
            push!(obs_posterior_mean, μ)
            push!(obs_posterior_std, σ)
            obs_cov_blocks[obs_name] = Σ
        catch e
            @warn "[UQ] GP fitting failed for observable $obs_name: $e"
        end
    end

    if isempty(obs_names)
        return nothing
    end

    # Step 2: Build Σ_d by mapping data_labels to estimator covariance entries
    # Build obs_name → index in obs_names
    obs_name_to_idx = Dict(obs_names[i] => i for i in eachindex(obs_names))

    Σ_d = zeros(n_data, n_data)
    for i in 1:n_data
        base_i, order_i = _parse_data_label(data_labels[i])
        if isempty(base_i)
            msg = "Unparseable data label '$(data_labels[i])' — skipped in Σ_d"
            push!(warnings, msg)
            @warn "[UQ] $msg"
            continue
        end

        # Find which observable this corresponds to
        obs_idx_i = _match_obs_name(base_i, obs_name_to_idx)
        if isnothing(obs_idx_i)
            msg = "Data label '$(data_labels[i])' (base='$base_i') has no matching observable — skipped in Σ_d"
            push!(warnings, msg)
            @warn "[UQ] $msg"
            continue
        end
        obs_name_i = obs_names[obs_idx_i]

        for j in 1:n_data
            base_j, order_j = _parse_data_label(data_labels[j])
            if isempty(base_j)
                continue  # already warned on the outer loop
            end

            # Must be same observable (block-diagonal assumption)
            obs_idx_j = _match_obs_name(base_j, obs_name_to_idx)
            isnothing(obs_idx_j) && continue
            obs_idx_i != obs_idx_j && continue

            # Look up covariance from the estimator-sampling covariance block
            if haskey(obs_cov_blocks, obs_name_i)
                Σ_block = obs_cov_blocks[obs_name_i]
                if order_i + 1 <= size(Σ_block, 1) && order_j + 1 <= size(Σ_block, 2)
                    Σ_d[i, j] = Σ_block[order_i + 1, order_j + 1]
                end
            end
        end
    end

    # Enforce PSD on Σ_d
    Σ_d = Symmetric(Σ_d)
    evals = eigvals(Σ_d)
    if minimum(evals) < 0
        Σ_d = Σ_d + Matrix{Float64}(I, n_data, n_data) * (abs(minimum(evals)) + 1e-15)
    end

    # Check: zero-variance data variables with nonzero sensitivity
    xlabels = if !isempty(sensitivity_report.data_sensitivity_unknown_labels)
        sensitivity_report.data_sensitivity_unknown_labels
    else
        sensitivity_report.jacobian_col_labels  # fallback for legacy reports
    end
    zero_diag = findall(d -> d == 0.0, diag(Σ_d))
    if !isempty(zero_diag)
        for idx in zero_diag
            sensitive_params = findall(s -> abs(s) > 1e-10, S[:, idx])
            if !isempty(sensitive_params)
                pnames = [xlabels[k] for k in sensitive_params]
                msg = "Zero estimator covariance for '$(data_labels[idx])' but parameters [$(join(pnames, ", "))] depend on it — their uncertainty is UNDERESTIMATED"
                push!(warnings, msg)
                @warn "[UQ] $msg"
            end
        end
    end

    # Step 3: Compute Σ_x = S · Σ_d · S'
    Σ_x = S * Matrix(Σ_d) * S'
    Σ_x = Symmetric(Σ_x)

    # Enforce PSD
    evals_x = eigvals(Σ_x)
    if minimum(evals_x) < 0
        Σ_x = Σ_x + Matrix{Float64}(I, n_unknowns, n_unknowns) * (abs(minimum(evals_x)) + 1e-15)
    end

    # Warn on negative variance before clipping
    neg_diag_x = findall(d -> d < -1e-10, diag(Σ_x))
    if !isempty(neg_diag_x)
        neg_labels = [xlabels[k] for k in neg_diag_x if k <= length(xlabels)]
        msg = "Negative variance for parameters [$(join(neg_labels, ", "))] (values: $(diag(Σ_x)[neg_diag_x])) — numerical breakdown, clipping to zero"
        push!(warnings, msg)
        @warn "[UQ] $msg"
    end

    param_std = sqrt.(max.(diag(Σ_x), 0.0))

    # Step 4: Build param labels and true values (from data sensitivity unknown labels)
    param_labels = if !isempty(sensitivity_report.data_sensitivity_unknown_labels)
        sensitivity_report.data_sensitivity_unknown_labels
    else
        sensitivity_report.jacobian_col_labels  # fallback for legacy reports
    end
    param_roles = if !isempty(sensitivity_report.data_sensitivity_unknown_roles)
        sensitivity_report.data_sensitivity_unknown_roles
    else
        sensitivity_report.jacobian_col_roles  # fallback for legacy reports
    end

    # Look up true values for each unknown
    param_true_values = Float64[]
    for label in param_labels
        val = NaN
        # Try matching against p_true
        for (p, v) in pep.p_true
            if replace(string(p), "(t)" => "") == label
                val = v
                break
            end
        end
        # Try matching against ic
        if isnan(val)
            for (s, v) in pep.ic
                if replace(string(s), "(t)" => "") == label
                    val = v
                    break
                end
            end
        end
        # Try parsed name for derivative variables
        if isnan(val)
            parsed = parse_derivative_variable_name(label)
            if !isnothing(parsed)
                base, order = parsed
                if order == 0
                    for (p, v) in pep.p_true
                        if replace(string(p), "(t)" => "") == base
                            val = v
                            break
                        end
                    end
                    if isnan(val)
                        for (s, v) in pep.ic
                            if replace(string(s), "(t)" => "") == base
                                val = v
                                break
                            end
                        end
                    end
                end
            end
        end
        push!(param_true_values, val)
    end

    # Step 5: Correlation matrix
    corr = zeros(n_unknowns, n_unknowns)
    for i in 1:n_unknowns
        for j in 1:n_unknowns
            si = param_std[i]
            sj = param_std[j]
            if si > 0 && sj > 0
                corr[i, j] = clamp(Matrix(Σ_x)[i, j] / (si * sj), -1.0, 1.0)
            elseif i == j
                corr[i, j] = 1.0
            end
        end
    end

    # Step 6: Quality classification
    max_cv = _uq_max_cv(param_std, param_true_values)
    status = _uq_status_from_cv(max_cv)

    ia = compute_practical_identifiability_index(
        Matrix(Σ_x), param_labels, param_roles, param_true_values,
    )

    return UncertaintyReport(
        pep.name, t_eval,
        obs_names, obs_posterior_mean, obs_posterior_std,
        Matrix(Σ_d), data_labels,
        Matrix(Σ_x), param_std, param_labels, param_roles, param_true_values,
        corr,
        max_cv, status, warnings,
        :estimator_sampling, :learned_gp_homoscedastic, ia,
    ), uq_interps
end


# ─── Backsolve Uncertainty Propagation ─────────────────────────────────

_uq_clean_name(x) = replace(string(x), "(t)" => "")

function _uq_name_index(vars)
    return Dict(_uq_clean_name(v) => i for (i, v) in enumerate(vars))
end

function _uq_ordered_result_values(result_dict, vars; default = NaN)
    values = fill(Float64(default), length(vars))
    by_name = Dict(_uq_clean_name(k) => Float64(v) for (k, v) in result_dict)
    for (i, v) in enumerate(vars)
        values[i] = get(by_name, _uq_clean_name(v), Float64(default))
    end
    return values
end

function _uq_ordered_rhs_expressions(sys, states)
    t_var = ModelingToolkit.get_iv(sys)
    D_var = Differential(t_var)
    eqs = ModelingToolkit.equations(sys)
    rhs_by_state = Dict{String, Any}()

    for eq in eqs
        lhs_str = string(eq.lhs)
        for s in states
            if lhs_str == string(D_var(s))
                rhs_by_state[_uq_clean_name(s)] = eq.rhs
                break
            end
        end
    end

    if length(rhs_by_state) == length(states)
        return [rhs_by_state[_uq_clean_name(s)] for s in states]
    end

    if length(eqs) == length(states)
        return [eq.rhs for eq in eqs]
    end

    error("Could not align ODE RHS expressions to state order for variational backsolve UQ")
end

function _uq_build_rhs_eval(sys)
    t_var = ModelingToolkit.get_iv(sys)
    states = ModelingToolkit.unknowns(sys)
    params = ModelingToolkit.parameters(sys)
    rhs_exprs = _uq_ordered_rhs_expressions(sys, states)

    built = ModelingToolkit.build_function(rhs_exprs, states, params, t_var; expression = Val(false))
    f_raw = isa(built, Tuple) ? built[1] : built

    return function rhs_eval(x, p, tt)
        vals = f_raw(x, p, tt)
        return collect(vals)
    end
end

function _uq_state_values_at_time(
    pep::ParameterEstimationProblem,
    best_result::ParameterEstimationResult,
    completed_sys,
    completed_states,
    completed_params,
    completed_param_values::Vector{Float64},
    t0::Float64,
    t_shoot::Float64,
)
    state_values = try
        if !isnothing(best_result.solution)
            Float64[Float64(real(best_result.solution(t_shoot)[i])) for i in eachindex(completed_states)]
        else
            Float64[]
        end
    catch
        Float64[]
    end
    length(state_values) == length(completed_states) && return state_values

    ic_values = _uq_ordered_result_values(best_result.states, completed_states)
    if any(!isfinite, ic_values) || any(!isfinite, completed_param_values)
        error("Cannot compute shooting-point states for backsolve UQ: missing estimated states or parameters")
    end

    prob = ODEProblem(
        completed_sys,
        merge(Dict(completed_states .=> ic_values), Dict(completed_params .=> completed_param_values)),
        (t0, t_shoot),
    )
    sol = OrdinaryDiffEq.solve(prob, AutoVern9(Rodas5P()); abstol = 1e-12, reltol = 1e-12)
    SciMLBase.successful_retcode(sol) ||
        error("Forward solve to shooting point failed with retcode $(sol.retcode)")
    return Float64[Float64(real(sol(t_shoot)[i])) for i in eachindex(completed_states)]
end

function _uq_variational_backsolve_jacobian(
    pep::ParameterEstimationProblem,
    best_result::ParameterEstimationResult,
    t0::Float64,
    t_shoot::Float64,
)
    completed_sys = ModelingToolkit.complete(pep.model.system)
    completed_states = ModelingToolkit.unknowns(completed_sys)
    completed_params = ModelingToolkit.parameters(completed_sys)
    original_states = pep.model.original_states
    original_params = pep.model.original_parameters

    completed_param_values = _uq_ordered_result_values(best_result.parameters, completed_params)
    if any(!isfinite, completed_param_values)
        error("Cannot compute variational backsolve UQ: missing estimated parameters")
    end

    x_shoot = _uq_state_values_at_time(
        pep, best_result, completed_sys, completed_states, completed_params,
        completed_param_values, t0, t_shoot,
    )

    rhs_eval = _uq_build_rhs_eval(completed_sys)
    n_x = length(completed_states)
    n_p = length(completed_params)
    n_A = n_x * n_x
    n_B = n_x * n_p
    A0 = Matrix{Float64}(I, n_x, n_x)
    B0 = zeros(n_x, n_p)
    u_ext0 = vcat(x_shoot, vec(A0), vec(B0))

    function variational_rhs!(du, u, _, tt)
        T = eltype(u)
        x = collect(@view u[1:n_x])
        p = T.(completed_param_values)
        A = reshape(@view(u[(n_x + 1):(n_x + n_A)]), n_x, n_x)
        B = reshape(@view(u[(n_x + n_A + 1):(n_x + n_A + n_B)]), n_x, n_p)

        fx = ForwardDiff.jacobian(xx -> rhs_eval(xx, p, tt), x)
        fp = n_p == 0 ? zeros(T, n_x, 0) : ForwardDiff.jacobian(pp -> rhs_eval(x, pp, tt), p)
        dx = rhs_eval(x, p, tt)
        dA = fx * A
        dB = fx * B + fp

        du[1:n_x] .= dx
        du[(n_x + 1):(n_x + n_A)] .= vec(dA)
        if n_B > 0
            du[(n_x + n_A + 1):(n_x + n_A + n_B)] .= vec(dB)
        end
        return nothing
    end

    prob = ODEProblem(variational_rhs!, u_ext0, (t_shoot, t0), nothing)
    sol = OrdinaryDiffEq.solve(
        prob, AutoVern9(Rodas5P());
        abstol = 1e-12, reltol = 1e-12, save_everystep = false,
    )
    SciMLBase.successful_retcode(sol) ||
        error("Variational backsolve failed with retcode $(sol.retcode)")

    u_final = sol.u[end]
    x_t0 = Vector{Float64}(u_final[1:n_x])
    A_t0 = Matrix{Float64}(reshape(u_final[(n_x + 1):(n_x + n_A)], n_x, n_x))
    B_t0 = Matrix{Float64}(reshape(u_final[(n_x + n_A + 1):(n_x + n_A + n_B)], n_x, n_p))

    state_index = _uq_name_index(completed_states)
    param_index = _uq_name_index(completed_params)
    original_state_indices = [state_index[_uq_clean_name(s)] for s in original_states]
    original_param_indices = [param_index[_uq_clean_name(p)] for p in original_params]

    J = hcat(B_t0[original_state_indices, original_param_indices],
             A_t0[original_state_indices, original_state_indices])
    return J, x_t0[original_state_indices]
end

function _uq_correlation_matrix(Σ::AbstractMatrix{<:Real}, σ::Vector{Float64})
    n = length(σ)
    corr = zeros(n, n)
    for i in 1:n
        for j in 1:n
            if σ[i] > 0 && σ[j] > 0
                corr[i, j] = clamp(Float64(Σ[i, j]) / (σ[i] * σ[j]), -1.0, 1.0)
            elseif i == j
                corr[i, j] = 1.0
            end
        end
    end
    return corr
end

function _uq_matrix_amplification(M::AbstractMatrix)
    isempty(M) && return NaN
    try
        svs = svdvals(M)
        isempty(svs) ? NaN : maximum(svs)
    catch
        NaN
    end
end

function _uq_status_from_cv(max_cv::Float64)
    return if !isfinite(max_cv)
        :degenerate
    elseif max_cv < 0.5
        :ok
    elseif max_cv < 2.0
        :wide_ci
    else
        :degenerate
    end
end

function _uq_max_cv(param_std::Vector{Float64}, values::Vector{Float64}; scale_floor::Float64 = 1e-8)
	max_cv = isempty(param_std) ? NaN : 0.0
	for i in eachindex(param_std)
		i <= length(values) || return Inf
		v = values[i]
		isfinite(v) || return Inf
		max_cv = max(max_cv, param_std[i] / max(abs(v), scale_floor))
	end
	return max_cv
end

function _uq_lookup_value(result_dict, truth_dict, name::String)
    for (k, v) in result_dict
        _uq_clean_name(k) == name && return Float64(v)
    end
    for (k, v) in truth_dict
        _uq_clean_name(k) == name && return Float64(v)
    end
    return NaN
end

function _uq_state_values_at_eval_by_name(
    pep::ParameterEstimationProblem,
    best_result::ParameterEstimationResult,
    t_eval::Float64,
)
    completed_sys = ModelingToolkit.complete(pep.model.system)
    completed_states = ModelingToolkit.unknowns(completed_sys)
    completed_params = ModelingToolkit.parameters(completed_sys)
    completed_param_values = _uq_ordered_result_values(best_result.parameters, completed_params)
    if any(!isfinite, completed_param_values)
        return Dict{String, Float64}()
    end

    t0 = Float64(pep.data_sample["t"][1])
    values = try
        _uq_state_values_at_time(
            pep, best_result, completed_sys, completed_states, completed_params,
            completed_param_values, t0, t_eval,
        )
    catch
        Float64[]
    end
    length(values) == length(completed_states) || return Dict{String, Float64}()
    return Dict(_uq_clean_name(s) => Float64(values[i]) for (i, s) in enumerate(completed_states))
end

function _uq_local_snapshot(
    pep::ParameterEstimationProblem,
    best_result::ParameterEstimationResult,
    uq::UncertaintyReport,
)
    local_roles = Dict{String, Symbol}()
    state_values = _uq_state_values_at_eval_by_name(pep, best_result, uq.t_eval)
    coordinate_values = Float64[]

    for label in uq.param_labels
        role = get(uq.param_roles, label, :unknown)
        parsed = parse_derivative_variable_name(label)
        if role == :state_ic
            role = :state_at_eval
        end
        local_roles[label] = role

        value = NaN
        if role == :parameter
            base = isnothing(parsed) ? label : String(parsed[1])
            value = _uq_lookup_value(best_result.parameters, pep.p_true, base)
        elseif role == :state_at_eval
            if !isnothing(parsed) && parsed[2] == 0
                value = get(state_values, String(parsed[1]), NaN)
            else
                value = get(state_values, label, NaN)
            end
        end
        push!(coordinate_values, value)
    end

    return LocalUQSnapshot(
        uq.t_eval,
        copy(uq.param_covariance),
        copy(uq.param_std),
        copy(uq.param_labels),
        local_roles,
        coordinate_values,
        copy(uq.correlation_matrix),
        uq.max_cv,
        uq.status,
        nothing,
    )
end

function _uq_physical_truth_values(
    pep::ParameterEstimationProblem,
    best_result::ParameterEstimationResult,
    labels::Vector{String},
    roles::Dict{String, Symbol},
)
    values = Float64[]
    for label in labels
        role = get(roles, label, :unknown)
        if role == :parameter
            push!(values, _uq_lookup_value(OrderedDict{Any, Any}(), pep.p_true, label))
        elseif role == :state_ic
            push!(values, _uq_lookup_value(OrderedDict{Any, Any}(), pep.ic, label))
        else
            push!(values, NaN)
        end
    end
    return values
end

function _uq_physical_estimate_values(
    best_result::ParameterEstimationResult,
    labels::Vector{String},
    roles::Dict{String, Symbol},
)
    values = Float64[]
    empty_truth = OrderedDict{Any, Any}()
    for label in labels
        role = get(roles, label, :unknown)
        if role == :parameter
            push!(values, _uq_lookup_value(best_result.parameters, empty_truth, label))
        elseif role == :state_ic
            push!(values, _uq_lookup_value(best_result.states, empty_truth, label))
        else
            push!(values, NaN)
        end
    end
    return values
end

function _uq_source_covariance(
    uq::UncertaintyReport,
    source_labels::Vector{String},
    warnings::Vector{String},
)
    n = length(source_labels)
    Σ = zeros(n, n)
    source_to_uq = Vector{Union{Nothing, Int}}(undef, n)
    for (i, label) in enumerate(source_labels)
        idx = _find_uq_param_index(label, uq)
        source_to_uq[i] = idx
        if isnothing(idx)
            push!(warnings, "Physical UQ source coordinate '$label' was not present in the local IFT covariance; its covariance entries were set to zero.")
        end
    end

    used = Set{Int}()
    for i in 1:n
        ii = source_to_uq[i]
        isnothing(ii) && continue
        push!(used, ii)
        for j in 1:n
            jj = source_to_uq[j]
            isnothing(jj) && continue
            Σ[i, j] = uq.param_covariance[ii, jj]
        end
    end

    dropped = [uq.param_labels[i] for i in eachindex(uq.param_labels) if !(i in used)]
    if !isempty(dropped)
        preview = length(dropped) <= 8 ? join(dropped, ", ") : join(dropped[1:8], ", ") * ", ..."
        push!(warnings, "Physical UQ projected out $(length(dropped)) local helper/derivative coordinate(s): $preview.")
    end
    return _psd_symmetric_matrix(Σ)
end

function _uq_failed_physicalization_report(
    uq::UncertaintyReport,
    local_snapshot::Union{Nothing, LocalUQSnapshot},
    msg::String,
)
    warnings = vcat(uq.warnings, ["Physical-coordinate UQ transform failed: $msg"])
    return UncertaintyReport(
        uq.model_name, uq.t_eval,
        uq.obs_names, uq.obs_posterior_mean, uq.obs_posterior_std,
        uq.data_covariance, uq.data_labels,
        uq.param_covariance, uq.param_std, uq.param_labels, uq.param_roles, uq.param_true_values,
        uq.correlation_matrix, uq.max_cv, :failed, warnings,
        uq.covariance_kind, uq.noise_source, uq.practical_identifiability_index,
        :local_at_eval, local_snapshot, nothing,
        uq.target, copy(uq.estimate_values), uq.linearization_diagnostics,
    )
end

"""
    physicalize_uncertainty_report(pep, best_result, local_uq) -> UncertaintyReport

Convert the direct IFT UQ report from local coordinates `[p, x(t_eval)]` to
the public physical coordinates `[p, x(t0)]`.
"""
function physicalize_uncertainty_report(
    pep::ParameterEstimationProblem,
    best_result::ParameterEstimationResult,
    local_uq::UncertaintyReport,
)
    local_snapshot = _uq_local_snapshot(pep, best_result, local_uq)

    try
        t0 = Float64(pep.data_sample["t"][1])
        t_eval = Float64(local_uq.t_eval)
        t_best = Float64(best_result.at_time)
        warnings = copy(local_uq.warnings)
        if abs(t_eval - t_best) > 1e-8
            push!(warnings, "UQ t_eval=$t_eval differs from best result at_time=$t_best; physicalization used UQ t_eval.")
        end

        params = pep.model.original_parameters
        states = pep.model.original_states
        real_indices = [i for (i, s) in enumerate(states)
                        if !startswith(_uq_clean_name(s), "_trfn_")]
        real_states = states[real_indices]
        n_params = length(params)
        n_states = length(states)
        n_real = length(real_states)

        source_labels = vcat([_uq_clean_name(p) for p in params],
                             [_uq_clean_name(s) for s in states])
        target_labels = vcat([_uq_clean_name(p) for p in params],
                             [_uq_clean_name(s) for s in real_states])

        source_roles = Dict{String, Symbol}()
        for p in params
            source_roles[_uq_clean_name(p)] = :parameter
        end
        for s in states
            name = _uq_clean_name(s)
            source_roles[name] = startswith(name, "_trfn_") ? :transcendental : :state_at_eval
        end

        target_roles = Dict{String, Symbol}()
        for p in params
            target_roles[_uq_clean_name(p)] = :parameter
        end
        for s in real_states
            target_roles[_uq_clean_name(s)] = :state_ic
        end

        transform = zeros(n_params + n_real, n_params + n_states)
        for i in 1:n_params
            transform[i, i] = 1.0
        end

        transform_status = :identity
        if abs(t_eval - t0) < 1e-10
            for (out_i, state_i) in enumerate(real_indices)
                transform[n_params + out_i, n_params + state_i] = 1.0
            end
        else
            J_state, _ = _uq_variational_backsolve_jacobian(pep, best_result, t0, t_eval)
            transform[(n_params + 1):end, :] .= J_state[real_indices, :]
            transform_status = :variational
        end

        source_cov = _uq_source_covariance(local_uq, source_labels, warnings)
        Σ_phys = _psd_symmetric_matrix(transform * source_cov * transform')
        param_std = sqrt.(max.(diag(Σ_phys), 0.0))
        truth_values = _uq_physical_truth_values(pep, best_result, target_labels, target_roles)
        estimate_values = _uq_physical_estimate_values(best_result, target_labels, target_roles)
        corr = _uq_correlation_matrix(Σ_phys, param_std)
        max_cv = _uq_max_cv(param_std, estimate_values)
        status = _uq_status_from_cv(max_cv)
        ia = compute_practical_identifiability_index(
            Matrix(Σ_phys), target_labels, target_roles, estimate_values,
        )
        append!(warnings, ia.warnings)

        amplification = _uq_matrix_amplification(transform)
        backsolve = UQBacksolveTransform(
            t_eval, t0, source_labels, target_labels,
            source_roles, target_roles, transform,
            amplification, transform_status, warnings,
        )

        return UncertaintyReport(
            local_uq.model_name, local_uq.t_eval,
            local_uq.obs_names, local_uq.obs_posterior_mean, local_uq.obs_posterior_std,
            local_uq.data_covariance, local_uq.data_labels,
            Matrix(Σ_phys), param_std, target_labels, target_roles, truth_values,
            corr, max_cv, status, warnings,
            local_uq.covariance_kind, local_uq.noise_source, ia,
            :physical_initial_conditions, local_snapshot, backsolve,
            local_uq.target, estimate_values, local_uq.linearization_diagnostics,
        )
    catch e
        return _uq_failed_physicalization_report(local_uq, local_snapshot, sprint(showerror, e))
    end
end

function _uq_scale_for_name(scales, label::AbstractString)
    base = label
    parsed = parse_derivative_variable_name(String(label))
    if !isnothing(parsed)
        base = String(parsed[1])
    end
    for (sym, scale) in scales
        if _uq_clean_name(sym) == base
            return Float64(scale)
        end
    end
    return 1.0
end

function _uq_scale_factor(label::AbstractString, role::Symbol, info::ScaleInfo)
    if role == :parameter
        return _uq_scale_for_name(info.param_scales, label)
    elseif role in (:state_ic, :state_at_eval, :state_derivative, :transcendental)
        return _uq_scale_for_name(info.state_scales, label)
    else
        return 1.0
    end
end

function _uq_scaled_values(values::Vector{Float64}, factors::Vector{Float64})
    out = copy(values)
    for i in eachindex(out)
        if i <= length(factors) && isfinite(out[i])
            out[i] *= factors[i]
        end
    end
    return out
end

function _uq_rescale_snapshot(
    snapshot::LocalUQSnapshot,
    info::ScaleInfo,
)
    factors = [
        _uq_scale_factor(label, get(snapshot.param_roles, label, :unknown), info)
        for label in snapshot.param_labels
    ]
    D = Diagonal(factors)
    Σ = _psd_symmetric_matrix(D * snapshot.param_covariance * D)
    σ = sqrt.(max.(diag(Σ), 0.0))
    values = _uq_scaled_values(snapshot.coordinate_values, factors)
    corr = _uq_correlation_matrix(Σ, σ)
    max_cv = _uq_max_cv(σ, values)
    status = snapshot.status in (:degenerate, :failed) ? snapshot.status : _uq_status_from_cv(max_cv)

    return LocalUQSnapshot(
        snapshot.t_eval,
        Matrix(Σ),
        σ,
        copy(snapshot.param_labels),
        copy(snapshot.param_roles),
        values,
        corr,
        max_cv,
        status,
        snapshot.practical_identifiability_index,
    )
end

"""
    unrescale_uncertainty_report(uq, info) -> UncertaintyReport

Convert a UQ report produced inside the automatic power-of-two rescaled problem
back to original parameter/state units. The estimation pipeline computes UQ
before `unrescale_results`, because the solver trajectory and local algebraic
coordinates are still in the scaled problem at that point.
"""
function unrescale_uncertainty_report(
    uq::Union{Nothing, AbstractUQOutcome},
    info::ScaleInfo,
)
    isnothing(uq) && return nothing
    uq isa UQUnavailable && return uq
    uq isa UncertaintyReport || return uq

    factors = [
        _uq_scale_factor(label, get(uq.param_roles, label, :unknown), info)
        for label in uq.param_labels
    ]
    D = Diagonal(factors)
    Σ = _psd_symmetric_matrix(D * uq.param_covariance * D)
    σ = sqrt.(max.(diag(Σ), 0.0))
    truth_values = _uq_scaled_values(uq.param_true_values, factors)
    estimate_values = isempty(uq.estimate_values) ?
        _uq_scaled_values(uq.param_true_values, factors) :
        _uq_scaled_values(uq.estimate_values, factors)
    corr = _uq_correlation_matrix(Σ, σ)
    max_cv = _uq_max_cv(σ, estimate_values)
    status = (uq.status in (:degenerate, :failed) || uq.linearization_diagnostics.degraded) ?
        :degenerate : _uq_status_from_cv(max_cv)
    warnings = copy(uq.warnings)
    if any(!=(1.0), factors)
        push!(warnings, "UQ covariance and reported values were converted from internal power-of-two rescaled coordinates to original units.")
    end

    local_snapshot = isnothing(uq.local_coordinate_report) ?
        nothing :
        _uq_rescale_snapshot(uq.local_coordinate_report, info)

    backsolve = uq.backsolve_transform
    if !isnothing(backsolve)
        source_factors = [
            _uq_scale_factor(label, get(backsolve.source_roles, label, :unknown), info)
            for label in backsolve.source_labels
        ]
        target_factors = [
            _uq_scale_factor(label, get(backsolve.target_roles, label, :unknown), info)
            for label in backsolve.target_labels
        ]
        D_source_inv = Diagonal(1.0 ./ source_factors)
        D_target = Diagonal(target_factors)
        transform = Matrix(D_target * backsolve.transform_matrix * D_source_inv)
        amplification = _uq_matrix_amplification(transform)
        backsolve = UQBacksolveTransform(
            backsolve.t_eval,
            backsolve.t0,
            copy(backsolve.source_labels),
            copy(backsolve.target_labels),
            copy(backsolve.source_roles),
            copy(backsolve.target_roles),
            transform,
            amplification,
            backsolve.status,
            warnings,
        )
    end

    ia = compute_practical_identifiability_index(
        Matrix(Σ), copy(uq.param_labels), copy(uq.param_roles), estimate_values,
    )
    append!(warnings, ia.warnings)

    return UncertaintyReport(
        uq.model_name,
        uq.t_eval,
        copy(uq.obs_names),
        deepcopy(uq.obs_posterior_mean),
        deepcopy(uq.obs_posterior_std),
        copy(uq.data_covariance),
        copy(uq.data_labels),
        Matrix(Σ),
        σ,
        copy(uq.param_labels),
        copy(uq.param_roles),
        truth_values,
        corr,
        max_cv,
        status,
        warnings,
        uq.covariance_kind,
        uq.noise_source,
        ia,
        uq.coordinate_system,
        local_snapshot,
        backsolve,
        uq.target,
        estimate_values,
        uq.linearization_diagnostics,
    )
end

"""
    propagate_backsolve_uncertainty(pep, best_result, uq_report) → BacksolveUQReport

Propagate parameter uncertainty from the shooting point through the backward
ODE integration to initial conditions at t₀ using the delta method:

    Σ_{s(t₀)} = J_g · Σ_{p, s(t_eval)} · J_g'

where J_g = ∂g/∂(p, s(t_eval)) is computed by integrating the variational
equations backward along the estimated trajectory.
"""
function propagate_backsolve_uncertainty(pep::ParameterEstimationProblem,
    best_result::ParameterEstimationResult,
    uq_report::UncertaintyReport)

    t_data = pep.data_sample["t"]
    t0 = t_data[1]
    t_shoot = best_result.at_time

    # If estimation was at t0, no backsolve needed — return actual values
    if abs(t_shoot - t0) < 1e-10
        real_states = [s for s in pep.model.original_states
                        if !startswith(replace(string(s), "(t)" => ""), "_trfn_")]
        ic_names = [replace(string(s), "(t)" => "") for s in real_states]
        n_ic = length(ic_names)

        # Get actual estimated ICs
        ic_est = Float64[]
        for s in real_states
            s_name = replace(string(s), "(t)" => "")
            found = false
            for (es, ev) in best_result.states
                if replace(string(es), "(t)" => "") == s_name
                    push!(ic_est, ev); found = true; break
                end
            end
            found || push!(ic_est, NaN)
        end

        # Get true ICs
        ic_true = [get(pep.ic, s, NaN) for s in real_states]

        # At t0, uncertainty comes directly from UQ (no backsolve propagation)
        ic_std = Float64[]
        for s_name in ic_names
            push!(ic_std, _find_uq_sigma(s_name, uq_report))
        end
        # Replace NaN with 0 for missing entries
        ic_std = [isnan(s) ? 0.0 : s for s in ic_std]

        ic_covers = [ic_std[i] > 0 ? abs(ic_true[i] - ic_est[i]) < UQ_CI_Z * ic_std[i] : true
                     for i in 1:n_ic]

        return BacksolveUQReport(
            t_shoot, t0, ic_names, ic_est, ic_true, ic_std,
            ic_covers, Matrix{Float64}(undef, 0, 0), 1.0, true,
        )
    end

    params = pep.model.original_parameters
    states = pep.model.original_states

    # Filter out _trfn_ states
    real_states = [s for s in states if !startswith(replace(string(s), "(t)" => ""), "_trfn_")]
    ic_names = [replace(string(s), "(t)" => "") for s in real_states]
    n_params = length(params)
    n_states = length(real_states)

    # Extract estimated state ICs for fallback reporting if backsolve propagation fails.
    est_ics = Float64[]
    for s in real_states
        s_name = replace(string(s), "(t)" => "")
        found = false
        for (es, ev) in best_result.states
            if replace(string(es), "(t)" => "") == s_name
                push!(est_ics, ev)
                found = true
                break
            end
        end
        found || push!(est_ics, NaN)
    end

    # True ICs for coverage check
    ic_true = Float64[]
    for s in real_states
        push!(ic_true, get(pep.ic, s, NaN))
    end

    real_indices = [i for (i, s) in enumerate(states)
                    if !startswith(replace(string(s), "(t)" => ""), "_trfn_")]

    # Compute J_g = ∂s(t0)/∂(p, s(t_shoot)) using variational equations.
    J_g, backsolved_states = try
        _uq_variational_backsolve_jacobian(pep, best_result, t0, t_shoot)
    catch e
        @warn "[BACKSOLVE_UQ] Variational backsolve Jacobian failed: $e"
        return BacksolveUQReport(t_shoot, t0, ic_names, est_ics, ic_true,
            fill(NaN, n_states), fill(false, n_states),
            Matrix{Float64}(undef, 0, 0), NaN, false)
    end

    # Extract the sub-block of Σ_x corresponding to [params, order-0 states]
    # from the UQ report. UQ param_labels include derivative variables — we only
    # want params and order-0 state ICs.
    n_theta = n_params + length(states)
    Σ_sub = zeros(n_theta, n_theta)

    # Build mapping: θ index → UQ param index
    theta_labels = String[]
    for p in params
        push!(theta_labels, replace(string(p), "(t)" => ""))
    end
    for s in states
        push!(theta_labels, replace(string(s), "(t)" => ""))
    end

    for i in 1:n_theta
        for j in 1:n_theta
            # Find matching indices in UQ param_labels
            ui = _find_uq_param_index(theta_labels[i], uq_report)
            uj = _find_uq_param_index(theta_labels[j], uq_report)
            if !isnothing(ui) && !isnothing(uj)
                Σ_sub[i, j] = uq_report.param_covariance[ui, uj]
            end
        end
    end

    # Propagate: Σ_ic = J_g · Σ_sub · J_g'
    Σ_ic = J_g * Σ_sub * J_g'

    ic_std = [sqrt(max(Σ_ic[i, i], 0.0)) for i in real_indices]
    ic_estimated = [backsolved_states[i] for i in real_indices]

    ic_ci_covers = [abs(ic_true[i] - ic_estimated[i]) < UQ_CI_Z * ic_std[i] for i in 1:n_states]

    # Amplification = max singular value of J_g
    amplification = _uq_matrix_amplification(J_g[real_indices, :])

    return BacksolveUQReport(
        t_shoot, t0, ic_names, ic_estimated, ic_true,
        ic_std, ic_ci_covers,
        J_g[real_indices, :], amplification, true,
    )
end

"""Find the index of a variable name in the UQ report's param_labels."""
function _find_uq_param_index(name::String, uq::UncertaintyReport)
    for (i, label) in enumerate(uq.param_labels)
        if label == name
            return i
        end
        # Also try parsed base name for derivative vars like "x1_0"
        parsed = parse_derivative_variable_name(label)
        if !isnothing(parsed)
            base, order = parsed
            if order == 0 && String(base) == name
                return i
            end
        end
    end
    return nothing
end

# ─── UQ Name-Matching Helper ─────────────────────────────────────────

"""
    _find_uq_sigma(name, uq) → Float64

Look up the standard deviation for variable `name` in the UQ report,
trying exact match first, then SIAN-style `_0` suffix match (e.g. `a_0` → `a`).
Returns `NaN` if no match found.
"""
function _find_uq_sigma(name::String, uq::UncertaintyReport)
    # Exact match
    for j in eachindex(uq.param_labels)
        if uq.param_labels[j] == name
            return uq.param_std[j]
        end
    end
    # SIAN-style match: "a_0" matches "a"
    for j in eachindex(uq.param_labels)
        parsed = parse_derivative_variable_name(uq.param_labels[j])
        if !isnothing(parsed)
            base, order = parsed
            if order == 0 && String(base) == name
                return uq.param_std[j]
            end
        end
    end
    return NaN
end

# ─── Estimation Results HTML Section ──────────────────────────────────

"""
Write the estimation results section to the HTML report.
Shows a comparison table of true vs estimated values with CI coverage.
"""
function _write_html_estimation_section(io, est::EstimationResultsReport;
    uq::Union{Nothing, UncertaintyReport} = nothing)

    println(io, "<details open><summary>Estimation Results</summary><div class=\"detail-body\">")

    # Summary line
    err_badge = est.best_error < 0.01 ? """<span class="badge badge-easy">Low Error</span>""" :
        est.best_error < 0.10 ? """<span class="badge badge-moderate">Moderate Error</span>""" :
        """<span class="badge badge-hard">High Error</span>"""
    println(io, """<div class="provenance">Best error: $(_fmt(est.best_error)) &middot; $(est.n_results) solution(s) &middot; $(round(est.estimation_time_seconds; digits=1))s $err_badge</div>""")

    has_uq = !isnothing(uq) && !isempty(uq.param_std)

    # Parameters table
    if !isempty(est.param_comparison)
        println(io, "<h4>Parameters</h4>")
        println(io, "<table><tr><th>Parameter</th><th>True</th><th>Estimated</th><th>Rel Error</th>")
        if has_uq
            println(io, "<th>95% CI</th><th>Coverage</th>")
        end
        println(io, "</tr>")

        for pc in est.param_comparison
            is_unident = pc.is_unidentifiable
            err_cls = is_unident ? "" : pc.rel_error < 0.01 ? "err-ok" : pc.rel_error < 0.10 ? "err-warn" : "err-bad"
            pretty = _pretty_name(pc.name)
            raw_esc = replace(pc.name, "&" => "&amp;", "<" => "&lt;", ">" => "&gt;")

            unident_badge = is_unident ? """ <span class="badge badge-moderate">unident.</span>""" : ""
            row_style = is_unident ? " style=\"opacity:0.6;\"" : ""
            print(io, """<tr$row_style><td><span title="$raw_esc" class="math" style="font-weight:600;">$pretty</span>$unident_badge</td>""")
            print(io, "<td>$(_fmt(pc.true_val))</td>")
            print(io, "<td>$(_fmt(pc.est_val))</td>")
            print(io, """<td class="$err_cls">$(_fmt_pct(pc.rel_error))</td>""")

            if has_uq
                if is_unident
                    # Skip CI for unidentifiable parameters
                    print(io, "<td>—</td><td>—</td>")
                else
                    σ = _find_uq_sigma(pc.name, uq)
                    if isfinite(σ)
                        ci_lo = pc.est_val - UQ_CI_Z * σ
                        ci_hi = pc.est_val + UQ_CI_Z * σ
                        within = abs(pc.true_val - pc.est_val) < UQ_CI_Z * σ
                        cov_mark = within ? """<span style="color:var(--easy);">✓</span>""" :
                            """<span style="color:var(--hard);">✗</span>"""
                        print(io, "<td>[$(_fmt(ci_lo)), $(_fmt(ci_hi))]</td><td>$cov_mark</td>")
                    else
                        print(io, "<td>—</td><td>—</td>")
                    end
                end
            end
            println(io, "</tr>")
        end
        println(io, "</table>")
    end

    # States (ICs) table — with CI columns when UQ is available
    if !isempty(est.state_comparison)
        println(io, "<h4>Initial Conditions</h4>")
        println(io, "<table><tr><th>State</th><th>True</th><th>Estimated</th><th>Rel Error</th>")
        if has_uq
            println(io, "<th>95% CI</th><th>Coverage</th>")
        end
        println(io, "</tr>")

        for sc in est.state_comparison
            err_cls = sc.rel_error < 0.01 ? "err-ok" : sc.rel_error < 0.10 ? "err-warn" : "err-bad"
            pretty = _pretty_name(sc.name)
            raw_esc = replace(sc.name, "&" => "&amp;", "<" => "&lt;", ">" => "&gt;")

            print(io, """<tr><td><span title="$raw_esc" class="math" style="font-weight:600;">$pretty</span>(0)</td>""")
            print(io, "<td>$(_fmt(sc.true_val))</td>")
            print(io, "<td>$(_fmt(sc.est_val))</td>")
            print(io, """<td class="$err_cls">$(_fmt_pct(sc.rel_error))</td>""")

            if has_uq
                σ = _find_uq_sigma(sc.name, uq)
                if isfinite(σ)
                    ci_lo = sc.est_val - UQ_CI_Z * σ
                    ci_hi = sc.est_val + UQ_CI_Z * σ
                    within = abs(sc.true_val - sc.est_val) < UQ_CI_Z * σ
                    cov_mark = within ? """<span style="color:var(--easy);">✓</span>""" :
                        """<span style="color:var(--hard);">✗</span>"""
                    print(io, "<td>[$(_fmt(ci_lo)), $(_fmt(ci_hi))]</td><td>$cov_mark</td>")
                else
                    print(io, "<td>—</td><td>—</td>")
                end
            end
            println(io, "</tr>")
        end
        println(io, "</table>")
    end

    # Cross-solution spread (practical identifiability)
    if !isnothing(est.spread) && est.spread.n_solutions > 1
        _write_html_spread_section(io, est.spread)
    end

    println(io, "</div></details>")
end

"""
Write the cross-solution spread table showing practical identifiability.
"""
function _write_html_spread_section(io, spread::CrossSolutionSpread)
    println(io, "<details open><summary>Practical Identifiability (Cross-Solution Spread, N=$(spread.n_solutions))</summary><div class=\"detail-body\">")
    println(io, """<div class="provenance"><b>What this shows:</b> How much each parameter/state varies across all $(spread.n_solutions) HC solutions from different interpolators and shooting points. Small CV (coefficient of variation) = practically identifiable at this noise level. Large CV = practically non-identifiable — multiple parameter values fit the data equally well.<br><b>Classification:</b> <span class="err-ok">tight</span> (CV &lt; 5%), <span class="err-warn">moderate</span> (5–50%), <span class="err-bad">loose</span> (CV &gt; 50%).</div>""")

    for (label, entries) in [("Parameters", spread.param_spread), ("Initial Conditions", spread.state_spread)]
        isempty(entries) && continue
        println(io, "<h4>$label</h4>")
        println(io, "<table><tr><th>Name</th><th>True</th><th>Median</th><th>CV</th><th>IQR</th><th>Range</th><th>N</th><th>Class</th></tr>")
        for e in entries
            cls_str = e.classification == :tight ? "tight" :
                      e.classification == :moderate ? "moderate" :
                      e.classification == :loose ? "LOOSE" :
                      e.classification == :unidentifiable ? "unident." : "?"
            cls_css = e.classification == :tight ? "err-ok" :
                      e.classification == :moderate ? "err-warn" : "err-bad"
            cv_str = isinf(e.cv) ? "∞" : isnan(e.cv) ? "—" : @sprintf("%.1f%%", e.cv * 100)
            row_style = e.is_unidentifiable ? " style=\"opacity:0.6;\"" : ""
            pretty = _pretty_name(e.name)

            println(io, """<tr$row_style><td class="math" style="font-weight:600;">$pretty</td>""" *
                "<td>$(_fmt(e.true_val))</td>" *
                "<td>$(_fmt(e.median))</td>" *
                """<td class="$cls_css">$cv_str</td>""" *
                "<td>[$(_fmt(e.iqr_low)), $(_fmt(e.iqr_high))]</td>" *
                "<td>[$(_fmt(e.min_val)), $(_fmt(e.max_val))]</td>" *
                "<td>$(e.n_solutions)</td>" *
                """<td class="$cls_css" style="font-weight:600;">$cls_str</td></tr>""")
        end
        println(io, "</table>")
    end
    println(io, "</div></details>")
end

# ─── Backsolve UQ HTML Section ──────────────────────────────────────────

"""
Write the backsolve UQ section to the HTML report.
Shows uncertainty propagation from shooting point through backward ODE to t₀.
"""
function _write_html_backsolve_uq_section(io, bq::BacksolveUQReport)
    println(io, "<details><summary>Backsolve Uncertainty (t=$(round(bq.t_shoot; digits=4)) → t₀=$(round(bq.t0; digits=4)))</summary><div class=\"detail-body\">")

    amp_color = bq.amplification < 10 ? "var(--easy)" : bq.amplification < 100 ? "var(--moderate)" : "var(--hard)"
    println(io, """<div class="provenance">Uncertainty propagated from shooting point t = $(round(bq.t_shoot; digits=4)) to t₀ = $(round(bq.t0; digits=4)) via backward ODE Jacobian (ForwardDiff). Amplification factor: <span style="color:$amp_color;font-weight:600;">$(_fmt(bq.amplification))×</span> (max singular value of J<sub>g</sub>).</div>""")

    # IC table
    println(io, "<table><tr><th>State IC</th><th>True</th><th>Estimated</th><th>σ (propagated)</th><th>95% CI</th><th>Coverage</th></tr>")

    for i in eachindex(bq.ic_names)
        pretty = _pretty_name(bq.ic_names[i])
        raw_esc = replace(bq.ic_names[i], "&" => "&amp;", "<" => "&lt;", ">" => "&gt;")

        ci_lo = bq.ic_estimated[i] - UQ_CI_Z * bq.ic_std[i]
        ci_hi = bq.ic_estimated[i] + UQ_CI_Z * bq.ic_std[i]
        cov_mark = bq.ic_ci_covers[i] ? """<span style="color:var(--easy);">✓</span>""" :
            """<span style="color:var(--hard);">✗</span>"""

        print(io, """<tr><td><span title="$raw_esc" class="math" style="font-weight:600;">$pretty</span>(0)</td>""")
        print(io, "<td>$(_fmt(bq.ic_true[i]))</td>")
        print(io, "<td>$(_fmt(bq.ic_estimated[i]))</td>")
        print(io, "<td>$(_fmt(bq.ic_std[i]))</td>")
        print(io, "<td>[$(_fmt(ci_lo)), $(_fmt(ci_hi))]</td>")
        println(io, "<td>$cov_mark</td></tr>")
    end
    println(io, "</table>")

    println(io, "</div></details>")
end
