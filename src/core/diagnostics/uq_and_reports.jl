# Helpers shared by the uncertainty report: name parsing, the practical
# identifiability index, the backsolve Jacobian, and unrescaling.
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
    uq isa UncertaintyUnavailable && return uq
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
