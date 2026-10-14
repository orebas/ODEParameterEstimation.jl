# Estimator-aware production UQ
#
# This file is downstream of the shared UQ helpers in uq_and_reports.jl. It
# routes from the exact rank-one estimator artifact retained during production;
# it never refits an interpolator or independently chooses a candidate.

function _uq_target_with_match(target::UQTargetSnapshot, match::Symbol)
    return UQTargetSnapshot(
        selected_rank = target.selected_rank,
        identity = target.identity,
        lineage = target.lineage,
        estimand = target.estimand,
        artifact_match = match,
    )
end

function _uq_unavailable(
    pep::ParameterEstimationProblem,
    target::UQTargetSnapshot,
    reason::Symbol,
    message::AbstractString;
    warnings::Vector{String} = String[],
)
    return UQUnavailable(pep.name, target, reason, String(message), warnings)
end

function _uq_residual_edf_variance(
    H::AbstractMatrix{<:Real},
    y::AbstractVector{<:Real},
)::Float64
    n = length(y)
    size(H) == (n, n) || throw(ArgumentError(
        "smoother matrix has size $(size(H)); expected ($n, $n)",
    ))
    residual = y - H * y
    df_resid = n - 2 * tr(H) + tr(H' * H)
    isfinite(df_resid) && df_resid > 0 ||
        throw(ArgumentError("smoother residual effective degrees of freedom is $df_resid"))
    sigma2 = sum(abs2, residual) / df_resid
    isfinite(sigma2) && sigma2 >= 0 ||
        throw(ArgumentError("smoother residual variance is invalid: $sigma2"))
    return Float64(sigma2)
end

function _uq_observation_covariance_from_source(
    interp::AGPInterpolatorUQ,
    source::Symbol,
)
    if source == :learned_gp_homoscedastic
        return nothing
    elseif source != :smoother_residual_edf
        throw(ArgumentError("Unsupported UQ noise source: $source"))
    end

    xs = interp.xs_train
    n = length(xs)
    n > 0 || throw(ArgumentError("Cannot estimate residual noise from an empty GP fit"))
    H = Matrix{Float64}(undef, n, n)
    for (i, x) in enumerate(xs)
        _, W = gp_derivative_influence_matrix(interp, x, 0)
        H[i, :] .= @view W[1, :]
    end
    y = _raw_training_values(interp)
    sigma2 = _uq_residual_edf_variance(H, y)
    return Diagonal(fill(sigma2, n))
end

function _uq_stacked_jet(
    interp::AGPInterpolatorUQ,
    times::Vector{Float64},
    max_order::Int,
    obs_name::String,
    noise_source::Symbol,
)
    covariance = _uq_observation_covariance_from_source(interp, noise_source)
    if isnothing(covariance)
        return joint_derivative_estimator_covariance(
            interp, times, max_order;
            observable_name = obs_name,
            noise_source = noise_source,
        )
    end
    return joint_derivative_estimator_covariance(
        interp, times, max_order;
        observable_name = obs_name,
        noise_source = noise_source,
        observation_covariance = covariance,
    )
end

function _uq_observable_index(pep::ParameterEstimationProblem)
    return Dict{String, Int}(
        replace(string(mq.lhs), r"\(.*\)" => "") => i
        for (i, mq) in enumerate(pep.measured_quantities)
        if !startswith(replace(string(mq.lhs), r"\(.*\)" => ""), "_obs_trfn_")
    )
end

function _uq_single_point_meta(artifact::SinglePointUQArtifact, pep::ParameterEstimationProblem)
    obs_index = _uq_observable_index(pep)
    metadata = DataVarMeta[]
    for variable in artifact.data_vars
        name = string(variable)
        if occursin("_trfn_", name)
            push!(metadata, DataVarMeta(name, nothing, 0, 1, :transcendental))
            continue
        end
        base, order = _parse_data_label(name)
        idx = isempty(base) ? nothing : get(obs_index, base, nothing)
        if isnothing(idx)
            positional = match(r"^y(\d+)$", base)
            if !isnothing(positional)
                candidate = parse(Int, positional.captures[1])
                idx = 1 <= candidate <= length(pep.measured_quantities) ? candidate : nothing
            end
        end
        if isnothing(idx)
            push!(metadata, DataVarMeta(name, nothing, order, 1, :unresolved))
        else
            push!(metadata, DataVarMeta(name, idx, order, 1, :observable_jet))
        end
    end
    return metadata
end

function _uq_exact_interpolant(interpolants::AbstractDict, mq)
    rhs = Symbolics.diff2term(mq.rhs)
    if haskey(interpolants, rhs)
        return interpolants[rhs]
    end
    rhs_name = replace(string(rhs), "(t)" => "")
    for (key, value) in interpolants
        replace(string(key), "(t)" => "") == rhs_name && return value
    end
    return nothing
end

function _uq_gp_factorization_assessment(
    pep::ParameterEstimationProblem,
    interpolants::AbstractDict,
)
    ratios = Float64[]
    residuals = Float64[]
    warnings = String[]
    degraded = false
    for mq in pep.measured_quantities
        _is_trfn_observable(Symbolics.wrap(mq.rhs)) && continue
        interp = _uq_exact_interpolant(interpolants, mq)
        interp isa AGPInterpolatorUQ || continue
        diagnostics = gp_factorization_diagnostics(interp)
        push!(ratios, diagnostics.jitter_to_noise)
        push!(residuals, diagnostics.factorization_residual)
        if diagnostics.status == :material_regularization
            push!(warnings,
                "Observable '$(_uq_observation_name(mq))' required GP Cholesky jitter $(diagnostics.jitter_to_noise)× its learned observation-noise variance.")
        elseif diagnostics.status == :factorization_mismatch
            degraded = true
            push!(warnings,
                "Observable '$(_uq_observation_name(mq))' has GP factorization residual $(diagnostics.factorization_residual).")
        end
    end
    max_ratio = isempty(ratios) ? NaN : maximum(ratios)
    max_residual = isempty(residuals) ? NaN : maximum(residuals)
    return max_ratio, max_residual, degraded, warnings
end

function _uq_append_linearization_warnings!(
    warnings::Vector{String},
    diagnostics::UQLinearizationDiagnostics,
)
    if diagnostics.reason == :scale_sensitive_conditioning
        push!(warnings,
            "The raw linearization condition number $(diagnostics.jacobian_condition) is scale-sensitive; one-pass row/column equilibration gives $(diagnostics.jacobian_condition_equilibrated) and the solve backward error is $(diagnostics.linear_solve_backward_error).")
    elseif diagnostics.reason in (
        :nonfinite_linearization,
        :intrinsic_ill_conditioning,
        :large_backward_error,
        :forward_error_risk,
        :factorization_failed,
    )
        push!(warnings,
            "The retained linear solve failed numerical reliability checks ($(diagnostics.reason)): raw condition=$(diagnostics.jacobian_condition), equilibrated condition=$(diagnostics.jacobian_condition_equilibrated), backward error=$(diagnostics.linear_solve_backward_error).")
    end
    return warnings
end

_uq_observation_name(mq) = replace(string(mq.lhs), r"\(.*\)" => "")
_uq_raw_observation_label(obs_name::AbstractString, time_index::Integer) =
    "$(obs_name)(t_index=$(Int(time_index)))"

function _uq_measurement_series(pep::ParameterEstimationProblem, mq)::Vector{Float64}
    rhs = Symbolics.diff2term(mq.rhs)
    lhs_wrapped = Symbolics.wrap(mq.lhs)
    candidates = Any[
        rhs,
        mq.rhs,
        lhs_wrapped,
        _uq_observation_name(mq),
        replace(string(rhs), "(t)" => ""),
    ]
    for key in candidates
        if haskey(pep.data_sample, key)
            return Float64.(pep.data_sample[key])
        end
    end
    throw(ArgumentError(
        "raw measurements for observable '$(_uq_observation_name(mq))' are absent from data_sample",
    ))
end

function _uq_nontranscendental_observation_indices(pep::ParameterEstimationProblem)
    return Int[
        idx for (idx, mq) in enumerate(pep.measured_quantities)
        if !_is_trfn_observable(Symbolics.wrap(mq.rhs))
    ]
end

function _uq_training_index(
    interp::AGPInterpolatorUQ,
    requested_index::Int,
    requested_time::Float64,
)
    if 1 <= requested_index <= length(interp.xs_train) &&
       isapprox(interp.xs_train[requested_index], requested_time; rtol = 1e-10, atol = 1e-12)
        return requested_index
    end
    matches = findall(x -> isapprox(x, requested_time; rtol = 1e-10, atol = 1e-12),
                      interp.xs_train)
    length(matches) == 1 || throw(ArgumentError(
        "selected time $requested_time does not map uniquely to the retained AGPUQ training grid",
    ))
    return only(matches)
end

function _uq_solve_coordinate_index(solve_vars::AbstractVector, name::String)
    return findfirst(eachindex(solve_vars)) do idx
        base, order, point = _uq_var_base_order_point(solve_vars[idx])
        base == name && order == 0 && point == 1
    end
end

function _uq_direct_observation_for_state(pep::ParameterEstimationProblem, state)
    return findfirst(pep.measured_quantities) do mq
        isequal(Symbolics.diff2term(mq.rhs), state)
    end
end

"""
Append the raw measurements that production uses when a directly observed
state was eliminated from the algebraic root. These rows do not enter the
polynomial equations (their root sensitivity columns are zero), but they do
enter the reported estimator through the output map with unit influence.
"""
function _uq_augment_direct_observation_rows(
    pep::ParameterEstimationProblem,
    solve_vars::AbstractVector,
    data_values::Vector{Float64},
    metadata::Vector{DataVarMeta},
    data_labels::Vector{String},
    time_indices::Vector{Int},
    times::Vector{Float64},
)
    length(time_indices) == length(times) ||
        throw(ArgumentError("selected time-index/time-value length mismatch"))
    isempty(time_indices) && throw(ArgumentError("selected algebraic estimator has no shooting time"))
    augmented_values = copy(data_values)
    augmented_metadata = copy(metadata)
    augmented_labels = copy(data_labels)
    source_index = first(time_indices)
    source_time = first(times)
    t_vector = Float64.(pep.data_sample["t"])
    1 <= source_index <= length(t_vector) ||
        throw(ArgumentError("selected shooting index $source_index is outside the observed time grid"))
    isapprox(t_vector[source_index], source_time; rtol = 1e-10, atol = 1e-12) ||
        throw(ArgumentError("selected shooting index/time provenance is inconsistent"))

    for state in pep.model.original_states
        name = _uq_clean_name(state)
        startswith(name, "_trfn_") && continue
        !isnothing(_uq_solve_coordinate_index(solve_vars, name)) && continue
        obs_idx = _uq_direct_observation_for_state(pep, state)
        isnothing(obs_idx) && throw(ArgumentError(
            "physical state '$name' is absent from the retained root and is not directly observed",
        ))
        existing = findfirst(augmented_metadata) do meta
            meta.kind == :raw_observation && meta.obs_idx == obs_idx && meta.point == 1
        end
        !isnothing(existing) && continue
        mq = pep.measured_quantities[obs_idx]
        values = _uq_measurement_series(pep, mq)
        source_index <= length(values) || throw(ArgumentError(
            "raw observable '$(_uq_observation_name(mq))' has no value at index $source_index",
        ))
        push!(augmented_values, values[source_index])
        push!(augmented_metadata,
              DataVarMeta(_uq_observation_name(mq), obs_idx, 0, 1, :raw_observation))
        push!(augmented_labels,
              "raw::$(_uq_raw_observation_label(_uq_observation_name(mq), source_index))")
    end
    return augmented_values, augmented_metadata, augmented_labels
end

function _uq_assemble_data_covariance(
    pep::ParameterEstimationProblem,
    interpolants::AbstractDict,
    time_indices::Vector{Int},
    times::Vector{Float64},
    metadata::Vector{DataVarMeta},
    data_labels::Vector{String},
    noise_source::Symbol,
)
    length(metadata) == length(data_labels) ||
        throw(ArgumentError("data metadata/label length mismatch"))
    unresolved = findall(meta -> meta.kind == :unresolved, metadata)
    isempty(unresolved) ||
        throw(ArgumentError("unresolved data-variable metadata at rows $(unresolved)"))
    length(time_indices) == length(times) ||
        throw(ArgumentError("selected time-index/time-value length mismatch"))

    n_data = length(metadata)
    estimates = Dict{Int, StackedJetInfluenceEstimate}()
    warnings = String[]

    all_obs = _uq_nontranscendental_observation_indices(pep)
    used_obs = sort(unique(Int[meta.obs_idx for meta in metadata
                               if meta.kind in (:observable_jet, :raw_observation) &&
                                  !isnothing(meta.obs_idx)]))
    all(obs_idx -> obs_idx in all_obs, used_obs) ||
        throw(ArgumentError("data metadata references a transcendental or unknown observable"))

    raw_offsets = Dict{Int, UnitRange{Int}}()
    raw_blocks = Dict{Int, Matrix{Float64}}()
    raw_labels = String[]
    raw_total = 0
    for obs_idx in all_obs
        mq = pep.measured_quantities[obs_idx]
        obs_name = _uq_observation_name(mq)
        interp = _uq_exact_interpolant(interpolants, mq)
        interp isa AGPInterpolatorUQ ||
            throw(ArgumentError("observable '$obs_name' was not estimated by an exact AGPInterpolatorUQ"))
        block = _uq_observation_covariance_from_source(interp, noise_source)
        if isnothing(block)
            block = Diagonal(fill(
                learned_observation_noise_variance(interp), length(interp.xs_train),
            ))
        end
        block_matrix = Matrix{Float64}(block)
        n_raw = size(block_matrix, 1)
        size(block_matrix, 2) == n_raw ||
            throw(ArgumentError("raw observation covariance for '$obs_name' is not square"))
        length(_uq_measurement_series(pep, mq)) == n_raw ||
            throw(ArgumentError("raw observation/training-grid length mismatch for '$obs_name'"))
        raw_offsets[obs_idx] = (raw_total + 1):(raw_total + n_raw)
        raw_blocks[obs_idx] = block_matrix
        append!(raw_labels, [_uq_raw_observation_label(obs_name, i) for i in 1:n_raw])
        raw_total += n_raw
    end
    observation_covariance = zeros(raw_total, raw_total)
    for obs_idx in all_obs
        rows = raw_offsets[obs_idx]
        observation_covariance[rows, rows] .= raw_blocks[obs_idx]
    end

    for obs_idx in used_obs
        mq = pep.measured_quantities[obs_idx]
        obs_name = _uq_observation_name(mq)
        interp = _uq_exact_interpolant(interpolants, mq)
        max_order = maximum((meta.order for meta in metadata
                             if meta.kind == :observable_jet && meta.obs_idx == obs_idx);
                            init = 0)
        estimate = _uq_stacked_jet(interp, times, max_order, obs_name, noise_source)
        estimates[obs_idx] = estimate
        append!(warnings, ["Observable '$obs_name': $warning" for warning in estimate.warnings])
    end

    data_to_raw = zeros(n_data, raw_total)
    for i in eachindex(metadata)
        mi = metadata[i]
        if mi.kind == :observable_jet
            obs_idx = something(mi.obs_idx)
            estimate = estimates[obs_idx]
            ii = stacked_jet_index(estimate, mi.point, mi.order)
            data_to_raw[i, raw_offsets[obs_idx]] .= @view estimate.W_stack[ii, :]
        elseif mi.kind == :raw_observation
            obs_idx = something(mi.obs_idx)
            1 <= mi.point <= length(times) ||
                throw(ArgumentError("raw-observation point $(mi.point) is outside the selected point set"))
            interp = _uq_exact_interpolant(interpolants, pep.measured_quantities[obs_idx])
            train_idx = _uq_training_index(
                interp, time_indices[mi.point], times[mi.point],
            )
            data_to_raw[i, first(raw_offsets[obs_idx]) + train_idx - 1] = 1.0
        elseif mi.kind != :transcendental
            throw(ArgumentError("unsupported data-variable metadata kind $(mi.kind)"))
        end
    end

    covariance = _psd_symmetric_matrix(
        data_to_raw * observation_covariance * data_to_raw',
    )

    obs_names = String[]
    obs_means = Vector{Float64}[]
    obs_stds = Vector{Float64}[]
    for obs_idx in used_obs
        estimate = estimates[obs_idx]
        push!(obs_names, estimate.observable_name)
        push!(obs_means, copy(estimate.mean))
        push!(obs_stds, sqrt.(max.(diag(estimate.jet_covariance), 0.0)))
    end
    return covariance, obs_names, obs_means, obs_stds, warnings,
           data_to_raw, observation_covariance, raw_labels
end

function _uq_exact_ift(
    equations::AbstractVector,
    solve_vars::AbstractVector,
    data_vars::AbstractVector,
    root::AbstractVector,
    data_values::Vector{Float64};
    root_residual_rtol::Float64 = 1e-3,
)
    length(equations) == length(solve_vars) ||
        throw(ArgumentError("exact algebraic UQ requires a square production system"))
    length(root) == length(solve_vars) ||
        throw(ArgumentError("retained root length does not match solve variables"))
    length(data_values) == length(data_vars) ||
        throw(ArgumentError("retained data length does not match data variables"))
    all(isfinite, root) || throw(ArgumentError("retained root contains non-finite values"))
    all(isfinite, data_values) || throw(ArgumentError("retained data contains non-finite values"))

    combined_vars = vcat(solve_vars, data_vars)
    combined_values = vcat(Float64.(root), data_values)
    fn = _compile_system_function(equations, combined_vars)
    residual = Float64.(fn(combined_values))
    J = Float64.(ForwardDiff.jacobian(fn, combined_values))
    residual_abs = norm(residual, Inf)
    term_scale = maximum(abs.(J) * abs.(combined_values); init = 0.0)
    residual_rel = residual_abs / max(term_scale, 1e-300)
    n_x = length(solve_vars)
    assessment = _ift_solve_assessment(
        J[:, 1:n_x], J[:, (n_x + 1):end],
    )
    S = assessment.sensitivity
    isempty(S) && throw(LinearAlgebra.SingularException(0))
    residual_degraded = !(residual_rel <= root_residual_rtol)
    reason = if assessment.degraded
        assessment.reason
    elseif residual_degraded
        :root_residual
    else
        assessment.reason
    end
    diagnostics = UQLinearizationDiagnostics(
        reason, residual_abs, residual_rel,
        assessment.raw_condition, assessment.equilibrated_condition,
        assessment.backward_error, NaN, Int[],
        assessment.degraded || residual_degraded, NaN, NaN,
    )
    return S, diagnostics
end

"""
    _multipoint_solve_var_point(name) → (point::Int, clean_name::String)

Point index and `_ptK`-stripped name for one multipoint solve/data variable
(point 1 = unsuffixed).
"""
function _multipoint_solve_var_point(name::AbstractString)
    m = match(r"_pt(\d+)$", name)
    isnothing(m) && return 1, String(name)
    return parse(Int, m.captures[1]), replace(String(name), r"_pt\d+$" => "")
end

function _uq_var_base_order_point(variable)
    point, clean = _multipoint_solve_var_point(string(variable))
    parsed = parse_derivative_variable_name(clean)
    if isnothing(parsed)
        return clean, 0, point
    end
    return String(parsed[1]), Int(parsed[2]), point
end

function _uq_direct_observation_index(
    pep::ParameterEstimationProblem,
    state,
    metadata::Vector{DataVarMeta},
)
    obs_idx = _uq_direct_observation_for_state(pep, state)
    isnothing(obs_idx) && return nothing
    raw_idx = findfirst(metadata) do meta
        meta.kind == :raw_observation && meta.obs_idx == obs_idx && meta.point == 1
    end
    !isnothing(raw_idx) && return raw_idx
    # Compatibility for direct callers of `_uq_local_output_map`. Production
    # estimator-aware UQ always augments an explicit raw-observation row first.
    return findfirst(metadata) do meta
        meta.kind == :observable_jet && meta.obs_idx == obs_idx &&
            meta.order == 0 && meta.point == 1
    end
end

function _uq_local_output_map(
    pep::ParameterEstimationProblem,
    solve_vars::AbstractVector,
    data_values::Vector{Float64},
    root::AbstractVector,
    metadata::Vector{DataVarMeta},
    S::Matrix{Float64},
)
    params = pep.model.original_parameters
    states = pep.model.original_states
    n_data = size(S, 2)
    labels = vcat([_uq_clean_name(p) for p in params], [_uq_clean_name(s) for s in states])
    roles = Dict{String, Symbol}()
    L = zeros(length(labels), n_data)
    values = fill(NaN, length(labels))

    for (i, param) in enumerate(params)
        name = _uq_clean_name(param)
        roles[name] = :parameter
        idx = _uq_solve_coordinate_index(solve_vars, name)
        isnothing(idx) && throw(ArgumentError("physical parameter '$name' is absent from the retained algebraic root"))
        L[i, :] .= @view S[idx, :]
        values[i] = Float64(root[idx])
    end

    offset = length(params)
    for (state_idx, state) in enumerate(states)
        row = offset + state_idx
        name = _uq_clean_name(state)
        if startswith(name, "_trfn_")
            roles[name] = :transcendental
            values[row] = 0.0
            continue
        end
        roles[name] = :state_at_eval
        idx = _uq_solve_coordinate_index(solve_vars, name)
        if !isnothing(idx)
            L[row, :] .= @view S[idx, :]
            values[row] = Float64(root[idx])
            continue
        end
        data_idx = _uq_direct_observation_index(pep, state, metadata)
        isnothing(data_idx) &&
            throw(ArgumentError("physical state '$name' is neither a solve coordinate nor a direct observed data coordinate"))
        L[row, data_idx] = 1.0
        values[row] = data_values[data_idx]
    end
    return L, labels, roles, values
end

function _uq_selected_values(
    pep::ParameterEstimationProblem,
    result::ParameterEstimationResult,
    real_state_indices::Vector{Int},
)
    params = pep.model.original_parameters
    states = pep.model.original_states[real_state_indices]
    values = Float64[]
    for param in params
        haskey(result.parameters, param) || throw(ArgumentError("selected result lacks parameter $param"))
        push!(values, Float64(result.parameters[param]))
    end
    for state in states
        haskey(result.states, state) || throw(ArgumentError("selected result lacks state $state"))
        push!(values, Float64(result.states[state]))
    end
    return values
end

function _uq_truth_values(
    pep::ParameterEstimationProblem,
    real_state_indices::Vector{Int},
)
    params = pep.model.original_parameters
    states = pep.model.original_states[real_state_indices]
    return vcat(
        Float64[get(pep.p_true, param, NaN) for param in params],
        Float64[get(pep.ic, state, NaN) for state in states],
    )
end

function _uq_build_algebraic_report(
    pep::ParameterEstimationProblem,
    result::ParameterEstimationResult,
    target::UQTargetSnapshot,
    equations::AbstractVector,
    solve_vars::AbstractVector,
    data_vars::AbstractVector,
    data_values::Vector{Float64},
    root::AbstractVector,
    interpolants::AbstractDict,
    time_indices::Vector{Int},
    times::Vector{Float64},
    metadata::Vector{DataVarMeta},
    opts::EstimationOptions,
)
    S, diagnostics = _uq_exact_ift(
        equations, solve_vars, data_vars, root, data_values,
    )
    data_labels = String[string(variable) for variable in data_vars]
    augmented_values, augmented_metadata, augmented_labels =
        _uq_augment_direct_observation_rows(
            pep, solve_vars, data_values, metadata, data_labels,
            time_indices, times,
        )
    n_supplemental = length(augmented_values) - length(data_values)
    S_augmented = n_supplemental == 0 ? S :
        hcat(S, zeros(size(S, 1), n_supplemental))
    Sigma_d, obs_names, obs_means, obs_stds, warnings,
        data_to_raw, observation_covariance, observation_labels =
        _uq_assemble_data_covariance(
            pep, interpolants, time_indices, times, augmented_metadata,
            augmented_labels, opts.uq_noise_source,
        )
    gp_ratio, gp_residual, gp_degraded, gp_warnings =
        _uq_gp_factorization_assessment(pep, interpolants)
    append!(warnings, gp_warnings)
    diagnostic_reason = gp_degraded && !diagnostics.degraded ?
        :gp_factorization_regularization : diagnostics.reason
    diagnostics = UQLinearizationDiagnostics(
        diagnostic_reason, diagnostics.root_residual_abs,
        diagnostics.root_residual_rel, diagnostics.jacobian_condition,
        diagnostics.jacobian_condition_equilibrated,
        diagnostics.linear_solve_backward_error,
        diagnostics.gradient_norm, copy(diagnostics.active_bounds),
        diagnostics.degraded || gp_degraded, gp_ratio, gp_residual,
    )
    _uq_append_linearization_warnings!(warnings, diagnostics)
    L, local_labels, local_roles, local_values = _uq_local_output_map(
        pep, solve_vars, augmented_values, root, augmented_metadata, S_augmented,
    )
    local_influence = L * data_to_raw
    Sigma_local = _psd_symmetric_matrix(
        local_influence * observation_covariance * local_influence',
    )
    local_std = sqrt.(max.(diag(Sigma_local), 0.0))
    local_corr = _uq_correlation_matrix(Sigma_local, local_std)
    local_cv = _uq_max_cv(local_std, local_values)
    local_status = diagnostics.degraded ? :degenerate : _uq_status_from_cv(local_cv)
    local_ia = compute_practical_identifiability_index(
        Sigma_local, local_labels, local_roles, local_values,
    )
    local_snapshot = LocalUQSnapshot(
        first(times), Sigma_local, local_std, local_labels, local_roles,
        local_values, local_corr, local_cv, local_status, local_ia,
    )

    params = pep.model.original_parameters
    states = pep.model.original_states
    real_indices = [i for (i, state) in enumerate(states)
                    if !startswith(_uq_clean_name(state), "_trfn_")]
    n_params = length(params)
    n_states = length(states)
    t_eval = first(times)
    t0 = Float64(pep.data_sample["t"][1])
    transform = zeros(n_params + length(real_indices), n_params + n_states)
    transform[1:n_params, 1:n_params] .= Matrix{Float64}(I, n_params, n_params)
    transform_status = :identity
    if abs(t_eval - t0) <= 1e-10
        for (out_idx, state_idx) in enumerate(real_indices)
            transform[n_params + out_idx, n_params + state_idx] = 1.0
        end
    else
        J_state, _ = _uq_variational_backsolve_jacobian(pep, result, t0, t_eval)
        transform[(n_params + 1):end, :] .= J_state[real_indices, :]
        transform_status = :variational
    end

    physical_influence = transform * local_influence
    Sigma_physical = _psd_symmetric_matrix(
        physical_influence * observation_covariance * physical_influence',
    )
    physical_std = sqrt.(max.(diag(Sigma_physical), 0.0))
    physical_labels = vcat(
        [_uq_clean_name(param) for param in params],
        [_uq_clean_name(states[idx]) for idx in real_indices],
    )
    physical_roles = Dict{String, Symbol}(
        vcat(
            [(_uq_clean_name(param), :parameter) for param in params],
            [(_uq_clean_name(states[idx]), :state_ic) for idx in real_indices],
        ),
    )
    estimate_values = _uq_selected_values(pep, result, real_indices)
    true_values = _uq_truth_values(pep, real_indices)
    physical_corr = _uq_correlation_matrix(Sigma_physical, physical_std)
    max_cv = _uq_max_cv(physical_std, estimate_values)
    status = diagnostics.degraded ? :degenerate : _uq_status_from_cv(max_cv)
    ia = compute_practical_identifiability_index(
        Sigma_physical, physical_labels, physical_roles, estimate_values,
    )
    append!(warnings, ia.warnings)
    diagnostics.degraded && push!(warnings,
        "The exact selected-root linearization exceeded a residual or conditioning gate; covariance is reported for audit but is not reliable.")

    source_roles = copy(local_roles)
    target_roles = copy(physical_roles)
    backsolve = UQBacksolveTransform(
        t_eval, t0, local_labels, physical_labels,
        source_roles, target_roles, transform,
        _uq_matrix_amplification(transform), transform_status, copy(warnings),
    )
    _run_ctx_register_uq_influence!(
        target.identity.candidate_id,
        UQInfluenceArtifact(
            Matrix{Float64}(physical_influence),
            Matrix{Float64}(observation_covariance),
            copy(observation_labels), copy(physical_labels),
        ),
    )

    return UncertaintyReport(
        pep.name, t_eval,
        obs_names, obs_means, obs_stds,
        Sigma_d, augmented_labels,
        Sigma_physical, physical_std, physical_labels, physical_roles, true_values,
        physical_corr, max_cv, status, warnings,
        :estimator_sampling, opts.uq_noise_source, ia,
        :physical_initial_conditions, local_snapshot, backsolve,
        _uq_target_with_match(target, :exact), estimate_values, diagnostics,
    )
end

function _uq_single_point_report(
    pep::ParameterEstimationProblem,
    result::ParameterEstimationResult,
    target::UQTargetSnapshot,
    artifact::SinglePointUQArtifact,
    opts::EstimationOptions,
)
    metadata = _uq_single_point_meta(artifact, pep)
    return _uq_build_algebraic_report(
        pep, result, target,
        artifact.equations, artifact.solve_vars, artifact.data_vars,
        artifact.data_values, artifact.root, artifact.interpolants,
        Int[artifact.time_index], Float64[artifact.time_value], metadata, opts,
    )
end

function _uq_multipoint_report(
    pep::ParameterEstimationProblem,
    result::ParameterEstimationResult,
    target::UQTargetSnapshot,
    artifact::MultipointUQArtifact,
    opts::EstimationOptions,
)
    evaluation = artifact.evaluation
    template = evaluation.template
    return _uq_build_algebraic_report(
        pep, result, target,
        template.stripped_equations, template.solve_vars, template.data_vars,
        evaluation.data_values, artifact.root, artifact.interpolants,
        evaluation.time_indices, evaluation.t_values, template.data_var_meta, opts,
    )
end

function _uq_polish_predictions(ctx::PolishContext, internal::AbstractVector{<:Real})
    external = _polish_internal_to_external(
        internal, ctx.coordinate_transforms, ctx.coordinate_shifts,
    )
    ic = @view external[1:ctx.n_ic]
    params = @view external[(ctx.n_ic + 1):end]
    problem = remake(
        ctx.base_ode_prob;
        u0 = Dict(ctx.unknown_syms .=> ic),
        p = Dict(ctx.param_syms .=> params),
        build_initializeprob = false,
    )
    solution = ModelingToolkit.solve(
        problem, ctx.solver;
        saveat = ctx.t_vector, abstol = ctx.abstol, reltol = ctx.reltol,
        maxiters = ctx.polish_ode_maxiters,
    )
    SciMLBase.successful_retcode(solution) ||
        throw(ErrorException("trajectory solve failed at the selected polish optimum: $(solution.retcode)"))
    predictions = Vector{eltype(external)}()
    sizehint!(predictions, sum(length, ctx.data_targets))
    for f in ctx.obs_funcs
        for state in solution.u
            push!(predictions, f(state, params))
        end
    end
    return predictions
end

function _uq_polish_penalty(ctx::PolishContext, internal::AbstractVector{<:Real})
    penalty = max(ctx.regularization_lambda, 0.0) * sum(abs2, internal)
    lambda_softwall = max(ctx.softwall_lambda, 0.0)
    if lambda_softwall <= 0 || isnothing(ctx.internal_lb) || isnothing(ctx.internal_ub)
        return penalty
    end
    epsilon = ctx.softwall_epsilon
    (0.0 <= epsilon < 0.5) || return penalty
    for i in eachindex(internal)
        midpoint = (ctx.internal_lb[i] + ctx.internal_ub[i]) / 2
        halfrange = (ctx.internal_ub[i] - ctx.internal_lb[i]) / 2
        threshold = (1 - epsilon) * halfrange
        deviation = abs(internal[i] - midpoint)
        if halfrange > 0 && deviation > threshold
            over = (deviation - threshold) / halfrange
            penalty += lambda_softwall * over^2
        end
    end
    return penalty
end

function _uq_polish_objective(
    ctx::PolishContext,
    internal::AbstractVector{<:Real},
    observations::AbstractVector{<:Real},
    objective_kind::Symbol,
)
    predictions = _uq_polish_predictions(ctx, internal)
    length(predictions) == length(observations) ||
        throw(ArgumentError("polish prediction/observation length mismatch"))
    penalty = if objective_kind == :residual_least_squares
        _uq_polish_penalty(ctx, internal)
    elseif objective_kind == :trajectory_sse
        zero(eltype(internal))
    else
        throw(ArgumentError("unsupported retained polish objective '$objective_kind'"))
    end
    return sum(abs2, predictions .- observations) + penalty
end

function _uq_finite_difference_score_hessian(objective, x::Vector{Float64})
    n = length(x)
    steps = eps(Float64)^(1 / 4) .* max.(abs.(x), 1.0)
    center = Float64(objective(x))
    forward = zeros(n)
    backward = zeros(n)
    gradient = zeros(n)
    H = zeros(n, n)

    for i in eachindex(x)
        plus = copy(x)
        minus = copy(x)
        plus[i] += steps[i]
        minus[i] -= steps[i]
        forward[i] = Float64(objective(plus))
        backward[i] = Float64(objective(minus))
        gradient[i] = (forward[i] - backward[i]) / (2steps[i])
        H[i, i] = (forward[i] - 2center + backward[i]) / steps[i]^2
    end

    for i in 1:(n - 1), j in (i + 1):n
        plus_plus = copy(x)
        plus_minus = copy(x)
        minus_plus = copy(x)
        minus_minus = copy(x)
        plus_plus[i] += steps[i]
        plus_plus[j] += steps[j]
        plus_minus[i] += steps[i]
        plus_minus[j] -= steps[j]
        minus_plus[i] -= steps[i]
        minus_plus[j] += steps[j]
        minus_minus[i] -= steps[i]
        minus_minus[j] -= steps[j]
        mixed = (
            Float64(objective(plus_plus)) - Float64(objective(plus_minus)) -
            Float64(objective(minus_plus)) + Float64(objective(minus_minus))
        ) / (4steps[i] * steps[j])
        H[i, j] = mixed
        H[j, i] = mixed
    end
    return gradient, Matrix{Float64}(Symmetric(H))
end

function _uq_least_squares_score(
    predictions::Vector{Float64},
    observations::Vector{Float64},
    prediction_jacobian::Matrix{Float64},
    penalty_gradient::Vector{Float64},
)
    length(predictions) == length(observations) == size(prediction_jacobian, 1) ||
        throw(ArgumentError("polish prediction/observation/Jacobian row mismatch"))
    length(penalty_gradient) == size(prediction_jacobian, 2) ||
        throw(ArgumentError("polish penalty-gradient/Jacobian column mismatch"))
    return 2 .* (prediction_jacobian' * (predictions .- observations)) .+
        penalty_gradient
end

function _uq_full_observed_hessian(
    ctx::PolishContext,
    internal::Vector{Float64},
    observations::Vector{Float64},
    objective_kind::Symbol,
)
    objective = u -> _uq_polish_objective(ctx, u, observations, objective_kind)
    # Do not nest ForwardDiff through a ModelingToolkit ODE solve here. On
    # Julia 1.12 that live path can fail in code generation with a corrupt
    # `Dual` field-offset error even though first-order polish Jacobians work.
    # Central differences evaluate the exact retained scalar objective in
    # Float64 coordinates and therefore still produce the full observed
    # Hessian (including residual-curvature and penalty terms).
    _, hessian = _uq_finite_difference_score_hessian(objective, internal)
    return hessian
end

function _uq_polish_score_gradient(
    ctx::PolishContext,
    internal::Vector{Float64},
    observations::Vector{Float64},
    objective_kind::Symbol,
    prediction_jacobian::Matrix{Float64},
)
    predictions = Float64.(_uq_polish_predictions(ctx, internal))
    penalty_gradient = if objective_kind == :residual_least_squares
        ForwardDiff.gradient(
            u -> _uq_polish_penalty(ctx, u), internal,
        )
    elseif objective_kind != :trajectory_sse
        throw(ArgumentError("unsupported retained polish objective '$objective_kind'"))
    else
        zeros(length(internal))
    end
    return _uq_least_squares_score(
        predictions, observations, prediction_jacobian, penalty_gradient,
    )
end

function _uq_active_bounds(ctx::PolishContext, internal::Vector{Float64})
    (isnothing(ctx.internal_lb) || isnothing(ctx.internal_ub)) && return Int[]
    active = Int[]
    for i in eachindex(internal)
        tolerance = sqrt(eps(Float64)) * max(
            1.0, abs(internal[i]), abs(ctx.internal_lb[i]), abs(ctx.internal_ub[i]),
        )
        if abs(internal[i] - ctx.internal_lb[i]) <= tolerance ||
           abs(internal[i] - ctx.internal_ub[i]) <= tolerance
            push!(active, i)
        end
    end
    return active
end

function _uq_optimizer_success(result)::Bool
    isnothing(result) && return false
    if hasproperty(result, :retcode)
        return try
            SciMLBase.successful_retcode(result)
        catch
            try
                SciMLBase.successful_retcode(result.retcode)
            catch
                result.retcode in (:Success, :success)
            end
        end
    elseif hasproperty(result, :converged)
        return Bool(result.converged)
    elseif hasproperty(result, :original) && hasproperty(result.original, :converged)
        return Bool(result.original.converged)
    end
    # Some residual solvers expose neither a common retcode nor a convergence
    # field. The score-norm gate below remains mandatory for those artifacts.
    return true
end

function _uq_polish_observation_covariance(
    pep::ParameterEstimationProblem,
    opts::EstimationOptions,
)
    interpolants = _run_ctx_noise_interpolants(:agp_uq)
    isnothing(interpolants) &&
        throw(ArgumentError("no explicitly configured AGPUQ observation-noise provider was retained"))
    gp_ratio, gp_residual, gp_degraded, gp_warnings =
        _uq_gp_factorization_assessment(pep, interpolants)
    n_total = sum(length(_uq_measurement_series(pep, mq)) for mq in pep.measured_quantities)
    covariance = zeros(n_total, n_total)
    labels = String[]
    obs_names = String[]
    obs_means = Vector{Float64}[]
    obs_stds = Vector{Float64}[]
    offset = 0
    for mq in pep.measured_quantities
        name = _uq_observation_name(mq)
        if _is_trfn_observable(Symbolics.wrap(mq.rhs))
            # Transcendental helper observables are evaluated analytically by
            # production and deliberately have no fitted interpolator. Keep
            # their trajectory-loss rows in the score, but assign them zero
            # observation variance: they are deterministic inputs, not noisy
            # measurement channels.
            raw = _uq_measurement_series(pep, mq)
            n = length(raw)
            append!(labels, ["$name(t_index=$i)" for i in 1:n])
            push!(obs_names, name)
            push!(obs_means, raw)
            push!(obs_stds, zeros(n))
            offset += n
            continue
        end
        interp = _uq_exact_interpolant(interpolants, mq)
        interp isa AGPInterpolatorUQ ||
            throw(ArgumentError("observation-noise provider lacks AGPUQ fit for '$name'"))
        block = _uq_observation_covariance_from_source(interp, opts.uq_noise_source)
        if isnothing(block)
            block = Diagonal(fill(
                learned_observation_noise_variance(interp), length(interp.xs_train),
            ))
        end
        n = size(block, 1)
        covariance[(offset + 1):(offset + n), (offset + 1):(offset + n)] .= block
        append!(labels, ["$name(t_index=$i)" for i in 1:n])
        push!(obs_names, name)
        push!(obs_means, _raw_training_values(interp))
        push!(obs_stds, sqrt.(max.(diag(block), 0.0)))
        offset += n
    end
    offset == n_total || throw(ArgumentError("observation covariance assembly length mismatch"))
    return covariance, labels, obs_names, obs_means, obs_stds,
        gp_ratio, gp_residual, gp_degraded, gp_warnings
end

function _uq_polish_physical_map(
    ctx::PolishContext,
    internal::AbstractVector{<:Real},
)
    external = _polish_internal_to_external(
        internal, ctx.coordinate_transforms, ctx.coordinate_shifts,
    )
    states = @view external[1:ctx.n_ic]
    params = @view external[(ctx.n_ic + 1):end]
    values = Vector{eltype(external)}()
    append!(values, params)
    append!(values, states)
    return values
end

function _uq_polish_report(
    pep::ParameterEstimationProblem,
    result::ParameterEstimationResult,
    target::UQTargetSnapshot,
    artifact,
    opts::EstimationOptions,
)
    artifact isa PolishUQArtifact ||
        return _uq_unavailable(pep, target, :artifact_mismatch,
            "polish identity is paired with $(typeof(artifact))")
    ctx = artifact.context
    internal = Float64.(artifact.internal_optimum)
    active = _uq_active_bounds(ctx, internal)
    isempty(active) || return _uq_unavailable(
        pep, target, :active_bounds,
        "selected optimum has active internal bounds at coordinates $active",
    )
    _uq_optimizer_success(artifact.optimizer_result) || return _uq_unavailable(
        pep, target, :optimizer_not_converged,
        "the selected optimizer did not report convergence",
    )

    observations = reduce(vcat, ctx.data_targets; init = Float64[])
    observation_covariance, data_labels, obs_names, obs_means, obs_stds,
        gp_ratio, gp_residual, gp_degraded, gp_warnings = try
        _uq_polish_observation_covariance(pep, opts)
    catch err
        _rethrow_if_interrupt(err)
        return _uq_unavailable(pep, target, :unsupported_noise_provider, sprint(showerror, err))
    end

    J_prediction = try
        Matrix{Float64}(_outer_jacobian(
            u -> _uq_polish_predictions(ctx, u), internal,
        ))
    catch err
        _rethrow_if_interrupt(err)
        return _uq_unavailable(pep, target, :numerical_failure,
            "trajectory prediction Jacobian failed: $(sprint(showerror, err))")
    end
    gradient = try
        _uq_polish_score_gradient(
            ctx, internal, observations, artifact.objective_kind, J_prediction,
        )
    catch err
        _rethrow_if_interrupt(err)
        return _uq_unavailable(pep, target, :numerical_failure,
            "trajectory score evaluation failed: $(sprint(showerror, err))")
    end
    H = try
        _uq_full_observed_hessian(
            ctx, internal, observations, artifact.objective_kind,
        )
    catch err
        _rethrow_if_interrupt(err)
        return _uq_unavailable(pep, target, :numerical_failure,
            "full observed-Hessian evaluation failed: $(sprint(showerror, err))")
    end
    gradient_norm = norm(gradient, Inf)
    objective = _uq_polish_objective(
        ctx, internal, observations, artifact.objective_kind,
    )
    gradient_tolerance = 1e-4 * max(1.0, sqrt(max(objective, 0.0)))
    gradient_norm <= gradient_tolerance || return _uq_unavailable(
        pep, target, :optimizer_not_converged,
        "selected optimum has score norm $gradient_norm above $gradient_tolerance",
    )

    all(isfinite, H) || return _uq_unavailable(
        pep, target, :numerical_failure, "observed Hessian contains non-finite values",
    )
    curvature = eigvals(Symmetric(H))
    isempty(curvature) && return _uq_unavailable(
        pep, target, :numerical_failure, "empty observed Hessian",
    )
    max_curvature = maximum(curvature)
    min_curvature = minimum(curvature)
    curvature_tolerance = sqrt(eps(Float64)) * max(abs(max_curvature), 1.0)
    min_curvature > curvature_tolerance || return _uq_unavailable(
        pep, target, :nonpositive_hessian,
        "selected interior optimum has minimum observed-Hessian curvature $min_curvature (required > $curvature_tolerance)",
    )
    B = -2 .* J_prediction'
    influence_internal = try
        factor = cholesky(Symmetric(H))
        -(factor \ B)
    catch err
        _rethrow_if_interrupt(err)
        return _uq_unavailable(pep, target, :numerical_failure,
            "observed Hessian factorization failed: $(sprint(showerror, err))")
    end
    all(isfinite, influence_internal) || return _uq_unavailable(
        pep, target, :numerical_failure, "optimizer influence contains non-finite values",
    )
    linear_assessment = _linear_solve_assessment(
        H, influence_internal, -B,
    )

    Q = ForwardDiff.jacobian(u -> _uq_polish_physical_map(ctx, u), internal)
    influence_physical_all = Q * influence_internal
    all_states = ctx.unknown_syms
    real_indices = [i for (i, state) in enumerate(all_states)
                    if !startswith(_uq_clean_name(state), "_trfn_")]
    n_params = ctx.n_param
    keep = vcat(collect(1:n_params), n_params .+ real_indices)
    influence = influence_physical_all[keep, :]
    Sigma = _psd_symmetric_matrix(influence * observation_covariance * influence')
    standard_deviation = sqrt.(max.(diag(Sigma), 0.0))
    labels = vcat(
        [_uq_clean_name(param) for param in ctx.param_syms],
        [_uq_clean_name(all_states[i]) for i in real_indices],
    )
    roles = Dict{String, Symbol}(
        vcat(
            [(_uq_clean_name(param), :parameter) for param in ctx.param_syms],
            [(_uq_clean_name(all_states[i]), :state_ic) for i in real_indices],
        ),
    )
    estimate_values_all = Float64.(_uq_polish_physical_map(ctx, internal))
    estimate_values = estimate_values_all[keep]
    truth_values = vcat(
        Float64[get(pep.p_true, param, NaN) for param in ctx.param_syms],
        Float64[get(pep.ic, all_states[i], NaN) for i in real_indices],
    )
    correlation = _uq_correlation_matrix(Sigma, standard_deviation)
    max_cv = _uq_max_cv(standard_deviation, estimate_values)
    degraded = linear_assessment.degraded || gp_degraded
    status = degraded ? :degenerate : _uq_status_from_cv(max_cv)
    diagnostic_reason = linear_assessment.degraded ? linear_assessment.reason :
        gp_degraded ? :gp_factorization_regularization : linear_assessment.reason
    diagnostics = UQLinearizationDiagnostics(
        diagnostic_reason, NaN, NaN,
        linear_assessment.raw_condition,
        linear_assessment.equilibrated_condition,
        linear_assessment.backward_error,
        gradient_norm, active, degraded, gp_ratio, gp_residual,
    )
    warnings = String[
        "Polish/direct UQ is conditional on the selected optimizer basin and unchanged active set.",
        "Observation covariance treats GP hyperparameters as fixed plug-in estimates.",
        "Trajectory score and prediction Jacobian use the production first-order AD path; the full observed Hessian uses central finite differences in retained optimizer coordinates.",
    ]
    append!(warnings, gp_warnings)
    _uq_append_linearization_warnings!(warnings, diagnostics)
    ia = compute_practical_identifiability_index(Sigma, labels, roles, estimate_values)
    append!(warnings, ia.warnings)

    snapshot = LocalUQSnapshot(
        Float64(ctx.t_vector[1]), Sigma, standard_deviation, labels, roles,
        estimate_values, correlation, max_cv, status, ia,
    )
    transform = Matrix{Float64}(I, length(labels), length(labels))
    backsolve = UQBacksolveTransform(
        Float64(ctx.t_vector[1]), Float64(ctx.t_vector[1]), labels, labels,
        roles, roles, transform, 1.0, :identity, copy(warnings),
    )
    _run_ctx_register_uq_influence!(
        target.identity.candidate_id,
        UQInfluenceArtifact(
            Matrix{Float64}(influence), Matrix{Float64}(observation_covariance),
            copy(data_labels), copy(labels),
        ),
    )
    return UncertaintyReport(
        pep.name, Float64(ctx.t_vector[1]),
        obs_names, obs_means, obs_stds,
        observation_covariance, data_labels,
        Sigma, standard_deviation, labels, roles, truth_values,
        correlation, max_cv, status, warnings,
        :estimator_sampling, opts.uq_noise_source, ia,
        :physical_initial_conditions, snapshot, backsolve,
        _uq_target_with_match(target, :exact), estimate_values, diagnostics,
    )
end

function _uq_result_with_physical_coordinates(
    source::ParameterEstimationResult,
    labels::Vector{String},
    values::AbstractVector{<:Real},
)
    length(labels) == length(values) ||
        throw(ArgumentError("physical-coordinate label/value length mismatch"))
    result = deepcopy(source)
    parameter_keys = Dict(_uq_clean_name(key) => key for key in keys(result.parameters))
    state_keys = Dict(_uq_clean_name(key) => key for key in keys(result.states))
    for (label, value) in zip(labels, values)
        if haskey(parameter_keys, label)
            result.parameters[parameter_keys[label]] = Float64(value)
        elseif haskey(state_keys, label)
            result.states[state_keys[label]] = Float64(value)
        else
            throw(ArgumentError("parent UQ coordinate '$label' is absent from the branch anchor"))
        end
    end
    return result
end

function _uq_branch_data_values(
    pep::ParameterEstimationProblem,
    anchor::ParameterEstimationResult,
    physical_labels::Vector{String},
    physical_values::AbstractVector{<:Real},
    template,
    t0::Float64,
    max_required_deriv::Int,
    opts::EstimationOptions,
)
    perturbed = _uq_result_with_physical_coordinates(anchor, physical_labels, physical_values)
    perturbed_pep = _consensus_candidate_pep(pep, perturbed)
    interpolants = build_perfect_interpolants(
        perturbed_pep, t0, max_required_deriv;
        solver = pep.solver, abstol = opts.abstol, reltol = opts.reltol,
    )
    return evaluate_data_vars_at_point(
        interpolants, template.extended_data_vars, template.template_DD,
        pep.measured_quantities, t0,
    )
end

function _uq_branch_data_jacobian(
    pep::ParameterEstimationProblem,
    anchor::ParameterEstimationResult,
    physical_labels::Vector{String},
    physical_values::Vector{Float64},
    template,
    t0::Float64,
    max_required_deriv::Int,
    opts::EstimationOptions,
)
    center = _uq_branch_data_values(
        pep, anchor, physical_labels, physical_values,
        template, t0, max_required_deriv, opts,
    )
    all(isfinite, center) || throw(ArgumentError("branch anchor produced non-finite exact jet data"))
    coarse = zeros(length(center), length(physical_values))
    fine = similar(coarse)
    for j in eachindex(physical_values)
        h = cbrt(eps(Float64)) * max(abs(physical_values[j]), 1.0)
        plus = copy(physical_values)
        minus = copy(physical_values)
        plus[j] += h
        minus[j] -= h
        coarse[:, j] .= (
            _uq_branch_data_values(pep, anchor, physical_labels, plus,
                template, t0, max_required_deriv, opts) .-
            _uq_branch_data_values(pep, anchor, physical_labels, minus,
                template, t0, max_required_deriv, opts)
        ) ./ (2h)

        h2 = h / 2
        plus[j] = physical_values[j] + h2
        minus[j] = physical_values[j] - h2
        fine[:, j] .= (
            _uq_branch_data_values(pep, anchor, physical_labels, plus,
                template, t0, max_required_deriv, opts) .-
            _uq_branch_data_values(pep, anchor, physical_labels, minus,
                template, t0, max_required_deriv, opts)
        ) ./ (2h2)
    end
    all(isfinite, fine) || throw(ArgumentError("branch jet Jacobian contains non-finite values"))
    richardson = fine .+ (fine .- coarse) ./ 3
    discrepancy = norm(fine - coarse, Inf) / max(norm(richardson, Inf), 1e-12)
    return center, richardson, discrepancy
end

function _uq_retained_branch_root(payload, solve_vars::AbstractVector)
    source_vars = payload.solve_vars
    source_root = payload.root
    length(source_vars) == length(source_root) || throw(ArgumentError(
        "retained branch root has $(length(source_root)) values for $(length(source_vars)) variables",
    ))
    length(solve_vars) == length(source_vars) || throw(ArgumentError(
        "reconstructed branch template has $(length(solve_vars)) variables; retained solve had $(length(source_vars))",
    ))

    # solve_with_hc promises root coordinates in its varlist order. Reorder by
    # exact symbolic identity (with a unique string fallback for reconstructed
    # Symbolics objects) instead of rebuilding the selected root from an oracle
    # ODE solve, which would target a nearby but different estimator.
    source_strings = string.(source_vars)
    indices = Int[]
    for variable in solve_vars
        exact = findall(v -> isequal(v, variable), source_vars)
        matches = isempty(exact) ? findall(==(string(variable)), source_strings) : exact
        length(matches) == 1 || throw(ArgumentError(
            "retained branch variable '$variable' is absent or ambiguous in the HC solve ordering",
        ))
        push!(indices, only(matches))
    end
    length(unique(indices)) == length(indices) || throw(ArgumentError(
        "reconstructed branch variables do not map injectively to the retained HC solve ordering",
    ))
    return Float64[source_root[index] for index in indices]
end

# A completed sibling is a deterministic estimator transform of its retained
# anchor. Propagate the anchor's selected-estimator covariance through
#
#     q_anchor -> exact observable jets -> selected sibling root -> q_sibling
#
# without rerunning clustering, ranking, or branch selection.
function _uq_branch_completion_report(
    pep::ParameterEstimationProblem,
    result::ParameterEstimationResult,
    target::UQTargetSnapshot,
    artifact,
    opts::EstimationOptions,
)
    artifact isa BranchCompletionUQArtifact || return _uq_unavailable(
        pep, target, :artifact_mismatch,
        "branch-completion identity is paired with $(typeof(artifact))",
    )
    payload = artifact.payload
    parent_identity = _run_ctx_identity(artifact.parent_candidate_id)
    isnothing(parent_identity) && return _uq_unavailable(
        pep, target, :missing_parent,
        "branch-completion parent identity $(artifact.parent_candidate_id) is unavailable",
    )
    parent_target = UQTargetSnapshot(
        selected_rank = 0,
        identity = parent_identity,
        lineage = _run_ctx_lineage(parent_identity),
        estimand = :conditional_on_branch_anchor,
        artifact_match = isnothing(_run_ctx_artifact(parent_identity.candidate_id)) ? :missing : :exact,
    )
    parent_uq = _compute_estimator_aware_uq(
        pep, payload.anchor, parent_target, opts,
    )
    if parent_uq isa UQUnavailable
        return _uq_unavailable(
            pep, target, :parent_uq_unavailable,
            "branch anchor UQ failed [$(parent_uq.reason)]: $(parent_uq.message)",
            warnings = copy(parent_uq.warnings),
        )
    end
    parent_uq isa UncertaintyReport || return _uq_unavailable(
        pep, target, :parent_uq_unavailable,
        "branch anchor did not produce a covariance report",
    )
    parent_influence = _run_ctx_uq_influence(parent_identity.candidate_id)
    isnothing(parent_influence) && return _uq_unavailable(
        pep, target, :parent_uq_unavailable,
        "branch anchor did not retain its raw-observation influence map",
    )
    parent_influence.coordinate_labels == parent_uq.param_labels ||
        return _uq_unavailable(
            pep, target, :artifact_mismatch,
            "branch anchor influence coordinates do not match its UQ report",
        )

    template = try
        _derive_param_homotopy_template(payload.si_template, payload.si_template.template_DD, opts)
    catch err
        _rethrow_if_interrupt(err)
        return _uq_unavailable(pep, target, :artifact_mismatch,
            "could not reconstruct retained branch template: $(sprint(showerror, err))")
    end
    length(template.template_equations) == length(template.solve_vars) ||
        return _uq_unavailable(pep, target, :non_square_branch_map,
            "retained branch template has $(length(template.template_equations)) equations for $(length(template.solve_vars)) solve variables")

    parent_values = copy(parent_uq.estimate_values)
    length(parent_values) == length(parent_uq.param_labels) ||
        return _uq_unavailable(pep, target, :parent_uq_unavailable,
            "branch anchor report lacks estimator-scale coordinates")
    data_values, data_jacobian, fd_discrepancy = try
        _uq_branch_data_jacobian(
            pep, payload.anchor, parent_uq.param_labels, parent_values,
            template, payload.t0, payload.max_required_deriv, opts,
        )
    catch err
        _rethrow_if_interrupt(err)
        return _uq_unavailable(pep, target, :numerical_failure,
            "branch anchor-to-jet linearization failed: $(sprint(showerror, err))")
    end
    child_root = try
        _uq_retained_branch_root(payload, template.solve_vars)
    catch err
        _rethrow_if_interrupt(err)
        return _uq_unavailable(pep, target, :artifact_mismatch,
            "selected sibling retained-root alignment failed: $(sprint(showerror, err))")
    end
    S, diagnostics = try
        _uq_exact_ift(
            template.template_equations, template.solve_vars,
            template.extended_data_vars, child_root, data_values,
        )
    catch err
        _rethrow_if_interrupt(err)
        return _uq_unavailable(pep, target, :numerical_failure,
            "selected sibling IFT failed: $(sprint(showerror, err))")
    end
    metadata = _uq_single_point_meta(
        SinglePointUQArtifact(
            template.template_equations, template.solve_vars,
            template.extended_data_vars, data_values, child_root,
            payload.perfect_interpolants, 1, payload.t0,
        ),
        pep,
    )
    template_labels = String[string(variable) for variable in template.extended_data_vars]
    augmented_values, augmented_metadata, augmented_labels = try
        _uq_augment_direct_observation_rows(
            pep, template.solve_vars, data_values, metadata, template_labels,
            Int[1], Float64[payload.t0],
        )
    catch err
        _rethrow_if_interrupt(err)
        return _uq_unavailable(pep, target, :artifact_mismatch,
            "selected sibling raw-output map failed: $(sprint(showerror, err))")
    end
    n_template_data = length(data_values)
    n_supplemental = length(augmented_values) - n_template_data
    S_augmented = n_supplemental == 0 ? S :
        hcat(S, zeros(size(S, 1), n_supplemental))

    size(data_jacobian, 2) == size(parent_influence.influence, 1) ||
        return _uq_unavailable(pep, target, :artifact_mismatch,
            "branch jet Jacobian and anchor influence dimensions do not agree")
    data_to_raw = zeros(length(augmented_values), size(parent_influence.influence, 2))
    data_to_raw[1:n_template_data, :] .= data_jacobian * parent_influence.influence
    for row in (n_template_data + 1):length(augmented_metadata)
        meta = augmented_metadata[row]
        meta.kind == :raw_observation || return _uq_unavailable(
            pep, target, :artifact_mismatch,
            "unexpected supplemental branch-data kind $(meta.kind)",
        )
        obs_idx = something(meta.obs_idx)
        obs_name = _uq_observation_name(pep.measured_quantities[obs_idx])
        raw_label = _uq_raw_observation_label(obs_name, 1)
        matches = findall(==(raw_label), parent_influence.observation_labels)
        length(matches) == 1 || return _uq_unavailable(
            pep, target, :artifact_mismatch,
            "branch raw output '$raw_label' is absent or ambiguous in the anchor influence map",
        )
        data_to_raw[row, only(matches)] = 1.0
    end
    local_map, local_labels, local_roles, _ = try
        _uq_local_output_map(
            pep, template.solve_vars, augmented_values, child_root,
            augmented_metadata, S_augmented,
        )
    catch err
        _rethrow_if_interrupt(err)
        return _uq_unavailable(pep, target, :artifact_mismatch,
            "selected sibling output map failed: $(sprint(showerror, err))")
    end
    real_rows = [
        i for i in eachindex(local_labels)
        if get(local_roles, local_labels[i], :unknown) != :transcendental
    ]
    influence = local_map[real_rows, :] * data_to_raw
    covariance = _psd_symmetric_matrix(
        influence * parent_influence.observation_covariance * influence',
    )
    child_data_covariance = _psd_symmetric_matrix(
        data_to_raw * parent_influence.observation_covariance * data_to_raw',
    )
    labels = local_labels[real_rows]
    roles = Dict(label => local_roles[label] for label in labels)
    states = pep.model.original_states
    real_state_indices = [
        i for (i, state) in enumerate(states)
        if !startswith(_uq_clean_name(state), "_trfn_")
    ]
    estimate_values = try
        _uq_selected_values(pep, result, real_state_indices)
    catch err
        _rethrow_if_interrupt(err)
        return _uq_unavailable(pep, target, :artifact_mismatch, sprint(showerror, err))
    end
    truth_values = _uq_truth_values(pep, real_state_indices)
    standard_deviation = sqrt.(max.(diag(covariance), 0.0))
    correlation = _uq_correlation_matrix(covariance, standard_deviation)
    max_cv = _uq_max_cv(standard_deviation, estimate_values)
    fd_degraded = !(isfinite(fd_discrepancy) && fd_discrepancy <= 1e-3)
    parent_degraded = parent_uq.linearization_diagnostics.degraded
    degraded = diagnostics.degraded || fd_degraded || parent_degraded
    status = degraded ? :degenerate : _uq_status_from_cv(max_cv)
    diagnostic_reason = fd_degraded ? :branch_map_unstable :
        parent_degraded && !diagnostics.degraded ? :parent_degraded : diagnostics.reason
    diagnostics = UQLinearizationDiagnostics(
        diagnostic_reason, diagnostics.root_residual_abs,
        diagnostics.root_residual_rel, diagnostics.jacobian_condition,
        diagnostics.jacobian_condition_equilibrated,
        diagnostics.linear_solve_backward_error,
        fd_discrepancy, Int[], degraded,
        diagnostics.gp_jitter_to_noise,
        diagnostics.gp_factorization_residual,
    )
    warnings = copy(parent_uq.warnings)
    push!(warnings,
        "Branch-completion UQ is conditional on the retained anchor, selected sibling, and unchanged algebraic branch.")
    push!(warnings,
        "The anchor-to-exact-jet Jacobian used two centered step sizes; its relative discrepancy was $(fd_discrepancy).")
    fd_degraded && push!(warnings,
        "The branch-map finite-difference stability gate exceeded 1e-3; covariance is reported for audit but is unreliable.")
    _uq_append_linearization_warnings!(warnings, diagnostics)
    ia = compute_practical_identifiability_index(
        covariance, labels, roles, estimate_values,
    )
    append!(warnings, ia.warnings)
    snapshot = LocalUQSnapshot(
        payload.t0, covariance, standard_deviation, labels, roles,
        estimate_values, correlation, max_cv, status, ia,
    )
    _run_ctx_register_uq_influence!(
        target.identity.candidate_id,
        UQInfluenceArtifact(
            Matrix{Float64}(influence),
            copy(parent_influence.observation_covariance),
            copy(parent_influence.observation_labels), copy(labels),
        ),
    )
    return UncertaintyReport(
        pep.name, payload.t0,
        copy(parent_uq.obs_names), deepcopy(parent_uq.obs_posterior_mean),
        deepcopy(parent_uq.obs_posterior_std),
        child_data_covariance, augmented_labels,
        covariance, standard_deviation, labels, roles, truth_values,
        correlation, max_cv, status, warnings,
        parent_uq.covariance_kind, parent_uq.noise_source, ia,
        :physical_initial_conditions, snapshot, nothing,
        _uq_target_with_match(target, :exact), estimate_values, diagnostics,
    )
end

function _compute_estimator_aware_uq(
    pep::ParameterEstimationProblem,
    result::ParameterEstimationResult,
    target::UQTargetSnapshot,
    opts::EstimationOptions,
)
    identity = target.identity
    if identity.estimator_kind in (
            :model_assisted_linear,
            :model_assisted_local_resolve,
        )
        return _uq_unavailable(
            pep,
            _uq_target_with_match(target, :unsupported),
            :unsupported_estimator,
            "model-assisted correction UQ is unavailable pending the pilot-through-correction influence map",
        )
    end
    artifact = _run_ctx_artifact(identity.candidate_id)
    if isnothing(artifact)
        return _uq_unavailable(
            pep, _uq_target_with_match(target, :missing), :missing_artifact,
            "the selected $(identity.estimator_kind) candidate has no exact retained production artifact",
        )
    end

    if identity.estimator_kind == :single_point_algebraic
        artifact isa SinglePointUQArtifact ||
            return _uq_unavailable(pep, target, :artifact_mismatch,
                "single-point identity is paired with $(typeof(artifact))")
        try
            return _uq_single_point_report(pep, result, target, artifact, opts)
        catch err
            _rethrow_if_interrupt(err)
            reason = occursin("AGPInterpolatorUQ", sprint(showerror, err)) ?
                :unsupported_interpolator : :numerical_failure
            return _uq_unavailable(pep, target, reason, sprint(showerror, err))
        end
    elseif identity.estimator_kind == :multipoint_algebraic
        artifact isa MultipointUQArtifact ||
            return _uq_unavailable(pep, target, :artifact_mismatch,
                "multipoint identity is paired with $(typeof(artifact))")
        try
            return _uq_multipoint_report(pep, result, target, artifact, opts)
        catch err
            _rethrow_if_interrupt(err)
            reason = occursin("AGPInterpolatorUQ", sprint(showerror, err)) ?
                :unsupported_interpolator : :numerical_failure
            return _uq_unavailable(pep, target, reason, sprint(showerror, err))
        end
    elseif identity.estimator_kind in (:trajectory_polish, :direct_optimization)
        return _uq_polish_report(pep, result, target, artifact, opts)
    elseif identity.estimator_kind == :branch_completed
        return _uq_branch_completion_report(pep, result, target, artifact, opts)
    end

    return _uq_unavailable(
        pep, target, :unsupported_estimator,
        "no estimator-matched covariance is implemented for $(identity.estimator_kind)",
    )
end
