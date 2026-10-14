# σ_d-aware sensitivity-based seed generation (Layer 2 of the sensitivity-seeds plan).
#
# Given a candidate pool from the polynomial pipeline and a single reference
# sensitivity matrix S (computed at the best candidate's local geometry), plus
# σ_d at the reference time, this module:
#
# - Computes a reference predicted parameter covariance Σ_x = S · diag(σ_d²) · S'.
# - Eigendecomposes Σ_x to identify the local sloppy directions.
# - For each candidate, emits the candidate as a seed plus probe seeds at
#   ±√λ · u along significant eigenvectors (sloppy-direction probes).
# - For pairs of candidates within Mahalanobis distance under Σ_x, emits
#   the average as a blend seed (denoising same-branch candidates with
#   different polynomial noise).
#
# v1 approximation: a single reference Σ_x for the whole pool. v2 (per-candidate
# Σ_x) requires a sensitivity-at-candidate refactor that's not in this PR.

"""
    SensitivitySeedReport

Provenance + diagnostics for a sensitivity-seed generation run. Returned by
`generate_sensitivity_seeds` so the caller can log what was emitted and why.
"""
struct SensitivitySeedReport
    n_candidates::Int
    n_probe_seeds::Int
    n_blend_seeds::Int
    eigenvalues::Vector{Float64}
    significant_eigenvalue_indices::Vector{Int}
    sigma_d_floor_active_count::Int
    mahalanobis_threshold_used::Float64
    notes::String
end

"""
    _build_obs_name_to_idx(measured_quantities) -> Dict{String, Int}

Map an observable's base name (e.g. `"y1"`, derived from `mq.lhs` minus `(t)`)
to its index in the measured-quantities vector. Used to align σ_d (keyed by
obs_idx) with sensitivity matrix columns (labeled by base-name + order).
"""
function _build_obs_name_to_idx(measured_quantities)
    out = Dict{String, Int}()
    for (i, mq) in enumerate(measured_quantities)
        name = replace(string(mq.lhs), r"\(.*\)$" => "")
        out[name] = i
    end
    return out
end

"""
    _align_sigma_d_to_labels(estimate, data_labels, obs_name_to_idx) -> Vector{Float64}

For each label in `data_labels` (column ordering of S), return the corresponding
σ_d value from `estimate`. Returns a vector of length `length(data_labels)`. Labels
that fail to parse or that don't match a known observable get a fallback value
of `0.0` (drops out of the diag, contributing nothing to Σ_x along that column).
"""
function _align_sigma_d_to_labels(
    estimate::DerivativeUncertaintyEstimate,
    data_labels::Vector{String},
    obs_name_to_idx::Dict{String, Int},
)
    n = length(data_labels)
    out = zeros(Float64, n)
    @inbounds for i in 1:n
        parsed = parse_sensitivity_label(data_labels[i])
        isnothing(parsed) && continue
        base_name, order = parsed
        haskey(obs_name_to_idx, base_name) || continue
        obs_idx = obs_name_to_idx[base_name]
        val = get_sigma_d(estimate, obs_idx, order)
        out[i] = isfinite(val) ? val : 0.0
    end
    return out
end

"""
    _chi2_quantile_95(dof::Int) -> Float64

Quick approximation of the chi-squared 95th percentile for `dof` degrees of
freedom. Uses the Wilson-Hilferty approximation `χ²_{0.95}(dof) ≈ dof·(1 - 2/(9·dof) + 1.6449·√(2/(9·dof)))³`.
For our purposes (a heuristic threshold for "same algebraic branch"), this is
within a few percent of the true quantile and avoids a Distributions.jl dep.
"""
function _chi2_quantile_95(dof::Int)
    dof <= 0 && return Inf
    h = 2.0 / (9.0 * dof)
    factor = 1.0 - h + 1.6449 * sqrt(h)
    return dof * factor^3
end

"""
    generate_sensitivity_seeds(candidates, S, sigma_d_estimate, polish_ctx, measured_quantities;
                               probe_scale=1.0, eigenvalue_threshold=0.01,
                               mahalanobis_threshold=-1.0)
        -> (Vector{Vector{Float64}}, SensitivitySeedReport)

Generate additional polish seeds from a candidate pool using a reference
sensitivity matrix `S` and per-derivative uncertainty `sigma_d_estimate`.

# Arguments
- `candidates::Vector{ParameterEstimationResult}` — pool from the polynomial
  pipeline (post-cluster, pre-polish).
- `S::Matrix{Float64}` — reference data-sensitivity matrix from
  `_compute_data_sensitivity`. Shape `(n_unknowns, n_data)`.
- `data_labels::Vector{String}` — column labels of `S`.
- `unknown_labels::Vector{String}` — row labels of `S`. Used to align with
  the candidate's parameter ordering via `polish_ctx.unknown_syms`/`param_syms`.
- `sigma_d_estimate::DerivativeUncertaintyEstimate` — at the reference t_eval.
- `polish_ctx::PolishContext` — for converting candidates to parameter vectors.
- `measured_quantities` — for `_build_obs_name_to_idx`.

# Keywords
- `probe_scale::Float64 = 1.0` — multiplier on √λ for ±σ probes.
- `eigenvalue_threshold::Float64 = 0.01` — only emit probes for eigenvalues
  with `λ_i / λ_max > eigenvalue_threshold`.
- `mahalanobis_threshold::Float64 = -1.0` — Mahalanobis-d² cutoff for blends.
  Negative means "use chi² 95% quantile for n_unknowns DoF."

# Returns
A tuple `(seeds, report)`:
- `seeds::Vector{Vector{Float64}}` — the emitted polish seeds in **external**
  (user-space) coordinates, one per row, one row per seed.
- `report::SensitivitySeedReport` — counts and diagnostics.
"""
function generate_sensitivity_seeds(
    candidates::Vector,
    S::AbstractMatrix{<:Real},
    data_labels::Vector{String},
    unknown_labels::Vector{String},
    sigma_d_estimate::DerivativeUncertaintyEstimate,
    polish_ctx,
    measured_quantities;
    probe_scale::Float64 = 1.0,
    eigenvalue_threshold::Float64 = 0.01,
    mahalanobis_threshold::Float64 = -1.0,
)
    # Coerce S to Float64 — `_compute_data_sensitivity` may return BigFloat for
    # ill-conditioned cases, but downstream eigendecomposition + Mahalanobis math
    # is fastest in Float64.
    S = Matrix{Float64}(S)
    seeds = Vector{Float64}[]
    n_probe = 0
    n_blend = 0
    notes_parts = String[]

    if isempty(candidates) || size(S, 1) == 0 || size(S, 2) == 0
        push!(notes_parts, "empty input — no seeds emitted")
        return seeds, SensitivitySeedReport(
            length(candidates), 0, 0, Float64[], Int[], 0,
            mahalanobis_threshold, join(notes_parts, "; "),
        )
    end

    obs_name_to_idx = _build_obs_name_to_idx(measured_quantities)
    sigma_d_vec = _align_sigma_d_to_labels(sigma_d_estimate, data_labels, obs_name_to_idx)

    # The sensitivity matrix S has rows for ALL SI unknowns (state derivatives, params,
    # ICs, plus any auxiliary algebraic variables introduced by the template). The polish
    # context vector is just (states; params). We project S down to the polish-context
    # ordering by matching unknown_labels to polish_ctx.unknown_syms / param_syms by name.
    # Rows of S that don't match (state derivatives etc.) are dropped — sensitivity to
    # data perturbations along those rows isn't directly actionable for seed generation.
    polish_names = vcat(
        [string(s) for s in polish_ctx.unknown_syms],
        [string(p) for p in polish_ctx.param_syms],
    )
    # Strip "(t)" wrapping that ModelingToolkit applies to states.
    polish_names_stripped = [replace(n, r"\(.*\)$" => "") for n in polish_names]
    S_label_lookup = Dict{String, Int}()
    for (i, label) in enumerate(unknown_labels)
        # SI labels look like "x1_0" for the 0-th derivative of x1 (i.e., x1 itself),
        # "a12_0" for parameter a12. Strip the "_0" suffix when matching.
        base_label = replace(label, r"_0$" => "")
        if !haskey(S_label_lookup, base_label)
            S_label_lookup[base_label] = i
        end
    end
    row_perm = Int[]
    matched_polish_indices = Int[]
    for (j, pname) in enumerate(polish_names_stripped)
        idx = get(S_label_lookup, pname, 0)
        if idx > 0
            push!(row_perm, idx)
            push!(matched_polish_indices, j)
        end
    end

    if isempty(row_perm)
        push!(notes_parts, "no S rows matched polish-context names")
        return seeds, SensitivitySeedReport(
            length(candidates), 0, 0, Float64[], Int[], 0,
            mahalanobis_threshold, join(notes_parts, "; "),
        )
    end

    n_unknowns_ctx = length(polish_names_stripped)
    # Project: S_proj is (n_matched, n_data); sensitivity of MATCHED params/states only.
    S_proj = S[row_perm, :]

    # Predict Σ_x_proj = S_proj · diag(σ_d²) · S_proj'  — covariance over MATCHED unknowns.
    sigma_d_sq = sigma_d_vec .^ 2
    Sigma_x_proj = S_proj * Diagonal(sigma_d_sq) * S_proj'
    Sigma_x_proj = Symmetric((Sigma_x_proj + Sigma_x_proj') / 2)

    # Embed Σ_x_proj into the full (n_unknowns_ctx × n_unknowns_ctx) space with zeros
    # in unmatched rows/cols. Probes along eigenvectors of this embedded matrix only
    # move the matched coordinates — exactly what we want.
    Sigma_x = zeros(Float64, n_unknowns_ctx, n_unknowns_ctx)
    for (a, ia) in enumerate(matched_polish_indices)
        for (b, ib) in enumerate(matched_polish_indices)
            Sigma_x[ia, ib] = Sigma_x_proj[a, b]
        end
    end
    Sigma_x = Symmetric(Sigma_x)
    push!(notes_parts, "S row projection: $(length(matched_polish_indices))/$n_unknowns_ctx polish vars matched ($(size(S, 1)) total in S)")

    # Eigendecomposition. eigen returns eigenvalues in ascending order.
    F = try
        eigen(Sigma_x)
    catch err
        push!(notes_parts, "eigendecomposition failed: $(err)")
        return seeds, SensitivitySeedReport(
            length(candidates), 0, 0, Float64[], Int[], 0,
            mahalanobis_threshold, join(notes_parts, "; "),
        )
    end
    eigenvalues = collect(F.values)
    eigenvectors = F.vectors

    λ_max = maximum(abs.(eigenvalues))
    significant_idx = Int[]
    if λ_max > 0.0
        for k in 1:length(eigenvalues)
            λ_k = abs(eigenvalues[k])
            if λ_k > 0.0 && (λ_k / λ_max) >= eigenvalue_threshold
                push!(significant_idx, k)
            end
        end
    end

    sigma_d_floor_count = count(sigma_d_estimate.floor_active)

    # Per-candidate emission: candidate + L2-minimum probe along each sloppy direction.
    #
    # For each significant eigenvector u_k (a sloppy direction), instead of probing both
    # x_c ± √λ_k · u_k (which doubles seed count and tends to overshoot when truth is small
    # along u_k), emit the SINGLE point on the credibility line that minimizes ‖x‖².
    # The minimizer of ‖x_c + t·u_k‖² is t* = −(x_c · u_k); we clip to the
    # ±(probe_scale · σ_k) credibility ball. This uses "small-along-sloppy-direction"
    # as a soft prior and concretely fixes the fitzhugh-style overshoot where x_c had a
    # large b and ±1σ flipped its sign.
    for candidate in candidates
        candidate_vec = _candidate_to_external_vector(candidate, polish_ctx)
        any(isnan, candidate_vec) && continue
        push!(seeds, candidate_vec)
        for k in significant_idx
            σ_k = sqrt(max(eigenvalues[k], 0.0))
            σ_k <= 0.0 && continue
            u_k = eigenvectors[:, k]
            proj = dot(candidate_vec, u_k)
            t_star = clamp(-proj, -probe_scale * σ_k, probe_scale * σ_k)
            push!(seeds, candidate_vec .+ t_star .* u_k)
            n_probe += 1
        end
    end

    # Pairwise Mahalanobis-gated blends. Use Σ_x as the metric for the test.
    threshold = mahalanobis_threshold > 0 ? mahalanobis_threshold : _chi2_quantile_95(n_unknowns_ctx)
    Sigma_x_inv = try
        inv(Sigma_x)
    catch
        # Fall back to pseudoinverse for ill-conditioned Σ_x
        pinv(Matrix(Sigma_x))
    end

    candidate_vectors = Vector{Vector{Float64}}(undef, length(candidates))
    for (i, c) in enumerate(candidates)
        candidate_vectors[i] = _candidate_to_external_vector(c, polish_ctx)
    end

    for i in 1:length(candidates)
        v_i = candidate_vectors[i]
        any(isnan, v_i) && continue
        for j in (i + 1):length(candidates)
            v_j = candidate_vectors[j]
            any(isnan, v_j) && continue
            δ = v_i .- v_j
            d2 = (δ' * Sigma_x_inv * δ) |> first |> Float64
            if isfinite(d2) && d2 <= threshold
                push!(seeds, (v_i .+ v_j) ./ 2)
                n_blend += 1
            end
        end
    end

    push!(notes_parts, "λ_max=$(round(λ_max; sigdigits=3))")
    push!(notes_parts, "n_significant_eigvecs=$(length(significant_idx))")
    if !isempty(significant_idx)
        push!(notes_parts, "λ_range=[$(round(minimum(abs.(eigenvalues[significant_idx])); sigdigits=3)), $(round(λ_max; sigdigits=3))]")
    end

    return seeds, SensitivitySeedReport(
        length(candidates), n_probe, n_blend, eigenvalues, significant_idx,
        sigma_d_floor_count, threshold, join(notes_parts, "; "),
    )
end

"""
    _candidate_to_external_vector(candidate, polish_ctx) -> Vector{Float64}

Convert a candidate's parameters/states to the parameter-vector ordering used
by `polish_ctx` (`[unknown_syms; param_syms]`), in **external** (user-space)
coordinates. Mirrors `_candidate_seed_vector` from `benchmark_sweeps.jl` but
without bound clamping — caller decides clamping policy at seed-emission time.
"""
function _candidate_to_external_vector(candidate, polish_ctx)
    state_lookup = Dict{String, Float64}()
    for (k, v) in candidate.states
        state_lookup[string(k)] = Float64(v)
    end
    param_lookup = Dict{String, Float64}()
    for (k, v) in candidate.parameters
        param_lookup[string(k)] = Float64(v)
    end
    ic_vec = Float64[get(state_lookup, string(s), NaN) for s in polish_ctx.unknown_syms]
    p_vec = Float64[get(param_lookup, string(p), NaN) for p in polish_ctx.param_syms]
    return vcat(ic_vec, p_vec)
end

"""
    _maybe_augment_with_sensitivity_seeds(PEP, solved_res, ctx, setup_data, opts)
        -> Vector{ParameterEstimationResult}

Augment the candidate pool with sensitivity-aware seeds.

This is the wiring entry point used by `optimized_multishot_parameter_estimation`
when `opts.use_sensitivity_seeds = true`. The strategy:

1. Pick the best-fit candidate as a reference (its local Σ_x is the proxy for
   the whole pool's local geometry — a v1 simplification; per-candidate Σ_x is
   v2 work).
2. Build a "candidate-as-truth" PEP via `_consensus_candidate_pep` so that
   `compute_oracle_taylor_coefficients` and `_compute_data_sensitivity` evaluate
   at the candidate's parameters/states rather than the user's `pep.p_true`.
3. Compute σ_d at the reference t_eval. Use `setup_data.interpolants` as a
   single-source cache (cross-interpolator spread benefits require multi-source
   accumulation, which is a follow-up; for now σ_d falls through to the
   order-decay floor `noise_level · κ^order` for one-source runs).
4. Call `generate_sensitivity_seeds` and wrap the resulting vectors as
   `ParameterEstimationResult`s.
5. Append the new seeds to `solved_res` and return.

Failures (any of the underlying calls throwing, NaN values, mismatched
dimensions) are caught upstream — this function returns `solved_res` unchanged
on error.
"""
function _maybe_augment_with_sensitivity_seeds(
    PEP::ParameterEstimationProblem,
    solved_res::Vector,
    ctx,
    setup_data,
    opts::EstimationOptions;
    interpolant_cache::Union{Nothing, AbstractDict} = nothing,
)
    # Pick reference candidate: lowest finite fit error.
    err_key(c) = (isnothing(c.err) || !isfinite(c.err)) ? Inf : Float64(c.err)
    ref_idx = argmin([err_key(c) for c in solved_res])
    ref_candidate = solved_res[ref_idx]

    # Candidate-as-truth PEP: lets `compute_oracle_taylor_coefficients` evaluate
    # the ODE at the candidate's parameters/IC rather than `PEP.p_true`/`PEP.ic`.
    ref_pep = _consensus_candidate_pep(PEP, ref_candidate)

    # Pick a single t_eval for the reference. Use the first stored time index,
    # falling back to the data sample's first time.
    t_vec = PEP.data_sample["t"]
    t_eval = if !isempty(setup_data.time_index_set)
        idx = first(setup_data.time_index_set)
        Float64(t_vec[clamp(idx, 1, length(t_vec))])
    else
        Float64(t_vec[1])
    end

    # Maximum derivative order present in the SI template.
    max_order = if isempty(setup_data.good_deriv_level)
        2
    else
        Int(maximum(values(setup_data.good_deriv_level)))
    end

    # Taylor coefficients at the candidate's local geometry.
    state_taylor = compute_oracle_taylor_coefficients(ref_pep, t_eval, max_order + 2)
    obs_taylor = compute_observable_taylor_coefficients(ref_pep, state_taylor, t_eval, max_order + 2)

    # Reference sensitivity matrix S = -(∂F/∂x)⁻¹·(∂F/∂d) at the candidate.
    S, data_labels, _data_roles, unknown_labels, _unknown_roles = _compute_data_sensitivity(
        ref_pep, setup_data, t_eval, [], [];
        state_taylor = state_taylor, obs_taylor = obs_taylor,
    )

    if size(S, 1) == 0 || size(S, 2) == 0
        if !opts.nooutput
            println("[Sensitivity seeds] sensitivity matrix empty; skipping augmentation")
        end
        return solved_res
    end

    # σ_d: prefer multi-source interpolant cache (cross-interpolator spread is the
    # primary signal). Fall back to single-source from setup_data.interpolants when
    # the caller didn't pass a cache (e.g. direct calls from tests).
    σd_cache = if !isnothing(interpolant_cache) && !isempty(interpolant_cache)
        Dict{Symbol, AbstractDict}(k => v for (k, v) in interpolant_cache)
    else
        Dict{Symbol, AbstractDict}(:setup => setup_data.interpolants)
    end
    sigma_d_est = compute_sigma_d(
        σd_cache, PEP.measured_quantities, t_eval, max_order;
        noise_level = max(opts.noise_level, 0.0),
        order_decay_kappa = opts.sensitivity_sigma_d_kappa,
    )

    seed_vecs, report = generate_sensitivity_seeds(
        Vector(solved_res), S, data_labels, unknown_labels, sigma_d_est, ctx, PEP.measured_quantities;
        probe_scale = opts.sensitivity_seed_probe_scale,
        eigenvalue_threshold = opts.sensitivity_seed_eigenvalue_threshold,
        mahalanobis_threshold = opts.sensitivity_seed_mahalanobis_threshold,
    )

    if isempty(seed_vecs)
        if !opts.nooutput
            println("[Sensitivity seeds] no seeds emitted; skipping augmentation")
        end
        return solved_res
    end

    new_candidates = seed_vectors_to_candidates(
        seed_vecs, ctx, t_eval; source_type = :sensitivity_seed,
        parent_candidate_id = ensure_result_estimator_identity!(ref_candidate).candidate_id,
    )

    # Evaluate the ODE-trajectory residual for each seed using the polish-context
    # scalar loss. This makes seeds rank against algebraic candidates even when
    # `polish_solutions = false` — i.e., the "no-polish wins" use case where
    # sensitivity seeds compete on raw algebraic quality.
    n_evaluated = 0
    for c in new_candidates
        try
            p_external = _candidate_to_external_vector(c, ctx)
            any(!isfinite, p_external) && continue
            # Apply external bound clamping if available, then transform to internal
            # coords matching the polish_ctx's transform setup.
            if !isnothing(ctx.lb) && !isnothing(ctx.ub)
                p_external = clamp.(p_external, ctx.lb, ctx.ub)
            end
            # Coordinate-INDEPENDENT scoring (same fix as synthesize_aggregates): score
            # the seed's true loss at its external params, not via the search-coordinate
            # round-trip (which distorts err for :shifted_log + large shift).
            loss = Float64(_trajectory_sse(ctx, p_external))
            if isfinite(loss)
                c.err = loss
                if !isnothing(c.provenance)
                    c.provenance.post_polish_error = loss
                    c.provenance.pre_polish_error = loss
                end
                n_evaluated += 1
            end
        catch
            # Leave c.err as nothing if evaluation fails; downstream cluster/sort tolerates this.
        end
    end

    if !opts.nooutput
        n_sources = length(σd_cache)
        floor_count = report.sigma_d_floor_active_count
        n_sig_eigs = length(report.significant_eigenvalue_indices)
        println("[Sensitivity seeds] $(length(solved_res)) → $(length(solved_res) + length(new_candidates)) seeds " *
                "(probes=$(report.n_probe_seeds), blends=$(report.n_blend_seeds), " *
                "sources=$n_sources, sig_eigvecs=$n_sig_eigs, σ_d_floor_cells=$floor_count, " *
                "err_evaluated=$n_evaluated/$(length(new_candidates)))")
    end
    return vcat(solved_res, new_candidates)
end

"""
    seed_vectors_to_candidates(seeds, polish_ctx, t0; source_type=:sensitivity_seed)
        -> Vector{ParameterEstimationResult}

Wrap each `Vector{Float64}` seed (in `[states; params]` order matching
`polish_ctx.unknown_syms ++ polish_ctx.param_syms`) into a synthetic
`ParameterEstimationResult` with `err = nothing` and provenance tagged
`source_type=:sensitivity_seed`. The polish phase will compute the actual
err post-polish via `set_result_lineage!`.
"""
function seed_vectors_to_candidates(
    seeds::AbstractVector{<:AbstractVector{<:Real}},
    polish_ctx,
    t0::Real;
    source_type::Symbol = :sensitivity_seed,
    parent_candidate_id::Int = 0,
)
    n_ic = length(polish_ctx.unknown_syms)
    n_p = length(polish_ctx.param_syms)
    out = ParameterEstimationResult[]
    sizehint!(out, length(seeds))

    for seed in seeds
        length(seed) == n_ic + n_p || continue
        any(!isfinite, seed) && continue
        states = OrderedDict{Num, Float64}(
            polish_ctx.unknown_syms[i] => Float64(seed[i]) for i in 1:n_ic
        )
        params = OrderedDict{Num, Float64}(
            polish_ctx.param_syms[i] => Float64(seed[n_ic + i]) for i in 1:n_p
        )

        result = ParameterEstimationResult(
            params,
            states,
            Float64(t0),
            nothing,                 # err — filled in by polish
            nothing,                 # return_code
            length(polish_ctx.t_vector),
            Float64(t0),
            OrderedDict{Num, Float64}(),
            Set{Num}(),
            nothing,                 # solution
        )
        result.provenance = ResultProvenance(
            primary_method = :direct_opt,
            source_type = source_type,
            polish_applied = false,
        )
		set_result_estimator_identity!(result;
			estimator_kind = :sensitivity_seed,
			data_scope = :derived,
			time_values = Float64[t0],
			parent_candidate_ids = parent_candidate_id > 0 ? Int[parent_candidate_id] : Int[],
		)
        sync_result_contract!(result)
        push!(out, result)
    end
    return out
end
