# Per-derivative uncertainty estimation (σ_d).
#
# Motivation: the polynomial systems we solve are constructed from noisy
# derivative estimates produced by interpolants. Different interpolators
# disagree on the same (variable, time, order). Existing UQ uses GP posterior
# variance which under-reports actual error (correlated kernel bias, no
# accounting for boundary failure or higher-order amplification).
#
# This module provides a cheap, mostly-honest σ_d estimate from two sources:
# 1. **Cross-interpolator empirical spread** — std across all available
#    interpolators of nth_deriv(interp, order, t_eval). Data-driven; biased
#    when interpolators share failure modes (e.g. all hate boundaries).
# 2. **Order-decay floor** — `noise_level · κ^order`. Guards against a
#    cross-interpolator estimate that's spuriously small (e.g. only one
#    interpolator in run, or a fluke agreement).
#
# Final σ_d at (obs_idx, order, t) = `max(σ_empirical, σ_floor)`. The floor is a
# lower bound; we never claim a derivative is more certain than physics permits.

"""
    DerivativeUncertaintyEstimate

Per-(observable index, derivative order) uncertainty σ_d at a single evaluation
time `t_eval`. Σ_d for downstream covariance propagation is `diag(σ_d²)` flattened
in the same order as `SensitivityReport.data_sensitivity_data_labels`.

# Fields
- `t_eval::Float64` — the time at which σ_d is evaluated.
- `n_observables::Int`, `max_order::Int` — shape of the σ_d matrix.
- `sigma_d::Matrix{Float64}` — `sigma_d[obs_idx, order+1]` is the uncertainty
  estimate for the `order`-th derivative of observable `obs_idx` at `t_eval`.
- `source_values::Matrix{Vector{Tuple{Symbol, Float64}}}` — for each
  (obs_idx, order+1), the list of (interpolator source, derivative value) pairs
  that contributed. Useful for diagnostic inspection.
- `sources_used::Vector{Symbol}` — interpolator sources actually consulted.
- `floor_active::Matrix{Bool}` — `true` where the order-decay floor exceeded
  the cross-interpolator spread; signals "we believe derivatives are at least
  this uncertain regardless of interpolator agreement."
- `noise_level::Float64`, `order_decay_kappa::Float64` — provenance for the floor.
"""
struct DerivativeUncertaintyEstimate
    t_eval::Float64
    n_observables::Int
    max_order::Int
    sigma_d::Matrix{Float64}
    source_values::Matrix{Vector{Tuple{Symbol, Float64}}}
    sources_used::Vector{Symbol}
    floor_active::Matrix{Bool}
    noise_level::Float64
    order_decay_kappa::Float64
end

"""
    get_sigma_d(estimate, obs_idx, order) -> Float64

Per-(observable, order) lookup. Returns `NaN` if the request is out of range.
"""
function get_sigma_d(estimate::DerivativeUncertaintyEstimate, obs_idx::Int, order::Int)
    1 <= obs_idx <= estimate.n_observables || return NaN
    0 <= order <= estimate.max_order || return NaN
    return estimate.sigma_d[obs_idx, order + 1]
end


"""
    compute_sigma_d(interpolant_cache, measured_quantities, t_eval, max_order;
                    noise_level=0.0, order_decay_kappa=2.0) -> DerivativeUncertaintyEstimate

Build a `DerivativeUncertaintyEstimate` at a single time `t_eval`.

# Arguments
- `interpolant_cache::Dict{Symbol, <:AbstractDict}` — outer keyed by interpolator
  source symbol (e.g. `:aaad`, `:gpr_pivot`), inner keyed by Symbolics RHS of the
  measured-quantity equations (matches `pep.measured_quantities[i].rhs` after
  `Symbolics.diff2term` if necessary).
- `measured_quantities` — vector of measured-quantity equations (typically
  `pep.measured_quantities`); used to look up RHS keys.
- `t_eval::Float64` — evaluation time.
- `max_order::Int` — highest derivative order to compute σ_d for.

# Keyword args
- `noise_level::Float64 = 0.0` — measurement noise level (typically `opts.noise_level`).
- `order_decay_kappa::Float64 = 2.0` — multiplicative factor for the order-decay
  floor. Higher = more pessimistic about higher orders. Empirically reasonable
  default; not tuned to a specific model.

# Algorithm
For each (observable, derivative order) cell:
1. Pull `nth_deriv(interp, order, t_eval)` from each interpolator in the cache
   that has the observable's RHS as a key. Skip silently if a source can't be
   evaluated (e.g. the interpolator throws on a boundary).
2. Compute `σ_empirical = std(values_across_sources)` if at least 2 finite
   values were collected; otherwise `0.0`.
3. Compute `σ_floor = noise_level · order_decay_kappa^order`.
4. Final `σ_d = max(σ_empirical, σ_floor)`. Record which one dominated.

Returns a `DerivativeUncertaintyEstimate` with σ_d, full provenance of source
values used (for diagnostics), and a per-cell `floor_active` flag.
"""
function compute_sigma_d(
    interpolant_cache::Dict{Symbol, <:AbstractDict},
    measured_quantities,
    t_eval::Real,
    max_order::Int;
    noise_level::Real = 0.0,
    order_decay_kappa::Real = 2.0,
)
    n_obs = length(measured_quantities)
    sources_used = sort(collect(keys(interpolant_cache)))
    sigma_d = zeros(Float64, n_obs, max_order + 1)
    floor_active = falses(n_obs, max_order + 1)
    source_values = Matrix{Vector{Tuple{Symbol, Float64}}}(undef, n_obs, max_order + 1)
    @inbounds for i in 1:n_obs, j in 1:(max_order + 1)
        source_values[i, j] = Tuple{Symbol, Float64}[]
    end

    obs_rhs_lookup = [Symbolics.diff2term(eq.rhs) for eq in measured_quantities]

    for obs_idx in 1:n_obs
        obs_rhs = obs_rhs_lookup[obs_idx]
        for order in 0:max_order
            cell_values = source_values[obs_idx, order + 1]
            for source in sources_used
                inner = interpolant_cache[source]
                haskey(inner, obs_rhs) || continue
                interp = inner[obs_rhs]
                val = try
                    Float64(_estimation_derivative(interp, order, Float64(t_eval)))
                catch
                    NaN
                end
                isfinite(val) || continue
                push!(cell_values, (source, val))
            end

            σ_empirical = if length(cell_values) >= 2
                vals = Float64[v for (_, v) in cell_values]
                Float64(std(vals))
            else
                0.0
            end
            σ_floor = Float64(noise_level) * (Float64(order_decay_kappa)^order)
            if σ_floor > σ_empirical
                sigma_d[obs_idx, order + 1] = σ_floor
                floor_active[obs_idx, order + 1] = true
            else
                sigma_d[obs_idx, order + 1] = σ_empirical
                floor_active[obs_idx, order + 1] = false
            end
        end
    end

    return DerivativeUncertaintyEstimate(
        Float64(t_eval),
        n_obs,
        max_order,
        sigma_d,
        source_values,
        sources_used,
        floor_active,
        Float64(noise_level),
        Float64(order_decay_kappa),
    )
end

"""
    parse_sensitivity_label(label::AbstractString) -> Union{Nothing, Tuple{Int, Int}}

Parse a `data_sensitivity_data_labels` entry (e.g. `"y1_2"` for the 2nd derivative
of observable `y1`) into `(obs_idx, order)`. Returns `nothing` if the label
doesn't match the expected pattern.

The mapping from observable name (e.g. `"y1"`) to its index in `measured_quantities`
is required separately — this function only extracts the suffix order. The caller
typically holds an `obs_name_to_idx::Dict{String, Int}` built from the measured
quantities at the same time the SensitivityReport is constructed.
"""
function parse_sensitivity_label(label::AbstractString)
    # Thin adapter over the single shared parser (Phase D1); this caller's
    # failure contract is `nothing`.
    return parse_jet_label(label)
end
