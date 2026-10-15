using OrderedCollections
using Symbolics
using OrdinaryDiffEq
using SciMLBase

"""
	AbstractInterpolator

Abstract type for interpolation function objects.
All interpolators should be callable with a single argument and return the interpolated value.

Defined here (not in core/derivatives.jl) so the type exists before any file that
annotates with it — `parameter_estimation.jl` (included earlier than derivatives.jl)
references `Dict{Num, AbstractInterpolator}`, which previously survived only by lazy
in-function evaluation.
"""
abstract type AbstractInterpolator end

"""
    OrderedODESystem

A ModelingToolkit system together with its parameters and states in a fixed
order. Results list their values in this order.

The keyword constructor of [`ParameterEstimationProblem`](@ref) builds one from
a `System`, so you rarely need to.

# Fields
- `system`: the ModelingToolkit system.
- `original_parameters`: the parameters, in order.
- `original_states`: the states, in order.
"""
struct OrderedODESystem
    system::ModelingToolkit.AbstractSystem
    original_parameters::Vector{Num}
    original_states::Vector{Num}
end

#function remake(pep::OrderedODESystem; p=nothing, u0=nothing)
#    sys = pep.system
#    if !isnothing(p)
#        sys = remake(sys, p=p)
#    end
#    if !isnothing(u0)
#        sys = remake(sys, u0=u0)
#    end
#    return OrderedODESystem(sys, pep.original_parameters, pep.original_states)
#end

"""
    ParameterEstimationProblem

A model, the quantities measured on it, and the measurements.

Build one from a ModelingToolkit `System` with
`ParameterEstimationProblem(system, measured_quantities; data)`. The example
models in `ODEParameterEstimation.Examples`, such as `lotka_volterra()`, return
ready-made problems without data.

# Fields
- `name`: a label for the problem.
- `model`: the system, as an [`OrderedODESystem`](@ref).
- `measured_quantities`: equations such as `y1 ~ x1`, one per measured series.
- `data_sample`: the measurements, or `nothing` before any are attached. Either
  a dictionary with the time points under `"t"` and each series under the
  right-hand side of its measured quantity, or an [`ObservationData`](@ref).
- `solver`: the ODE solver used to simulate the model while estimating.
- `p_true`, `ic`: the true parameter values and initial conditions, `NaN` where
  they are not known.
- `recommended_time_interval`, `unident_count`: notes carried by the built-in
  example models. Estimation does not read them.

The struct can also be built by passing all nine fields in the order `name`,
`model`, `measured_quantities`, `data_sample`, `recommended_time_interval`,
`solver`, `p_true`, `ic`, `unident_count`.
"""
struct ParameterEstimationProblem
    name::String
    model::OrderedODESystem
    measured_quantities::Vector{ModelingToolkit.Equation}
    data_sample::Union{Nothing, OrderedDict{Union{String, Num}, Vector{Float64}}, ObservationData}
    recommended_time_interval::Union{Nothing, Vector{Float64}}
    solver::Any  # Use Any for now since the exact type hierarchy can be complex
    p_true::OrderedDict{Symbolics.Num, Float64}
    ic::OrderedDict{Symbolics.Num, Float64}
    unident_count::Int
end

# Constants for analysis and clustering
const CLUSTERING_THRESHOLD = 0.00001  # 0.001% relative difference threshold
const MAX_ERROR_THRESHOLD = 0.5       # Maximum acceptable error
const IMAG_THRESHOLD = 1e-8           # Threshold for ignoring imaginary components
const MAX_SOLUTIONS = 20              # Maximum number of solutions to consider if no good ones found
const DEFAULT_BOUND_MULTIPLIER = 1e9  # Multiplier for data scale: blown-backsolve DETECTION threshold (kept wide on purpose)
const DEFAULT_POLISH_BOUND_MULTIPLIER = 1e6  # Narrower multiplier for the no-user-bounds POLISH search box (decoupled from detection; tune toward 1e3 if needed)
const UQ_CI_Z = 1.96                  # Normal 95% quantile — used for BOTH the displayed CI half-width and the coverage check (must stay identical)
const UQ_EQUILIBRATED_CONDITION_THRESHOLD = 1e6
const UQ_BACKWARD_ERROR_THRESHOLD = 1e-10
const UQ_CONDITION_ERROR_PRODUCT_THRESHOLD = 1e-4

"""
    NumericalIdentifiabilityAdvisory

Structured best-effort numerical identifiability diagnostics.

# Fields
- `status::Symbol`: `:available`, `:failed`, or `:unavailable`
- `recommended_num_points::Union{Nothing, Int}`: Recommended point count heuristic
- `recommended_deriv_level::Dict{Int, Int}`: Recommended derivative depth by observable index
- `flagged_variables::Set{Num}`: Variables flagged as numerically fragile at the probe point
- `notes::Vector{Symbol}`: Advisory notes such as `:heuristic_fallback`
- `failure_reason::Union{Nothing, String}`: Best-effort failure summary when advisory analysis failed
"""
struct NumericalIdentifiabilityAdvisory
    status::Symbol
    recommended_num_points::Union{Nothing, Int}
    recommended_deriv_level::Dict{Int, Int}
    flagged_variables::Set{Num}
    notes::Vector{Symbol}
    failure_reason::Union{Nothing, String}
end

"""
    EstimatorIdentity

Canonical, run-scoped identity of the estimator that produced one candidate.
Unlike the legacy flat provenance fields, this record describes the estimator
that produced the CURRENT values and preserves parent IDs whenever a later
stage transforms a candidate.
"""
struct EstimatorIdentity
    candidate_id::Int
    estimator_kind::Symbol
    data_scope::Symbol
    time_indices::Vector{Int}
    time_values::Vector{Float64}
    interpolator_source::Union{Nothing, Symbol}
    parent_candidate_ids::Vector{Int}
end

function EstimatorIdentity(;
    candidate_id::Integer = 0,
    estimator_kind::Symbol = :unknown,
    data_scope::Symbol = :unknown,
    time_indices::AbstractVector{<:Integer} = Int[],
    time_values::AbstractVector{<:Real} = Float64[],
    interpolator_source::Union{Nothing, Symbol} = nothing,
    parent_candidate_ids::AbstractVector{<:Integer} = Int[],
)
    return EstimatorIdentity(
        Int(candidate_id), estimator_kind, data_scope,
        Int[time_indices...], Float64[time_values...], interpolator_source,
        Int[parent_candidate_ids...],
    )
end

copy_estimator_identity(identity::EstimatorIdentity) = EstimatorIdentity(
    candidate_id = identity.candidate_id,
    estimator_kind = identity.estimator_kind,
    data_scope = identity.data_scope,
    time_indices = identity.time_indices,
    time_values = identity.time_values,
    interpolator_source = identity.interpolator_source,
    parent_candidate_ids = identity.parent_candidate_ids,
)

function NumericalIdentifiabilityAdvisory(;
    status::Symbol = :unavailable,
    recommended_num_points::Union{Nothing, Integer} = nothing,
    recommended_deriv_level = Dict{Int, Int}(),
    flagged_variables = Set{Num}(),
    notes = Symbol[],
    failure_reason::Union{Nothing, AbstractString} = nothing,
)
    return NumericalIdentifiabilityAdvisory(
        status,
        isnothing(recommended_num_points) ? nothing : Int(recommended_num_points),
        Dict{Int, Int}(Int(k) => Int(v) for (k, v) in pairs(recommended_deriv_level)),
        Set{Num}(flagged_variables),
        Symbol[notes...],
        isnothing(failure_reason) ? nothing : String(failure_reason),
    )
end

"""
    ResultProvenance

Structured lineage metadata for a parameter-estimation result.

# Fields
- `primary_method::Symbol`: `:algebraic` or `:direct_opt`
- `interpolator_source::Union{Nothing, Symbol}`: Which interpolator produced the candidate
- `rescue_path::Symbol`: `:none`, `:algebraic_resolve_t0`, `:algebraic_resolve_seeded`, `:direct_opt_fallback`
- `source_shooting_index::Union{Nothing, Int}`: Shooting-point index that produced the candidate
- `source_candidate_index::Union{Nothing, Int}`: Stable candidate index within the producing phase
- `pre_polish_error::Union{Nothing, Float64}`: Error before polishing, when applicable
- `post_polish_error::Union{Nothing, Float64}`: Error after polishing, when applicable
- `polish_applied::Bool`: Whether a polishing stage was applied
- `representative_assignments::OrderedDict{Num, Float64}`: Values assigned only because variables were already known structurally unidentifiable
- `structural_fix_set::OrderedDict{Num, Float64}`: Representative structural fix set derived from SI structural outputs
- `template_status::Union{Nothing, Symbol}`: Template dimension status after structural fixing (2026-08: replaced the vestigial residual_fix_set + before/after status pair — no residual-repair mechanism ever existed, so the pair was identical by construction)
- `equations_dropped_by_rank_trimming::Vector{Int}`: Equation indices removed by rank-based template trimming
- `practical_identifiability_status::Symbol`: Practical/numerical identifiability assessment status for this flow
- `numerical_advisory::Union{Nothing, NumericalIdentifiabilityAdvisory}`: Best-effort advisory numerical diagnostics and heuristic recommendations
- `notes::Vector{Symbol}`: Additional lineage/debug notes
"""
mutable struct ResultProvenance
    primary_method::Symbol
    interpolator_source::Union{Nothing, Symbol}
    rescue_path::Symbol
    source_shooting_index::Union{Nothing, Int}
    source_candidate_index::Union{Nothing, Int}
    pre_polish_error::Union{Nothing, Float64}
    post_polish_error::Union{Nothing, Float64}
    polish_applied::Bool
    representative_assignments::OrderedDict{Num, Float64}
    structural_fix_set::OrderedDict{Num, Float64}
    template_status::Union{Nothing, Symbol}
    equations_dropped_by_rank_trimming::Vector{Int}
    practical_identifiability_status::Symbol
    numerical_advisory::Union{Nothing, NumericalIdentifiabilityAdvisory}
    notes::Vector{Symbol}
    # Multipoint provenance (SP/MP origin tracking)
    source_type::Symbol                                    # :single_point | :multipoint | :synthesized_aggregate | :imported | ...
    multipoint_time_indices::Union{Nothing, Vector{Int}}   # time indices in combo, e.g. [376, 1126]
    multipoint_combo_index::Union{Nothing, Int}            # which combo (1-based) produced this solution
    # Synthesized-aggregate provenance (set when source_type == :synthesized_aggregate)
    aggregation_strategy::Symbol                           # :median, :trim25_mean, :mean, :weighted_median, ... ; :none for non-synthesized candidates
    aggregation_source_indices::Vector{Int}                # indices into the upstream candidate pool that were aggregated
    # Polish-stage HC source index (set by _polish_batch_from_context after polish);
    # 1-based index into the raw HC candidate list passed to _polish_cluster_metadata.
    # Used for offline branch-clustering analysis (paired with EstimationOptions.dump_raw_candidates_path).
    polish_source_hc_idx::Union{Nothing, Int}
    # Canonical identity of the estimator that produced the current values.
    # Appended for positional-constructor compatibility.
    estimator_identity::EstimatorIdentity
end

function ResultProvenance(;
    primary_method::Symbol = :algebraic,
    interpolator_source::Union{Nothing, Symbol} = nothing,
    rescue_path::Symbol = :none,
    source_shooting_index::Union{Nothing, Int} = nothing,
    source_candidate_index::Union{Nothing, Int} = nothing,
    pre_polish_error::Union{Nothing, Real} = nothing,
    post_polish_error::Union{Nothing, Real} = nothing,
    polish_applied::Bool = false,
    representative_assignments = OrderedDict{Num, Float64}(),
    structural_fix_set = OrderedDict{Num, Float64}(),
    template_status::Union{Nothing, Symbol} = nothing,
    equations_dropped_by_rank_trimming = Int[],
    practical_identifiability_status::Symbol = :not_assessed,
    numerical_advisory::Union{Nothing, NumericalIdentifiabilityAdvisory} = nothing,
    notes = Symbol[],
    source_type::Symbol = :single_point,
    multipoint_time_indices::Union{Nothing, Vector{Int}} = nothing,
    multipoint_combo_index::Union{Nothing, Int} = nothing,
    aggregation_strategy::Symbol = :none,
    aggregation_source_indices::AbstractVector = Int[],
    polish_source_hc_idx::Union{Nothing, Int} = nothing,
    estimator_identity::EstimatorIdentity = EstimatorIdentity(),
)
    return ResultProvenance(
        primary_method,
        interpolator_source,
        rescue_path,
        source_shooting_index,
        source_candidate_index,
        isnothing(pre_polish_error) ? nothing : Float64(pre_polish_error),
        isnothing(post_polish_error) ? nothing : Float64(post_polish_error),
        polish_applied,
        OrderedDict{Num, Float64}(k => Float64(v) for (k, v) in representative_assignments),
        OrderedDict{Num, Float64}(k => Float64(v) for (k, v) in structural_fix_set),
        template_status,
        Int[equations_dropped_by_rank_trimming...],
        practical_identifiability_status,
        isnothing(numerical_advisory) ? nothing : deepcopy(numerical_advisory),
        Symbol[notes...],
        source_type,
        isnothing(multipoint_time_indices) ? nothing : Int[multipoint_time_indices...],
        multipoint_combo_index,
        aggregation_strategy,
        Int[aggregation_source_indices...],
        polish_source_hc_idx,
        copy_estimator_identity(estimator_identity),
    )
end

function copy_provenance(
    provenance::ResultProvenance;
    primary_method = provenance.primary_method,
    interpolator_source = provenance.interpolator_source,
    rescue_path = provenance.rescue_path,
    source_shooting_index = provenance.source_shooting_index,
    source_candidate_index = provenance.source_candidate_index,
    pre_polish_error = provenance.pre_polish_error,
    post_polish_error = provenance.post_polish_error,
    polish_applied = provenance.polish_applied,
    representative_assignments = provenance.representative_assignments,
    structural_fix_set = provenance.structural_fix_set,
    template_status = provenance.template_status,
    equations_dropped_by_rank_trimming = provenance.equations_dropped_by_rank_trimming,
    practical_identifiability_status = provenance.practical_identifiability_status,
    numerical_advisory = provenance.numerical_advisory,
    notes = provenance.notes,
    source_type = provenance.source_type,
    multipoint_time_indices = provenance.multipoint_time_indices,
    multipoint_combo_index = provenance.multipoint_combo_index,
    aggregation_strategy = provenance.aggregation_strategy,
    aggregation_source_indices = provenance.aggregation_source_indices,
    polish_source_hc_idx = provenance.polish_source_hc_idx,
    estimator_identity = provenance.estimator_identity,
)
    return ResultProvenance(
        primary_method = primary_method,
        interpolator_source = interpolator_source,
        rescue_path = rescue_path,
        source_shooting_index = source_shooting_index,
        source_candidate_index = source_candidate_index,
        pre_polish_error = pre_polish_error,
        post_polish_error = post_polish_error,
        polish_applied = polish_applied,
        representative_assignments = deepcopy(representative_assignments),
        structural_fix_set = deepcopy(structural_fix_set),
        template_status = template_status,
        equations_dropped_by_rank_trimming = copy(equations_dropped_by_rank_trimming),
        practical_identifiability_status = practical_identifiability_status,
        numerical_advisory = isnothing(numerical_advisory) ? nothing : deepcopy(numerical_advisory),
        notes = copy(notes),
        source_type = source_type,
        multipoint_time_indices = isnothing(multipoint_time_indices) ? nothing : copy(multipoint_time_indices),
        multipoint_combo_index = multipoint_combo_index,
        aggregation_strategy = aggregation_strategy,
        aggregation_source_indices = copy(aggregation_source_indices),
        polish_source_hc_idx = polish_source_hc_idx,
        estimator_identity = copy_estimator_identity(estimator_identity),
    )
end

"""
    compatibility_return_code(provenance::ResultProvenance) -> Symbol

Map canonical provenance metadata onto the legacy `return_code` compatibility field.
"""
function compatibility_return_code(provenance::ResultProvenance)::Symbol
    provenance.rescue_path != :none && return provenance.rescue_path
    provenance.primary_method == :direct_opt && return :direct_opt
    return :algebraic
end

"""
    sync_result_contract!(result) -> result

Synchronize compatibility fields that are now derived from provenance.
"""
function sync_result_contract!(result)
    result.interpolator_source = result.provenance.interpolator_source
    result.return_code = compatibility_return_code(result.provenance)
    return result
end

"""
    lineage_summary(result) -> String

Compact human-readable summary of result provenance for logs and diagnostics.
"""
function lineage_summary(result)::String
    prov = result.provenance
    parts = String["method=$(prov.primary_method)"]
    identity = prov.estimator_identity
    identity.candidate_id > 0 && push!(parts, "id=$(identity.candidate_id)")
    identity.estimator_kind != :unknown && push!(parts, "estimator=$(identity.estimator_kind)")
    push!(parts, "source=$(prov.source_type)")
    if prov.source_type == :multipoint
        !isnothing(prov.multipoint_combo_index) && push!(parts, "combo=$(prov.multipoint_combo_index)")
        !isnothing(prov.multipoint_time_indices) && push!(parts, "mp_times=$(prov.multipoint_time_indices)")
    end
    prov.rescue_path != :none && push!(parts, "rescue=$(prov.rescue_path)")
    !isnothing(prov.source_shooting_index) && push!(parts, "shoot=$(prov.source_shooting_index)")
    !isnothing(prov.source_candidate_index) && push!(parts, "candidate=$(prov.source_candidate_index)")
    !isnothing(prov.interpolator_source) && push!(parts, "interp=$(prov.interpolator_source)")
    prov.polish_applied && push!(parts, "polished=true")
    !isempty(prov.representative_assignments) && push!(parts, "representative=$(length(prov.representative_assignments))")
    !isempty(prov.structural_fix_set) && push!(parts, "structural_fix=$(length(prov.structural_fix_set))")
    !isnothing(prov.template_status) && push!(parts, "template=$(prov.template_status)")
    prov.practical_identifiability_status != :not_assessed && push!(parts, "practical=$(prov.practical_identifiability_status)")
    !isnothing(prov.numerical_advisory) && push!(parts, "advisory=$(prov.numerical_advisory.status)")
    return join(parts, ", ")
end

"""
    provenance_metadata_dict(prov::ResultProvenance) -> OrderedDict{String, Any}

Single source of the provenance block written into `odepe_metadata.json` by the
PEB / experiments benchmark templates. This block used to be copy-pasted into
~8 template writers across two repos, which produced stragglers on every
provenance-schema change (2026-08-12 dedup). Values are JSON-friendly (Symbols
stringified, OrderedDicts re-keyed by string). Template-local extras
(parameters, states, all_unidentifiable, …) are merged on top by the caller.
"""
function provenance_metadata_dict(prov::ResultProvenance)
    notes_strs = [string(x) for x in prov.notes]
    _string_keyed(d) = OrderedDict{String, Any}(string(k) => v for (k, v) in d)
    return OrderedDict{String, Any}(
        "primary_method" => string(prov.primary_method),
        "interpolator_source" => isnothing(prov.interpolator_source) ? nothing : string(prov.interpolator_source),
        "rescue_path" => string(prov.rescue_path),
        "source_shooting_index" => prov.source_shooting_index,
        "source_candidate_index" => prov.source_candidate_index,
        "structural_fix_set" => _string_keyed(prov.structural_fix_set),
        "representative_assignments" => _string_keyed(prov.representative_assignments),
        "template_status" => isnothing(prov.template_status) ? nothing : string(prov.template_status),
        "equations_dropped_by_rank_trimming" => prov.equations_dropped_by_rank_trimming,
        "practical_identifiability_status" => string(prov.practical_identifiability_status),
        "notes" => notes_strs,
        "was_terminal_fallback" => ("terminal_fallback" in notes_strs) || (string(prov.rescue_path) == "direct_opt_fallback"),
        "source_type" => string(prov.source_type),
        "multipoint_time_indices" => isnothing(prov.multipoint_time_indices) ? nothing : prov.multipoint_time_indices,
        "multipoint_combo_index" => prov.multipoint_combo_index,
        "aggregation_strategy" => string(prov.aggregation_strategy),
        "aggregation_source_indices" => prov.aggregation_source_indices,
        "polish_applied" => prov.polish_applied,
        "pre_polish_error" => prov.pre_polish_error,
        "post_polish_error" => prov.post_polish_error,
        "polish_source_hc_idx" => prov.polish_source_hc_idx,
        "estimator_identity" => _estimator_identity_metadata(prov.estimator_identity),
    )
end

"""
    ParameterEstimationResult

One solution: parameter values and initial conditions with which the model
reproduces the data.

Printing a result shows its values. `result[a]`, `result[:a]` and `result["a"]`
look one up.

# Fields
- `parameters`: the estimated parameter values, keyed by symbol.
- `states`: the estimated states at `report_time`, keyed by symbol.
- `report_time`: the time those states belong to: the first time point of the
  data, or the `initial_time` of an [`ObservationData`](@ref).
- `err`: the fit error, the sum of squared differences between the data and the
  simulated model over all series and time points. Results are ranked by it.
- `all_unidentifiable`: the parameters and initial conditions that the measured
  quantities cannot determine. Their reported values are arbitrary.
- `unident_dict`: the values those were fixed at while solving.
- `solution`: the simulated trajectory.
- `datasize`: the number of time points.
- `at_time`: the time point at which the equations were solved.
- `interpolator_source`: the interpolator whose derivative estimates led here.
- `provenance`: a [`ResultProvenance`](@ref) record of how the solution was
  produced.
- `return_code`: an older summary of the same: `:algebraic`, `:direct_opt`, or
  the name of a recovery path.
- `branch_size`: how many candidate solutions were merged into this one.

When the package rescales a badly scaled problem internally (`auto_rescale`),
`err`, `solution` and `unident_dict` stay in the rescaled units. `parameters`
and `states` are always in yours.
"""
mutable struct ParameterEstimationResult
    parameters::OrderedDict{Num, Float64}
    states::OrderedDict{Num, Float64}
    at_time::Float64
    err::Union{Nothing, Float64}
    return_code::Union{Nothing, Symbol}
    datasize::Int64
    report_time::Union{Nothing, Float64}
    unident_dict::Union{Nothing, OrderedDict{Num, Float64}}
    all_unidentifiable::Set{Num}
    solution::Union{Nothing, SciMLBase.AbstractODESolution}
    interpolator_source::Union{Nothing, Symbol}   # Which interpolator produced this result
    provenance::ResultProvenance
    branch_size::Int   # Phase B candidate-reduction: cluster size at output time (1 = singleton or undetected)

    function ParameterEstimationResult(
        parameters::AbstractDict, states::AbstractDict, at_time, err, return_code, datasize,
        report_time, unident_dict::Union{Nothing, AbstractDict}, all_unidentifiable, solution,
        interpolator_source, provenance, branch_size,
    )
        # OrderedCollections 2 rejects implicit Dict -> OrderedDict conversion.
        # Construct mappings explicitly while preserving already-normalized
        # ordered mappings, including their insertion order and object identity.
        return new(
            _result_ordered_dict(parameters), _result_ordered_dict(states),
            at_time, err, return_code, datasize, report_time,
            _result_ordered_dict(unident_dict), all_unidentifiable, solution,
            interpolator_source, provenance, branch_size,
        )
    end
end

_result_ordered_dict(values::OrderedDict{Num, Float64}) = values
_result_ordered_dict(values::AbstractDict) = OrderedDict{Num, Float64}(values)
_result_ordered_dict(::Nothing) = nothing

# Backward-compatible constructor (interpolator_source defaults to nothing, provenance to an empty record,
# branch_size to 1 = singleton)
# Symbolic keys and numerical values are normalized by the full constructor.
function ParameterEstimationResult(
    parameters, states, at_time, err, return_code, datasize,
    report_time, unident_dict, all_unidentifiable, solution,
)
    return ParameterEstimationResult(
        parameters, states, at_time, err, return_code, datasize,
        report_time, unident_dict, all_unidentifiable, solution, nothing, ResultProvenance(), 1,
    )
end

# Constructor with interpolator_source + provenance (no branch_size) — for callers that
# already populate those but were written before branch_size was added.
function ParameterEstimationResult(
    parameters, states, at_time, err, return_code, datasize,
    report_time, unident_dict, all_unidentifiable, solution,
    interpolator_source, provenance,
)
    return ParameterEstimationResult(
        parameters, states, at_time, err, return_code, datasize,
        report_time, unident_dict, all_unidentifiable, solution,
        interpolator_source, provenance, 1,
    )
end

"""
    DerivativeData

Struct to store derivative data of state variable equations and measured quantity equations.
No substitutions are made.
The "cleared" versions are produced from versions of the state equations and measured quantity equations
which have had their denominators cleared, i.e. they should be polynomial and never rational.
Their tables may be empty or shorter than the rational tables when construction is
deferred. Call `ensure_cleared_derivatives!` before reading a required cleared order.

# Fields
- `states_lhs_cleared::Vector{Vector{Num}}`: Left-hand side of cleared state equations (indexed by [derivative_order+1])
- `states_rhs_cleared::Vector{Vector{Num}}`: Right-hand side of cleared state equations
- `obs_lhs_cleared::Vector{Vector{Num}}`: Left-hand side of cleared observation equations
- `obs_rhs_cleared::Vector{Vector{Num}}`: Right-hand side of cleared observation equations
- `states_lhs::Vector{Vector{Num}}`: Left-hand side of state equations
- `states_rhs::Vector{Vector{Num}}`: Right-hand side of state equations
- `obs_lhs::Vector{Vector{Num}}`: Left-hand side of observation equations
- `obs_rhs::Vector{Vector{Num}}`: Right-hand side of observation equations
- `all_unidentifiable::Set{Num}`: Set of all unidentifiable parameters
"""
mutable struct DerivativeData
    states_lhs_cleared::Vector{Vector{Num}}
    states_rhs_cleared::Vector{Vector{Num}}
    obs_lhs_cleared::Vector{Vector{Num}}
    obs_rhs_cleared::Vector{Vector{Num}}
    states_lhs::Vector{Vector{Num}}
    states_rhs::Vector{Vector{Num}}
    obs_lhs::Vector{Vector{Num}}
    obs_rhs::Vector{Vector{Num}}
    all_unidentifiable::Set{Num}
end

# ─── Taylor-polynomial interpolant ─────────────────────────────────────

"""
    PerfectInterpolant

Stores Taylor coefficients `c[k+1] = f^(k)(t0) / k!` and evaluates as a
polynomial via Horner's method.  TaylorDiff on this interpolant yields
machine-precision derivatives up to the stored order.
"""
struct PerfectInterpolant
    t0::Float64
    coeffs::Vector{Float64}  # coeffs[k+1] = f^(k)(t0) / k!
end

function (p::PerfectInterpolant)(t)
    dt = t - p.t0
    result = p.coeffs[end]
    for k in (length(p.coeffs) - 1):-1:1
        result = muladd(result, dt, p.coeffs[k])
    end
    return result
end

"""
    JetInfluenceEstimate

Linear influence representation for observable jets estimated from data.

For fixed GP hyperparameters, the derivative estimator is linear in the raw
observations: `jet_mean = W * y`.  The estimator sampling covariance is then
`jet_covariance = W * observation_covariance * W'`.
"""
struct JetInfluenceEstimate{M<:AbstractMatrix{Float64}}
    observable_name::String
    t_eval::Float64
    orders::Vector{Int}
    labels::Vector{String}
    mean::Vector{Float64}
    W::Matrix{Float64}
    observation_covariance::M
    jet_covariance::Matrix{Float64}
    covariance_kind::Symbol
    noise_source::Symbol
    warnings::Vector{String}
end

"""
    StackedJetInfluenceEstimate

Multi-time sibling of [`JetInfluenceEstimate`](@ref): the estimator-sampling
covariance of one observable's posterior-mean jets at SEVERAL evaluation
times, factorized as `W_stack · Σ_y · W_stackᵀ`. Cross-time blocks are nonzero
because every jet is a linear functional of the SAME training observations.
PSD by construction; genuine singularity (e.g. repeated times) is preserved
rather than repaired.

Row order is time-major: for each time in `t_evals`, orders `0:max_deriv`.
Use [`stacked_jet_index`](@ref) to map (time_idx, order) → row/column.
"""
struct StackedJetInfluenceEstimate{M<:AbstractMatrix{Float64}}
    observable_name::String
    t_evals::Vector{Float64}
    orders::Vector{Int}            # 0:max_deriv, shared by every time
    labels::Vector{String}         # length = length(t_evals) * length(orders)
    mean::Vector{Float64}
    W_stack::Matrix{Float64}
    observation_covariance::M
    jet_covariance::Matrix{Float64}
    covariance_kind::Symbol
    noise_source::Symbol
    warnings::Vector{String}
end

"""
    PracticalIdentifiabilityIndex

Scale-normalized covariance summary for the selected physical unknowns.

`i_a = sqrt(lambda_max(normalized_covariance))`, where
`normalized_covariance = D^{-1} P Σ_x P' D^{-1}`.
"""
struct PracticalIdentifiabilityIndex
    labels::Vector{String}
    roles::Vector{Symbol}
    scale_values::Vector{Float64}
    projected_covariance::Matrix{Float64}
    normalized_covariance::Matrix{Float64}
    lambda_max::Float64
    i_a::Float64
    status::Symbol
    warnings::Vector{String}
end

"""
    LocalUQSnapshot

Audit copy of the direct IFT uncertainty calculation in the local algebraic
coordinates at `t_eval`.
"""
struct LocalUQSnapshot
    t_eval::Float64
    param_covariance::Matrix{Float64}
    param_std::Vector{Float64}
    param_labels::Vector{String}
    param_roles::Dict{String, Symbol}
    coordinate_values::Vector{Float64}
    correlation_matrix::Matrix{Float64}
    max_cv::Float64
    status::Symbol
    practical_identifiability_index::Union{Nothing, PracticalIdentifiabilityIndex}
end

"""
    UQBacksolveTransform

Linear delta-method map from local coordinates `[p, x(t_eval)]` to physical
coordinates `[p, x(t0)]`.
"""
struct UQBacksolveTransform
    t_eval::Float64
    t0::Float64
    source_labels::Vector{String}
    target_labels::Vector{String}
    source_roles::Dict{String, Symbol}
    target_roles::Dict{String, Symbol}
    transform_matrix::Matrix{Float64}
    amplification::Float64
    status::Symbol
    warnings::Vector{String}
end

"""Common supertype for requested uncertainty-quantification outcomes."""
abstract type AbstractUQOutcome end

"""
    UQReliabilityAssessment

Orthogonal reliability axes for a UQ outcome.  These deliberately do not
collapse software availability, local numerical validity, interval width,
selection conditioning, and empirical calibration into a single status flag.
`UncertaintyReport.status` remains as a backward-compatible summary.
"""
struct UQReliabilityAssessment
    availability::Symbol
    numerical_linearization::Symbol
    interval_width::Symbol
    selection_scope::Symbol
    empirical_calibration::Symbol
end

"""
    UQTargetSnapshot

Serializable audit record for the exact rank-one estimator targeted by UQ.
`lineage` contains the selected identity followed by any retained ancestors.
"""
struct UQTargetSnapshot
    selected_rank::Int
    identity::EstimatorIdentity
    lineage::Vector{EstimatorIdentity}
    estimand::Symbol
    artifact_match::Symbol
end

function UQTargetSnapshot(;
    selected_rank::Integer = 1,
    identity::EstimatorIdentity = EstimatorIdentity(),
    lineage::AbstractVector{EstimatorIdentity} = EstimatorIdentity[identity],
    estimand::Symbol = :conditional_on_selected_estimator,
    artifact_match::Symbol = :unknown,
)
    return UQTargetSnapshot(
        Int(selected_rank), copy_estimator_identity(identity),
        EstimatorIdentity[copy_estimator_identity(x) for x in lineage],
        estimand, artifact_match,
    )
end

"""Numerical audit values for the local estimator linearization."""
struct UQLinearizationDiagnostics
    reason::Symbol
    root_residual_abs::Float64
    root_residual_rel::Float64
    jacobian_condition::Float64
    jacobian_condition_equilibrated::Float64
    linear_solve_backward_error::Float64
    gradient_norm::Float64
    active_bounds::Vector{Int}
    degraded::Bool
    gp_jitter_to_noise::Float64
    gp_factorization_residual::Float64
end

# Backward-compatible constructor from before scale-aware linear-solve
# diagnostics were retained.  Callers using the historical raw condition only
# continue to work; the two new values are explicitly unknown rather than
# silently inferred.
function UQLinearizationDiagnostics(
    reason, root_residual_abs, root_residual_rel, jacobian_condition,
    gradient_norm, active_bounds, degraded,
    gp_jitter_to_noise, gp_factorization_residual,
)
    return UQLinearizationDiagnostics(
        reason, root_residual_abs, root_residual_rel, jacobian_condition,
        NaN, NaN, gradient_norm, active_bounds, degraded,
        gp_jitter_to_noise, gp_factorization_residual,
    )
end

function UQLinearizationDiagnostics(
    reason, root_residual_abs, root_residual_rel, jacobian_condition,
    gradient_norm, active_bounds, degraded,
)
    return UQLinearizationDiagnostics(
        reason, root_residual_abs, root_residual_rel, jacobian_condition,
        gradient_norm, active_bounds, degraded, NaN, NaN,
    )
end

UQLinearizationDiagnostics() = UQLinearizationDiagnostics(
    :not_recorded, NaN, NaN, NaN, NaN, Int[], false,
)

"""Typed outcome returned when UQ was requested but cannot be computed honestly."""
struct UncertaintyUnavailable <: AbstractUQOutcome
    model_name::String
    target::Union{Nothing, UQTargetSnapshot}
    reason::Symbol
    message::String
    warnings::Vector{String}
end

"""
    UncertaintyError

Thrown instead of returning a [`UncertaintyUnavailable`](@ref) when the uncertainty
cannot be worked out and the options say `uq_failure_policy = :throw`. Its
`outcome` field is that `UncertaintyUnavailable`.
"""
struct UncertaintyError <: Exception
    outcome::UncertaintyUnavailable
end

function Base.showerror(io::IO, err::UncertaintyError)
    print(io, "Uncertainty quantification failed [", err.outcome.reason, "]: ", err.outcome.message)
end

"""
    UncertaintyReport

Propagated parameter uncertainty through the implicit function theorem
sensitivity matrix.

Σ_x = S · Σ_d · S' where S = -(∂F/∂x)⁻¹·(∂F/∂d).  For the default package UQ,
Σ_d is the estimator sampling covariance of the observable jet, conditional on
learned GP hyperparameters.  Legacy latent GP posterior covariance is kept as a
separate diagnostic concept.
"""
struct UncertaintyReport <: AbstractUQOutcome
    model_name::String
    t_eval::Float64
    # Per-observable jet estimate at t_eval
    obs_names::Vector{String}
    obs_posterior_mean::Vector{Vector{Float64}}    # [obs_idx][deriv_order+1]
    obs_posterior_std::Vector{Vector{Float64}}     # [obs_idx][deriv_order+1]
    # Data/jet covariance
    data_covariance::Matrix{Float64}              # Σ_d (n_data × n_data)
    data_labels::Vector{String}
    # Parameter covariance from IFT
    param_covariance::Matrix{Float64}             # Σ_x (n_unknowns × n_unknowns)
    param_std::Vector{Float64}                    # √diag(Σ_x)
    param_labels::Vector{String}
    param_roles::Dict{String, Symbol}
    param_true_values::Vector{Float64}
    # Correlation matrix
    correlation_matrix::Matrix{Float64}           # Σ_x normalized to [-1,1]
    # Quality
    max_cv::Float64                               # max coefficient of variation
    status::Symbol                                # :ok, :wide_ci, :degenerate
    warnings::Vector{String}
    covariance_kind::Symbol                       # :estimator_sampling or :latent_gp_posterior
    noise_source::Symbol                          # e.g. :learned_gp_homoscedastic
    practical_identifiability_index::Union{Nothing, PracticalIdentifiabilityIndex}
    coordinate_system::Symbol                     # :physical_initial_conditions or :local_at_eval
    local_coordinate_report::Union{Nothing, LocalUQSnapshot}
    backsolve_transform::Union{Nothing, UQBacksolveTransform}
    target::Union{Nothing, UQTargetSnapshot}
    estimate_values::Vector{Float64}
    linearization_diagnostics::UQLinearizationDiagnostics
end

function _estimator_identity_metadata(identity::EstimatorIdentity)
    return OrderedDict{String, Any}(
        "candidate_id" => identity.candidate_id,
        "estimator_kind" => string(identity.estimator_kind),
        "data_scope" => string(identity.data_scope),
        "time_indices" => copy(identity.time_indices),
        "time_values" => copy(identity.time_values),
        "interpolator_source" => isnothing(identity.interpolator_source) ? nothing : string(identity.interpolator_source),
        "parent_candidate_ids" => copy(identity.parent_candidate_ids),
    )
end

function _uq_target_metadata(target::Union{Nothing, UQTargetSnapshot})
    isnothing(target) && return nothing
    return OrderedDict{String, Any}(
        "selected_rank" => target.selected_rank,
        "estimand" => string(target.estimand),
        "artifact_match" => string(target.artifact_match),
        "identity" => _estimator_identity_metadata(target.identity),
        "lineage" => [_estimator_identity_metadata(identity) for identity in target.lineage],
    )
end

_uq_selection_scope(::Nothing) = :unknown
_uq_selection_scope(target::UQTargetSnapshot) = target.estimand

"""
    uq_reliability(outcome) -> UQReliabilityAssessment

Return the independent reliability axes for a requested UQ outcome.  A single
run can establish availability and pass local numerical gates, but it cannot
certify repeated-sampling coverage; empirical calibration therefore remains
`:not_assessed_by_single_run` for every individual report.
"""
function uq_reliability(::Nothing)
    return UQReliabilityAssessment(
        :not_requested, :not_computed, :not_computed, :unknown, :not_assessed,
    )
end

function uq_reliability(outcome::UncertaintyUnavailable)
    return UQReliabilityAssessment(
        :unavailable, :not_computed, :not_computed,
        _uq_selection_scope(outcome.target), :not_assessed,
    )
end

function uq_reliability(report::UncertaintyReport)
    diagnostics = report.linearization_diagnostics
    numerical = if diagnostics.reason == :not_recorded
        :not_recorded
    elseif diagnostics.degraded
        :degraded
    else
        :accepted
    end
    interval_width = if isempty(report.param_std)
        :not_computed
    elseif any(x -> !isfinite(x) || x < 0.0, report.param_std)
        :invalid
    elseif !isfinite(report.max_cv)
        :undefined_scale
    elseif report.max_cv < 0.5
        :narrow
    elseif report.max_cv < 2.0
        :wide
    else
        :extreme
    end
    return UQReliabilityAssessment(
        :available, numerical, interval_width,
        _uq_selection_scope(report.target), :not_assessed_by_single_run,
    )
end

function _uq_reliability_metadata(assessment::UQReliabilityAssessment)
    return OrderedDict{String, Any}(
        "availability" => string(assessment.availability),
        "numerical_linearization" => string(assessment.numerical_linearization),
        "interval_width" => string(assessment.interval_width),
        "selection_scope" => string(assessment.selection_scope),
        "empirical_calibration" => string(assessment.empirical_calibration),
    )
end

# JSON has no representation for NaN or infinities.  Keep the public sidecar
# dependency-free and recursively normalize all non-finite floating-point
# values to `nothing` (serialized as JSON null by JSON3 and other encoders).
_uq_json_safe(value::AbstractFloat) = isfinite(value) ? value : nothing
_uq_json_safe(value::Tuple) = [_uq_json_safe(item) for item in value]
_uq_json_safe(values::AbstractArray) = map(_uq_json_safe, values)
function _uq_json_safe(values::AbstractDict)
    return OrderedDict{String, Any}(
        string(key) => _uq_json_safe(value) for (key, value) in values
    )
end
_uq_json_safe(value) = value

"""JSON-friendly additive sidecar block for a requested UQ outcome."""
uq_metadata_dict(::Nothing) = nothing

function uq_metadata_dict(outcome::UncertaintyUnavailable)
    metadata = OrderedDict{String, Any}(
        "outcome" => "unavailable",
        "model_name" => outcome.model_name,
        "target" => _uq_target_metadata(outcome.target),
        "reason" => string(outcome.reason),
        "message" => outcome.message,
        "reliability" => _uq_reliability_metadata(uq_reliability(outcome)),
        "warnings" => copy(outcome.warnings),
    )
    return _uq_json_safe(metadata)
end

function uq_metadata_dict(report::UncertaintyReport)
    ia = report.practical_identifiability_index
    diagnostics = report.linearization_diagnostics
    metadata = OrderedDict{String, Any}(
        "outcome" => "report",
        "model_name" => report.model_name,
        "target" => _uq_target_metadata(report.target),
        "status" => string(report.status),
        "reliability" => _uq_reliability_metadata(uq_reliability(report)),
        "coordinate_system" => string(report.coordinate_system),
        "t_eval" => report.t_eval,
        "covariance_kind" => string(report.covariance_kind),
        "noise_source" => string(report.noise_source),
        "param_labels" => copy(report.param_labels),
        "param_roles" => OrderedDict{String, Any}(k => string(v) for (k, v) in report.param_roles),
        "estimate_values" => copy(report.estimate_values),
        "truth_values" => copy(report.param_true_values),
        "param_std" => copy(report.param_std),
        "param_covariance" => [collect(row) for row in eachrow(report.param_covariance)],
        "max_cv" => report.max_cv,
        "practical_identifiability_index" => isnothing(ia) ? nothing : ia.i_a,
        "linearization" => OrderedDict{String, Any}(
            "reason" => string(diagnostics.reason),
            "root_residual_abs" => diagnostics.root_residual_abs,
            "root_residual_rel" => diagnostics.root_residual_rel,
            "jacobian_condition" => diagnostics.jacobian_condition,
            "jacobian_condition_equilibrated" => diagnostics.jacobian_condition_equilibrated,
            "linear_solve_backward_error" => diagnostics.linear_solve_backward_error,
            "gradient_norm" => diagnostics.gradient_norm,
            "active_bounds" => copy(diagnostics.active_bounds),
            "degraded" => diagnostics.degraded,
            "gp_jitter_to_noise" => diagnostics.gp_jitter_to_noise,
            "gp_factorization_residual" => diagnostics.gp_factorization_residual,
        ),
        "warnings" => copy(report.warnings),
    )
    return _uq_json_safe(metadata)
end

# Backward-compatible full constructor for callers written before the
# estimator-aware target fields were added.
function UncertaintyReport(
    model_name, t_eval, obs_names, obs_posterior_mean, obs_posterior_std,
    data_covariance, data_labels, param_covariance, param_std, param_labels,
    param_roles, param_true_values, correlation_matrix, max_cv, status,
    warnings, covariance_kind, noise_source, practical_identifiability_index,
    coordinate_system, local_coordinate_report, backsolve_transform,
)
    return UncertaintyReport(
        model_name, t_eval, obs_names, obs_posterior_mean, obs_posterior_std,
        data_covariance, data_labels, param_covariance, param_std, param_labels,
        param_roles, param_true_values, correlation_matrix, max_cv, status,
        warnings, covariance_kind, noise_source, practical_identifiability_index,
        coordinate_system, local_coordinate_report, backsolve_transform,
        nothing, Float64[], UQLinearizationDiagnostics(),
    )
end

# Backward-compatible constructor for callers that predate explicit coordinate
# semantics. The direct IFT calculation lives in local coordinates at t_eval.
function UncertaintyReport(
    model_name, t_eval, obs_names, obs_posterior_mean, obs_posterior_std,
    data_covariance, data_labels, param_covariance, param_std, param_labels,
    param_roles, param_true_values, correlation_matrix, max_cv, status,
    warnings, covariance_kind, noise_source, practical_identifiability_index,
)
    return UncertaintyReport(
        model_name, t_eval, obs_names, obs_posterior_mean, obs_posterior_std,
        data_covariance, data_labels, param_covariance, param_std, param_labels,
        param_roles, param_true_values, correlation_matrix, max_cv, status,
        warnings, covariance_kind, noise_source, practical_identifiability_index,
        :local_at_eval, nothing, nothing,
    )
end

# Backward-compatible constructor for callers that predate explicit covariance
# semantics. New package-produced reports should pass these fields explicitly.
function UncertaintyReport(
    model_name, t_eval, obs_names, obs_posterior_mean, obs_posterior_std,
    data_covariance, data_labels, param_covariance, param_std, param_labels,
    param_roles, param_true_values, correlation_matrix, max_cv, status,
    warnings,
)
    return UncertaintyReport(
        model_name, t_eval, obs_names, obs_posterior_mean, obs_posterior_std,
        data_covariance, data_labels, param_covariance, param_std, param_labels,
        param_roles, param_true_values, correlation_matrix, max_cv, status,
        warnings, :unspecified, :unspecified, nothing,
    )
end

# Backward-compatible constructor (warnings defaults to empty)
function UncertaintyReport(
    model_name, t_eval, obs_names, obs_posterior_mean, obs_posterior_std,
    data_covariance, data_labels, param_covariance, param_std, param_labels,
    param_roles, param_true_values, correlation_matrix, max_cv, status,
)
    return UncertaintyReport(
        model_name, t_eval, obs_names, obs_posterior_mean, obs_posterior_std,
        data_covariance, data_labels, param_covariance, param_std, param_labels,
        param_roles, param_true_values, correlation_matrix, max_cv, status,
        String[],
    )
end

struct TimingPhaseEntry
    name::String
    seconds::Float64
    bytes::Int64
    gctime::Float64
end

Base.@kwdef struct TimingBreakdown
    label::Symbol = :unknown
    total_seconds::Float64 = 0.0
    phases::Vector{TimingPhaseEntry} = TimingPhaseEntry[]
    details::OrderedDict{Symbol, Any} = OrderedDict{Symbol, Any}()
end

# ─── Multi-point template types ──────────────────────────────────────

"""
    DataVarMeta

Construction-time metadata for one multipoint data variable — the
authoritative (observable, order, point, kind) record consumed by UQ Σ_d
assembly. Downstream label/name parsing is display-only.

# Fields
- `clean_name::String`: `_ptK`-stripped SIAN-style name (e.g. `"y1_2"`)
- `obs_idx::Union{Nothing, Int}`: index into `measured_quantities`
  (`nothing` for `:transcendental`/`:unresolved`)
- `order::Int`: derivative order (0 for transcendental/unresolved)
- `point::Int`: 1-based multipoint index (1 = unsuffixed)
- `kind::Symbol`: `:observable_jet` | `:raw_observation` |
  `:transcendental` | `:unresolved`. `:raw_observation` is appended only by
  estimator-aware UQ when production reports a directly observed state that
  was eliminated from the algebraic solve coordinates.
"""
struct DataVarMeta
    clean_name::String
    obs_idx::Union{Nothing, Int}
    order::Int
    point::Int
    kind::Symbol
end

"""
    MultiPointTemplate

Pre-computed multi-point polynomial template. Built once per model from the
single-point SI template. Stores the structural recipe (which equations to keep)
so that evaluation at specific time points is cheap.

The template stores equations in **symbolic** form (data variables not substituted),
enabling HC.jl parameter homotopy across different time point pairs.

Invariant: `length(stripped_equations) == length(solve_vars)` (square system).
"""
struct MultiPointTemplate
    n_points::Int
    base_si_template::Any  # NamedTuple from prepare_si_template_with_structural_fix

    # Combined symbolic system (N copies, renamed, stripped to square)
    all_equations::Vector{Num}
    stripped_equations::Vector{Num}
    solve_vars::Vector{Any}    # Shared params + per-point state derivs (HC variables)
    data_vars::Vector{Any}     # Per-point observable derivs + _trfn_ vars (HC parameters)

    # Metadata for parameter extraction
    param_var_indices::Vector{Int}     # Indices into solve_vars that are shared parameters
    param_names::Vector{String}        # Clean parameter names (e.g., "a", "b") for error scoring

    # Equation metadata
    eq_metadata::Vector{@NamedTuple{point::Int, is_data::Bool, order::Int}}
    total_equation_count::Int
    kept_equation_indices::Vector{Int}
    dropped_equation_indices::Vector{Int}

    # For data evaluation at arbitrary time points
    # Exact per-point index lists into data_vars. Deliberately NOT ranges: the
    # old first:last collapse silently mis-assigned data when per-point indices
    # interleave (they only happen to be contiguous under point-major ordering).
    per_point_data_var_indices::Vector{Vector{Int}}
    template_DD::Any           # DerivativeData for observable → variable mapping
    measured_quantities::Vector{ModelingToolkit.Equation}
    # Authoritative per-data-var metadata (aligned 1:1 with data_vars) for UQ
    # Σ_d assembly — label/name parsing downstream is display-only. Appended
    # last to keep the positional constructors stable.
    data_var_meta::Vector{DataVarMeta}
end

"""
    MultiPointEvaluation

Result of evaluating a MultiPointTemplate at specific time point indices.
Contains the data variable values needed for HC.jl parameter substitution or homotopy.
"""
struct MultiPointEvaluation
    template::MultiPointTemplate
    time_indices::Vector{Int}
    t_values::Vector{Float64}
    data_values::Vector{Float64}  # Ordered to match template.data_vars
end
