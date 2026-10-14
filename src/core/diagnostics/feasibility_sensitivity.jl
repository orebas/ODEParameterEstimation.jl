# Variable classification, true-value lookup, residuals, and the sensitivity of
# the solved unknowns to the data (implicit function theorem).
# ─── Variable classification ───────────────────────────────────────────

"""
    _classify_polynomial_variables(var_names, pep)

Classify each variable name in the polynomial system into a role:
  :parameter, :state_ic, :state_derivative, :data_derivative, :transcendental

Uses `parse_derivative_variable_name` to decompose SIAN-style names (e.g. `x1_0`, `y1_2`).
"""
function _classify_polynomial_variables(var_names::Vector{String}, pep::ParameterEstimationProblem)
    roles = Dict{String, Symbol}()

    # Build lookup sets
    param_bases = Set{String}()
    for p in pep.model.original_parameters
        push!(param_bases, replace(string(p), "(t)" => ""))
    end
    state_bases = Set{String}()
    for s in pep.model.original_states
        push!(state_bases, replace(string(s), "(t)" => ""))
    end
    obs_bases = Set{String}()
    for mq in pep.measured_quantities
        obs_name = replace(string(mq.lhs), "(t)" => "")
        # Strip _obs_trfn_ prefix for matching
        if !startswith(obs_name, "_obs_trfn_")
            push!(obs_bases, obs_name)
        end
    end

    for vn in var_names
        # Check transcendental first
        if contains(vn, "_trfn_")
            roles[vn] = :transcendental
            continue
        end

        parsed = parse_derivative_variable_name(vn)
        if !isnothing(parsed)
            base_name, deriv_order = parsed

            # Parameter (order 0 only in SI template)
            if base_name in param_bases && deriv_order == 0
                roles[vn] = :parameter
                continue
            end

            # State IC (order 0) or state derivative (order > 0)
            if base_name in state_bases
                roles[vn] = deriv_order == 0 ? :state_ic : :state_derivative
                continue
            end

            # Observable derivative (data variable)
            if base_name in obs_bases
                roles[vn] = :data_derivative
                continue
            end
        end

        # Direct parameter match (no _N suffix)
        if vn in param_bases
            roles[vn] = :parameter
            continue
        end

        # Direct state match
        if vn in state_bases
            roles[vn] = :state_ic
            continue
        end

        # Default to state_derivative (unknown role)
        roles[vn] = :state_derivative
    end

    return roles
end

"""
Build a vector of evaluation-point values in the order of `varlist` by matching
variable names to parameters, states, and their derivatives.
Uses the supplied Taylor coefficients to resolve derivative variables (e.g., x1_1, x2_2).

The values are whatever `pep` carries in `p_true` and `ic`.
"""
function _build_true_value_vector(pep::ParameterEstimationProblem, varlist;
    state_taylor::Union{Nothing, Dict{Num, Vector{Float64}}} = nothing,
    obs_taylor::Union{Nothing, Dict{Num, Vector{Float64}}} = nothing,
    t_eval::Float64 = 0.0,
)
    true_vals = Float64[]
    for v in varlist
        val = _lookup_true_value(pep, v; state_taylor = state_taylor, obs_taylor = obs_taylor, t_eval = t_eval)
        push!(true_vals, val)
    end
    return true_vals
end

function _lookup_true_value(pep::ParameterEstimationProblem, var;
    state_taylor::Union{Nothing, Dict{Num, Vector{Float64}}} = nothing,
    obs_taylor::Union{Nothing, Dict{Num, Vector{Float64}}} = nothing,
    t_eval::Float64 = 0.0,
)
    # Direct parameter match
    for (p, v) in pep.p_true
        if isequal(var, p)
            return v
        end
    end
    # Direct state match
    for (s, v) in pep.ic
        if isequal(var, s)
            return v
        end
    end

    var_name = string(var)

    # Check bare _obs_trfn_ names (Symbolics form, no _N suffix) before parsing
    clean_name = replace(var_name, r"\(.*\)" => "")
    obs_trfn_val = evaluate_obs_trfn_template_variable(clean_name, t_eval)
    if !isnothing(obs_trfn_val)
        return obs_trfn_val
    end

    # Try parse as derivative variable (e.g., y1_0, x1_0, x2_2)
    parsed = parse_derivative_variable_name(var_name)
    if !isnothing(parsed)
        base_name, deriv_order = parsed

        # Check if it's a _trfn_ or _obs_trfn_ variable
        trfn_val = evaluate_trfn_template_variable(var_name, t_eval)
        if isnothing(trfn_val)
            trfn_val = evaluate_obs_trfn_template_variable(var_name, t_eval)
        end
        if !isnothing(trfn_val)
            return trfn_val
        end

        # Match against parameters by base name (order 0 only)
        for (p, v) in pep.p_true
            p_str = string(p)
            p_base = replace(p_str, "(t)" => "")
            if p_base == base_name && deriv_order == 0
                return v
            end
        end

        # Match against states by base name — use Taylor coefficients for all orders
        # SIAN variables store actual derivatives f^(k)(t_eval), not Taylor coeffs f^(k)(t0)/k!
        # Taylor coeffs are at t_eval, so tc[k+1] * k! = f^(k)(t_eval)
        for (s, v) in pep.ic
            s_str = string(s)
            s_base = replace(s_str, "(t)" => "")
            if s_base == base_name
                if !isnothing(state_taylor) && haskey(state_taylor, s)
                    tc = state_taylor[s]
                    if deriv_order + 1 <= length(tc)
                        return tc[deriv_order + 1] * factorial(deriv_order)
                    end
                end
                # Fallback: IC only valid for order 0 at t=0
                if deriv_order == 0
                    return v
                end
            end
        end

        # Match against observable names — use observable Taylor coefficients
        if !isnothing(obs_taylor)
            for mq in pep.measured_quantities
                obs_lhs_name = replace(string(mq.lhs), "(t)" => "")
                if obs_lhs_name == base_name
                    key = Symbolics.diff2term(mq.rhs)
                    if haskey(obs_taylor, key)
                        tc = obs_taylor[key]
                        if deriv_order + 1 <= length(tc)
                            return tc[deriv_order + 1] * factorial(deriv_order)
                        end
                    end
                end
            end
        end
    end

    # Unknown variable — return NaN
    @warn "[DIAG] Could not find true value for variable: $var"
    return NaN
end

function _compute_residual(equations, varlist, vals)
    if isempty(equations) || isempty(vals) || any(isnan, vals)
        return NaN
    end
    try
        f = _compile_system_function(equations, varlist)
        r = f(vals)
        return norm(r)
    catch
        # Fallback to substitution
        subst_dict = Dict(varlist[i] => vals[i] for i in eachindex(varlist))
        total = 0.0
        for eq in equations
            r = try
                Float64(Symbolics.value(Symbolics.substitute(eq, subst_dict)))
            catch
                NaN
            end
            total += r^2
        end
        return sqrt(total)
    end
end

# ─── Sensitivity analysis ─────────────────────────────────────────────

function _condition_2(A::AbstractMatrix{<:Real})::Float64
    return try
        singular_values = svdvals(Matrix{Float64}(A))
        isempty(singular_values) ? Inf :
            singular_values[1] / max(singular_values[end], floatmin(Float64))
    catch err
        _rethrow_if_interrupt(err)
        Inf
    end
end

"""
    _equilibrated_condition_2(A) -> Float64

Condition number after one L2 row-normalization pass followed by one L2
column-normalization pass. This is diagnostic coordinate equilibration only:
it never changes the system that was solved. A zero or non-finite row/column
norm makes the diagnostic infinite.
"""
function _equilibrated_condition_2(A::AbstractMatrix{<:Real})::Float64
    equilibrated = Matrix{Float64}(A)
    isempty(equilibrated) && return Inf
    all(isfinite, equilibrated) || return Inf
    for row in axes(equilibrated, 1)
        row_norm = norm(@view equilibrated[row, :])
        isfinite(row_norm) && row_norm > 0 || return Inf
        @views equilibrated[row, :] ./= row_norm
    end
    for column in axes(equilibrated, 2)
        column_norm = norm(@view equilibrated[:, column])
        isfinite(column_norm) && column_norm > 0 || return Inf
        @views equilibrated[:, column] ./= column_norm
    end
    return _condition_2(equilibrated)
end

"""
    _linear_solve_backward_error(A, X, B) -> Float64

Normwise backward error for the already-computed solution of `A*X=B`.
"""
function _linear_solve_backward_error(
    A::AbstractMatrix{<:Real},
    X::AbstractMatrix{<:Real},
    B::AbstractMatrix{<:Real},
)::Float64
    try
        residual = Matrix{Float64}(A) * Matrix{Float64}(X) - Matrix{Float64}(B)
        numerator = opnorm(residual, Inf)
        denominator = opnorm(Matrix{Float64}(A), Inf) *
            opnorm(Matrix{Float64}(X), Inf) + opnorm(Matrix{Float64}(B), Inf)
        return numerator / max(denominator, floatmin(Float64))
    catch err
        _rethrow_if_interrupt(err)
        return Inf
    end
end

function _linear_solve_assessment(
    A::AbstractMatrix{<:Real},
    X::AbstractMatrix{<:Real},
    B::AbstractMatrix{<:Real};
    raw_condition_warning_threshold::Float64 = 1e6,
    equilibrated_condition_threshold::Float64 = UQ_EQUILIBRATED_CONDITION_THRESHOLD,
    backward_error_threshold::Float64 = UQ_BACKWARD_ERROR_THRESHOLD,
    condition_error_product_threshold::Float64 = UQ_CONDITION_ERROR_PRODUCT_THRESHOLD,
)
    raw_condition = _condition_2(A)
    equilibrated_condition = _equilibrated_condition_2(A)
    backward_error = _linear_solve_backward_error(A, X, B)
    condition_error_product = raw_condition * backward_error
    reason = if !all(isfinite,
        (raw_condition, equilibrated_condition, backward_error,
         condition_error_product))
        :nonfinite_linearization
    elseif equilibrated_condition > equilibrated_condition_threshold
        :intrinsic_ill_conditioning
    elseif backward_error > backward_error_threshold
        :large_backward_error
    elseif condition_error_product > condition_error_product_threshold
        :forward_error_risk
    elseif raw_condition > raw_condition_warning_threshold
        :scale_sensitive_conditioning
    else
        :ok
    end
    degraded = reason in (
        :nonfinite_linearization,
        :intrinsic_ill_conditioning,
        :large_backward_error,
        :forward_error_risk,
    )
    return (;
        raw_condition,
        equilibrated_condition,
        backward_error,
        condition_error_product,
        reason,
        degraded,
    )
end

function _ift_solve_assessment(
    J_x::AbstractMatrix{<:Real},
    J_d::AbstractMatrix{<:Real};
    cond_threshold::Float64 = 1e6,
)
    S = try
        -(J_x \ J_d)
    catch err
        _rethrow_if_interrupt(err)
        raw_condition = _condition_2(J_x)
        equilibrated_condition = _equilibrated_condition_2(J_x)
        @warn "[DIAG] J_x factorization failed ($err) — returning empty S"
        return (
            sensitivity = Matrix{Float64}(undef, 0, 0),
            raw_condition = raw_condition,
            equilibrated_condition = equilibrated_condition,
            backward_error = Inf,
            condition_error_product = Inf,
            reason = :factorization_failed,
            degraded = true,
        )
    end
    assessment = _linear_solve_assessment(
        J_x, S, -Matrix{Float64}(J_d);
        raw_condition_warning_threshold = cond_threshold,
    )
    if assessment.degraded
        @warn "[DIAG] J_x linearization is numerically unreliable ($(assessment.reason)); raw condition=$(assessment.raw_condition), equilibrated condition=$(assessment.equilibrated_condition), backward error=$(assessment.backward_error)" maxlog = 10
    elseif assessment.reason == :scale_sensitive_conditioning
        @warn "[DIAG] J_x has scale-sensitive raw conditioning ($(assessment.raw_condition)) but the equilibrated condition ($(assessment.equilibrated_condition)) and backward error ($(assessment.backward_error)) pass" maxlog = 10
    end
    return merge((sensitivity = Matrix{Float64}(S),), assessment)
end

"""
    _ift_solve(J_x, J_d) → (S, raw_condition, degraded::Bool)

Factorized IFT sensitivity `S = -(J_x \\ J_d)`. Numerical degradation is based
on equilibrated conditioning and solve backward error, while the raw condition
is retained as a scale-sensitivity warning. This avoids both the historical
silent pseudoinverse and false failure caused solely by heterogeneous units.
"""
function _ift_solve(J_x::AbstractMatrix{<:Real}, J_d::AbstractMatrix{<:Real};
    cond_threshold::Float64 = 1e6,
)
    assessment = _ift_solve_assessment(
        J_x, J_d; cond_threshold = cond_threshold,
    )
    return assessment.sensitivity, assessment.raw_condition, assessment.degraded
end

"""
    _compute_data_sensitivity(pep, setup_data, t_eval, prod_vars, true_vals; kwargs...)

Compute the parameter-data sensitivity matrix S = -(∂F/∂x)⁻¹ · (∂F/∂d) via the
implicit function theorem.  Here F(x, d) = 0 is the SI polynomial system, x are
the unknowns (params, ICs, state derivatives), and d are the data variables
(interpolated observable derivatives).

S[i,j] tells you: a unit perturbation in data variable j causes S[i,j] displacement
in unknown i.  This directly quantifies how interpolation errors propagate to
recovered parameter values.

Returns `(S_matrix, data_labels, data_roles)`.
"""
function _compute_data_sensitivity(
    pep::ParameterEstimationProblem,
    setup_data,
    t_eval::Float64,
    prod_vars,
    true_vals;
    state_taylor = nothing,
    obs_taylor = nothing,
    kwargs...,
)
    model = pep.model.system
    mq = pep.measured_quantities
    DD = setup_data.good_DD

    # Step 1: Get SI template equations (pre-substitution)
    ordered_model = if isa(model, OrderedODESystem)
        model
    else
        (_, _, model_states, model_ps) = unpack_ODE(model)
        OrderedODESystem(model, model_states, model_ps)
    end

    template_equations, derivative_dict, _, _, _, _ = get_si_equation_system(
        ordered_model, mq, pep.data_sample;
        DD = DD, infolevel = 0,
        compute_multiplicity = false,  # M metadata is discarded here — skip the Groebner step
    )
    template_DD = ensure_si_template_dd_support(ordered_model, mq, DD, derivative_dict)

    # Step 2: Collect all variables from template equations
    all_template_vars = OrderedCollections.OrderedSet{Any}()
    for eq in template_equations
        union!(all_template_vars, Symbolics.get_variables(eq))
    end

    # Step 3: Identify data variables from DD.obs_lhs
    data_var_set = Set{Any}()
    for (level_idx, level_vars) in enumerate(template_DD.obs_lhs)
        for v in level_vars
            if v in all_template_vars
                push!(data_var_set, v)
            end
        end
    end

    # Step 4: Substitute _trfn_ vars only (known functions of time, not data)
    t_vec = pep.data_sample["t"]
    time_idx = argmin(abs.(t_vec .- t_eval))
    t_point = t_vec[time_idx]

    trfn_subst = Dict{Any, Any}()
    trfn_substituted_vars = Set{Any}()
    for v in all_template_vars
        vname = string(v)
        trfn_val = evaluate_trfn_template_variable(vname, t_point)
        if isnothing(trfn_val)
            # Also try _obs_trfn_ pattern (observable wrappers for transcendental inputs)
            trfn_val = evaluate_obs_trfn_template_variable(vname, t_point)
        end
        if !isnothing(trfn_val)
            trfn_subst[v] = trfn_val
            push!(trfn_substituted_vars, v)
        end
    end

    working_equations = if !isempty(trfn_subst)
        Symbolics.substitute.(template_equations, Ref(trfn_subst))
    else
        template_equations
    end

    # Remove trivial equations (0 variables after _trfn_ substitution)
    kept_equations = eltype(working_equations)[]
    for eq in working_equations
        if !isempty(Symbolics.get_variables(eq))
            push!(kept_equations, eq)
        end
    end

    if isempty(kept_equations)
        return Matrix{Float64}(undef, 0, 0), String[], Dict{String, Symbol}(), String[], Dict{String, Symbol}()
    end

    # Re-collect variables after substitution
    remaining_vars = OrderedCollections.OrderedSet{Any}()
    for eq in kept_equations
        union!(remaining_vars, Symbolics.get_variables(eq))
    end

    # Step 5: Separate unknowns (solved-for) and data (interpolated) variables
    # Remove _trfn_/_obs_trfn_ vars from data set (they're known functions, not data)
    for v in trfn_substituted_vars
        delete!(data_var_set, v)
    end
    # Template vars and prod_vars may be different Symbolics objects with the same name,
    # so use isequal for matching (structural comparison, not reference equality).
    data_var_names = Set(string(v) for v in data_var_set)
    unknown_vars = [v for v in remaining_vars if !(string(v) in data_var_names)]
    data_vars = [v for v in remaining_vars if string(v) in data_var_names]

    n_x = length(unknown_vars)
    n_d = length(data_vars)

    if n_x == 0 || n_d == 0
        return Matrix{Float64}(undef, 0, 0), String[], Dict{String, Symbol}(), String[], Dict{String, Symbol}()
    end

    # Step 6: Build evaluation-point values for all variables from the values
    # `pep` carries.
    # Unknowns use _build_true_value_vector (SIAN naming).
    # Data vars use the obs Taylor coefficients directly via DD.obs_lhs mapping.
    unknown_true = _build_true_value_vector(pep, unknown_vars;
        state_taylor = state_taylor, obs_taylor = obs_taylor, t_eval = t_eval)

    # Build oracle values for data variables from obs_taylor
    data_true = Float64[]
    for v in data_vars
        val = NaN
        # Match data variable against DD.obs_lhs to find (obs_idx, deriv_level)
        for (level_idx, level_vars) in enumerate(template_DD.obs_lhs)
            deriv_level = level_idx - 1
            for (obs_idx, lhs_var) in enumerate(level_vars)
                if isequal(v, lhs_var)
                    # Get oracle value: obs_taylor gives Taylor coefficients
                    # Taylor coeff[k+1] * k! = f^(k)(t_eval)
                    if !isnothing(obs_taylor) && obs_idx <= length(mq)
                        obs_rhs_key = Symbolics.diff2term(mq[obs_idx].rhs)
                        if haskey(obs_taylor, obs_rhs_key)
                            tc = obs_taylor[obs_rhs_key]
                            if deriv_level + 1 <= length(tc)
                                val = tc[deriv_level + 1] * factorial(deriv_level)
                            end
                        end
                    end
                    @goto found_data_val
                end
            end
        end
        @label found_data_val
        push!(data_true, val)
    end

    combined_vars = [unknown_vars..., data_vars...]
    combined_true = [unknown_true..., data_true...]

    if any(isnan, combined_true)
        @warn "[DIAG] NaN in combined true values for data sensitivity" nan_count = count(isnan, combined_true)
        return Matrix{Float64}(undef, 0, 0), String[], Dict{String, Symbol}(), String[], Dict{String, Symbol}()
    end

    # Step 7: Build combined function and compute full Jacobian
    combined_fn = _compile_system_function(kept_equations, combined_vars)
    J_full = ForwardDiff.jacobian(combined_fn, combined_true)

    # Partition: J_x = J_full[:, 1:n_x], J_d = J_full[:, n_x+1:end]
    J_x = J_full[:, 1:n_x]
    J_d = J_full[:, (n_x + 1):end]

    # Step 8: IFT solve (factorized; degrades loudly instead of silent pinv)
    S, cond_Jx, ift_degraded = _ift_solve(J_x, J_d)

    # Build labels and roles
    d_labels = [string(v) for v in data_vars]
    d_roles = _classify_polynomial_variables(d_labels, pep)
    x_labels = [string(v) for v in unknown_vars]
    x_roles = _classify_polynomial_variables(x_labels, pep)

    return S, d_labels, d_roles, x_labels, x_roles
end
