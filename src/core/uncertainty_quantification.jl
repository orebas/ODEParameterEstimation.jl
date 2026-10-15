"""
Uncertainty Quantification for ODE Parameter Estimation

This module provides functions for propagating uncertainty from GP-fitted observables
and their derivatives to estimated parameters using the implicit function theorem.

# Theory
For a GP f ~ GP(m, k), derivatives are jointly Gaussian. The key result:
    Cov(f^(i)(t), f^(j)(t')) = ∂^(i+j) k(t, t') / (∂t^i ∂t'^j)

Given parameter estimation as F(θ, z) = 0, the IFT gives:
    Cov(θ) ≈ (J_θ⁻¹ J_z) Σ_z (J_θ⁻¹ J_z)ᵀ

# References
- Rasmussen & Williams (2006) - "Gaussian Processes for Machine Learning", Ch. 9
- Solak et al. (2003) - "Derivative observations in Gaussian process models"
"""

#==========================================================================
 Helper Functions for Symbol Handling
==========================================================================#


#==========================================================================
 SE Kernel Derivatives - Core Building Block
==========================================================================#

"""
    se_kernel_derivative(σ²::Real, ℓ::Real, Δt::Real, i::Int, j::Int) -> Real

Compute the SE kernel derivative: ∂^(i+j)/∂t^i∂t'^j k(t, t') evaluated at Δt = t - t'.

The SE kernel is k(Δt) = σ² exp(-Δt²/(2ℓ²)).

Uses closed-form expressions derived from the product rule and the fact that
∂ⁿ/∂xⁿ exp(-x²/2) = Hₙ(x) exp(-x²/2) where Hₙ is the probabilist's Hermite polynomial.

# Arguments
- `σ²::Real`: Signal variance
- `ℓ::Real`: Lengthscale
- `Δt::Real`: Time difference t - t'
- `i::Int`: Order of derivative with respect to t (0 ≤ i ≤ 4)
- `j::Int`: Order of derivative with respect to t' (0 ≤ j ≤ 4)

# Returns
- The value of the kernel derivative at Δt

# Examples
```julia
# Prior variance of f(t)
se_kernel_derivative(1.0, 0.5, 0.0, 0, 0)  # = σ² = 1.0

# Prior variance of f'(t)
se_kernel_derivative(1.0, 0.5, 0.0, 1, 1)  # = σ²/ℓ² = 4.0

# Covariance between f(t) and f''(t) at same point
se_kernel_derivative(1.0, 0.5, 0.0, 0, 2)  # = -σ²/ℓ² = -4.0
```
"""
function se_kernel_derivative(σ²::Real, ℓ::Real, Δt::Real, i::Int, j::Int)::Float64
	@assert i >= 0 && j >= 0 "Derivative orders must be non-negative"

	# Normalized distance
	u = Δt / ℓ
	u² = u^2

	# Base kernel value: k(Δt) = σ² exp(-Δt²/(2ℓ²)) = σ² exp(-u²/2)
	base = σ² * exp(-u² / 2)

	# Total derivative order
	n = i + j

	# General sign pattern: (-1)^i · σ²/ℓ^(i+j) · He_{i+j}(u) · exp(-u²/2)
	# where He_n is the probabilist's Hermite polynomial
	# This comes from ∂/∂t = +∂/∂Δ and ∂/∂t' = -∂/∂Δ

	if n == 0
		# k(Δt) = σ² exp(-u²/2)
		return base

	elseif n == 1
		# First derivative: ∂k/∂t = -σ² u/ℓ exp(-u²/2)
		# ∂k/∂t' = σ² u/ℓ exp(-u²/2)  (opposite sign due to chain rule with -)
		sign = (i == 1) ? -1 : 1
		return sign * base * u / ℓ

	elseif n == 2
		if i == 1 && j == 1
			# ∂²k/∂t∂t' = σ²/ℓ² (1 - u²) exp(-u²/2)
			return base / ℓ^2 * (1 - u²)
		else
			# ∂²k/∂t² or ∂²k/∂t'² = σ²/ℓ² (u² - 1) exp(-u²/2)
			return base / ℓ^2 * (u² - 1)
		end
	end

	# For n ≥ 3: general formula (-1)^i · σ²/ℓⁿ · Heₙ(u) · exp(-u²/2)
	# Hermite recurrence: He₀=1, He₁=u, He_{k+1} = u·Heₖ - k·He_{k-1}
	He_prev, He_curr = 1.0, u
	for k in 1:(n - 1)
		He_prev, He_curr = He_curr, u * He_curr - k * He_prev
	end
	return (iseven(i) ? 1 : -1) * base / ℓ^n * He_curr
end

"""
    se_kernel_prior_covariance_matrix(σ²::Real, ℓ::Real, max_deriv::Int=2) -> Matrix{Float64}

Compute the prior covariance matrix for [f(t), f'(t), f''(t), ...] at a single point t.

For the SE kernel at t = t' (Δt = 0), many cross-covariances are zero due to symmetry:
- Cov(f, f') = 0 (odd symmetry)
- Cov(f', f'') = 0 (odd symmetry)

# Arguments
- `σ²::Real`: Signal variance
- `ℓ::Real`: Lengthscale
- `max_deriv::Int`: Maximum derivative order (default 2)

# Returns
- Symmetric matrix of size (max_deriv+1) × (max_deriv+1)

# Example
For SE kernel with σ²=1, ℓ=1, the matrix for [f, f', f''] is:
```
[  1      0    -1   ]
[  0      1     0   ]
[ -1      0     3   ]
```
"""
function se_kernel_prior_covariance_matrix(σ²::Real, ℓ::Real, max_deriv::Int = 2)::Matrix{Float64}
	n = max_deriv + 1
	Σ = zeros(n, n)

	for i in 0:max_deriv
		for j in 0:max_deriv
			Σ[i+1, j+1] = se_kernel_derivative(σ², ℓ, 0.0, i, j)
		end
	end

	return Σ
end

"""
    se_kernel_cross_time_covariance_matrix(σ²::Real, ℓ::Real, delta_t::Real, max_deriv::Int=2) -> Matrix{Float64}

Compute the prior covariance matrix between derivative vectors at two time points separated by `delta_t`.

Entry [a+1, b+1] = Cov(f^(a)(t), f^(b)(t')) where t - t' = delta_t.

Reduces to `se_kernel_prior_covariance_matrix` when `delta_t == 0`.

# Arguments
- `σ²::Real`: Signal variance
- `ℓ::Real`: Lengthscale
- `delta_t::Real`: Time difference t - t'
- `max_deriv::Int`: Maximum derivative order (default 2)

# Returns
- Matrix of size (max_deriv+1) × (max_deriv+1)
"""
function se_kernel_cross_time_covariance_matrix(σ²::Real, ℓ::Real, delta_t::Real, max_deriv::Int = 2)::Matrix{Float64}
	n = max_deriv + 1
	Σ = zeros(n, n)

	for i in 0:max_deriv
		for j in 0:max_deriv
			Σ[i+1, j+1] = se_kernel_derivative(σ², ℓ, delta_t, i, j)
		end
	end

	return Σ
end

#==========================================================================
 AGP Extended Interpolator with Full UQ Support
==========================================================================#

"""
    AGPInterpolatorUQ

Extended GP interpolator with full uncertainty quantification support.
Stores pre-computed values needed for efficient derivative covariance computation.

# Fields
- `mean_function::Function`: Returns mean prediction (denormalized)
- `std_function::Function`: Returns standard deviation (denormalized)
- `xs_train::Vector{Float64}`: Training x values (raw, not normalized)
- `ys_train::Vector{Float64}`: Training y values (normalized)
- `alpha::Vector{Float64}`: Pre-computed weights (K⁻¹ y)
- `chol::Cholesky`: Cholesky factorization of K + σₙ²I
- `lengthscale::Float64`: Optimized lengthscale
- `fitted_lengthscale::Float64`: Function-value marginal-likelihood lengthscale
- `lengthscale_factor::Float64`: Explicit derivative-oriented multiplier
- `signal_var::Float64`: Optimized signal variance (σ²)
- `noise_var::Float64`: Optimized noise variance (σₙ²)
- `y_mean::Float64`: Mean of original y values
- `y_std::Float64`: Std of original y values
- `cholesky_jitter::Float64`: Absolute diagonal regularizer used by final factorization
- `cholesky_jitter_ratio::Float64`: Jitter divided by learned observation-noise variance
- `factorization_residual::Float64`: Relative residual of the retained Cholesky factor
"""
struct AGPInterpolatorUQ <: AbstractInterpolator
	mean_function::Function
	std_function::Function
	xs_train::Vector{Float64}
	ys_train::Vector{Float64}
	alpha::Vector{Float64}
	chol::Cholesky
	lengthscale::Float64
	fitted_lengthscale::Float64
	lengthscale_factor::Float64
	signal_var::Float64
	noise_var::Float64
	y_mean::Float64
	y_std::Float64
	cholesky_jitter::Float64
	cholesky_jitter_ratio::Float64
	factorization_residual::Float64
	hyperparams_optimized::Bool
end

# Backward-compatible 13-arg constructor from before factorization telemetry.
function AGPInterpolatorUQ(
	mean_function, std_function, xs_train, ys_train, alpha, chol,
	lengthscale, signal_var, noise_var, y_mean, y_std,
	cholesky_jitter, hyperparams_optimized,
)
	ratio = cholesky_jitter == 0 ? 0.0 :
		noise_var > 0 ? cholesky_jitter / noise_var : Inf
	return AGPInterpolatorUQ(
		mean_function, std_function, xs_train, ys_train, alpha, chol,
		lengthscale, lengthscale, 1.0, signal_var, noise_var, y_mean, y_std,
		cholesky_jitter, ratio, NaN, hyperparams_optimized,
	)
end

# Backward-compatible 11-arg constructor (defaults: jitter=0.0, optimized=true)
function AGPInterpolatorUQ(
	mean_function, std_function, xs_train, ys_train, alpha, chol,
	lengthscale, signal_var, noise_var, y_mean, y_std,
)
	return AGPInterpolatorUQ(
		mean_function, std_function, xs_train, ys_train, alpha, chol,
		lengthscale, signal_var, noise_var, y_mean, y_std,
		0.0, true,
	)
end

const GP_UQ_MATERIAL_JITTER_RATIO = 1.0

"""
	gp_factorization_diagnostics(interp::AGPInterpolatorUQ)

Return scale-aware numerical telemetry for the GP factorization used by both
the estimator mean and derivative UQ. `:material_regularization` means the
added Cholesky jitter exceeded the learned observation-noise variance. This is
a warning axis, not by itself proof that the retained linear solve is invalid.
"""
function gp_factorization_diagnostics(interp::AGPInterpolatorUQ)
	status = if !isfinite(interp.factorization_residual)
		:not_recorded
	elseif interp.factorization_residual > 1e-10
		:factorization_mismatch
	elseif interp.cholesky_jitter_ratio > GP_UQ_MATERIAL_JITTER_RATIO
		:material_regularization
	else
		:ok
	end
	return (
		lengthscale = interp.lengthscale,
		fitted_lengthscale = interp.fitted_lengthscale,
		lengthscale_factor = interp.lengthscale_factor,
		signal_variance = interp.signal_var,
		noise_variance = interp.noise_var,
		jitter = interp.cholesky_jitter,
		jitter_to_noise = interp.cholesky_jitter_ratio,
		factorization_residual = interp.factorization_residual,
		status = status,
	)
end

# Make it callable like other interpolators
(interp::AGPInterpolatorUQ)(x) = interp.mean_function(x)

"""
    _build_K_star_n(interp::AGPInterpolatorUQ, t::Real, max_deriv::Int) -> Matrix{Float64}

Build the K_*n matrix: covariance between test derivatives at `t` and training points.

K_star_n[d+1, k] = Cov(f^(d)(t), f(x_k)) = ∂^d/∂t^d k(t, x_k)

# Returns
- Matrix of size (max_deriv+1) × n_train
"""
function _build_K_star_n(interp::AGPInterpolatorUQ, t::Real, max_deriv::Int)::Matrix{Float64}
	σ² = interp.signal_var
	ℓ = interp.lengthscale
	xs = interp.xs_train
	n_train = length(xs)
	n_derivs = max_deriv + 1

	K_star_n = zeros(n_derivs, n_train)
	for d in 0:max_deriv
		for k in 1:n_train
			Δt = t - xs[k]
			K_star_n[d+1, k] = se_kernel_derivative(σ², ℓ, Δt, d, 0)
		end
	end

	return K_star_n
end

"""Raw training observations represented by an `AGPInterpolatorUQ`."""
function _raw_training_values(interp::AGPInterpolatorUQ)::Vector{Float64}
	return interp.ys_train .* interp.y_std .+ interp.y_mean
end

"""
	learned_observation_noise_variance(interp::AGPInterpolatorUQ) -> Float64

Return the GP-learned observation noise variance in the original observation
units.  `AGPInterpolatorUQ.noise_var` is learned after z-scoring the observable,
so it must be multiplied by `y_std^2` before it is used with raw-data influence
weights.
"""
function learned_observation_noise_variance(interp::AGPInterpolatorUQ)::Float64
	return max(interp.noise_var, 0.0) * interp.y_std^2
end

"""
	gp_derivative_influence_matrix(interp, t, max_deriv) -> (mean, W)

For fixed GP hyperparameters, compute the linear map from raw observations `y`
to the posterior-mean jet `[f(t), f'(t), ...]`.

The current GP implementation centers and scales `y` before fitting.  This
routine folds the centering operation into `W`, so `W * raw_y` reproduces the
implemented derivative estimator exactly up to floating-point roundoff.
"""
function gp_derivative_influence_matrix(
	interp::AGPInterpolatorUQ,
	t::Real,
	max_deriv::Int = 2,
)::Tuple{Vector{Float64}, Matrix{Float64}}
	max_deriv < 0 && throw(ArgumentError("max_deriv must be non-negative"))

	K_star_n = _build_K_star_n(interp, t, max_deriv)
	# A^{-1} K_*' where A = K_train + σ_n² I.  Transposing gives rows
	# k_d(t)' A^{-1}, one for each derivative order.
	W = Matrix((interp.chol \ K_star_n')')

	n = length(interp.xs_train)
	n == 0 && throw(ArgumentError("AGPInterpolatorUQ has no training points"))

	# The fit uses ys_norm = (y - mean(y)) / std(y).  The final multiplication
	# by y_std cancels the scaling, leaving only the centering projection.
	for r in axes(W, 1)
		W[r, :] .-= mean(@view W[r, :])
	end
	# Order zero adds the data mean back to the posterior mean.
	W[1, :] .+= 1.0 / n

	raw_y = _raw_training_values(interp)
	return W * raw_y, W
end

function _make_jet_labels(obs_name::AbstractString, t::Real, max_deriv::Int)
	t_str = string(round(Float64(t), digits = 3))
	return [d == 0 ? "$(obs_name)(t=$(t_str))" : "$(obs_name)$(repeat("'", d))(t=$(t_str))" for d in 0:max_deriv]
end

function _psd_symmetric_matrix(Σ::AbstractMatrix{<:Real}; jitter_floor::Float64 = 1e-15)
	Σ_sym = Symmetric(Matrix{Float64}(Σ))
	evals = eigvals(Σ_sym)
	if !isempty(evals) && minimum(evals) < 0
		return Matrix(Σ_sym) + Matrix{Float64}(I, size(Σ_sym, 1), size(Σ_sym, 2)) *
			   (abs(minimum(evals)) + jitter_floor)
	end
	return Matrix(Σ_sym)
end

"""
Resolve the observation-noise covariance Σ_y for estimator-sampling
propagation: GP-learned homoscedastic σ²I by default, or a caller-supplied
n×n matrix (validated; provenance relabeled). Shared by the scalar and
stacked `joint_derivative_estimator_covariance` methods.
"""
function _resolve_observation_covariance(
	interp::AGPInterpolatorUQ,
	noise_source::Symbol,
	observation_covariance::Union{Nothing, AbstractMatrix{<:Real}},
)
	n = length(interp.xs_train)
	if observation_covariance === nothing
		if noise_source != :learned_gp_homoscedastic
			throw(ArgumentError("noise_source=$noise_source requires explicit observation_covariance"))
		end
		return Diagonal(fill(learned_observation_noise_variance(interp), n)), noise_source
	end
	size(observation_covariance) == (n, n) ||
		throw(ArgumentError("observation_covariance has size $(size(observation_covariance)); expected ($n, $n)"))
	resolved = noise_source == :learned_gp_homoscedastic ? :user_supplied_covariance : noise_source
	return Matrix{Float64}(observation_covariance), resolved
end

"""
	joint_derivative_estimator_covariance(interp, t, max_deriv=2; kwargs...)

Compute the sampling covariance of the GP posterior-mean derivative estimator.
More precisely: the repeated-observation sampling covariance evaluated at
plug-in (fitted) hyperparameters — how the jet estimates would fluctuate if
the observations were redrawn around a fixed underlying signal with noise
covariance Σ_y. (Hyperparameter-estimation variability is omitted; the
direction and magnitude of the resulting covariance error are not guaranteed.)

This is intentionally different from `joint_derivative_covariance`, which
returns the latent GP posterior covariance.
"""
function joint_derivative_estimator_covariance(
	interp::AGPInterpolatorUQ,
	t::Real,
	max_deriv::Int = 2;
	observable_name::AbstractString = "",
	noise_source::Symbol = :learned_gp_homoscedastic,
	observation_covariance::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
)::JetInfluenceEstimate
	μ, W = gp_derivative_influence_matrix(interp, t, max_deriv)
	Σ_y, resolved_noise_source = _resolve_observation_covariance(interp, noise_source, observation_covariance)

	Σ_jet = _psd_symmetric_matrix(W * Σ_y * W')
	warnings = String[]
	if !interp.hyperparams_optimized
		push!(warnings, "GP hyperparameter optimization did not converge; estimator covariance uses fallback hyperparameters.")
	end
	if resolved_noise_source == :learned_gp_homoscedastic && learned_observation_noise_variance(interp) <= 0
		push!(warnings, "Learned GP observation noise variance is zero; estimator covariance may be underestimated.")
	end

	obs_name = String(observable_name)
	labels = isempty(obs_name) ? ["order_$d" for d in 0:max_deriv] : _make_jet_labels(obs_name, t, max_deriv)
	return JetInfluenceEstimate(
		obs_name,
		Float64(t),
		collect(0:max_deriv),
		labels,
		μ,
		W,
		Σ_y,
		Σ_jet,
		:estimator_sampling,
		resolved_noise_source,
		warnings,
	)
end

"""
	joint_derivative_estimator_covariance(interp, ts::AbstractVector, max_deriv=2; kwargs...)

Stacked multi-time estimator-sampling covariance of one observable's jets:
`Σ = W_stack · Σ_y · W_stackᵀ` with `W_stack = vcat(W(t₁), …, W(t_T))`.
Cross-time blocks `W(tᵢ) Σ_y W(tⱼ)ᵀ` are nonzero because every jet is a
linear functional of the SAME training observations — this is the
repeated-observation sampling covariance at plug-in hyperparameters, NOT the
latent GP posterior cross-time covariance (`build_observation_covariance`
computes that distinct object; the two must never be mixed).

Factorized construction is PSD up to roundoff whenever Σ_y is PSD. Tiny
negative eigenvalues are clipped to zero; materially negative ones warn
loudly (construction-bug indicator). Genuine singularity — e.g. repeated
evaluation times — is preserved, never shifted away: `S Σ_d Sᵀ` downstream
needs no inverse of this matrix.

Row order is time-major (per time, orders `0:max_deriv`); map indices with
[`stacked_jet_index`](@ref).
"""
function joint_derivative_estimator_covariance(
	interp::AGPInterpolatorUQ,
	ts::AbstractVector{<:Real},
	max_deriv::Int = 2;
	observable_name::AbstractString = "",
	noise_source::Symbol = :learned_gp_homoscedastic,
	observation_covariance::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
)::StackedJetInfluenceEstimate
	isempty(ts) && throw(ArgumentError("ts must be non-empty"))
	Σ_y, resolved_noise_source = _resolve_observation_covariance(interp, noise_source, observation_covariance)

	means = Vector{Float64}[]
	Ws = Matrix{Float64}[]
	for t in ts
		μ_t, W_t = gp_derivative_influence_matrix(interp, t, max_deriv)
		push!(means, μ_t)
		push!(Ws, W_t)
	end
	W_stack = vcat(Ws...)
	μ = vcat(means...)

	Σ_sym = Symmetric(W_stack * Σ_y * W_stack')
	warnings = String[]
	evals, evecs = eigen(Σ_sym)
	min_eig = isempty(evals) ? 0.0 : minimum(evals)
	max_eig = isempty(evals) ? 0.0 : maximum(abs, evals)
	Σ_jet = if min_eig < 0
		if min_eig < -1e-10 * max(max_eig, 1e-300)
			msg = "Stacked jet covariance materially indefinite (min eig $(min_eig), max $(max_eig)) — clipped to zero; investigate construction."
			push!(warnings, msg)
			@warn "[UQ] $msg"
		end
		Matrix{Float64}(Symmetric(evecs * Diagonal(max.(evals, 0.0)) * evecs'))
	else
		Matrix{Float64}(Σ_sym)
	end

	if !interp.hyperparams_optimized
		push!(warnings, "GP hyperparameter optimization did not converge; estimator covariance uses fallback hyperparameters.")
	end
	if resolved_noise_source == :learned_gp_homoscedastic && learned_observation_noise_variance(interp) <= 0
		push!(warnings, "Learned GP observation noise variance is zero; estimator covariance may be underestimated.")
	end

	obs_name = String(observable_name)
	labels = String[]
	for t in ts
		append!(labels,
			isempty(obs_name) ? ["order_$d" for d in 0:max_deriv] : _make_jet_labels(obs_name, t, max_deriv))
	end

	return StackedJetInfluenceEstimate(
		obs_name,
		Float64.(collect(ts)),
		collect(0:max_deriv),
		labels,
		μ,
		W_stack,
		Σ_y,
		Σ_jet,
		:estimator_sampling,
		resolved_noise_source,
		warnings,
	)
end

"""
	stacked_jet_index(est::StackedJetInfluenceEstimate, time_idx, order) -> Int

Row/column index of (time `time_idx`, derivative `order`) in the time-major
stacked jet ordering.
"""
function stacked_jet_index(est::StackedJetInfluenceEstimate, time_idx::Int, order::Int)
	n_orders = length(est.orders)
	1 <= time_idx <= length(est.t_evals) ||
		throw(ArgumentError("time_idx $time_idx out of range 1:$(length(est.t_evals))"))
	0 <= order < n_orders ||
		throw(ArgumentError("order $order out of range 0:$(n_orders - 1)"))
	return (time_idx - 1) * n_orders + order + 1
end

"""
	nth_deriv(f::AGPInterpolatorUQ, n::Int, t::Real) -> Real

Analytic nth derivative of the GP posterior mean. Uses the closed-form SE kernel
derivative formula via `_build_K_star_n` — avoids TaylorDiff and is exact.

The GP posterior mean is: μ(t) = y_std * K_*n(t) · α + y_mean
Its nth derivative is:  μ^(n)(t) = y_std * K_*n[n+1, :] · α
"""
function nth_deriv(f::AGPInterpolatorUQ, n::Int, t::Real)::Real
	if n == 0
		return f(t)
	end
	K_star_n = _build_K_star_n(f, t, n)
	return f.y_std * dot(K_star_n[n+1, :], f.alpha)
end

# Do not hide the analytic AGPUQ derivative behind an anonymous closure on
# production estimation paths.  UQ propagates this exact fixed-smoother
# influence operator, so the estimator must consume the same operator.
function _estimation_derivative(f::AGPInterpolatorUQ, n::Int, t::Real)::Real
	return nth_deriv(f, n, t)
end

"""
    joint_derivative_covariance(interp::AGPInterpolatorUQ, t::Real, max_deriv::Int=2)

Compute the joint posterior covariance matrix for [f(t), f'(t), ..., f^(max_deriv)(t)].

This is the key function for UQ: it gives us the covariance structure of the
observable and its derivatives at a single time point.

# Arguments
- `interp::AGPInterpolatorUQ`: The GP interpolator with stored hyperparameters
- `t::Real`: Time point at which to compute covariance
- `max_deriv::Int`: Maximum derivative order to include (default 2)

# Returns
- `μ::Vector{Float64}`: Posterior means [f(t), f'(t), ..., f^(max_deriv)(t)]
- `Σ::Matrix{Float64}`: Posterior covariance matrix

# Theory
The posterior covariance is:
    Σ_posterior = Σ_prior - K_*n K⁻¹ K_n*

where:
- Σ_prior is the prior covariance matrix for derivatives at t
- K_*n is the covariance between test derivatives and training points
- K⁻¹ is the inverse of the training covariance (via Cholesky)
"""
function joint_derivative_covariance(interp::AGPInterpolatorUQ, t::Real, max_deriv::Int = 2)
	σ² = interp.signal_var
	ℓ = interp.lengthscale
	alpha = interp.alpha
	C = interp.chol
	n_derivs = max_deriv + 1

	# 1. Build prior covariance for [f(t), f'(t), ..., f^(max_deriv)(t)]
	Σ_prior = se_kernel_prior_covariance_matrix(σ², ℓ, max_deriv)

	# 2. Build K_*n matrix using extracted helper
	K_star_n = _build_K_star_n(interp, t, max_deriv)

	# 3. Posterior mean: μ = K_*n @ alpha
	μ_norm = K_star_n * alpha

	# 4. Posterior covariance: Σ = Σ_prior - K_*n @ K⁻¹ @ K_n*
	# Using Cholesky: K⁻¹ @ K_n* = C \ (C' \ K_n*')' = (C \ K_star_n')'
	V = C \ K_star_n'  # n_train × n_derivs
	Σ_posterior = Σ_prior - K_star_n * V

	# 5. Ensure positive semi-definiteness (numerical safety)
	Σ_posterior = Symmetric(Σ_posterior)

	# Add small jitter if needed for numerical stability
	min_eig = minimum(eigvals(Σ_posterior))
	if min_eig < 0
		Σ_posterior = Σ_posterior + Matrix{Float64}(I, n_derivs, n_derivs) * (abs(min_eig) + 1e-10)
	end

	# 6. De-normalize to original scale
	# f scales by y_std, f' scales by y_std (x is not normalized)
	# f'' scales by y_std, etc.
	scale_factors = fill(interp.y_std, n_derivs)

	μ_scaled = μ_norm .* scale_factors
	μ_scaled[1] += interp.y_mean  # Only add mean to f(t), not derivatives

	Σ_scaled = Diagonal(scale_factors) * Σ_posterior * Diagonal(scale_factors)

	return μ_scaled, Matrix(Σ_scaled)
end


#==========================================================================
 Building Full Observation Covariance Matrix Σ_z
==========================================================================#

"""
    build_observation_covariance(gp_results::Dict, times::Vector, max_deriv::Int=2)

Build the full covariance matrix Σ_z for all observables and derivatives at given times.

Includes **within-observable cross-time covariance blocks** from the GP posterior.
Cross-observable blocks remain zero (independent GPs for different sensors).

# Arguments
- `gp_results::Dict{String, AGPInterpolatorUQ}`: GP results keyed by observable name
- `times::Vector{<:Real}`: Time points at which to evaluate
- `max_deriv::Int`: Maximum derivative order (default 2)

# Returns
- `μ_z::Vector{Float64}`: Full mean vector for all observables and derivatives
- `Σ_z::Matrix{Float64}`: Full covariance matrix
- `labels::Vector{String}`: Labels for each component (e.g., "y1(t=0.5)", "y1'(t=0.5)")

# Structure
The ordering is: for each observable, for each time, [f, f', f'', ...]
So for 2 observables, 3 times, max_deriv=1:
  [y1(t1), y1'(t1), y1(t2), y1'(t2), y1(t3), y1'(t3), y2(t1), y2'(t1), ...]
"""
function build_observation_covariance(
	gp_results::Dict{String, <:AGPInterpolatorUQ},
	times::Vector{<:Real},
	max_deriv::Int = 2,
)
	n_obs = length(gp_results)
	n_times = length(times)
	n_derivs = max_deriv + 1
	obs_block_size = n_times * n_derivs  # size of per-observable block
	total_dim = n_obs * obs_block_size

	μ_z = zeros(total_dim)
	Σ_z = zeros(total_dim, total_dim)
	labels = String[]

	obs_offset = 0  # tracks starting row/col for current observable

	for (obs_name, gp) in gp_results
		σ² = gp.signal_var
		ℓ = gp.lengthscale
		C = gp.chol
		alpha = gp.alpha
		scale_factors = fill(gp.y_std, n_derivs)
		scale_diag = Diagonal(scale_factors)

		# Pre-compute K_star_n and V = C \ K_star_n' for all time points
		K_stars = Vector{Matrix{Float64}}(undef, n_times)
		Vs = Vector{Matrix{Float64}}(undef, n_times)
		for ti in 1:n_times
			K_stars[ti] = _build_K_star_n(gp, times[ti], max_deriv)
			Vs[ti] = C \ K_stars[ti]'  # n_train × n_derivs
		end

		for ti in 1:n_times
			row_start = obs_offset + (ti - 1) * n_derivs + 1
			row_range = row_start:(row_start + n_derivs - 1)

			# Posterior mean
			μ_norm = K_stars[ti] * alpha
			μ_scaled = μ_norm .* scale_factors
			μ_scaled[1] += gp.y_mean
			μ_z[row_range] = μ_scaled

			# Generate labels for this time point
			for d in 0:max_deriv
				deriv_label = d == 0 ? "" : "'" ^ d
				push!(labels, "$(obs_name)$(deriv_label)(t=$(round(times[ti], digits=3)))")
			end

			# Diagonal block: same-time covariance (ti, ti)
			Σ_prior_diag = se_kernel_prior_covariance_matrix(σ², ℓ, max_deriv)
			Σ_post_diag = Σ_prior_diag - K_stars[ti] * Vs[ti]

			# PSD enforcement on diagonal block
			Σ_post_diag = Symmetric(Σ_post_diag)
			min_eig = minimum(eigvals(Σ_post_diag))
			if min_eig < 0
				Σ_post_diag = Σ_post_diag + Matrix{Float64}(I, n_derivs, n_derivs) * (abs(min_eig) + 1e-10)
			end

			# De-normalize
			Σ_z[row_range, row_range] = scale_diag * Σ_post_diag * scale_diag

			# Off-diagonal (cross-time) blocks within this observable
			for tj in (ti+1):n_times
				col_start = obs_offset + (tj - 1) * n_derivs + 1
				col_range = col_start:(col_start + n_derivs - 1)

				# Prior cross-covariance at delta_t = times[ti] - times[tj]
				Σ_prior_cross = se_kernel_cross_time_covariance_matrix(σ², ℓ, times[ti] - times[tj], max_deriv)

				# Posterior cross-covariance: Σ_prior_cross - K_*n(ti) K⁻¹ K_n*(tj)
				Σ_cross = Σ_prior_cross - K_stars[ti] * Vs[tj]

				# De-normalize (no PSD enforcement for off-diagonal blocks)
				Σ_cross_scaled = scale_diag * Σ_cross * scale_diag

				Σ_z[row_range, col_range] = Σ_cross_scaled
				Σ_z[col_range, row_range] = Σ_cross_scaled'  # symmetric
			end
		end

		# PSD enforcement on the full per-observable block
		obs_range = (obs_offset + 1):(obs_offset + obs_block_size)
		Σ_obs = Symmetric(Σ_z[obs_range, obs_range])
		min_eig_obs = minimum(eigvals(Σ_obs))
		if min_eig_obs < 0
			Σ_z[obs_range, obs_range] .+= Matrix{Float64}(I, obs_block_size, obs_block_size) * (abs(min_eig_obs) + 1e-10)
		end

		obs_offset += obs_block_size
	end

	return μ_z, Symmetric(Σ_z), labels
end

#==========================================================================
 Parameter Covariance via Implicit Function Theorem
==========================================================================#

"""
    agp_gpr_uq(xs::AbstractArray, ys::AbstractArray;
        kernel_type=:se, lengthscale_factor=1.0) -> AGPInterpolatorUQ

Create a GP interpolator with full uncertainty quantification support.

Like `agp_gpr_robust`, but it returns an `AGPInterpolatorUQ` that stores all the
information needed for computing derivative covariances.

# Arguments
- `xs::AbstractArray`: X coordinates (e.g., time points)
- `ys::AbstractArray`: Y coordinates (observations)
- `kernel_type::Symbol`: currently only `:se` (Squared Exponential)
- `lengthscale_factor::Float64`: opt-in multiplier applied after fitting the
  function-value marginal-likelihood lengthscale (default: `1.0`)

# Returns
- `AGPInterpolatorUQ` with full UQ capability
"""
function agp_gpr_uq(xs::AbstractArray{T}, ys::AbstractArray{T};
	kernel_type::Symbol = :se,
	lengthscale_factor::Float64 = 1.0,
)::AGPInterpolatorUQ where {T}
	@assert length(xs) == length(ys) "Input arrays must have same length"
	@assert length(xs) >= 3 "Need at least 3 points for GP interpolation"
	kernel_type == :se ||
		throw(ArgumentError("agp_gpr_uq currently supports only kernel_type=:se for derivative estimator UQ; got $kernel_type"))
	isfinite(lengthscale_factor) && lengthscale_factor > 0 ||
		throw(ArgumentError("lengthscale_factor must be finite and positive"))

	# Handle constant data edge case
	y_std_raw = std(ys)
	if y_std_raw < 1e-10
		constant_val = mean(ys)
		# Return a dummy interpolator for constant data
		xs_vec = collect(Float64, xs)
		ys_norm = zeros(length(xs))
		alpha = zeros(length(xs))
		K = Matrix{Float64}(I, length(xs), length(xs))
		C = cholesky(K)
		return AGPInterpolatorUQ(
			x -> constant_val,
			x -> 0.0,
			xs_vec, ys_norm, alpha, C,
			1.0, 1.0, lengthscale_factor, 0.0, 1e-6,
			constant_val, 1.0,
			0.0, 0.0, 0.0, true,
		)
	end

	# Use raw X values (no normalization), as agp_gpr_robust does
	xs_raw = collect(Float64, xs)
	n = length(xs_raw)

	# Standardize Y (zero mean, unit variance)
	y_mean = mean(ys)
	y_std = max(y_std_raw, 1e-8)
	ys_norm = (collect(Float64, ys) .- y_mean) ./ y_std

	# No data jitter — adding noise to observations destroys high-order derivatives.
	# Numerical stability is handled by adaptive Cholesky jitter on the kernel matrix.

	# Use shared SE hyperparameter optimizer (same as agp_gpr_robust :se path)
	D_sq = [abs2(xs_raw[i] - xs_raw[j]) for i in 1:n, j in 1:n]
	hp = _optimize_se_hyperparams(
		xs_raw, ys_norm, D_sq; explicit_se_recipe = true,
	)
	l_fitted = hp.l
	l_opt = l_fitted * lengthscale_factor
	σ²_opt, σₙ²_opt, hp_optimized = hp.sigma2, hp.noise, hp.converged
	if !hp_optimized
		@warn "[UQ] GP hyperparameter optimization did not converge"
	end

	# Build the final matrix with the exact same arithmetic recipe as the NLL.
	# Reconstructing through KernelFunctions here used to differ by a few ulps;
	# on the audited LV cell that was enough to require jitter 59× the fitted
	# noise variance and materially change the derivative estimator.
	K_train = Matrix{Float64}(_se_covariance_matrix(D_sq, l_opt, σ²_opt))
	K_noisy = _add_diagonal_variance(K_train, σₙ²_opt)
	C, jitter_used = _cholesky_adaptive(K_noisy; relative_jitter = true)
	alpha = C \ ys_norm
	jitter_ratio = jitter_used == 0 ? 0.0 :
		σₙ²_opt > 0 ? jitter_used / σₙ²_opt : Inf
	factorization_residual = _cholesky_relative_residual(C, K_noisy, jitter_used)

	if jitter_ratio > GP_UQ_MATERIAL_JITTER_RATIO
		@warn "[UQ] GP kernel matrix required material Cholesky regularization" jitter = jitter_used jitter_to_noise = jitter_ratio lengthscale = l_opt
	end

	# Prediction functions
	# mean_pred uses explicit SE kernel math (no KernelFunctions dependency)
	# so that TaylorDiff can differentiate through it for the estimation pipeline.
	inv_2l2 = 1.0 / (2.0 * l_opt^2)
	function mean_pred(x)
		k_star = [σ²_opt * exp(-(x - xi)^2 * inv_2l2) for xi in xs_raw]
		return y_std * dot(k_star, alpha) + y_mean
	end

	function std_pred(x::Real)
		k_star = [σ²_opt * exp(-(x - xi)^2 * inv_2l2) for xi in xs_raw]
		v = C \ k_star
		prior_var = σ²_opt  # k(x, x) for SE kernel
		post_var = prior_var - dot(k_star, v)
		return y_std * sqrt(max(post_var, 0.0))
	end

	return AGPInterpolatorUQ(
		mean_pred, std_pred,
		xs_raw, ys_norm, alpha, C,
		l_opt, l_fitted, lengthscale_factor, σ²_opt, σₙ²_opt,
		y_mean, y_std,
		jitter_used, jitter_ratio, factorization_residual, hp_optimized,
	)
end

#==========================================================================
 Validation and Testing Utilities
==========================================================================#


"""
	print_uncertainty_results(uq::UncertaintyReport; io = stdout)

Print the standard errors of an [`UncertaintyReport`](@ref), with its status
and warnings. Experimental.
"""
function print_uncertainty_results(uq::UncertaintyReport; io = stdout)
	title = uq.coordinate_system == :physical_initial_conditions ?
		"Parameter/IC Uncertainty (physical coordinates)" :
		"Parameter Uncertainty (local IFT coordinates)"
	println(io, "\n$title")
	println(io, "-"^72)
	println(io, "Status: $(uq.status)   max CV: $(round(uq.max_cv; sigdigits = 3))")
	println(io, "Coordinate system: $(uq.coordinate_system)   t_eval: $(round(uq.t_eval; sigdigits = 6))")
	println(io, "Covariance: $(uq.covariance_kind)   noise source: $(uq.noise_source)")
	if !isnothing(uq.target)
		target = uq.target
		identity = target.identity
		println(io, "Target: rank $(target.selected_rank) $(identity.estimator_kind) candidate $(identity.candidate_id)   estimand: $(target.estimand)")
		!isempty(identity.time_indices) && println(io,
			"Estimator points: indices=$(identity.time_indices) times=$(identity.time_values)")
	end
	if !isnothing(uq.backsolve_transform)
		bt = uq.backsolve_transform
		println(io, "Backsolve transform: $(bt.status)   t0: $(round(bt.t0; sigdigits = 6))   amplification: $(round(bt.amplification; sigdigits = 4))")
	end
	if !isnothing(uq.practical_identifiability_index)
		ia = uq.practical_identifiability_index
		println(io, "I_A: $(round(ia.i_a; sigdigits = 3))   status: $(ia.status)")
	end
	println(io, rpad("Quantity", 16), " | ", rpad("Role", 14), " | ", rpad("Estimate", 12), " | ", rpad("Std. Dev.", 12), " | 95% CI half-width")
	println(io, "-"^72)
	for (i, name) in enumerate(uq.param_labels)
		σ = uq.param_std[i]
		role = get(uq.param_roles, name, :unknown)
		center = i <= length(uq.estimate_values) ? uq.estimate_values[i] : NaN
		@printf(io, "%-16s | %-14s | %12.6g | %12.6g | %12.6g\n", name, string(role), center, σ, UQ_CI_Z * σ)
	end
	if !isnothing(uq.local_coordinate_report)
		local_snapshot = uq.local_coordinate_report
		println(io, "-"^72)
		println(io, "Local IFT audit at t_eval=$(round(local_snapshot.t_eval; sigdigits = 6))")
		println(io, rpad("Local coord", 16), " | ", rpad("Role", 14), " | ", rpad("Std. Dev.", 12), " | coordinate value")
		for (i, name) in enumerate(local_snapshot.param_labels)
			σ = local_snapshot.param_std[i]
			role = get(local_snapshot.param_roles, name, :unknown)
			value = i <= length(local_snapshot.coordinate_values) ? local_snapshot.coordinate_values[i] : NaN
			@printf(io, "%-16s | %-14s | %12.6g | %12.6g\n", name, string(role), σ, value)
		end
	end
	for w in uq.warnings
		println(io, "  ! ", w)
	end
	println(io, "-"^72)
end
