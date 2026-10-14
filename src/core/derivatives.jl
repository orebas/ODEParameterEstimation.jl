# `abstract type AbstractInterpolator` now lives in types/core_types.jl (loaded
# before this file) — it is annotated by parameter_estimation.jl, which is included
# earlier than this file, so the definition must precede the whole core/ cluster.

const TAYLORDIFF_MAX_DERIVATIVE_ORDER = 20

"""
	UnsupportedDerivativeOrderError

Thrown when a model needs a higher derivative of the measured data than the
package computes (order $TAYLORDIFF_MAX_DERIVATIVE_ORDER). This happens when a few measured quantities have to
account for many states and parameters.
"""
struct UnsupportedDerivativeOrderError <: Exception
	requested_order::Int
	supported_order::Int
	backend::Symbol
	context::Union{Nothing, String}
end

function Base.showerror(io::IO, err::UnsupportedDerivativeOrderError)
	print(io, "This model needs derivative $(err.requested_order) of the measured data, and orders up to $(err.supported_order) are supported. ")
	print(io, "This happens when a few measured quantities have to account for many states and parameters.")
	!isnothing(err.context) && print(io, " It came up $(err.context).")
end


######### TODO(orebas)  REFACTOR into bary_derivs or something similiar
#to use the below, you can just pass vectors of xvalues and yvalues like so:
#F = aaad(xdata, ydata)
#and then F will be a callable, i.e. F(0.5) should work.  Let's please restrict to real xdata, and if so it should be sorted.  F is only defined in the range of the xvalues.

#To construct a derivative, you can do
#derivf(z) = ForwardDiff.derivative(F, z)
#I hope and suspect that other AD frameworks should work as well.

# The barycentric and rational interpolation helpers from here on were adapted
# from ParameterEstimation.jl (GPL-3.0), revision
# `99f4bd59d9c6cedcfeb644672168aaa3f0088984`,
# `src/rational_interpolation/bary_derivs.jl`. See the repository's
# `docs/internal/registry_preparation.md` attribution inventory.
function baryEval(z, f::Vector{T}, x::Vector{T}, w::Vector{T}, tol = 1e-13) where {T}
	@assert(length(f) == length(x) == length(w))
	num = zero(T)
	den = zero(T)
	breakflag = false
	breakindex = -1
	for j in eachindex(f)
		t = w[j] / (z - x[j])
		num += t * f[j]
		den += t
		# Near-node special case: fire only essentially AT a support point — the
		# barycentric form is forward-stable arbitrarily close to nodes; only an
		# exact hit divides by zero. The old `(z-x)^2 < sqrt(tol)` test was a
		# |z-x| < ~5.6e-4 window, wide enough to contain a NEIGHBORING node on
		# dense grids, and its last-match overwrite then corrupted the
		# special-case formula. Record the first match; a second match within
		# ~1e-13 would mean duplicate support points.
		if (breakindex == -1) && (abs(z - x[j]) < tol)
			breakflag = true
			breakindex = j
		end
	end
	fz = num / den
	if (breakflag)
		num = zero(T)
		den = zero(T)
		for j in eachindex(f)
			if (j != breakindex)
				t = w[j] / (z - x[j])
				num += t * f[j]
				den += t
			end
		end
		m = z - x[breakindex]
		fz = (w[breakindex] * f[breakindex] + m * num) / (w[breakindex] + m * den)
	end
	return fz
end

"""
	AAADapprox{T} <: AbstractInterpolator

Interpolator using AAA algorithm from BaryRational package.

# Fields
- `internalAAA::T`: Internal AAA representation from BaryRational
"""
struct AAADapprox{T} <: AbstractInterpolator
	internalAAA::T
end

"""
	FHDapprox{T} <: AbstractInterpolator

Interpolator using Floater-Hormann algorithm from BaryRational package.

# Fields
- `internalFHD::T`: Internal Floater-Hormann representation from BaryRational
"""
struct FHDapprox{T} <: AbstractInterpolator
	internalFHD::T
end

"""
	GPRapprox <: AbstractInterpolator

Interpolator using Gaussian Process Regression.

# Fields
- `gp_function::Function`: Callable function for evaluating the GPR prediction
"""
struct GPRapprox <: AbstractInterpolator
	gp_function::Function
end

"""
	AGPInterpolator <: AbstractInterpolator

GP interpolator using AbstractGPs.jl with uncertainty quantification support.

# Fields
- `mean_function::Function`: Returns mean prediction (denormalized)
- `std_function::Function`: Returns standard deviation (denormalized)
- `posterior::Any`: AbstractGPs posterior GP object
- `x_min::Float64`: Minimum x value (for normalization)
- `x_range::Float64`: Range of x values (for normalization)
- `y_mean::Float64`: Mean of y values (for denormalization)
- `y_std::Float64`: Std of y values (for denormalization)

# Usage
When called as `interp(x)`, returns mean prediction only (drop-in compatible).
For uncertainty, use `mean_and_var(interp, x)` to get (mean, variance) tuple.
"""
struct AGPInterpolator <: AbstractInterpolator
	mean_function::Function
	std_function::Function
	posterior::Any
	x_min::Float64
	x_range::Float64
	y_mean::Float64
	y_std::Float64
end

# Define call methods for each interpolator type
(y::FHDapprox)(z) = baryEval(z, y.internalFHD.f, y.internalFHD.x, y.internalFHD.w)
(y::AAADapprox)(z) = baryEval(z, y.internalAAA.f, y.internalAAA.x, y.internalAAA.w)
(y::GPRapprox)(z) = y.gp_function(z)
(y::AGPInterpolator)(z) = y.mean_function(z)

# AbstractInterpolator is defined in types/core_types.jl (loaded before this file)


# Keep nth_deriv_at as an alias to nth_deriv for backward compatibility but with a warning
function nth_deriv_at(f, n::Int, t::Real)::Real
	@warn "nth_deriv_at is deprecated due to inefficient recursive ForwardDiff. Using nth_deriv (TaylorDiff) instead." maxlog=1
	return nth_deriv(f isa AbstractInterpolator ? (x -> f(x)) : f, n, t)
end

nth_deriv_at(f::AbstractInterpolator, n::Int, t::Real)::Real = nth_deriv_at(x -> f(x), n, t)

"""
	nth_deriv(f::Function, n::Int, t::Real) -> Real

Computes the nth derivative of function f at point t using TaylorDiff.

# Arguments
- `f::Function`: Function to differentiate
- `n::Int`: Derivative order
- `t::Real`: Point at which to compute the derivative

# Returns
- Value of the nth derivative of f at t
"""
function nth_deriv(f::Function, n::Int, t::Real)::Real
	if n == 0
		return f(t)
	end

	if n > TAYLORDIFF_MAX_DERIVATIVE_ORDER
		throw(UnsupportedDerivativeOrderError(n, TAYLORDIFF_MAX_DERIVATIVE_ORDER, :taylordiff, nothing))
	end

	# Use TaylorDiff - no silent fallbacks, fail loudly if it doesn't work
	return TaylorDiff.derivative(f, t, Val(n))
end

"""
	_estimation_derivative(interp, n, t) -> Real

Evaluate an interpolant derivative on estimation and diagnostic call paths.
The generic route preserves the historical TaylorDiff-through-callable
behavior. Interpolators with an exact derivative operator specialize this
helper so production data values and their retained influence map use the same
calculus.
"""
function _estimation_derivative(interp, n::Int, t::Real)::Real
	return nth_deriv(x -> interp(x), n, t)
end


"""
	aaad_old_reliable(xs::AbstractArray{T}, ys::AbstractArray{T}) -> AAADapprox

Creates an interpolation function using AAA algorithm from BaryRational.

# Arguments
- `xs::AbstractArray{T}`: X coordinates
- `ys::AbstractArray{T}`: Y coordinates (function values)

# Returns
- AAADapprox interpolator object that can be called as a function
"""
function aaad_old_reliable(xs::AbstractArray{T}, ys::AbstractArray{T})::AAADapprox where {T}
	@assert length(xs) == length(ys) "Input arrays must have same length"
	internalApprox = BaryRational.aaa(xs, ys, verbose = false)
	return AAADapprox(internalApprox)
end

"""
	aaad(xs::AbstractArray{T}, ys::AbstractArray{T}, force_gpr::Bool=false) -> AAADapprox

Standard interpolation function using AAA algorithm.
The force_gpr parameter is kept for backward compatibility but is not used.

# Arguments
- `xs::AbstractArray{T}`: X coordinates
- `ys::AbstractArray{T}`: Y coordinates (function values)
- `force_gpr::Bool`: Unused parameter (for backward compatibility)

# Returns
- AAADapprox interpolator object that can be called as a function
"""
function aaad(xs::AbstractArray{T}, ys::AbstractArray{T}, force_gpr::Bool = false)::AAADapprox where {T}
	return aaad_old_reliable(xs, ys)
end


function fhd(xs::AbstractArray{T}, ys::AbstractArray{T}, N::Int) where {T}
	#@suppress begin
	@assert length(xs) == length(ys)
	internalApprox = BaryRational.FHInterp(xs, ys, order = N)
	return FHDapprox(internalApprox)
	#end
end

function fhdn(n)
	fh(xs, ys) = fhd(xs, ys, n)
	return fh
end


####################


"""
	aaad_gpr_pivot(xs::AbstractArray{T}, ys::AbstractArray{T}) -> GPRapprox

Creates an interpolation function using Gaussian Process Regression.
Normalizes input data, applies GPR with optimized hyperparameters,
and returns a callable interpolator.

# Arguments
- `xs::AbstractArray{T}`: X coordinates
- `ys::AbstractArray{T}`: Y coordinates (function values)

# Returns
- GPRapprox interpolator object that can be called as a function
"""
function aaad_gpr_pivot(xs::AbstractArray{T}, ys::AbstractArray{T})::GPRapprox where {T}
	@assert length(xs) == length(ys) "Input arrays must have same length"

	# 1. Normalize y values
	y_mean = mean(ys)
	y_std = std(ys)
	ys_normalized = (ys .- y_mean) ./ max(y_std, 1e-8)  # Avoid division by very small numbers

	# Configure initial GP hyperparameters
	initial_lengthscale = log(std(xs) / 8)
	initial_variance = 0.0
	initial_noise = -2.0

	# This private backend preserves the original GP.jl fitting policy. In
	# particular initial_noise is log standard deviation, not log variance.
	gp = GPBackend.fit_se(vec(xs), vec(ys_normalized); log_lengthscale=initial_lengthscale,
		log_signal_std=initial_variance, log_noise_std=initial_noise)
	denormalized_gpr(x) = y_std * GPBackend.predict_mean(gp, x) + y_mean

	return GPRapprox(denormalized_gpr)
end

"""
	_cholesky_adaptive(K; base_jitter=1e-14, relative_jitter=false, max_steps=12)

Attempt Cholesky factorization with adaptive jitter escalation.
Tries `cholesky(Symmetric(K))` first; on failure, adds escalating diagonal
jitter starting from `base_jitter` (×10 each step) until success. With
`relative_jitter=true`, `base_jitter` is interpreted as a fraction of the
largest absolute diagonal entry. The historical default remains absolute so
ordinary non-UQ interpolators are unchanged; AGPUQ explicitly enables the
scale-relative policy.

Returns `(C, jitter_used)` where `C` is the Cholesky factor and
`jitter_used` is the actual jitter added (0.0 if none needed).
"""
function _cholesky_adaptive(
	K::AbstractMatrix;
	base_jitter::Real = 1e-14,
	relative_jitter::Bool = false,
	max_steps::Int = 12,
)
	n = size(K, 1)
	# Try without jitter first
	try
		return cholesky(Symmetric(K)), 0.0
	catch
	end
	# Escalate jitter
	diagonal_scale = maximum(abs, diag(K); init = 0.0)
	diagonal_scale = max(Float64(diagonal_scale), floatmin(Float64))
	jitter = Float64(base_jitter) * (relative_jitter ? diagonal_scale : 1.0)
	jitter > 0 || throw(ArgumentError("base_jitter must be positive"))
	K_work = copy(K)
	for step in 1:max_steps
		K_work .= K
		for i in 1:n
			K_work[i, i] += jitter
		end
		try
			return cholesky(Symmetric(K_work)), jitter
		catch
			jitter *= 10.0
		end
	end
	error("_cholesky_adaptive: Cholesky failed even with jitter=$jitter")
end

"""
	_se_covariance_matrix(D_sq, lengthscale, signal_variance)

Construct the squared-exponential covariance matrix from precomputed squared
distances. This is the single arithmetic recipe used by SE
marginal-likelihood optimization and the explicit AGPUQ final factorization;
keeping that estimator route shared prevents tiny rounding differences from
becoming a hidden regularizer near the PSD boundary. The ordinary non-UQ GP
paths retain their historical KernelFunctions assembly. The implementation is
`ForwardDiff` compatible.
"""
function _se_covariance_matrix(
	D_sq::AbstractMatrix,
	lengthscale::Real,
	signal_variance::Real,
)
	inv_2l2 = inv(2 * lengthscale * lengthscale)
	return signal_variance .* exp.((-inv_2l2) .* D_sq)
end

"""Add observation-noise variance to a covariance diagonal without changing its recipe."""
function _add_diagonal_variance(K::AbstractMatrix, variance::Real)
	K_noisy = copy(K)
	for i in axes(K_noisy, 1)
		K_noisy[i, i] += variance
	end
	return K_noisy
end

function _cholesky_relative_residual(C::Cholesky, K::AbstractMatrix, jitter::Real)
	regularized = _add_diagonal_variance(K, jitter)
	reconstructed = Matrix(C.L * C.L')
	denominator = max(norm(regularized, Inf), floatmin(Float64))
	return norm(reconstructed - regularized, Inf) / denominator
end

# --- Softplus helpers (overflow-safe, module-level for sharing) ---
_softplus(x::Real) = x > 34.0 ? x : log(1.0 + exp(x))
_inv_softplus(y::Real) = y > 34.0 ? y : log(exp(y) - 1.0)

"""
	_optimize_se_hyperparams(xs_raw, ys_norm, D_sq;
		init_noise_override=nothing, explicit_se_recipe=false)

Shared SE-kernel hyperparameter optimization used by both `agp_gpr_robust` and `agp_gpr_uq`.
Uses softplus reparameterization, Fminbox-bounded LBFGS, and ForwardDiff exact gradients.
The default preserves the historical arithmetic used by the production
interpolator/noise-filter path. `agp_gpr_uq` opts into `explicit_se_recipe=true`
so its likelihood and retained factor use exactly the same matrix recipe.

# Arguments
- `xs_raw::Vector{Float64}`: Raw (unnormalized) x coordinates
- `ys_norm::Vector{Float64}`: Standardized y values (zero mean, unit variance)
- `D_sq::Matrix{Float64}`: Precomputed pairwise squared distances
- `init_noise_override::Union{Nothing,Float64}`: Override initial noise (default: roughness-adaptive)
- `explicit_se_recipe::Bool`: Use the explicit covariance helper retained by AGPUQ (default: false)

# Returns
- NamedTuple `(l, sigma2, noise, converged)` with optimized hyperparameters
"""
function _optimize_se_hyperparams(
	xs_raw::Vector{Float64},
	ys_norm::Vector{Float64},
	D_sq::Matrix{Float64};
	init_noise_override::Union{Nothing, Float64} = nothing,
	explicit_se_recipe::Bool = false,
)
	n = length(xs_raw)

	# --- Smoothness detection via second finite differences ---
	if n >= 3
		d2 = [ys_norm[i+2] - 2.0 * ys_norm[i+1] + ys_norm[i] for i in 1:(n-2)]
		roughness = mean(abs.(d2))
	else
		roughness = 1.0
	end

	if !isnothing(init_noise_override)
		init_noise = init_noise_override
	elseif roughness < 0.1
		init_noise = 1e-6    # very smooth data
	elseif roughness < 1.0
		init_noise = 1e-4    # moderate
	else
		init_noise = 0.01    # noisy
	end

	init_lengthscale = std(xs_raw) / 8.0
	init_signal_var = 1.0

	θ0 = [_inv_softplus(init_lengthscale), _inv_softplus(init_signal_var), _inv_softplus(init_noise)]
	θ_lower = [_inv_softplus(1e-3), _inv_softplus(1e-14), _inv_softplus(1e-14)]
	θ_upper = [_inv_softplus(150.0), _inv_softplus(150.0), _inv_softplus(10.0)]

	# Allocating NLL — ForwardDiff-compatible (no in-place mutation on captured matrix)
	function neg_logpdf_se(θ)
		l = _softplus(θ[1]); σ² = _softplus(θ[2]); σₙ² = _softplus(θ[3])
		K = if explicit_se_recipe
			_add_diagonal_variance(_se_covariance_matrix(D_sq, l, σ²), σₙ²)
		else
			# Keep this byte-for-byte arithmetic shape: the production noise gate
			# is sensitive to optimizer movement around its 1e-5 threshold.
			inv_2l² = 1.0 / (2.0 * l * l)
			K_historical = σ² .* exp.((-inv_2l²) .* D_sq)
			for i in 1:n
				K_historical[i, i] += σₙ²
			end
			K_historical
		end
		try
			C = cholesky(Symmetric(K))
			α = C \ ys_norm
			return 0.5 * (dot(ys_norm, α) + 2.0 * sum(log.(diag(C.U))) + n * log(2π))
		catch
			return 1e8 + sum(abs2, θ)  # smooth penalty for ForwardDiff
		end
	end

	l_opt = init_lengthscale
	σ²_opt = init_signal_var
	σₙ²_opt = init_noise
	converged = false

	try
		od = Optim.OnceDifferentiable(neg_logpdf_se, θ0; autodiff = AutoForwardDiff())
		result = Optim.optimize(
			od, θ_lower, θ_upper, θ0,
			Fminbox(LBFGS(linesearch = LineSearches.BackTracking())),
			Optim.Options(iterations = 1000),
		)
		θ_opt = Optim.minimizer(result)
		l_opt = _softplus(θ_opt[1])
		σ²_opt = _softplus(θ_opt[2])
		σₙ²_opt = _softplus(θ_opt[3])
		converged = Optim.converged(result)
	catch e
		@warn "_optimize_se_hyperparams: optimization failed, using adaptive defaults" exception = e
	end

	return (l = l_opt, sigma2 = σ²_opt, noise = σₙ²_opt, converged = converged)
end

"""
	estimate_relative_noise(pep) -> Float64

Data-driven relative-noise estimate using SE-kernel GP hyperparameter
optimization on the first observable's data. Returns σₙ in normalized-y
units (so it's scale-invariant — `sqrt(σₙ²_opt)` where the GP fit was
on z-scored data). Returns 0.0 on failure (interpreted as "couldn't
estimate; treat as zero noise"). Used by the noise-based interpolator
filter in `maybe_filter_interpolators_by_noise`.

This is one extra SE-kernel GP fit (~50ms) per estimation run. Skipped
entirely by callers when the interpolator list contains no AAA-family
methods (since only those are noise-gated).
"""
function estimate_relative_noise(pep)
	isnothing(pep.data_sample) && return 0.0
	t_vec = try
		Float64.(pep.data_sample["t"])
	catch
		return 0.0
	end
	# Pick the first non-"t" key in data_sample as the noise-probe observable
	y_vec = nothing
	for (k, v) in pep.data_sample
		k == "t" && continue
		y_vec = Float64.(v)
		break
	end
	isnothing(y_vec) && return 0.0
	(length(y_vec) != length(t_vec) || length(t_vec) < 4) && return 0.0
	try
		y_std = Statistics.std(y_vec)
		y_std < 1e-12 && return 0.0
		ys_norm = (y_vec .- Statistics.mean(y_vec)) ./ y_std
		D_sq = [(t_vec[i] - t_vec[j])^2 for i in eachindex(t_vec), j in eachindex(t_vec)]
		result = _optimize_se_hyperparams(t_vec, ys_norm, D_sq)
		return sqrt(max(result.noise, 0.0))
	catch e
		@warn "[INTERP-NOISE-ESTIMATE] Falling back to σ̂=0 due to error: $e"
		return 0.0
	end
end


"""
	mean_and_var(interp::AGPInterpolator, x::Real) -> (Float64, Float64)

Returns both mean prediction and variance at point x.

# Arguments
- `interp::AGPInterpolator`: The GP interpolator
- `x::Real`: Point at which to evaluate

# Returns
- Tuple of (mean, variance) - both denormalized to original data scale
"""
function mean_and_var(interp::AGPInterpolator, x::Real)
	mean_val = interp.mean_function(x)
	std_val = interp.std_function(x)
	return (mean_val, std_val^2)
end

"""
	agp_gpr_robust(xs, ys; kernel_type=:se) -> AGPInterpolator

Robust GP interpolator that fixes failures on smooth/noiseless data.

Uses softplus reparameterization (hyperparams are always strictly positive),
bounded optimization via Fminbox(LBFGS), ForwardDiff exact gradients, and
direct Cholesky log-likelihood (avoids ScaledKernel in the optimization loop).
Automatically detects smooth data and initializes noise variance low.

For SE kernel, delegates to `_optimize_se_hyperparams` (shared with `agp_gpr_uq`).

Returns an `AGPInterpolator` compatible with TaylorDiff for derivative computation.

# Arguments
- `xs::AbstractArray{T}`: Input x-values
- `ys::AbstractArray{T}`: Output y-values
- `kernel_type::Symbol`: Kernel type, `:se` (default) or `:matern52`

# Returns
- `AGPInterpolator`: GP interpolator with mean and std prediction functions
"""
function agp_gpr_robust(xs::AbstractArray{T}, ys::AbstractArray{T};
                        kernel_type::Symbol = :se)::AGPInterpolator where {T}
	@assert length(xs) == length(ys) "Input arrays must have same length"
	@assert length(xs) >= 3 "Need at least 3 points for GP interpolation"

	# Handle constant data edge case
	y_std_raw = std(ys)
	if y_std_raw < 1e-10
		constant_val = mean(ys)
		return AGPInterpolator(
			x -> constant_val,
			x -> 0.0,
			nothing,
			0.0, 1.0, constant_val, 1.0
		)
	end

	# NO X normalization (required for TaylorDiff)
	xs_raw = collect(Float64, xs)
	n = length(xs_raw)

	# Standardize Y only (zero mean, unit variance)
	y_mean = mean(ys)
	y_std = max(y_std_raw, 1e-8)
	ys_norm = (collect(Float64, ys) .- y_mean) ./ y_std

	# Precompute squared distance matrix (done once, reused across all iterations)
	D² = Matrix{Float64}(undef, n, n)
	for j in 1:n
		for i in 1:n
			D²[i, j] = abs2(xs_raw[i] - xs_raw[j])
		end
	end

	# Identity matrix (pre-allocate for post-optimization)
	I_n = Matrix{Float64}(I, n, n)

	# Initial hyperparameters (shared across all kernels)
	init_lengthscale = std(xs_raw) / 8.0
	init_signal_var = 1.0
	init_alpha = 2.0  # RQ shape parameter (small = more multi-scale)

	# ---- SE kernel: delegate to shared helper ----
	if kernel_type == :se
		hp = _optimize_se_hyperparams(xs_raw, ys_norm, D²)
		l_opt = hp.l; σ²_opt = hp.sigma2; σₙ²_opt = hp.noise

		base_kernel = SqExponentialKernel()
		scaled_kernel_opt = base_kernel ∘ ScaleTransform(1.0 / l_opt)
		kernel_opt = σ²_opt * scaled_kernel_opt

	# ---- Non-SE kernels: inline optimization with ForwardDiff ----
	else
		# --- Smoothness detection via second finite differences ---
		if n >= 3
			d2 = [ys_norm[i+2] - 2.0 * ys_norm[i+1] + ys_norm[i] for i in 1:(n-2)]
			roughness = mean(abs.(d2))
		else
			roughness = 1.0
		end

		if roughness < 0.1
			init_noise = 1e-6
		elseif roughness < 1.0
			init_noise = 1e-4
		else
			init_noise = 0.01
		end

		# Constants for Matern52
		is_matern52 = (kernel_type == :matern52)
		const_sqrt5 = sqrt(5.0)

		# --- Optimization in softplus-reparameterized space ---
		if kernel_type == :matern52
			θ0 = [_inv_softplus(init_lengthscale), _inv_softplus(init_signal_var), _inv_softplus(init_noise)]
			θ_lower = [_inv_softplus(1e-3), _inv_softplus(1e-14), _inv_softplus(1e-14)]
			θ_upper = [_inv_softplus(150.0), _inv_softplus(150.0), _inv_softplus(10.0)]
		elseif kernel_type == :rq
			θ0 = [_inv_softplus(init_lengthscale), _inv_softplus(init_signal_var), _inv_softplus(init_noise), _inv_softplus(init_alpha)]
			θ_lower = [_inv_softplus(1e-3), _inv_softplus(1e-14), _inv_softplus(1e-14), _inv_softplus(0.1)]
			θ_upper = [_inv_softplus(150.0), _inv_softplus(150.0), _inv_softplus(10.0), _inv_softplus(100.0)]
		elseif kernel_type == :se_plus_rq
			θ0 = [_inv_softplus(init_lengthscale), _inv_softplus(init_signal_var / 2), _inv_softplus(init_noise),
			       _inv_softplus(init_lengthscale), _inv_softplus(init_signal_var / 2), _inv_softplus(init_alpha)]
			θ_lower = [_inv_softplus(1e-3), _inv_softplus(1e-14), _inv_softplus(1e-14),
			            _inv_softplus(1e-3), _inv_softplus(1e-14), _inv_softplus(0.1)]
			θ_upper = [_inv_softplus(150.0), _inv_softplus(150.0), _inv_softplus(10.0),
			            _inv_softplus(150.0), _inv_softplus(150.0), _inv_softplus(100.0)]
		elseif kernel_type == :se_times_rq
			θ0 = [_inv_softplus(init_lengthscale), _inv_softplus(init_signal_var), _inv_softplus(init_noise),
			       _inv_softplus(init_lengthscale), _inv_softplus(init_alpha)]
			θ_lower = [_inv_softplus(1e-3), _inv_softplus(1e-14), _inv_softplus(1e-14),
			            _inv_softplus(1e-3), _inv_softplus(0.1)]
			θ_upper = [_inv_softplus(150.0), _inv_softplus(150.0), _inv_softplus(10.0),
			            _inv_softplus(150.0), _inv_softplus(100.0)]
		else
			error("agp_gpr_robust: unknown kernel_type=$kernel_type. Supported: :se, :matern52, :rq, :se_plus_rq, :se_times_rq")
		end

		# Allocating NLL — ForwardDiff-compatible (no in-place mutation on captured matrix)
		function neg_logpdf_robust(θ)
			if kernel_type == :matern52
				l = _softplus(θ[1]); σ² = _softplus(θ[2]); σₙ² = _softplus(θ[3])
				inv_l = 1.0 / l
				R = sqrt.(D²) .* inv_l
				K = σ² .* (1.0 .+ const_sqrt5 .* R .+ 5.0 .* R .^ 2 ./ 3.0) .* exp.((-const_sqrt5) .* R)
			elseif kernel_type == :rq
				l = _softplus(θ[1]); σ² = _softplus(θ[2]); σₙ² = _softplus(θ[3]); α = _softplus(θ[4])
				inv_2αl² = 1.0 / (2.0 * α * l * l)
				K = σ² .* (1.0 .+ D² .* inv_2αl²) .^ (-α)
			elseif kernel_type == :se_plus_rq
				l₁ = _softplus(θ[1]); σ²₁ = _softplus(θ[2]); σₙ² = _softplus(θ[3])
				l₂ = _softplus(θ[4]); σ²₂ = _softplus(θ[5]); α = _softplus(θ[6])
				inv_2l₁² = 1.0 / (2.0 * l₁ * l₁)
				inv_2αl₂² = 1.0 / (2.0 * α * l₂ * l₂)
				K = σ²₁ .* exp.((-inv_2l₁²) .* D²) .+ σ²₂ .* (1.0 .+ D² .* inv_2αl₂²) .^ (-α)
			elseif kernel_type == :se_times_rq
				l₁ = _softplus(θ[1]); σ² = _softplus(θ[2]); σₙ² = _softplus(θ[3])
				l₂ = _softplus(θ[4]); α = _softplus(θ[5])
				inv_2l₁² = 1.0 / (2.0 * l₁ * l₁)
				inv_2αl₂² = 1.0 / (2.0 * α * l₂ * l₂)
				K = σ² .* exp.((-inv_2l₁²) .* D²) .* (1.0 .+ D² .* inv_2αl₂²) .^ (-α)
			end

			for i in 1:n
				K[i, i] += σₙ²
			end

			try
				C = cholesky(Symmetric(K))
				alpha_local = C \ ys_norm
				data_fit = dot(ys_norm, alpha_local)
				log_det = 2.0 * sum(log.(diag(C.U)))
				return 0.5 * (data_fit + log_det + n * log(2π))
			catch
				return 1e8 + sum(abs2, θ)  # smooth penalty for ForwardDiff
			end
		end

		# Defaults in case optimization fails
		l_opt = init_lengthscale
		σ²_opt = init_signal_var
		σₙ²_opt = init_noise
		α_opt = init_alpha
		l2_opt = init_lengthscale
		σ²_1_opt = init_signal_var / 2
		σ²_2_opt = init_signal_var / 2

		try
			od = Optim.OnceDifferentiable(neg_logpdf_robust, θ0; autodiff = AutoForwardDiff())
			result = Optim.optimize(
				od, θ_lower, θ_upper, θ0,
				Fminbox(LBFGS(linesearch = LineSearches.BackTracking())),
				Optim.Options(iterations = 1000),
			)
			θ_opt = Optim.minimizer(result)
			if kernel_type == :matern52
				l_opt = _softplus(θ_opt[1]); σ²_opt = _softplus(θ_opt[2]); σₙ²_opt = _softplus(θ_opt[3])
			elseif kernel_type == :rq
				l_opt = _softplus(θ_opt[1]); σ²_opt = _softplus(θ_opt[2]); σₙ²_opt = _softplus(θ_opt[3])
				α_opt = _softplus(θ_opt[4])
			elseif kernel_type == :se_plus_rq
				l_opt = _softplus(θ_opt[1]); σ²_1_opt = _softplus(θ_opt[2]); σₙ²_opt = _softplus(θ_opt[3])
				l2_opt = _softplus(θ_opt[4]); σ²_2_opt = _softplus(θ_opt[5]); α_opt = _softplus(θ_opt[6])
			elseif kernel_type == :se_times_rq
				l_opt = _softplus(θ_opt[1]); σ²_opt = _softplus(θ_opt[2]); σₙ²_opt = _softplus(θ_opt[3])
				l2_opt = _softplus(θ_opt[4]); α_opt = _softplus(θ_opt[5])
			end
		catch e
			@warn "agp_gpr_robust: hyperparameter optimization failed, using adaptive defaults" exception = e
		end

		# Build AbstractGPs-compatible kernel for posterior/std_pred
		if kernel_type == :matern52
			base_kernel = Matern52Kernel()
			scaled_kernel_opt = base_kernel ∘ ScaleTransform(1.0 / l_opt)
			kernel_opt = σ²_opt * scaled_kernel_opt
		elseif kernel_type == :rq
			rq_scaled = RationalQuadraticKernel(alpha = α_opt) ∘ ScaleTransform(1.0 / l_opt)
			kernel_opt = σ²_opt * rq_scaled
		elseif kernel_type == :se_plus_rq
			se_part = σ²_1_opt * (SqExponentialKernel() ∘ ScaleTransform(1.0 / l_opt))
			rq_part = σ²_2_opt * (RationalQuadraticKernel(alpha = α_opt) ∘ ScaleTransform(1.0 / l2_opt))
			kernel_opt = se_part + rq_part
		elseif kernel_type == :se_times_rq
			se_raw = SqExponentialKernel() ∘ ScaleTransform(1.0 / l_opt)
			rq_raw = RationalQuadraticKernel(alpha = α_opt) ∘ ScaleTransform(1.0 / l2_opt)
			kernel_opt = σ²_opt * (se_raw * rq_raw)
		end
	end

	f_opt = AbstractGPs.GP(kernel_opt)

	# Pre-compute alpha for TaylorDiff-compatible mean predictions
	# Use adaptive Cholesky: try K + σₙ²*I first, escalate jitter from 1e-14 if needed
	K_train = kernelmatrix(kernel_opt, xs_raw)
	K_noisy = K_train + σₙ²_opt * I_n
	C_opt, _jitter_used = _cholesky_adaptive(K_noisy)
	alpha = C_opt \ collect(ys_norm)

	# Build posterior for variance predictions (needs effective noise including any jitter)
	effective_noise = σₙ²_opt + _jitter_used
	f_posterior = nothing
	try
		f_posterior = AbstractGPs.posterior(f_opt(xs_raw, effective_noise), ys_norm)
	catch e
		@warn "agp_gpr_robust: posterior failed" σₙ²_opt _jitter_used exception = e
		# Fallback: try larger jitter for posterior only
		for extra_jitter in [1e-6, 1e-4, 1e-2]
			try
				f_posterior = AbstractGPs.posterior(f_opt(xs_raw, σₙ²_opt + extra_jitter), ys_norm)
				break
			catch
			end
		end
		if isnothing(f_posterior)
			rethrow(e)
		end
	end

	# Prediction using cached alpha - TaylorDiff compatible (inline arithmetic, no KF dispatch)
	function mean_pred(x::Real)
		if kernel_type == :se
			k_star = [σ²_opt * exp(-(x - xi)^2 / (2.0 * l_opt^2)) for xi in xs_raw]
		elseif kernel_type == :matern52
			k_star = [σ²_opt * scaled_kernel_opt(x, xi) for xi in xs_raw]
		elseif kernel_type == :rq
			k_star = [σ²_opt * (1.0 + (x - xi)^2 / (2.0 * α_opt * l_opt^2))^(-α_opt) for xi in xs_raw]
		elseif kernel_type == :se_plus_rq
			k_star = [σ²_1_opt * exp(-(x - xi)^2 / (2.0 * l_opt^2)) +
			           σ²_2_opt * (1.0 + (x - xi)^2 / (2.0 * α_opt * l2_opt^2))^(-α_opt) for xi in xs_raw]
		elseif kernel_type == :se_times_rq
			k_star = [σ²_opt * exp(-(x - xi)^2 / (2.0 * l_opt^2)) *
			           (1.0 + (x - xi)^2 / (2.0 * α_opt * l2_opt^2))^(-α_opt) for xi in xs_raw]
		end
		return y_std * dot(k_star, alpha) + y_mean
	end

	function std_pred(x::Real)
		pred_var = AbstractGPs.var(f_posterior([x]))[1]
		return y_std * sqrt(max(pred_var, 0.0))
	end

	return AGPInterpolator(mean_pred, std_pred, f_posterior, 0.0, 1.0, y_mean, y_std)
end

# ============================================================================
# S3 Refinement: GP → AAA → MLE composite interpolator
# ============================================================================

"""
	_BaryResult

Lightweight struct matching the BaryRational.AAAapprox field interface (x, w, f).
Used to wrap MLE-refined barycentric parameters into an AAADapprox.
"""
struct _BaryResult
	x::Vector{Float64}  # support points
	w::Vector{Float64}  # barycentric weights
	f::Vector{Float64}  # function values at support points
end

"""
	_mle_refine_bary(z, w_init, f_init, t_data, y_data; maxiter, g_tol) -> (z, w_opt, f_opt)

MLE refinement of barycentric weights and function values against raw noisy data.
Uses LBFGS with a hard gauge constraint (w₁ = 1) to avoid scale ambiguity.
"""
function _mle_refine_bary(z::Vector{Float64}, w_init::Vector{Float64}, f_init::Vector{Float64},
                          t_data::Vector{Float64}, y_data::Vector{Float64};
                          maxiter::Int = 20000, g_tol::Float64 = 1e-15)
	m = length(z)
	m < 2 && return z, copy(w_init), copy(f_init)

	# Hard gauge: fix w₁ = 1
	w_scaled = w_init ./ w_init[1]
	θ0 = vcat(w_scaled[2:end], copy(f_init))

	fg! = function (F, G, θ)
		w = vcat(one(eltype(θ)), θ[1:m-1])
		fv = θ[m:2m-1]
		loss = zero(eltype(θ))
		G !== nothing && fill!(G, 0.0)
		for j in eachindex(t_data)
			tj = t_data[j]
			exact = 0
			for i in 1:m
				tj == z[i] && (exact = i; break)
			end
			if exact > 0
				res = y_data[j] - fv[exact]
				loss += res^2
				G !== nothing && (G[m-1+exact] -= 2.0 * res)
			else
				num, den = zero(eltype(θ)), zero(eltype(θ))
				for i in 1:m
					a = w[i] / (tj - z[i])
					num += a * fv[i]; den += a
				end
				rv = num / den
				res = y_data[j] - rv
				loss += res^2
				if G !== nothing
					id = 1.0 / den
					for i in 1:m
						ai = 1.0 / (tj - z[i])
						G[m-1+i] -= 2.0 * res * w[i] * ai * id
						i >= 2 && (G[i-1] -= 2.0 * res * ai * (fv[i] - rv) * id)
					end
				end
			end
		end

		return loss
	end

	od = Optim.OnceDifferentiable(NLSolversBase.only_fg!(fg!), θ0)
	opts = Optim.Options(iterations = maxiter, g_tol = g_tol,
	                     f_reltol = 1e-16, x_reltol = 1e-16, show_trace = false)
	result = Optim.optimize(od, θ0, Optim.LBFGS(), opts)
	θ_best = Optim.minimizer(result)
	# Warm restart
	result2 = Optim.optimize(od, θ_best, Optim.LBFGS(), opts)
	θ_final = Optim.minimum(result2) < Optim.minimum(result) ?
	          Optim.minimizer(result2) : θ_best

	w_opt = vcat(1.0, θ_final[1:m-1])
	f_opt = θ_final[m:2m-1]
	return z, w_opt, f_opt
end

"""
	s3_refine_gp(gp_interpolant::AbstractInterpolator, t_vector, y_raw;
	             aaa_tol, mmax, maxiter, g_tol) -> AAADapprox

S3 strategy: take a GP interpolant, evaluate it at data points to get denoised values,
run AAA on the denoised values (tight tolerance for more support points), then
MLE-refine the barycentric weights against the RAW noisy data.

Returns an AAADapprox whose call method uses baryEval (AD-friendly at support points).
"""
function s3_refine_gp(gp_interpolant::AbstractInterpolator,
                      t_vector::Vector{Float64}, y_raw::Vector{Float64};
                      aaa_tol::Float64 = 1e-14, mmax::Int = 200,
                      maxiter::Int = 20000, g_tol::Float64 = 1e-15)
	# Step 1: Evaluate GP at data points → denoised values
	y_gp = [gp_interpolant(t) for t in t_vector]

	# Step 2: AAA on denoised data with tight tolerance → many support points
	aaa_result = BaryRational.aaa(t_vector, y_gp; tol = aaa_tol, mmax = mmax)
	z = copy(aaa_result.x)
	w = copy(aaa_result.w)
	f = copy(aaa_result.f)

	# Step 3: MLE refine weights+values against raw noisy data
	z, w, f = _mle_refine_bary(z, w, f, t_vector, y_raw; maxiter = maxiter, g_tol = g_tol)

	# Step 4: Wrap in AAADapprox for baryEval-based evaluation
	refined = _BaryResult(z, w, f)
	return AAADapprox(refined)
end

"""
	s3_refine_gp_adaptive(gp_interpolant, t_vector, y_raw; k, mmax, maxiter, g_tol) -> AAADapprox

Adaptive-tolerance S3 strategy: estimate noise from GP residuals, then set
`aaa_tol = max(k * noise_std / data_scale, 1e-14)`.  Higher `k` → fewer support
points (more aggressive denoising).  `k=10` is the validated default.
"""
function s3_refine_gp_adaptive(gp_interpolant::AbstractInterpolator,
                                t_vector::Vector{Float64}, y_raw::Vector{Float64};
                                k::Float64 = 10.0, mmax::Int = 200,
                                maxiter::Int = 20000, g_tol::Float64 = 1e-15)
	y_gp = [gp_interpolant(t) for t in t_vector]
	noise_std = std(y_raw .- y_gp)
	data_scale = max(maximum(abs.(y_gp)), 1e-15)
	aaa_tol = max(k * noise_std / data_scale, 1e-14)
	return s3_refine_gp(gp_interpolant, t_vector, y_raw;
	                     aaa_tol = aaa_tol, mmax = mmax, maxiter = maxiter, g_tol = g_tol)
end

"""
	s3_refine_gp_bic(gp_interpolant, t_vector, y_raw; mmax, maxiter, g_tol) -> AAADapprox

BIC model-selection S3 strategy: sweep 12 AAA tolerances (1e-3 … 1e-14), pick
the one minimising `BIC = n_params * log(M) + M * log(SSR / M)` against the raw
data.  Parameter-free alternative to adaptive tolerance.
"""
function s3_refine_gp_bic(gp_interpolant::AbstractInterpolator,
                           t_vector::Vector{Float64}, y_raw::Vector{Float64};
                           mmax::Int = 200, maxiter::Int = 20000, g_tol::Float64 = 1e-15)
	y_gp = [gp_interpolant(t) for t in t_vector]
	M = length(t_vector)
	best_bic = Inf
	best_z, best_w, best_f = Float64[], Float64[], Float64[]

	for log_tol in -3:-1:-14
		tol = 10.0^log_tol
		try
			aaa_result = BaryRational.aaa(t_vector, y_gp; tol = tol, mmax = mmax)
			z, w, f = copy(aaa_result.x), copy(aaa_result.w), copy(aaa_result.f)
			predicted = [baryEval(t, f, z, w) for t in t_vector]
			SSR = sum(abs2.(y_raw .- predicted))
			n_params = 2 * length(z)
			bic = n_params * log(M) + M * log(SSR / M + 1e-100)
			if bic < best_bic
				best_bic = bic
				best_z, best_w, best_f = z, w, f
			end
		catch
			continue
		end
	end

	if isempty(best_z)
		return s3_refine_gp(gp_interpolant, t_vector, y_raw; mmax = mmax, maxiter = maxiter, g_tol = g_tol)
	end

	z, w, f = _mle_refine_bary(best_z, best_w, best_f, t_vector, y_raw; maxiter = maxiter, g_tol = g_tol)
	return AAADapprox(_BaryResult(z, w, f))
end

"""
	s2_aaa_mle_interpolator(xs, ys; aaa_tol, mmax, maxiter, g_tol) -> AAADapprox

S2 strategy: run AAA directly on raw (noisy) data to harvest support points,
then MLE-refine barycentric weights against the same data. No GP denoising step.

This is the GP-free counterpart to S3. At high noise, skipping GP avoids
over-smoothing; at low noise, S3 typically wins.
"""
function s2_aaa_mle_interpolator(xs::AbstractArray, ys::AbstractArray;
                                  aaa_tol::Float64 = 1e-10, mmax::Int = 200,
                                  maxiter::Int = 20000, g_tol::Float64 = 1e-15)
	t_vec = Vector{Float64}(xs)
	y_vec = Vector{Float64}(ys)

	# AAA directly on raw data
	aaa_result = BaryRational.aaa(t_vec, y_vec; tol = aaa_tol, mmax = mmax)
	z = copy(aaa_result.x)
	w = copy(aaa_result.w)
	f = copy(aaa_result.f)

	# MLE refine weights+values against the same raw data
	z, w, f = _mle_refine_bary(z, w, f, t_vec, y_vec; maxiter = maxiter, g_tol = g_tol)

	refined = _BaryResult(z, w, f)
	return AAADapprox(refined)
end

"""
	_s3_interpolator(xs, ys; kernel_type=:se) -> AAADapprox

Standalone S3 composite interpolator factory (DEPRECATED — use `_s3_adapt_interpolator`).
Fits a GP with the given kernel, then refines via AAA + MLE barycentric weights.
"""
function _s3_interpolator(xs::AbstractArray, ys::AbstractArray; kernel_type::Symbol = :se)
	Base.depwarn("`_s3_interpolator` is deprecated, use `_s3_adapt_interpolator` instead.", :_s3_interpolator)
	return _s3_adapt_interpolator(xs, ys; kernel_type = kernel_type)
end

"""
	_s3_adapt_interpolator(xs, ys; kernel_type=:se, k=10.0) -> AAADapprox

Adaptive-tolerance S3 factory. GP → AAA(adaptive tol) → MLE.
"""
function _s3_adapt_interpolator(xs::AbstractArray, ys::AbstractArray; kernel_type::Symbol = :se, k::Float64 = 10.0)
	t_vec = Vector{Float64}(xs)
	y_vec = Vector{Float64}(ys)
	gp = agp_gpr_robust(t_vec, y_vec; kernel_type = kernel_type)
	return s3_refine_gp_adaptive(gp, t_vec, y_vec; k = k)
end

"""
	_s3_bic_interpolator(xs, ys; kernel_type=:se) -> AAADapprox

BIC model-selection S3 factory. GP → AAA(BIC-selected tol) → MLE.
"""
function _s3_bic_interpolator(xs::AbstractArray, ys::AbstractArray; kernel_type::Symbol = :se)
	t_vec = Vector{Float64}(xs)
	y_vec = Vector{Float64}(ys)
	gp = agp_gpr_robust(t_vec, y_vec; kernel_type = kernel_type)
	return s3_refine_gp_bic(gp, t_vec, y_vec)
end

# --- Adaptive tolerance convenience aliases (recommended) ---
s3_adapt_se_interpolator(xs, ys; k = 10.0) = _s3_adapt_interpolator(xs, ys; kernel_type = :se, k = k)
s3_adapt_rq_interpolator(xs, ys; k = 10.0) = _s3_adapt_interpolator(xs, ys; kernel_type = :rq, k = k)
s3_adapt_se_plus_rq_interpolator(xs, ys; k = 10.0) = _s3_adapt_interpolator(xs, ys; kernel_type = :se_plus_rq, k = k)
s3_adapt_se_times_rq_interpolator(xs, ys; k = 10.0) = _s3_adapt_interpolator(xs, ys; kernel_type = :se_times_rq, k = k)
s3_adapt_matern52_interpolator(xs, ys; k = 10.0) = _s3_adapt_interpolator(xs, ys; kernel_type = :matern52, k = k)

# --- BIC model selection convenience aliases ---
s3_bic_se_interpolator(xs, ys) = _s3_bic_interpolator(xs, ys; kernel_type = :se)
s3_bic_rq_interpolator(xs, ys) = _s3_bic_interpolator(xs, ys; kernel_type = :rq)
s3_bic_se_plus_rq_interpolator(xs, ys) = _s3_bic_interpolator(xs, ys; kernel_type = :se_plus_rq)
s3_bic_se_times_rq_interpolator(xs, ys) = _s3_bic_interpolator(xs, ys; kernel_type = :se_times_rq)
s3_bic_matern52_interpolator(xs, ys) = _s3_bic_interpolator(xs, ys; kernel_type = :matern52)

# --- Deprecated old S3 names (forward to adaptive with default k=10) ---
s3_se_interpolator(xs, ys) = (_s3_adapt_interpolator(xs, ys; kernel_type = :se))
s3_rq_interpolator(xs, ys) = (_s3_adapt_interpolator(xs, ys; kernel_type = :rq))
s3_se_plus_rq_interpolator(xs, ys) = (_s3_adapt_interpolator(xs, ys; kernel_type = :se_plus_rq))
s3_se_times_rq_interpolator(xs, ys) = (_s3_adapt_interpolator(xs, ys; kernel_type = :se_times_rq))
s3_matern52_interpolator(xs, ys) = (_s3_adapt_interpolator(xs, ys; kernel_type = :matern52))

# ═══════════════════════════════════════════════════════════════════════════════
# Chebyshev polynomial interpolation with AICc degree selection
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ChebyshevApprox <: AbstractInterpolator

Chebyshev polynomial least-squares fit with automatic degree selection.
Derivatives are exact via Chebyshev coefficient recurrence.
"""
struct ChebyshevApprox <: AbstractInterpolator
	coeffs::Vector{Float64}   # Chebyshev coefficients in mapped [-1,1] domain
	x_min::Float64
	x_max::Float64
	degree::Int
end

function _chebyshev_vandermonde(u::Vector{Float64}, deg::Int)
	n = length(u)
	V = zeros(n, deg + 1)
	V[:, 1] .= 1.0
	if deg >= 1
		V[:, 2] .= u
	end
	for j in 3:(deg + 1)
		@. V[:, j] = 2 * u * V[:, j-1] - V[:, j-2]
	end
	return V
end

function _clenshaw_eval(c::Vector{Float64}, u)
	d = length(c) - 1
	d < 0 && return zero(u)
	d == 0 && return c[1] * one(u)
	b_next = zero(u)
	b_curr = zero(u)
	for k in d:-1:1
		b_prev = 2 * u * b_curr - b_next + c[k+1]
		b_next = b_curr
		b_curr = b_prev
	end
	return u * b_curr - b_next + c[1]
end

function (cf::ChebyshevApprox)(x)
	L = cf.x_max - cf.x_min
	u = 2 * (x - cf.x_min) / L - 1
	return _clenshaw_eval(cf.coeffs, u)
end


function _select_degree_aicc(x::Vector{Float64}, y::Vector{Float64}; min_deg::Int = 3, max_deg::Int = 30)
	n = length(x)
	x_min, x_max = extrema(x)
	x_max == x_min && return min_deg, Inf
	u = @. 2 * (x - x_min) / (x_max - x_min) - 1
	max_deg_safe = min(max_deg, n - 2)
	min_deg_safe = max(min(min_deg, max_deg_safe), 1)
	V_full = _chebyshev_vandermonde(u, max_deg_safe)
	best_aicc = Inf
	best_deg = min_deg_safe
	for deg in min_deg_safe:max_deg_safe
		try
			V = @view V_full[:, 1:(deg+1)]
			coeffs = V \ y
			rss = sum((y - V * coeffs) .^ 2)
			k = deg + 2
			if n - k - 1 > 0
				aicc = n * log(rss / n + 1e-12) + 2k + 2k * (k + 1) / (n - k - 1)
			else
				aicc = Inf
			end
			if aicc < best_aicc
				best_aicc = aicc
				best_deg = deg
			end
		catch
			continue
		end
	end
	return best_deg, best_aicc
end

"""
    chebyshev_aicc(xs, ys) -> ChebyshevApprox

Fit a Chebyshev polynomial to data with automatic degree selection via AICc.
"""
function chebyshev_aicc(xs::AbstractArray{T}, ys::AbstractArray{T})::ChebyshevApprox where {T}
	x = Float64.(xs)
	y = Float64.(ys)
	deg, _ = _select_degree_aicc(x, y; min_deg = 3, max_deg = min(30, length(x) - 1))
	x_min, x_max = extrema(x)
	u = @. 2 * (x - x_min) / (x_max - x_min) - 1
	V = _chebyshev_vandermonde(u, deg)
	coeffs = V \ y
	return ChebyshevApprox(coeffs, x_min, x_max, deg)
end

function _select_degree_bic(x::Vector{Float64}, y::Vector{Float64}; min_deg::Int = 3, max_deg::Int = 30)
	n = length(x)
	x_min, x_max = extrema(x)
	x_max == x_min && return min_deg, Inf
	u = @. 2 * (x - x_min) / (x_max - x_min) - 1
	max_deg_safe = min(max_deg, n - 2)
	min_deg_safe = max(min(min_deg, max_deg_safe), 1)
	V_full = _chebyshev_vandermonde(u, max_deg_safe)
	best_bic = Inf
	best_deg = min_deg_safe
	for deg in min_deg_safe:max_deg_safe
		try
			V = @view V_full[:, 1:(deg+1)]
			coeffs = V \ y
			rss = sum((y - V * coeffs) .^ 2)
			k = deg + 2
			bic = n * log(rss / n + 1e-12) + k * log(n)
			if bic < best_bic
				best_bic = bic
				best_deg = deg
			end
		catch
			continue
		end
	end
	return best_deg, best_bic
end

"""
    chebyshev_bic(xs, ys) -> ChebyshevApprox

Fit a Chebyshev polynomial to data with automatic degree selection via BIC.
BIC penalizes complexity more than AICc — may prefer lower degree on noisy data.
"""
function chebyshev_bic(xs::AbstractArray{T}, ys::AbstractArray{T})::ChebyshevApprox where {T}
	x = Float64.(xs)
	y = Float64.(ys)
	deg, _ = _select_degree_bic(x, y; min_deg = 3, max_deg = min(30, length(x) - 1))
	x_min, x_max = extrema(x)
	u = @. 2 * (x - x_min) / (x_max - x_min) - 1
	V = _chebyshev_vandermonde(u, deg)
	coeffs = V \ y
	return ChebyshevApprox(coeffs, x_min, x_max, deg)
end

