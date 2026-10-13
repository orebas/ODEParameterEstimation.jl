# UQ coverage/calibration harness (Stream B item 2, 2026-08-13).
#
# Measures empirical CI coverage of the UQ chain over N independent noise
# replicates: noisy draw → estimate → GP jets → Σ_d → sensitivity S → Σ_x →
# physicalized report → z = (estimate − truth)/σ̂ per physical coordinate.
#
# Two conditioning modes for S (diagnose_sensitivity value_source):
#   :estimate — the production path (θ̂/x̂ jets + GP data jets). DEFAULT.
#   :oracle   — truth-conditioned; the TAC-theory two_exp anchor numbers
#               (95% well-conditioned block / 82% weak block) were this mode.
#
# Two estimators:
#   :nls_polish    — forward-solve least squares started at truth (fast; the
#                    estimator itself is not under test, the UQ chain is).
#   :full_pipeline — analyze_parameter_estimation_problem with
#                    compute_uncertainty=true (paper-grade; slow per replicate).
#
# The gate smoke (test/test_uq_coverage_smoke.jl) includes this file. Keep it
# load-light and side-effect free. Full calibration campaigns live on the
# research branch.

using ODEParameterEstimation
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrderedCollections
using OrdinaryDiffEq
using SciMLBase
using Optim
using Random
using Statistics
using Logging
using LinearAlgebra

# ─── Anchor model: two decaying exponentials, summed observation ──────
# y = x1 + x2 with k2 >> k1 makes (x2, k2) the weakly-identified block at
# late shooting times (the fast mode has died) — the 82%-coverage residual
# documented in the TAC-theory PHASE3_CALIBRATION anchor.

function two_exp_pep(; x01_true = 2.0, x02_true = 1.5, k1_true = 0.6, k2_true = 3.0)
	parameters = @parameters k1 k2
	states = @variables x1(t) x2(t)
	@variables y1(t)
	equations = [
		D(x1) ~ -k1 * x1,
		D(x2) ~ -k2 * x2,
	]
	measured_quantities = [y1 ~ x1 + x2]
	model, mq = ODEParameterEstimation.create_ordered_ode_system(
		"two_exp", states, parameters, equations, measured_quantities)
	return ParameterEstimationProblem(
		"two_exp", model, mq, nothing, nothing, nothing,
		OrderedDict(parameters .=> [k1_true, k2_true]),
		OrderedDict(states .=> [x01_true, x02_true]),
		0,
	)
end

# Well-conditioned companion: both states observed directly (simple rotation).
two_state_observed_pep() = ODEParameterEstimation.Examples.simple()

# ─── Quiet wrapper ────────────────────────────────────────────────────

function _cov_quiet(f)
	redirect_stdout(devnull) do
		redirect_stderr(devnull) do
			with_logger(NullLogger()) do
				return f()
			end
		end
	end
end

# ─── Estimators ───────────────────────────────────────────────────────

"""
Forward-solve least-squares fit of (t0 states, params), started at truth.
Returns a `ParameterEstimationResult` (states = t0 ICs, `at_time = t_eval`).
Not the production estimator — a cheap stand-in so N replicates are viable;
the UQ chain downstream is exactly the production one.

Assumes observable RHS are time-invariant expressions of (states, params).
"""
function nls_polish_estimate(pep_data::ParameterEstimationProblem, t_eval::Float64)
	sys = pep_data.model.system
	completed = ModelingToolkit.complete(sys)
	states = ModelingToolkit.unknowns(completed)
	params = ModelingToolkit.parameters(completed)
	n_x, n_p = length(states), length(params)

	t_vec = Float64.(pep_data.data_sample["t"])
	tspan = (t_vec[1], t_vec[end])

	obs_keys = [ODEParameterEstimation.Symbolics.diff2term(mq.rhs) for mq in pep_data.measured_quantities]
	y_data = [Float64.(pep_data.data_sample[k]) for k in obs_keys]

	p_truth = [pep_data.p_true[p] for p in params]
	x0_truth = [pep_data.ic[s] for s in states]

	obs_fns = [ModelingToolkit.build_function(mq.rhs, states, params; expression = Val(false))
	           for mq in pep_data.measured_quantities]

	base_prob = ODEProblem(completed,
		merge(Dict(states .=> x0_truth), Dict(params .=> p_truth)), tspan)

	function sse(θ::Vector{Float64})
		x0 = θ[1:n_x]
		pv = θ[(n_x + 1):(n_x + n_p)]
		prob = remake(base_prob;
			u0 = Dict(states .=> x0), p = Dict(params .=> pv),
			build_initializeprob = false)
		sol = try
			ModelingToolkit.solve(prob, AutoVern9(Rodas5P());
				abstol = 1e-10, reltol = 1e-10, saveat = t_vec)
		catch
			return 1e100
		end
		SciMLBase.successful_retcode(sol) || return 1e100
		total = 0.0
		for (oi, fn) in enumerate(obs_fns)
			for ti in eachindex(t_vec)
				total += (fn(sol.u[ti], pv) - y_data[oi][ti])^2
			end
		end
		return total
	end

	θ0 = vcat(x0_truth, p_truth)
	res = Optim.optimize(sse, θ0, Optim.NelderMead(),
		Optim.Options(iterations = 4000))
	θ̂ = Optim.minimizer(res)

	est_states = OrderedDict{Num, Float64}(s => θ̂[i] for (i, s) in enumerate(states))
	est_params = OrderedDict{Num, Float64}(p => θ̂[n_x + i] for (i, p) in enumerate(params))
	return ODEParameterEstimation.ParameterEstimationResult(
		est_params, est_states, t_eval, Optim.minimum(res), :NLSPolish,
		length(t_vec), nothing, nothing, Set{Num}(), nothing)
end

# ─── Single replicate ─────────────────────────────────────────────────

struct CoverageReplicate
	labels::Vector{String}
	truth::Vector{Float64}
	center::Vector{Float64}
	sigma::Vector{Float64}
	covariance::Matrix{Float64}
	status::Symbol
end

CoverageReplicate(status::Symbol) = CoverageReplicate(
	String[], Float64[], Float64[], Float64[], zeros(0, 0), status,
)

_base_name(x) = replace(string(x), r"\(.*\)" => "")

function _truth_for_label(pep::ParameterEstimationProblem, label::String)
	base = _base_name(label)
	for (p, v) in pep.p_true
		_base_name(p) == base && return v
	end
	for (s, v) in pep.ic
		_base_name(s) == base && return v
	end
	return NaN
end

function _center_for_label(est, label::String)
	base = _base_name(label)
	for (p, v) in est.parameters
		_base_name(p) == base && return v
	end
	for (s, v) in est.states
		_base_name(s) == base && return v
	end
	return NaN
end

function run_replicate(pep_ctor, opts::EstimationOptions;
	value_source::Symbol = :estimate,
	estimator::Symbol = :nls_polish,
)
	pep = pep_ctor()
	pep_data = ODEParameterEstimation.sample_problem_data(pep, opts)

	if estimator === :full_pipeline
		_raw, analysis, uq = _cov_quiet() do
			ODEParameterEstimation.analyze_parameter_estimation_problem(pep_data, opts)
		end
		isnothing(uq) && return CoverageReplicate(:no_report)
		uq isa ODEParameterEstimation.UQUnavailable &&
			return CoverageReplicate(uq.reason)
		uq isa ODEParameterEstimation.UncertaintyReport ||
			return CoverageReplicate(:no_report)
		best = isempty(analysis.returned_results) ? nothing : first(analysis.returned_results)
		isnothing(best) && return CoverageReplicate(:no_estimate)
		labels = uq.param_labels
		return CoverageReplicate(labels,
			[_truth_for_label(pep_data, l) for l in labels],
			copy(uq.estimate_values),
			copy(uq.param_std), copy(uq.param_covariance), uq.status)
	end

	setup = _cov_quiet() do
		ODEParameterEstimation.setup_parameter_estimation(
			pep_data; interpolator = ODEParameterEstimation.agp_gpr_uq, nooutput = true)
	end
	t_vec = pep_data.data_sample["t"]
	t_eval = t_vec[setup.time_index_set[1]]

	est = _cov_quiet() do
		nls_polish_estimate(pep_data, t_eval)
	end

	sens = _cov_quiet() do
		if value_source === :estimate
			ODEParameterEstimation.diagnose_sensitivity(
				pep_data; setup_data = setup, t_eval = t_eval, estimate_result = est)
		else
			ODEParameterEstimation.diagnose_sensitivity(
				pep_data; setup_data = setup, t_eval = t_eval)
		end
	end

	r = _cov_quiet() do
		ODEParameterEstimation.diagnose_uncertainty(pep_data, setup, t_eval, sens)
	end
	isnothing(r) && return CoverageReplicate(:no_report)
	uq_local = first(r)
	uq = _cov_quiet() do
		ODEParameterEstimation.physicalize_uncertainty_report(pep_data, est, uq_local)
	end
	isnothing(uq) && return CoverageReplicate(:no_physicalization)

	labels = uq.param_labels
	return CoverageReplicate(labels,
		[_truth_for_label(pep_data, l) for l in labels],
		[_center_for_label(est, l) for l in labels],
		copy(uq.param_std), copy(uq.param_covariance), uq.status)
end

# ─── N-replicate aggregation ──────────────────────────────────────────

struct CoverageResult
	labels::Vector{String}
	n_total::Int
	n_reported::Int
	covered::Dict{String, Int}        # |z| ≤ z_crit count per label
	usable::Dict{String, Int}         # finite-z replicate count per label
	zs::Dict{String, Vector{Float64}}
	z_crit::Float64
	statuses::Vector{Symbol}
	# Added 2026-08-14 after the single-point-regime run: mean/sd of z are
	# destroyed by a single σ̂≈0 replicate (|z| → 1e7), and coverage alone
	# cannot distinguish "calibrated" from "estimate is hopeless but the
	# interval is honestly enormous". Track the estimator's own error and
	# count the degenerate-σ̂ replicates instead of silently averaging them in.
	rel_errs::Dict{String, Vector{Float64}}   # |estimate − truth| / |truth|
	sigmas::Dict{String, Vector{Float64}}     # σ̂ per replicate
	n_zero_sigma::Dict{String, Int}           # σ̂ ≤ 0 or non-finite z
	covariances::Vector{Matrix{Float64}}      # full joint covariance per usable report
	joint_mahalanobis::Vector{Float64}        # e'Σ⁻¹e when finite/positive definite
end

function run_coverage(pep_ctor;
	N::Int = 20,
	noise_level::Float64 = 0.01,
	datasize::Int = 121,
	time_interval::Vector{Float64} = [0.0, 1.0],
	value_source::Symbol = :estimate,
	estimator::Symbol = :nls_polish,
	seed0::Int = 7000,
	z_crit::Float64 = ODEParameterEstimation.UQ_CI_Z,
	extra_opts::NamedTuple = NamedTuple(),
)
	base = (datasize = datasize, time_interval = time_interval,
		noise_level = noise_level, nooutput = true, diagnostics = false)
	uq_flag = estimator === :full_pipeline ? (compute_uncertainty = true,) : NamedTuple()
	opts = EstimationOptions(; base..., uq_flag..., extra_opts...)

	covered = Dict{String, Int}()
	usable = Dict{String, Int}()
	zs = Dict{String, Vector{Float64}}()
	rel_errs = Dict{String, Vector{Float64}}()
	sigmas = Dict{String, Vector{Float64}}()
	n_zero_sigma = Dict{String, Int}()
	statuses = Symbol[]
	covariances = Matrix{Float64}[]
	joint_mahalanobis = Float64[]
	labels_seen = String[]
	n_reported = 0

	for i in 1:N
		Random.seed!(seed0 + i)
		rep = run_replicate(pep_ctor, opts; value_source, estimator)
		push!(statuses, rep.status)
		isempty(rep.labels) && continue
		n_reported += 1
		push!(covariances, copy(rep.covariance))
		error = rep.center .- rep.truth
		if size(rep.covariance) == (length(error), length(error)) && all(isfinite, error) &&
				all(isfinite, rep.covariance)
			quadratic = try
				dot(error, cholesky(Symmetric(rep.covariance)) \ error)
			catch
				NaN
			end
			isfinite(quadratic) && push!(joint_mahalanobis, quadratic)
		end
		for (j, l) in enumerate(rep.labels)
			l in labels_seen || push!(labels_seen, l)
			tv = rep.truth[j]
			if isfinite(tv) && abs(tv) > 1e-15 && isfinite(rep.center[j])
				push!(get!(rel_errs, l, Float64[]), abs(rep.center[j] - tv) / abs(tv))
			end
			isfinite(rep.sigma[j]) && push!(get!(sigmas, l, Float64[]), rep.sigma[j])
			z = (rep.center[j] - tv) / rep.sigma[j]
			if !isfinite(z)
				# σ̂ ≈ 0 (or non-finite center/truth): an infinite z is not a
				# coverage miss, it is an unusable measurement. Count it.
				n_zero_sigma[l] = get(n_zero_sigma, l, 0) + 1
				continue
			end
			usable[l] = get(usable, l, 0) + 1
			push!(get!(zs, l, Float64[]), z)
			abs(z) <= z_crit && (covered[l] = get(covered, l, 0) + 1)
		end
	end

	return CoverageResult(labels_seen, N, n_reported, covered, usable, zs, z_crit, statuses,
		rel_errs, sigmas, n_zero_sigma, covariances, joint_mahalanobis)
end

coverage_fraction(res::CoverageResult, label::String) =
	get(res.usable, label, 0) == 0 ? NaN :
	get(res.covered, label, 0) / res.usable[label]

unconditional_coverage_fraction(res::CoverageResult, label::String) =
	res.n_total == 0 ? NaN : get(res.covered, label, 0) / res.n_total

usable_fraction(res::CoverageResult, label::String) =
	res.n_total == 0 ? NaN : get(res.usable, label, 0) / res.n_total

function coverage_reason_counts(res::CoverageResult)
	counts = Dict{Symbol, Int}()
	for status in res.statuses
		counts[status] = get(counts, status, 0) + 1
	end
	return counts
end

_fmt3(x) = isempty(x) ? "—" : string(round(median(x); sigdigits = 3))

"""
Robust per-coordinate summary. `med|z|` near 0.67 and `p90|z|` near 1.64 indicate
a calibrated interval (the median and 90th percentile of a standard normal);
much smaller means the interval is conservative, much larger means overconfident.
`med relerr` is the estimator's own accuracy — a coordinate can be perfectly
"covered" by an interval so wide it says nothing, which is exactly the
out-of-regime signature.
"""
function print_coverage(res::CoverageResult)
	println("UQ coverage: $(res.n_reported)/$(res.n_total) replicates produced a report; z_crit = $(res.z_crit)")
	println("Outcome/status taxonomy: ", coverage_reason_counts(res))
	println(rpad("coordinate", 14), rpad("cond cov", 10), rpad("all cov", 10),
		rpad("usable", 9), rpad("n", 4), rpad("bad_σ", 6),
		rpad("med|z|", 9), rpad("p90|z|", 9), rpad("med relerr", 12), "med σ̂")
	for l in res.labels
		zv = get(res.zs, l, Float64[])
		az = sort(abs.(zv))
		p90 = isempty(az) ? NaN : az[max(1, ceil(Int, 0.9 * length(az)))]
		cov = coverage_fraction(res, l)
		all_cov = unconditional_coverage_fraction(res, l)
		usable_rate = usable_fraction(res, l)
		println(rpad(l, 14),
			rpad(isnan(cov) ? "—" : string(round(100 * cov; digits = 1), "%"), 10),
			rpad(isnan(all_cov) ? "—" : string(round(100 * all_cov; digits = 1), "%"), 10),
			rpad(isnan(usable_rate) ? "—" : string(round(100 * usable_rate; digits = 1), "%"), 9),
			rpad(string(length(zv)), 4),
			rpad(string(get(res.n_zero_sigma, l, 0)), 6),
			rpad(_fmt3(az), 9),
			rpad(isnan(p90) ? "—" : string(round(p90; sigdigits = 3)), 9),
			rpad(_fmt3(get(res.rel_errs, l, Float64[])), 12),
			_fmt3(get(res.sigmas, l, Float64[])))
	end
	if !isempty(res.joint_mahalanobis)
		println("Joint Mahalanobis e'Σ⁻¹e: n=$(length(res.joint_mahalanobis)), median=$(round(median(res.joint_mahalanobis); sigdigits=4))")
	end
	return nothing
end

"""
    regime_verdict(res; relerr_tol = 0.05) → (:in_regime | :marginal | :out_of_regime, notes)

Classify a (model, config) cell for the single-point-unpolished UQ regime:
the estimator must be ACCURATE (median relative error ≤ tol on every
coordinate) and the interval CALIBRATED (median |z| in a loose band around
the standard-normal median 0.674). Wide-but-covering is NOT in-regime.
"""
function regime_verdict(res::CoverageResult; relerr_tol::Float64 = 0.05)
	res.n_reported < max(1, res.n_total ÷ 2) && return (:out_of_regime, "only $(res.n_reported)/$(res.n_total) replicates reported")
	notes = String[]
	worst_err = 0.0
	calibrated = true
	# Standard-normal median |z| = 0.674; accept a factor of 2 either way.
	# (The earlier [0.15, 2.5] band was too permissive — it passed a
	# coordinate with med|z| = 1.89 and 50% coverage.)
	lo, hi = 0.34, 1.35
	for l in res.labels
		usable_rate = usable_fraction(res, l)
		if !isfinite(usable_rate) || usable_rate < 0.9
			calibrated = false
			push!(notes, "$l: unconditional usable rate $(round(100 * usable_rate; sigdigits = 3))% < 90%")
		end
		re = get(res.rel_errs, l, Float64[])
		isempty(re) || (worst_err = max(worst_err, median(re)))
		az = abs.(get(res.zs, l, Float64[]))
		if isempty(az)
			calibrated = false
			push!(notes, "$l: no usable z")
			continue
		end
		m = median(az)
		if m > hi
			calibrated = false
			push!(notes, "$l: OVERCONFIDENT med|z|=$(round(m; sigdigits = 3))")
		elseif m < lo
			calibrated = false
			push!(notes, "$l: conservative med|z|=$(round(m; sigdigits = 3))")
		end
		get(res.n_zero_sigma, l, 0) > 0 && push!(notes, "$l: $(res.n_zero_sigma[l]) degenerate σ̂")
	end
	accurate = worst_err <= relerr_tol
	verdict = accurate && calibrated ? :in_regime :
			  accurate ? :marginal : :out_of_regime
	push!(notes, "worst med relerr = $(round(100 * worst_err; sigdigits = 3))%")
	return verdict, notes
end
