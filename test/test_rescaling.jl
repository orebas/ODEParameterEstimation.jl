using ODEParameterEstimation
using ODEParameterEstimation: choose_scales, rescale_pep, transform_pep_for_estimation,
    unrescale_results
using ODEParameterEstimation.Examples
# Fast unit/contract tests for the opt-in power-of-2 problem rescaling
# (src/core/problem_rescaling.jl). The slow end-to-end PAYOFF test (hiv recovery
# with auto_rescale=true) lives in test/benchmark_smoke.jl — NOT here.
#
# Run: julia --startup-file=no -e 'using ODEParameterEstimation; include("test/test_rescaling.jl")'

using Test
using Logging
using Random
using OrderedCollections
import Symbolics
import ModelingToolkit

const OPE = ODEParameterEstimation

@testset "Problem rescaling (power-of-2)" begin

	@testset "extract_monomials: coeff/power split" begin
		Symbolics.@variables x p z
		vs = Set{Any}(Symbolics.value.([x, p, z]))
		# 3*x*p - 2*x^2 + 5
		mons = OPE.extract_monomials(3.0 * x * p - 2.0 * x^2 + 5.0, vs)
		# collect into a comparable form: coeff => sorted powers
		findmon(predicate) = findfirst(predicate, mons)
		# term 3*x*p
		i = findmon(m -> isapprox(m[1], 3.0) && m[2] == Dict(Symbolics.value(x) => 1, Symbolics.value(p) => 1))
		@test i !== nothing
		# term -2*x^2
		j = findmon(m -> isapprox(m[1], -2.0) && m[2] == Dict(Symbolics.value(x) => 2))
		@test j !== nothing
		# constant term 5 (empty powers)
		k = findmon(m -> isapprox(m[1], 5.0) && isempty(m[2]))
		@test k !== nothing

		# division folds into the coefficient: (2*x)/4 -> 0.5*x
		md = OPE.extract_monomials((2.0 * x) / 4.0, vs)
		@test length(md) == 1
		@test isapprox(md[1][1], 0.5)
		@test md[1][2] == Dict(Symbolics.value(x) => 1)

		# variable in a denominator -> negative power
		mn = OPE.extract_monomials(6.0 * p / z, vs)
		@test length(mn) == 1
		@test isapprox(mn[1][1], 6.0)
		@test mn[1][2] == Dict(Symbolics.value(p) => 1, Symbolics.value(z) => -1)
	end

	@testset "choose_scales: powers of 2, constants shrink (hiv)" begin
		pep = OPE.Examples.hiv()
		opts = EstimationOptions(datasize = 21, noise_level = 0.0, nooutput = true)
		Random.seed!(1)
		sampled = OPE.sample_problem_data(pep, opts)
		info = choose_scales(sampled)
		# every scale is an exact power of 2
		for sc in Iterators.flatten((values(info.state_scales), values(info.param_scales), values(info.observable_scales)))
			@test sc > 0
			@test isapprox(log2(sc), round(log2(sc)); atol = 1e-9)   # integer exponent
		end
		# hiv is ill-scaled → some nontrivial scaling, and the rescaled equation
		# constants stay modest (the repo hiv has unit literals, so normalizing the
		# variables introduces small constants — the win is variable-spread, below).
		@test info.metadata.n_nontrivial > 0
		@test info.metadata.max_const_after < 10.0
		# THE CONDITIONING WIN: scaled truth/IC values have a smaller log2 spread than
		# the originals (variables pulled toward O(1)).
		rpep0, _ = rescale_pep(sampled)
		origvals = Float64[abs(v) for v in vcat(collect(values(sampled.p_true)), collect(values(sampled.ic))) if v != 0]
		sclvals = Float64[abs(v) for v in vcat(collect(values(rpep0.p_true)), collect(values(rpep0.ic))) if v != 0]
		origspread = log2(maximum(origvals)) - log2(minimum(origvals))
		sclspread = log2(maximum(sclvals)) - log2(minimum(sclvals))
		@test sclspread < origspread
	end

	@testset "well-scaled model: mild scaling + round-trip (lotka_volterra)" begin
		pep = OPE.Examples.lotka_volterra()
		opts = EstimationOptions(datasize = 21, noise_level = 0.0, nooutput = true)
		Random.seed!(2)
		sampled = OPE.sample_problem_data(pep, opts)
		rpep, info = rescale_pep(sampled)
		if info === nothing
			@test rpep === sampled                       # exact no-op
		else
			# already O(1) ⇒ scaling stays mild and round-trips exactly
			for sc in Iterators.flatten((values(info.state_scales), values(info.param_scales)))
				@test 2.0^-4 <= sc <= 2.0^4
			end
			for (k, v) in sampled.p_true
				@test rpep.p_true[k] * info.param_scales[k] == v
			end
		end
	end

	@testset "round-trip identity: rescale truth, un-rescale, recover exactly" begin
		pep = OPE.Examples.hiv()
		opts = EstimationOptions(datasize = 21, noise_level = 0.0, nooutput = true)
		Random.seed!(3)
		sampled = OPE.sample_problem_data(pep, opts)
		rpep, info = rescale_pep(sampled)
		@test info !== nothing                       # hiv IS ill-scaled
		# scaled truth × scale == original truth, exactly (powers of 2)
		for (k, v) in sampled.p_true
			@test rpep.p_true[k] * info.param_scales[k] == v
		end
		for (k, v) in sampled.ic
			@test rpep.ic[k] * info.state_scales[k] == v
		end

		# unrescale_results inverts on a synthetic result carrying the scaled truth
		res = OPE.ParameterEstimationResult(
			OrderedDict{Symbolics.Num, Float64}(rpep.p_true),
			OrderedDict{Symbolics.Num, Float64}(rpep.ic),
			0.0, 0.0, nothing, 21, 0.0,
			OrderedDict{Symbolics.Num, Float64}(), Set{Symbolics.Num}(), nothing,
		)
		unrescale_results([res], info)
		for (k, v) in sampled.p_true
			@test isapprox(res.parameters[k], v; rtol = 1e-12)
		end
		for (k, v) in sampled.ic
			@test isapprox(res.states[k], v; rtol = 1e-12)
		end
		# idempotent: a second un-rescale must NOT double-scale
		unrescale_results([res], info)
		for (k, v) in sampled.p_true
			@test isapprox(res.parameters[k], v; rtol = 1e-12)
		end
	end

	@testset "sum-observable data scaling (hiv y4 ~ y+v)" begin
		pep = OPE.Examples.hiv()
		opts = EstimationOptions(datasize = 21, noise_level = 0.0, nooutput = true)
		Random.seed!(4)
		sampled = OPE.sample_problem_data(pep, opts)
		rpep, info = rescale_pep(sampled)
		@test info !== nothing
		# each observable column is re-keyed by the NEW (scaled) rhs and divided by its
		# (power-of-2) scale; matched observable-by-observable.
		for (omq, nmq) in zip(sampled.measured_quantities, rpep.measured_quantities)
			okey = haskey(sampled.data_sample, Symbolics.Num(omq.rhs)) ? Symbolics.Num(omq.rhs) : Symbolics.Num(omq.lhs)
			nkey = Symbolics.Num(nmq.rhs)
			sc = get(info.observable_scales, okey, 1.0)
			@test haskey(rpep.data_sample, nkey)
			@test rpep.data_sample[nkey] ≈ sampled.data_sample[okey] ./ sc
		end
		# "t" is untouched (time not scaled in this MVP)
		@test rpep.data_sample["t"] == sampled.data_sample["t"]
	end

	@testset "user-bound rescaling preserves physical meaning" begin
		pep = OPE.Examples.hiv()
		opts0 = EstimationOptions(datasize = 21, noise_level = 0.0, nooutput = true)
		Random.seed!(7)
		sampled = OPE.sample_problem_data(pep, opts0)
		info = choose_scales(sampled)
		states = Symbolics.Num.(ModelingToolkit.unknowns(sampled.model.system))
		params = Symbolics.Num.(ModelingToolkit.parameters(sampled.model.system))
		nS = length(states)
		scales = vcat(Float64[info.state_scales[s] for s in states], Float64[info.param_scales[p] for p in params])
		# physical bounds wide enough to contain the RAW hiv truth (beta=2e-5 … x=1000)
		lb = fill(1e-8, length(scales))
		ub = fill(1e4, length(scales))
		opts = EstimationOptions(datasize = 21, noise_level = 0.0, nooutput = true, opt_lb = lb, opt_ub = ub)
		opts2 = OPE.rescale_option_bounds(opts, info, sampled)
		# (a) the transform is exactly physical / scale, per variable
		@test opts2.opt_lb ≈ lb ./ scales
		@test opts2.opt_ub ≈ ub ./ scales
		# (b) MEANING preserved: a physical bound containing the raw truth ⇒ the scaled
		# bound contains the SCALED truth (this is the whole point of the fix).
		rpep, _ = rescale_pep(sampled)
		for (i, p) in enumerate(params)
			sv = rpep.p_true[p]
			@test opts2.opt_lb[nS+i] <= sv <= opts2.opt_ub[nS+i]
		end
		# no-op when no bounds
		@test OPE.rescale_option_bounds(opts0, info, sampled) === opts0
	end

	@testset "bounds length mismatch FAILS FAST (silent-fallback trio canary)" begin
		# The class this kills: a wrong-length opt_lb/opt_ub used to degrade
		# silently at three separate sites (untransformed bounds into a scaled
		# solve; fallback to the ±1e6 box; skipped backsolve clamp) — CSTR
		# final_v2 shipped out-of-box polished results this way
		# (docs/internal/2026-06-19_transform_bounds_mismatch.md).
		pep = OPE.Examples.hiv()
		opts0 = EstimationOptions(datasize = 21, noise_level = 0.0, nooutput = true)
		Random.seed!(7)
		sampled = OPE.sample_problem_data(pep, opts0)
		info = choose_scales(sampled)
		nvars = length(ModelingToolkit.unknowns(sampled.model.system)) +
				length(ModelingToolkit.parameters(sampled.model.system))

		# helper contract: no-throw on match / unset, throw on mismatch
		ok = EstimationOptions(nooutput = true, opt_lb = fill(0.0, nvars), opt_ub = fill(1.0, nvars))
		@test OPE._assert_bounds_length(ok, nvars, "site", "layout") === nothing
		@test OPE._assert_bounds_length(opts0, nvars, "site", "layout") === nothing
		bad = EstimationOptions(nooutput = true, opt_lb = fill(0.0, nvars - 1), opt_ub = fill(1.0, nvars - 1))
		@test_throws ArgumentError OPE._assert_bounds_length(bad, nvars, "site", "layout")

		# site 1: rescale_option_bounds no longer passes mismatched bounds through
		@test_throws ArgumentError OPE.rescale_option_bounds(bad, info, sampled)

		# site 3: backsolve clamp no longer silently skips on mismatch
		params = Symbolics.Num.(ModelingToolkit.parameters(sampled.model.system))
		nS = length(ModelingToolkit.unknowns(sampled.model.system))
		p_map = Dict(p => 1.0 for p in params)
		@test_throws ArgumentError OPE._clamp_params_for_backsolve(p_map, bad, params, nS)
		# ...and still clamps correctly on a well-formed spec
		tight = EstimationOptions(nooutput = true,
			opt_lb = fill(2.0, nS + length(params)), opt_ub = fill(3.0, nS + length(params)))
		clamped = OPE._clamp_params_for_backsolve(p_map, tight, params, nS)
		@test all(v -> 2.0 <= v <= 3.0, values(clamped))
	end

	@testset "known inputs keep their scale (sin(c*t) on a short interval)" begin
		# On [-0.5, 0.5] sin(0.5t) stays below 1/4, so the scale chooser used to
		# halve the state that stands for it and double its coefficient. That
		# state is later given its exact, unscaled values, so the estimated
		# coefficient came back doubled (2026-10-06).
		t = ModelingToolkit.t_nounits
		sampled = sample_problem_data(forced_decay(), EstimationOptions(datasize = 101, time_interval = [-0.5, 0.5]))
		lifted, _ = Logging.with_logger(Logging.NullLogger()) do
			transform_pep_for_estimation(sampled, t)
		end
		info = OPE.choose_scales(lifted)
		inputs = [s for s in keys(info.state_scales) if OPE._is_trfn_observable(s)]
		@test length(inputs) == 2
		@test all(s -> info.state_scales[s] == 1.0, inputs)
		@test all(info.observable_scales[s] == 1.0 for s in inputs)
		# The states and parameters of the model itself are still free to scale.
		@test Set(keys(info.state_scales)) == Set(Symbolics.Num.(ModelingToolkit.unknowns(lifted.model.system)))
	end

end
