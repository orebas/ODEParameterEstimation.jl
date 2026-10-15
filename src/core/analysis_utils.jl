
"""
	solution_distance(sol1, sol2)

Calculate the relative distance between two solutions.

# Arguments
- `sol1`: First solution
- `sol2`: Second solution

# Returns
- Maximum relative difference between components matched by symbolic key.
  Results with different state or parameter keys have infinite distance.
"""
function solution_distance(sol1, sol2)
	distance = 0.0
	for (left, right) in ((sol1.states, sol2.states), (sol1.parameters, sol2.parameters))
		length(left) == length(right) || return Inf
		for (key, x) in left
			haskey(right, key) || return Inf
			y = right[key]
			denom = max(abs(x) + abs(y), 1.0)
			distance = max(distance, abs(x - y) / denom)
		end
	end
	return distance
end

"""
	cluster_solutions(sorted_results)

Group similar solutions into clusters based on relative distances.

# Arguments
- `sorted_results`: Vector of solutions sorted by error

# Returns
- Vector of solution clusters
"""
function cluster_solutions(sorted_results; threshold::Float64 = CLUSTERING_THRESHOLD)
	clusters = Vector{Vector{Any}}()

	for sol in sorted_results
		# Try to find a cluster for this solution
		cluster_idx = findfirst(cluster ->
				solution_distance(sol, cluster[1]) < threshold,
			clusters)

		if isnothing(cluster_idx)
			# Start new cluster
			push!(clusters, [sol])
		else
			# Add to existing cluster
			push!(clusters[cluster_idx], sol)
		end
	end

	return clusters
end

"""
	cluster_solutions_identifiable_subspace(sorted_results; rough_eps, subspace_eps) → Vector{Vector}

Two-stage clustering that collapses near-duplicate rows along
practical-non-identifiability axes while preserving algebraically-distinct
basins.

# Stages

1. **Rough basin separation** at relative-distance `rough_eps` (default 1.0):
   uses the same metric as `cluster_solutions` but at a much coarser scale.
   Candidates whose parameter vectors differ by more than ≈`rough_eps` of the
   typical magnitude land in different rough clusters. This step preserves
   genuine basin-to-basin separation (e.g. slow_fast's truth vs mirror).
2. **Within-basin MAD-normalized L∞ merging** at `subspace_eps` (default
   0.05): for each rough cluster of size ≥ 3, compute per-axis MAD across
   the cluster's rows; the L∞ distance in MAD-normalized log/linear space
   then expresses "spread relative to within-basin noise." Axes with large
   MAD (= near-null directions of the residual within this basin) get
   compressed; axes with tight MAD (= identifiable directions) remain
   discriminating. Rough clusters with < 3 rows skip stage 2.

# Why not global MAD-normalization

Global MAD across all candidates is what `_branch_cluster_linf` does in
`_detect_branches`; the 2026-05-14 numbat regression eval found this drops
truth-near candidates in 43% of regression cells because MAD is inflated by
between-basin separation, which then compresses within-basin near-truth
distinctions. The per-basin MAD here is computed only within an
already-similar group, so it reflects practical non-identifiability noise
rather than basin separation.

# Returns

Vector of clusters; within each cluster, sorted by input order (which is
err-ascending upstream). The first element of each cluster is the err-best
candidate.
"""
function cluster_solutions_identifiable_subspace(sorted_results;
	rough_eps::Float64 = 1.0,
	subspace_eps::Float64 = 0.05)
	# Stage 1: rough basin clustering using existing relative-distance metric
	rough_clusters = Vector{Vector{Any}}()
	for sol in sorted_results
		cluster_idx = findfirst(cluster ->
				solution_distance(sol, cluster[1]) < rough_eps,
			rough_clusters)
		if isnothing(cluster_idx)
			push!(rough_clusters, [sol])
		else
			push!(rough_clusters[cluster_idx], sol)
		end
	end

	# Stage 2: within each rough cluster of size ≥ 3, MAD-normalize and re-cluster
	fine_clusters = Vector{Vector{Any}}()
	for rough in rough_clusters
		if length(rough) < 3
			push!(fine_clusters, rough)
			continue
		end
		sub_clusters = _within_basin_subspace_cluster(rough, subspace_eps)
		append!(fine_clusters, sub_clusters)
	end
	return fine_clusters
end

"""
Within-basin MAD-normalized L∞ clustering. Internal helper for
`cluster_solutions_identifiable_subspace`. Operates on a single rough cluster
where global basin-separation is not a concern.
"""
function _within_basin_subspace_cluster(rough_cluster::Vector, subspace_eps::Float64)
	n = length(rough_cluster)
	sample = first(rough_cluster)
	state_keys = collect(keys(sample.states))
	param_keys = collect(keys(sample.parameters))
	all_keys = vcat(state_keys, param_keys)
	d = length(all_keys)
	if d == 0
		return [rough_cluster]
	end

	# Build (n × d) matrix. Use log10 for axes where all entries are positive
	# (matches the log-space polish coordinate convention for positive-bounded
	# parameters); linear otherwise.
	X = Matrix{Float64}(undef, n, d)
	for (i, r) in enumerate(rough_cluster)
		for (j, k) in enumerate(all_keys)
			v = haskey(r.states, k) ? r.states[k] : r.parameters[k]
			X[i, j] = Float64(v)
		end
	end
	for v in 1:d
		col = @view X[:, v]
		if all(c -> c > 0, col)
			X[:, v] .= log10.(col)
		end
	end

	# Per-axis median + MAD (within this rough basin)
	med = [median(X[:, v]) for v in 1:d]
	mad = [median(abs.(X[:, v] .- med[v])) for v in 1:d]
	scale = [max(abs(med[v]), mad[v], 1e-10) for v in 1:d]
	Xn = similar(X)
	for v in 1:d
		Xn[:, v] = (X[:, v] .- med[v]) ./ scale[v]
	end

	# Greedy single-link L∞ clustering in row order
	sub_clusters = Vector{Vector{Any}}()
	rep_rows = Int[]
	for i in 1:n
		merged = false
		for (k, rep_row) in enumerate(rep_rows)
			d_inf = 0.0
			for v in 1:d
				diff = abs(Xn[i, v] - Xn[rep_row, v])
				d_inf = max(d_inf, diff)
				if d_inf >= subspace_eps
					break
				end
			end
			if d_inf < subspace_eps
				push!(sub_clusters[k], rough_cluster[i])
				merged = true
				break
			end
		end
		if !merged
			push!(sub_clusters, [rough_cluster[i]])
			push!(rep_rows, i)
		end
	end
	return sub_clusters
end

function relative_error_value(est, true_val)
	if !isfinite(est)
		return NaN
	elseif abs(true_val) < 1e-6
		return abs(est - true_val)
	else
		return abs((est - true_val) / true_val)
	end
end

function oracle_error_stats(problem::ParameterEstimationProblem, candidate)
	param_names = collect(keys(candidate.parameters))
	non_init_params = filter(p -> !startswith(string(p), "init_"), param_names)
	unident_params = if hasfield(typeof(candidate), :all_unidentifiable)
		Set(candidate.all_unidentifiable)
	else
		Set()
	end

	errorvec = Float64[]

	for (state, value) in candidate.states
		if !(state in unident_params) && haskey(problem.ic, state)
			push!(errorvec, relative_error_value(value, problem.ic[state]))
		end
	end

	for p in non_init_params
		if !(p in unident_params) && haskey(problem.p_true, p)
			push!(errorvec, relative_error_value(candidate.parameters[p], problem.p_true[p]))
		end
	end

	finite_errors = filter(isfinite, errorvec)
	isempty(finite_errors) && return nothing

	return (
		minimum = minimum(finite_errors),
		mean = mean(finite_errors),
		median = median(finite_errors),
		maximum = maximum(finite_errors),
		rms = sqrt(mean(finite_errors .^ 2)),
	)
end

function oracle_sort_key(problem::ParameterEstimationProblem, candidate)
	stats = oracle_error_stats(problem, candidate)
	if isnothing(stats)
		return (Inf, Inf, isnothing(candidate.err) ? Inf : candidate.err)
	end
	return (stats.maximum, stats.median, isnothing(candidate.err) ? Inf : candidate.err)
end

"""
	select_branch_diverse_reps(sorted_reps, M_eff, opts) → Vector

Select up to `M_eff` representatives from an already-ranked candidate list.
When `opts.algebraic_multiplicity > 1`, prefer rows that are separated by
`opts.branch_diversity_eps` in relative solution distance. If there are too
few separated rows, fill remaining slots from the original ranking.

This is a best-effort output contract helper: it avoids returning obvious
near-duplicates when alternatives are already present, but it cannot guarantee
that the alternatives are the true algebraic branches.
"""
function select_branch_diverse_reps(sorted_reps, M_eff::Int, opts::EstimationOptions)
	if M_eff <= 0 || length(sorted_reps) <= M_eff
		return sorted_reps
	end
	if !opts.branch_diversity_selection ||
	   opts.algebraic_multiplicity === nothing ||
	   opts.algebraic_multiplicity <= 1 ||
	   M_eff <= 1 ||
	   opts.branch_diversity_eps <= 0
		return sorted_reps[1:M_eff]
	end

	selected = Any[]
	for candidate in sorted_reps
		if all(solution_distance(candidate, existing) >= opts.branch_diversity_eps for existing in selected)
			push!(selected, candidate)
			length(selected) >= M_eff && return selected
		end
	end

	for candidate in sorted_reps
		any(existing -> existing === candidate, selected) && continue
		push!(selected, candidate)
		length(selected) >= M_eff && return selected
	end
	return selected
end

"""
	rank_cluster_representatives(cluster_reps) -> Vector

Order already-clustered representatives by fit error, best first. Shared by
final result selection and branch-completion anchor selection, so branch
completion chooses the same anchor the normal output path would put first.
"""
rank_cluster_representatives(cluster_reps) = sort(cluster_reps, by = _result_err_key)

"""
	ranked_candidate_representatives(candidates, opts) -> Vector

Score, cluster, annotate cluster sizes, and rank candidates using the same
non-oracle path used for normal returned results. Candidates with non-finite
errors are excluded.
"""
function ranked_candidate_representatives(candidates, opts::EstimationOptions)
	scored = _scored_results(candidates)
	isempty(scored) && return Any[]
	sorted_results = sort(scored, by = _result_err_key)
	clusters = cluster_solutions_identifiable_subspace(sorted_results;
		rough_eps = opts.rough_cluster_eps,
		subspace_eps = opts.subspace_cluster_eps)
	cluster_reps = [first(cluster) for cluster in clusters]
	for (i, c) in enumerate(clusters)
		cluster_reps[i].branch_size = length(c)
	end
	return opts.branch_detection ? rank_cluster_representatives(cluster_reps) : cluster_reps
end

"""
	ranked_full_space_representatives(candidates, opts) -> Vector

Score, cluster in the full parameter/IC coordinate space, annotate cluster
sizes, and rank. Branch completion uses this instead of identifiable-subspace
clustering because algebraic sibling branches are observationally equivalent
by construction and may be collapsed by identifiable-only clustering.
"""
function ranked_full_space_representatives(candidates, opts::EstimationOptions)
	scored = _scored_results(candidates)
	isempty(scored) && return Any[]
	sorted_results = sort(scored, by = _result_err_key)
	clusters = cluster_solutions(sorted_results)
	cluster_reps = [first(cluster) for cluster in clusters]
	for (i, c) in enumerate(clusters)
		cluster_reps[i].branch_size = length(c)
	end
	return opts.branch_detection ? rank_cluster_representatives(cluster_reps) : cluster_reps
end

"""
	_pool_was_branch_completed(results) -> Bool

Return whether the candidate pool contains results produced by successful
algebraic branch completion.

# Arguments
- `results`: Candidate results carrying structured provenance.

# Returns
- `true` when at least one candidate has `source_type == :branch_completed`;
  otherwise `false`.
"""
function _pool_was_branch_completed(results)::Bool
	return any(results) do result
		hasproperty(result, :provenance) || return false
		provenance = result.provenance
		return hasproperty(provenance, :source_type) && provenance.source_type === :branch_completed
	end
end

"""
	_cluster_output_candidates(sorted_results, opts) -> Vector{Vector{Any}}

Apply the configured output-stage clustering policy. Full-coordinate-space
clustering is used whenever algebraic siblings may be present — a pool actually
replaced by branch completion, OR a pool whose detected/declared
`algebraic_multiplicity` is ≥ 2 (siblings can sit in a RAW pool too: the solver
finds both branches organically, and completion can be inactive or fail with a
single anchor — the protection follows the math, not the feature's bookkeeping).
Otherwise candidates are clustered in the identifiable subspace, whether or not
the `branch_completion` feature flag is enabled. Note the M≥2 arm only fires on
an affirmative detection: undetected M degrades to identifiable-subspace
clustering.

# Arguments
- `sorted_results`: Finite-error candidates sorted by ascending fit error.
- `opts`: Estimation configuration (multiplicity and clustering tolerances).

# Returns
- Candidate clusters in input rank order.
"""
function _cluster_output_candidates(sorted_results, opts::EstimationOptions)::Vector{Vector{Any}}
	if _pool_was_branch_completed(sorted_results) ||
	   (opts.algebraic_multiplicity !== nothing && opts.algebraic_multiplicity >= 2)
		return cluster_solutions(sorted_results; threshold = opts.clustering_threshold)
	end
	return cluster_solutions_identifiable_subspace(sorted_results;
		rough_eps = opts.rough_cluster_eps,
		subspace_eps = opts.subspace_cluster_eps)
end


function apply_uq_failure_policy(uq_result, opts::EstimationOptions)
	failed = uq_result isa UncertaintyUnavailable ||
			 (uq_result isa UncertaintyReport && uq_result.status in (:degenerate, :failed))
	if failed && opts.uq_failure_policy == :throw
		outcome = if uq_result isa UncertaintyUnavailable
			uq_result
		else
			UncertaintyUnavailable(
				uq_result.model_name, uq_result.target, :degenerate,
				"computed covariance was classified as $(uq_result.status)",
				copy(uq_result.warnings),
			)
		end
		throw(UncertaintyError(outcome))
	end
	return uq_result
end

_result_err_key(candidate) = (isnothing(candidate.err) || !isfinite(candidate.err)) ? Inf : candidate.err

function _scored_results(results)
	return filter(results) do candidate
		err = candidate.err
		!isnothing(err) && isfinite(err)
	end
end


# UQ kernel dispatch (_uq_kernel_for_result and friends) was deleted 2026-08-13:
# production-uncalled — agp_gpr_uq is SE-only and throws on anything else.
# Matérn UQ returns together with the coverage harness that can validate it.

function _compute_uq_result(
	PEP::ParameterEstimationProblem,
	analysis,
	opts::EstimationOptions,
)
	if !opts.compute_uncertainty
		return nothing
	end

	selected = hasproperty(analysis, :returned_results) ? analysis.returned_results : []
	if isempty(selected)
		if !opts.nooutput
			println("\nSkipping parameter uncertainty: no finite-error solution available.")
		end
		return apply_uq_failure_policy(UncertaintyUnavailable(
			PEP.name, nothing, :no_selected_result,
			"analysis returned no finite rank-one estimator", String[],
		), opts)
	end
	best_solution = first(selected)
	identity = ensure_result_estimator_identity!(best_solution)
	target = UQTargetSnapshot(
		identity = identity,
		lineage = _run_ctx_lineage(identity),
		estimand = :conditional_on_selected_estimator,
		artifact_match = isnothing(_run_ctx_artifact(identity.candidate_id)) ? :missing : :exact,
	)

	if !opts.nooutput
		println("\nComputing parameter uncertainty (experimental)...")
	end

	uq_result = try
		_compute_estimator_aware_uq(PEP, best_solution, target, opts)
	catch e
		_rethrow_if_interrupt(e)
		@warn "Parameter uncertainty computation failed" exception = (e, catch_backtrace())
		UncertaintyUnavailable(
			PEP.name, target, :numerical_failure,
			sprint(showerror, e), String["Estimator-aware UQ threw while processing the selected estimator."],
		)
	end
	uq_result = apply_uq_failure_policy(uq_result, opts)
	if !opts.nooutput && uq_result isa UncertaintyReport
		print_uncertainty_results(uq_result)
	elseif !opts.nooutput && uq_result isa UncertaintyUnavailable
		println("UQ unavailable for rank-one $(identity.estimator_kind): $(uq_result.reason) — $(uq_result.message)")
	end
	return uq_result
end


"""
	analyze_estimation_result(problem::ParameterEstimationProblem, result)

Analyze the results of parameter estimation, including clustering solutions and printing statistics.

# Arguments
- `problem`: The parameter estimation problem
- `result`: Vector of solution results
"""
function analyze_estimation_result(problem::ParameterEstimationProblem, result;
		nooutput = false, opts::EstimationOptions = EstimationOptions())
	for candidate in result
		candidate isa ParameterEstimationResult && ensure_result_estimator_identity!(candidate)
	end
	# Merge dictionaries into a single OrderedDict
	all_params = merge(OrderedDict(), problem.ic, problem.p_true)

	# Print header
	if !nooutput
		println("\n=== Model: $(problem.name) ===")
	end

	scored_results = _scored_results(result)
	sorted_results = if isempty(scored_results)
		Any[]
	else
		sort(scored_results, by = _result_err_key)
	end

	# Cluster ALL candidates first; rely on clustering to produce the set of
	# distinct algebraic-basin representatives. The legacy
	# `err < MAX_ERROR_THRESHOLD` gate silently dropped truth-near candidates
	# whose trajectory loss was high (e.g. on practical-non-identifiability
	# cases like flexible_arm where fit-best is anticorrelated with truth-best).
	# Cluster-first preserves them as separate basin reps.
	#
	# We do NOT cap cluster count: the legacy code's MAX_SOLUTIONS cap applied
	# only to raw candidates in the rare fallback branch (when no candidate had
	# err<0.5). Capping clusters too aggressively drops truth-near reps when the
	# natural cluster count exceeds the cap (flex_arm has 83 natural clusters).
	#
	# Two-stage rough/within-basin clustering collapses near-duplicates along
	# practical-non-identifiability axes (see RECOMMENDATIONS.md Tier 3.2).
	# Successful branch completion is the one exception: its observationally equivalent sibling
	# branches must remain distinct in full coordinate space. Key this exception
	# off result provenance, not the default-on feature flag, because completion
	# is a no-op for M <= 1 and may fail while retaining the original pool.
	clusters = _cluster_output_candidates(sorted_results, opts)

	# Show unidentifiable parameters if any
	if !isempty(sorted_results)
		first_result = first(sorted_results)

		# First show all structurally unidentifiable parameters
		if hasfield(typeof(first_result), :all_unidentifiable) && !isempty(first_result.all_unidentifiable)
			if !nooutput
				println("\nAll structurally unidentifiable parameters:")
				println("-"^50)
				println("These parameters cannot be uniquely determined from the data:")
				for param in first_result.all_unidentifiable
					println("  • $param")
				end
				println()
			end
		end

		# Then show the minimal set of fixed values
		if !isnothing(first_result.unident_dict) && !isempty(first_result.unident_dict)
			if !nooutput
				println("\nMinimal set of fixed values to make remaining parameters identifiable:")
				println("-"^50)
				println("These parameters were fixed to make the system identifiable:")
				for (param, value) in first_result.unident_dict
					# Format the value - handle complex numbers
					val_str = if value isa Complex
						abs(imag(value)) < 1e-10 ?
						@sprintf("%.6f", real(value)) :
						@sprintf("%.3f%+.3fi", real(value), imag(value))
					else
						@sprintf("%.6f", value)
					end
					println("  • $param = $val_str")
				end
				println()
			end
		end
	end

	# Print best solution from each cluster
	if !nooutput
		println("\nFound $(length(clusters)) distinct solution clusters:")
	end
	for (i, cluster) in enumerate(clusters)
		best_solution = first(cluster)  # cluster is sorted by error
		if !nooutput
			println("\nCluster $i: $(length(cluster)) similar solutions")
			println("Best solution (Error: $(round(best_solution.err, digits=6))):")
			if hasfield(typeof(best_solution), :provenance)
				println("Lineage: $(lineage_summary(best_solution))")
			end
			println("-"^50)
		end

		# Get parameters in original order from the model
		param_names = problem.model.original_parameters

		# Filter out init_ parameters from the parameter list since they're already in states
		non_init_params = filter(p -> !startswith(string(p), "init_"), param_names)

		# First get the state names in the order they appear in the model
		state_names = problem.model.original_states

		# Collect values in matching order
		estimates = vcat(
			[best_solution.states[s] for s in state_names],
			[best_solution.parameters[p] for p in non_init_params],
		)
		true_values = vcat(
			[problem.ic[s] for s in state_names],
			[problem.p_true[p] for p in non_init_params],
		)
		var_names = vcat(state_names, non_init_params)

		# Calculate relative errors with proper handling of near-zero values
		rel_errors = map(zip(estimates, true_values)) do (est, true_val)
			relative_error_value(est, true_val)
		end

		# Find maximum lengths for alignment
		max_var_len = maximum(length(string(var)) for var in var_names)
		max_var_len = max(max_var_len, 12) # minimum width for "Variable" header

		# Print table header with dynamic width
		header = @sprintf("%-*s | True Value  | Estimated   | Rel. Error", max_var_len, "Variable")
		if !nooutput
			println(header)
			println("-"^(length(header)))
		end

		# Print states and parameters
		for (var, true_val, est_val, rel_err) in zip(var_names, true_values, estimates, rel_errors)
			# Format the estimated value - handle complex numbers
			est_str = if est_val isa Complex
				abs(imag(est_val)) < 1e-10 ?
				@sprintf("%.6f", real(est_val)) :
				@sprintf("%.3f%+.3fi", real(est_val), imag(est_val))
			else
				@sprintf("%.6f", est_val)
			end

			# Print the row with dynamic width
			if !nooutput
				@printf("%-*s | %10.6f | %10s | %10.6f\n",
					max_var_len, var, true_val, est_str, rel_err)
			end
		end
		if !nooutput
			println()
		end
	end

	# Print best approximation error summary line
	best_approximation_error = isempty(sorted_results) ? Inf : first(sorted_results).err
	if !nooutput
		println("\nBest approximation error for $(problem.name): $(round(best_approximation_error, digits=6))")
	end

	# Calculate and return best error (excluding unidentifiable parameters)
	besterror = Inf
	best_min_error = Inf
	best_mean_error = Inf
	best_median_error = Inf
	best_max_error = Inf
	best_rms_error = Inf

	if !isempty(scored_results)
		for each in result
			stats = oracle_error_stats(problem, each)
			if !isnothing(stats)
				besterror = min(besterror, stats.maximum)
				best_min_error = min(best_min_error, stats.minimum)
				best_mean_error = min(best_mean_error, stats.mean)
				best_median_error = min(best_median_error, stats.median)
				best_max_error = min(best_max_error, stats.maximum)
				best_rms_error = min(best_rms_error, stats.rms)
			end
		end
	end
	if !nooutput
		println("\nBest maximum relative error for $(problem.name) (excluding ALL unidentifiable parameters): $(round(besterror, digits=6))")
		println("Best minimum relative error: $(round(best_min_error, digits=6))")
		println("Best mean relative error: $(round(best_mean_error, digits=6))")
		println("Best median relative error: $(round(best_median_error, digits=6))")
		println("Best maximum relative error: $(round(best_max_error, digits=6))")
		println("Best RMS relative error: $(round(best_rms_error, digits=6))")
	end
	# Return a tuple containing:
	# - Vector of best solutions from each cluster
	# - Best maximum relative error across all results
	# - Best minimum relative error across all results
	# - Best mean relative error across all results
	# - Best median relative error across all results
	# - Best maximum relative error across all results
	# - Best approximation error across all results
	# - Best RMS relative error across all results
	#
	# When opts.branch_detection is true (default), the returned representatives
	# are cluster reps ranked by fit error and capped at min(M, branch_top_k).
	# cluster_reps comes from _cluster_output_candidates (full-space when the pool
	# is branch-completed or M≥2; otherwise
	# cluster_solutions_identifiable_subspace). Each method preserves err
	# order within clusters (first member = err-best). branch_size is annotated below.
	#
	# Historically this stage called _detect_branches to do additional L∞-MAD
	# clustering on identifiable axes and filter by branch_resid_factor / branch_min_size.
	# The 2026-05-14 numbat regression eval (results/numbat_analysis/three_way/) found
	# _detect_branches drops the truth-near candidate in 43% of regression cells:
	# MAD-normalization can compress near-truth and far-from-truth into one cluster,
	# whose err-best rep then misses truth. Skipping it improves recovery from 3% to
	# 77% (within 2× of the legacy "06" benchmark's oracle) on 101 probe cells.
	# _detect_branches was archived 2026-06-09 to deprecated/branch_detection.jl
	# (no longer part of the build); kept there for reference only.
	#
	# When opts.branch_detection is false, fall back to legacy oracle-based sort (the
	# cheat) — retained for old-benchmark reproducibility.
	cluster_reps = [first(cluster) for cluster in clusters]
	for (i, c) in enumerate(clusters)
		cluster_reps[i].branch_size = length(c)
	end
	returned_results = if opts.branch_detection
		# Rank by fit error before top-K slicing (the 2026-06-14 selection study
		# found it near-optimal; the alternative orderings were removed 2026-10-15).
		sorted_reps = rank_cluster_representatives(cluster_reps)
		# Truncation policy:
		#   - `opts.algebraic_multiplicity = nothing` (default) → keep up to
		#     `branch_top_k` rows. Legacy K=20 behavior.
		#   - `opts.algebraic_multiplicity = M` (positive Int) → keep
		#     `min(M, branch_top_k, length(sorted_reps))` rows. M is the
		#     algebraic multiplicity of the problem; output should contain
		#     exactly that many algebraic branches. `branch_top_k` remains
		#     a safety cap in case M is set wrong.
		# See PEB `results/wallaby_analysis/multiplicity/M_INFERENCE_INVESTIGATION.md`
		# for the inference pipeline and the upstream-patch path.
		M_eff = if opts.algebraic_multiplicity === nothing
			opts.branch_top_k
		elseif opts.branch_top_k > 0
			min(opts.algebraic_multiplicity, opts.branch_top_k)
		else
			opts.algebraic_multiplicity
		end
		if M_eff > 0 && length(sorted_reps) > M_eff
			select_branch_diverse_reps(sorted_reps, M_eff, opts)
		else
			sorted_reps
		end
	else
		sort(cluster_reps, by = candidate -> oracle_sort_key(problem, candidate))
	end

	# `all_unidentifiable` is an equation-level invariant of the system, not a
	# per-candidate observation. Some upstream candidate-creating code paths
	# (notably synthesize_aggregates) historically forgot to propagate it, and
	# whichever rep the selection happened to land on then carried an empty set
	# into the output. Union across every input candidate to recover the system-
	# wide flag, then stamp it on each returned rep so consumers see consistent
	# output regardless of which code path produced the winning candidate.
	system_unid = Set{Num}()
	for r in sorted_results
		if hasfield(typeof(r), :all_unidentifiable)
			union!(system_unid, r.all_unidentifiable)
		end
	end
	for rep in returned_results
		rep.all_unidentifiable = system_unid
	end

	# NamedTuple — preserves positional destructure for existing callers
	# (`a, b, ... = analysis` still works) AND exposes algebraic_multiplicity
	# as a named field for new users (`analysis.algebraic_multiplicity`).
	return (
		returned_results = returned_results,
		besterror = besterror,
		best_min_error = best_min_error,
		best_mean_error = best_mean_error,
		best_median_error = best_median_error,
		best_max_error = best_max_error,
		best_approximation_error = best_approximation_error,
		best_rms_error = best_rms_error,
		algebraic_multiplicity = opts.algebraic_multiplicity,
	)
end


"""
	analyze_parameter_estimation_problem(problem, options = EstimationOptions())

Estimate the parameters and initial conditions of `problem` and return everything
the run produced, as `(raw_results, analysis, uq)`.

[`estimate`](@ref) is the short form: it returns `analysis.returned_results`,
without the states the package adds for inputs such as `sin(c * t)`.

# Returns
- `raw_results`: every candidate solution the solver found, before ranking, with
  the identifiability findings. Mostly useful for debugging.
- `analysis`: a named tuple with these fields, in this order, so it can also be
  unpacked by position:
  - `returned_results`: the solutions, best fit first, as
    [`ParameterEstimationResult`](@ref)s.
  - `besterror`, `best_min_error`, `best_mean_error`, `best_median_error`,
    `best_max_error`: relative errors against the problem's true values,
    leaving out the quantities the data cannot determine, each the best over
    all candidates; `besterror` is the same as `best_max_error`.
    They are `Inf` when the true values are not known.
  - `best_approximation_error`: the fit error of the best solution.
  - `best_rms_error`: the root-mean-square relative error, as above.
  - `algebraic_multiplicity`: the number of solutions the equations admit,
    when it was determined.
- `uq`: with `compute_uncertainty = true`, an [`UncertaintyReport`](@ref), or
  an [`UncertaintyUnavailable`](@ref) that says why there is none; otherwise
  `nothing`. Uncertainty estimation is experimental.
"""
function analyze_parameter_estimation_problem(PEP::ParameterEstimationProblem, opts::EstimationOptions = EstimationOptions())
	_validate_observation_options(PEP.data_sample, opts)
	# Validated here, before the quiet scope, so configuration warnings are shown.
	validate_options(opts) || throw(ArgumentError("Invalid EstimationOptions; fix the reported configuration errors before running estimation."))
	return _with_run_scopes(opts) do
		# Establish a per-run RunContext (auto-M hand-off, timing, sinks) unless an
		# outer scope (e.g. with_estimation_timing) already bound one.
		if _run_ctx() === nothing
			value, _ = _with_run_context(() -> _analyze_parameter_estimation_problem_impl(PEP, opts))
			return value
		end
		return _analyze_parameter_estimation_problem_impl(PEP, opts)
	end
end

function _analyze_parameter_estimation_problem_impl(PEP::ParameterEstimationProblem, opts::EstimationOptions)
	# Checked on the model as it was written, before it is transformed and
	# rescaled, so that an error quotes the caller's own terms.
	opts.flow == FlowStandard && validate_supported_model_class(PEP)
	_run_ctx_begin_uq!(opts.compute_uncertainty)
	# Carry the per-analysis HC solver config to _hc_solve via the bound context.
	_run_ctx_set_hc_opts!(opts.hc_threading, opts.hc_compile_mode)

	# Auto-handle transcendental functions (sin/cos/exp) at the top level
	# so the transformed PEP is used consistently for both estimation and analysis
	if opts.auto_handle_transcendentals
		t_var = ModelingToolkit.get_iv(PEP.model.system)
		PEP, _tr_info = transform_pep_for_estimation(PEP, t_var)
		if !isnothing(_tr_info) && !opts.nooutput
			println("Transcendental handling: transformed $(length(_tr_info.entries)) expression(s) into polynomial form")
		end
	end

	# Optional power-of-2 problem rescaling (opt-in). Composed AFTER the transcendental
	# transform so it sees the final symbol set; results are un-rescaled at the very end
	# (after clustering/ranking/UQ, which are self-consistent in scaled space).
	scale_info = nothing
	if opts.auto_rescale
		PEP, scale_info = rescale_pep(PEP)
		if !isnothing(scale_info)
			# Transform any user-supplied opt_lb/opt_ub into scaled coordinates so a
			# physical bound keeps its physical meaning (all bound consumers use opts).
			opts = rescale_option_bounds(opts, scale_info, PEP)
			if !opts.nooutput
				println("Auto-rescale: power-of-2 scaling ($(scale_info.metadata.n_nontrivial) nontrivial factors; max |log2 const| $(round(scale_info.metadata.max_const_before, digits = 1)) → $(round(scale_info.metadata.max_const_after, digits = 1)))")
			end
		end
	end

	if !opts.nooutput
		println("Starting model: ", PEP.name)
	end
	_heartbeat_run_header(opts, PEP.name)

	# Initialize variables outside try blocks
	solved_res = []
	unident_dict = Dict()
	trivial_dict = Dict()
	all_unidentifiable = []

	# Choose workflow based on opts.flow
	if opts.flow == FlowStandard
		if !opts.nooutput
			println("Using NEW optimized parameter estimation flow")
		end
		results_tuple = optimized_multishot_parameter_estimation(PEP, opts)
	elseif opts.flow == FlowDirectOpt
		if !opts.nooutput
			println("Using direct optimization workflow (LBFGS)")
		end
		# Align return signature with other workflows: (solutions, unident_dict, trivial_dict, all_unidentifiable)
		local direct_results = direct_optimization_parameter_estimation(PEP; opts = opts)
		results_tuple = (direct_results, Dict(), Dict(), [])
	else
		error("Unknown EstimationFlow: $(opts.flow)")
	end
	solved_res, unident_dict, trivial_dict, all_unidentifiable = results_tuple

	if !opts.nooutput
		println("\nUnidentifiability analysis from the algebraic estimation workflow:")
		println("All unidentifiable variables: ", all_unidentifiable)
		println("Unidentifiable variables substitution dictionary: ", unident_dict)
		println("Trivially solvable variables: ", trivial_dict)
	end

	# Auto-populate algebraic_multiplicity if the caller left it unset and the
	# SI template build produced a value (RunContext auto-M hand-off; see
	# run_context.jl). When set explicitly by the caller, we never override.
	# CONSUME-ONCE via _run_ctx_take_auto_m!() so a value written by this run's
	# SI estimation can never leak into a later analysis sharing an outer
	# scope (e.g. the direct-opt flow under one with_estimation_timing block).
	auto_m = _run_ctx_take_auto_m!()
	if isnothing(opts.algebraic_multiplicity) && !isnothing(auto_m)
		opts = EstimationOptions(;
			(name => getfield(opts, name) for name in fieldnames(EstimationOptions) if name !== :algebraic_multiplicity)...,
			algebraic_multiplicity = auto_m,
		)
		if !opts.nooutput
			println("Auto-detected algebraic multiplicity M = $(opts.algebraic_multiplicity); result.csv truncated to min(M, branch_top_k) rows.")
		end
	end

	_heartbeat(opts, "Clustering + analysis")
	results_tuple_to_return = analyze_estimation_result(PEP, solved_res, nooutput = opts.nooutput, opts = opts)
	_heartbeat(opts, "Clustering + analysis"; kind = :done)

	opts.compute_uncertainty && _heartbeat(opts, "UQ (GP/IFT sidecar)")
	uq_result = _compute_uq_result(PEP, results_tuple_to_return, opts)
	opts.compute_uncertainty && _heartbeat(opts, "UQ (GP/IFT sidecar)"; kind = :done)
	if !isnothing(scale_info)
		uq_result = unrescale_uncertainty_report(uq_result, scale_info)
	end
	_heartbeat(opts, "run"; kind = :done)

	# Un-rescale results to original units (params/states). Done AFTER clustering/
	# ranking (analyze_estimation_result) and UQ, which are self-consistent in scaled
	# space. The returned cluster reps alias the solved_res objects, so un-rescaling
	# solved_res un-rescales them too; unrescale_results is idempotent (via a
	# :rescaled provenance note) so the aliasing can't double-scale.
	if !isnothing(scale_info)
		unrescale_results(solved_res, scale_info)
	end

	return results_tuple, results_tuple_to_return, uq_result

end


# _compile_system_function: a general symbolic->callable compiler (Symbolics.build_function
# wrapper). Lives here, included before all callers (multipoint_template #86,
# noise_frontier_construction #87, diagnostics #98/#99, research/consensus #107), so every
# caller resolves it forward. Relocated out of diagnostics 2026-06-10.
"""
Compile a vector of symbolic equations into a callable `f(x) → Vector{Float64}`
via `Symbolics.build_function`. Falls back to substitution-based evaluation.
The compiled function is compatible with ForwardDiff dual numbers.
"""
function _compile_system_function(equations, varlist)
    try
        # build_function with expression=Val(false) returns a compiled Julia function
        fn = Symbolics.build_function(equations, varlist; expression = Val(false))
        # build_function returns (out-of-place, in-place); take out-of-place
        f_oop = fn isa Tuple ? fn[1] : fn
        return f_oop
    catch e
        _rethrow_if_interrupt(e)
        @warn "[compile_system_function] build_function failed, using substitution fallback: $e"
        # Fallback: closure over symbolic substitution (works but no AD)
        return function (vals)
            subst_dict = Dict(varlist[i] => vals[i] for i in eachindex(varlist))
            result = zeros(eltype(vals), length(equations))
            for (i, eq) in enumerate(equations)
                result[i] = try
                    Float64(Symbolics.value(Symbolics.substitute(eq, subst_dict)))
                catch err
                    _rethrow_if_interrupt(err)
                    eltype(vals)(NaN)
                end
            end
            return result
        end
    end
end
