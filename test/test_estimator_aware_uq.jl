using ODEParameterEstimation
using ODEParameterEstimation: AbstractInterpolator, UncertaintyError, uq_metadata_dict,
    uq_reliability
using Test
using Logging
using LinearAlgebra
using OrderedCollections
using ModelingToolkit
using JSON

@testset "estimator-aware UQ target contract" begin
    pep = ODEParameterEstimation.Examples.harmonic_oscillator()
    params = collect(keys(pep.p_true))
    states = collect(keys(pep.ic))

    function candidate(err::Float64)
        return ODEParameterEstimation.ParameterEstimationResult(
            OrderedDict{Num, Float64}(p => Float64(pep.p_true[p]) for p in params),
            OrderedDict{Num, Float64}(s => Float64(pep.ic[s]) for s in states),
            0.0, err, :Success, 3, 0.0,
            OrderedDict{Num, Float64}(), Set{Num}(), nothing,
        )
    end

    @testset "rank-one analysis row is the only target" begin
        outcome, _ = ODEParameterEstimation._with_run_context() do
            ODEParameterEstimation._run_ctx_set_capture_uq!(true)
            lowest_sse = candidate(1e-8)
            rank_one = candidate(1e-3)
            low_id = ODEParameterEstimation.set_result_estimator_identity!(lowest_sse;
                estimator_kind = :single_point_algebraic,
                data_scope = :point_set,
                time_indices = [1], time_values = [0.0],
                interpolator_source = :agp_uq)
            rank_id = ODEParameterEstimation.set_result_estimator_identity!(rank_one;
                estimator_kind = :multipoint_algebraic,
                data_scope = :point_set,
                time_indices = [1, 3], time_values = [0.0, 1.0],
                interpolator_source = :agp_uq)
            @test low_id.candidate_id != rank_id.candidate_id

            analysis = (returned_results = Any[rank_one],)
            ODEParameterEstimation._compute_uq_result(
                pep, analysis,
                EstimationOptions(compute_uncertainty = true, nooutput = true),
            )
        end
        @test outcome isa UncertaintyUnavailable
        @test outcome.reason == :missing_artifact
        @test outcome.target.selected_rank == 1
        @test outcome.target.identity.estimator_kind == :multipoint_algebraic
        @test outcome.target.identity.time_indices == [1, 3]
        @test outcome.target.identity.time_values == [0.0, 1.0]
    end

    @testset "identity lineage and scoped artifact registry" begin
        observed, _ = ODEParameterEstimation._with_run_context() do
            parent = candidate(0.1)
            child = candidate(0.01)
            parent_id = ODEParameterEstimation.set_result_estimator_identity!(parent;
                estimator_kind = :single_point_algebraic, data_scope = :point_set,
                time_indices = [2], time_values = [0.5])
            child_id = ODEParameterEstimation.set_result_estimator_identity!(child;
                estimator_kind = :trajectory_polish, data_scope = :full_trajectory,
                parent_candidate_ids = [parent_id.candidate_id])
            lineage = ODEParameterEstimation._run_ctx_lineage(child_id)
            @test [x.candidate_id for x in lineage] ==
                [child_id.candidate_id, parent_id.candidate_id]

            artifact = ODEParameterEstimation.BranchCompletionUQArtifact(
                parent_id.candidate_id, (tag = :test,),
            )
            # Heavy artifacts are not retained until UQ capture is explicitly on.
            ODEParameterEstimation._run_ctx_register_artifact!(child_id.candidate_id, artifact)
            @test isnothing(ODEParameterEstimation._run_ctx_artifact(child_id.candidate_id))
            ODEParameterEstimation._run_ctx_set_capture_uq!(true)
            ODEParameterEstimation._run_ctx_register_artifact!(child_id.candidate_id, artifact)
            @test ODEParameterEstimation._run_ctx_artifact(child_id.candidate_id) === artifact
            (parent_id, child_id)
        end
        @test isnothing(ODEParameterEstimation._run_ctx())
        @test observed[1].candidate_id > 0
    end

    @testset "logical runs cannot reuse prior UQ state" begin
        reset_state, _ = ODEParameterEstimation._with_run_context() do
            ODEParameterEstimation._run_ctx_begin_uq!(true)
            old = candidate(0.2)
            old_id = ODEParameterEstimation.set_result_estimator_identity!(old;
                estimator_kind = :branch_completed, data_scope = :derived)
            old_artifact = ODEParameterEstimation.BranchCompletionUQArtifact(
                old_id.candidate_id, (tag = :old,),
            )
            ODEParameterEstimation._run_ctx_register_artifact!(
                old_id.candidate_id, old_artifact,
            )
            ODEParameterEstimation._run_ctx_register_noise_interpolants!(
                :agp_uq, Dict{Num, AbstractInterpolator}(),
            )

            ODEParameterEstimation._run_ctx_begin_uq!(true)
            new = candidate(0.1)
            new_id = ODEParameterEstimation.set_result_estimator_identity!(new;
                estimator_kind = :single_point_algebraic, data_scope = :point_set)
            (
                old_artifact = ODEParameterEstimation._run_ctx_artifact(old_id.candidate_id),
                old_identity = ODEParameterEstimation._run_ctx_identity(old_id.candidate_id),
                noise = ODEParameterEstimation._run_ctx_noise_interpolants(:agp_uq),
                new_id = new_id.candidate_id,
            )
        end
        @test isnothing(reset_state.old_artifact)
        # Candidate IDs restart at one, so this lookup now refers only to the
        # freshly registered identity, never to the old run's identity.
        @test reset_state.old_identity.estimator_kind == :single_point_algebraic
        @test isnothing(reset_state.noise)
        @test reset_state.new_id == 1
    end

    @testset "legacy defaults stay unknown and UQ capture is non-invasive" begin
        legacy = candidate(0.2)
        legacy_identity, _ = ODEParameterEstimation._with_run_context() do
            ODEParameterEstimation.ensure_result_estimator_identity!(legacy)
        end
        @test legacy_identity.estimator_kind == :unknown
        @test legacy_identity.data_scope == :unknown

        captured, _ = ODEParameterEstimation._with_run_context() do
            ODEParameterEstimation._run_ctx_set_capture_uq!(true)
            with_logger(NullLogger()) do
                ODEParameterEstimation._run_ctx_try_uq_capture("synthetic capture failure") do
                    error("capture-only failure")
                end
            end
        end
        @test isnothing(captured)
        @test_throws InterruptException ODEParameterEstimation._with_run_context() do
            ODEParameterEstimation._run_ctx_set_capture_uq!(true)
            ODEParameterEstimation._run_ctx_try_uq_capture("interrupt") do
                throw(InterruptException())
            end
        end
    end

    @testset "retained branch roots follow the production HC ordering" begin
        @variables branch_u branch_v
        payload = (
            solve_vars = Num[branch_u, branch_v],
            root = [11.0, 22.0],
        )
        reordered = ODEParameterEstimation._uq_retained_branch_root(
            payload, Num[branch_v, branch_u],
        )
        @test reordered == [22.0, 11.0]

        ambiguous = (solve_vars = Num[branch_u, branch_u], root = [1.0, 2.0])
        @test_throws ArgumentError ODEParameterEstimation._uq_retained_branch_root(
            ambiguous, Num[branch_u, branch_u],
        )
    end

    @testset "typed failure, throw policy, and sidecar metadata" begin
        rank_one = candidate(0.1)
        analysis = (returned_results = Any[rank_one],)
        unavailable, _ = ODEParameterEstimation._with_run_context() do
            ODEParameterEstimation._run_ctx_set_capture_uq!(true)
            ODEParameterEstimation.set_result_estimator_identity!(rank_one;
                estimator_kind = :synthesized_aggregate, data_scope = :ensemble)
            ODEParameterEstimation._compute_uq_result(
                pep, analysis,
                EstimationOptions(compute_uncertainty = true, nooutput = true),
            )
        end
        @test unavailable isa UncertaintyUnavailable
        @test uq_metadata_dict(unavailable)["outcome"] == "unavailable"
        @test uq_metadata_dict(unavailable)["target"]["selected_rank"] == 1
        unavailable_reliability = uq_reliability(unavailable)
        @test unavailable_reliability.availability == :unavailable
        @test unavailable_reliability.selection_scope == :conditional_on_selected_estimator
        @test isnothing(uq_metadata_dict(nothing))

        @test_throws UncertaintyError ODEParameterEstimation.apply_uq_failure_policy(
            unavailable,
            EstimationOptions(compute_uncertainty = true,
                uq_failure_policy = :throw, nooutput = true),
        )
        @test startswith(sprint(showerror, UncertaintyError(unavailable)), "Uncertainty quantification failed [")

        report = UncertaintyReport(
            "json_sidecar", 0.0,
            String[], Vector{Float64}[], Vector{Float64}[],
            zeros(0, 0), String[],
            reshape([1.0], 1, 1), [1.0], ["p"], Dict("p" => :parameter),
            [NaN], reshape([1.0], 1, 1), Inf, :degenerate, String[],
            :estimator_sampling, :learned_gp_homoscedastic, nothing,
        )
        metadata = uq_metadata_dict(report)
        @test metadata["truth_values"] == Any[nothing]
        @test isnothing(metadata["max_cv"])
        @test isnothing(metadata["linearization"]["gradient_norm"])
        @test isnothing(metadata["linearization"]["jacobian_condition_equilibrated"])
        @test isnothing(metadata["linearization"]["linear_solve_backward_error"])
        @test metadata["reliability"]["availability"] == "available"
        @test metadata["reliability"]["interval_width"] == "undefined_scale"
        @test JSON.json(metadata) isa String
    end

    @testset "operational CV never treats missing truth/centers as zero" begin
        @test isinf(ODEParameterEstimation._uq_max_cv([0.1, 0.2], [NaN, NaN]))
        @test ODEParameterEstimation._uq_max_cv([0.1], [2.0]) == 0.05
    end

    @testset "smoother residual EDF noise is explicit and typed" begin
        xs = collect(range(0.0, 1.0; length = 21))
        ys = @. exp(0.4 * xs) + 1e-3 * sin(17 * xs)
        interp = with_logger(NullLogger()) do
            ODEParameterEstimation.agp_gpr_uq(xs, ys)
        end
        H = Matrix{Float64}(undef, length(xs), length(xs))
        for (i, x) in enumerate(xs)
            _, W = ODEParameterEstimation.gp_derivative_influence_matrix(
                interp, x, 0,
            )
            H[i, :] .= @view W[1, :]
        end
        raw = ODEParameterEstimation._raw_training_values(interp)
        residual = raw - H * raw
        df = length(xs) - 2 * tr(H) + tr(H' * H)
        sigma2 = sum(abs2, residual) / df
        covariance = ODEParameterEstimation._uq_observation_covariance_from_source(
            interp, :smoother_residual_edf,
        )
        @test Matrix(covariance) ≈ sigma2 .* Matrix{Float64}(I, length(xs), length(xs)) rtol = 5e-13
        @test ishermitian(Matrix(covariance))
        @test minimum(eigvals(Symmetric(Matrix(covariance)))) >= 0

        estimate = ODEParameterEstimation.joint_derivative_estimator_covariance(
            interp, [0.4], 1;
            noise_source = :smoother_residual_edf,
            observation_covariance = covariance,
        )
        @test estimate.noise_source == :smoother_residual_edf
        @test_throws ArgumentError ODEParameterEstimation._uq_observation_covariance_from_source(
            interp, :not_a_noise_source,
        )
        @test_throws ArgumentError ODEParameterEstimation._uq_residual_edf_variance(
            Matrix{Float64}(I, length(xs), length(xs)), raw,
        )
    end

    @testset "trajectory calculus helpers recover the retained score" begin
        x = [0.3, -0.7]
        objective = u -> (u[1] - 1)^2 + 3 * (u[2] + 2)^2 + 0.5 * u[1] * u[2]
        gradient, hessian = ODEParameterEstimation._uq_finite_difference_score_hessian(
            objective, x,
        )
        expected_gradient = [
            2 * (x[1] - 1) + 0.5 * x[2],
            6 * (x[2] + 2) + 0.5 * x[1],
        ]
        @test gradient ≈ expected_gradient atol = 1e-8 rtol = 1e-8
        @test hessian ≈ [2.0 0.5; 0.5 6.0] atol = 1e-6 rtol = 1e-6

        predictions = [1.0, -2.0, 0.5]
        observations = [0.8, -2.4, 0.1]
        jacobian = [1.0 2.0; -1.0 0.5; 0.25 -3.0]
        penalty_gradient = [0.1, -0.2]
        score = ODEParameterEstimation._uq_least_squares_score(
            predictions, observations, jacobian, penalty_gradient,
        )
        @test score ≈ 2 .* (jacobian' * (predictions .- observations)) .+
            penalty_gradient
    end
end
