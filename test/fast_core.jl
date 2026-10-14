using ODEParameterEstimation
using Symbolics
using Test
using Logging
using HomotopyContinuation
using ModelingToolkit
using OrderedCollections
using Random

@testset "Fast Core Contracts" begin
    @testset "Ordered ODE construction" begin
        @independent_variables t
        @parameters a b
        @variables x1(t) x2(t) y1(t) y2(t)
        D = Differential(t)

        eqs = [
            D(x1) ~ -a * x2,
            D(x2) ~ b * x1,
        ]
        states = [x1, x2]
        params = [a, b]
        measured_quantities = [y1 ~ x1, y2 ~ x2]

        ordered_system, mq = ODEParameterEstimation.create_ordered_ode_system(
            "TestSystem", states, params, eqs, measured_quantities,
        )
        time_var, equations, state_vars, parameters = ODEParameterEstimation.unpack_ODE(ordered_system.system)

        @test ordered_system isa ODEParameterEstimation.OrderedODESystem
        @test mq == measured_quantities
        @test isequal(time_var, t)
        @test length(equations) == 2
        @test string.(state_vars) == string.(states)
        @test string.(parameters) == string.(params)
        @test string.(ordered_system.original_parameters) == string.(params)
        @test string.(ordered_system.original_states) == string.(states)
    end

    @testset "Option helpers" begin
        base = EstimationOptions(
            flow = FlowStandard,
            shooting_points = 3,
            save_system = false,
        )
        merged = ODEParameterEstimation.merge_options(
            base;
            shooting_points = 2,
            interpolators = [InterpolatorAGPRobust, InterpolatorS3AdaptSE],
            terminal_fallback = :direct_opt,
            backsolve_recovery = :algebraic_resolve,
            t0_state_completion = :strict,
        )

        @test merged.flow == FlowStandard
        @test merged.shooting_points == 2
        @test ODEParameterEstimation.validate_options(merged)

        resolved = ODEParameterEstimation.resolve_interpolator_list(merged)
        @test length(resolved) == 2
        @test first.(resolved) == [InterpolatorAGPRobust, InterpolatorS3AdaptSE]

        custom_interp = (xs, ys) -> ODEParameterEstimation.aaad(xs, ys)
        custom_opts = EstimationOptions(
            interpolators = [InterpolatorCustom],
            custom_interpolators = Function[custom_interp],
        )
        resolved_custom = ODEParameterEstimation.resolve_interpolator_list(custom_opts)
        @test length(resolved_custom) == 1
        @test first(resolved_custom)[1] == InterpolatorCustom
        @test first(resolved_custom)[2] === custom_interp

        plural_custom_opts = EstimationOptions(
            interpolators = [InterpolatorCustom, InterpolatorAAAD, InterpolatorCustom],
            custom_interpolators = Function[custom_interp, custom_interp],
        )
        @test ODEParameterEstimation.validate_options(plural_custom_opts)
        # An empty list is refused: there is no single-interpolator option to fall back on.
        @test !ODEParameterEstimation.validate_options(EstimationOptions(interpolators = InterpolatorMethod[]))
        @test !ODEParameterEstimation.validate_options(EstimationOptions(interpolators = [InterpolatorCustom]))
        @test !ODEParameterEstimation.validate_options(EstimationOptions(
            interpolators = [InterpolatorCustom, InterpolatorCustom],
            custom_interpolators = Function[custom_interp],
        ))
        @test !ODEParameterEstimation.validate_options(EstimationOptions(opt_ad_backend = :bogus))

        @test ODEParameterEstimation.compute_shooting_indices(0, 21) == [10]
        @test ODEParameterEstimation.compute_shooting_indices(3, 21; warp = false) == [1, 11, 21]
        @test merged.terminal_fallback == :direct_opt
        @test merged.backsolve_recovery == :algebraic_resolve
        @test merged.t0_state_completion == :strict
        branch_completion_opts = EstimationOptions(
            branch_completion = true,
            branch_completion_max_anchors = 2,
            branch_completion_residual_tol = 1e-7,
        )
        @test branch_completion_opts.branch_completion
        @test ODEParameterEstimation.validate_options(branch_completion_opts)

        @test instances(EstimationFlow) == (FlowStandard, FlowDirectOpt)
        @test !isdefined(ODEParameterEstimation, :FlowDeprecated)
        @test !isdefined(ODEParameterEstimation, :multipoint_parameter_estimation)
        @test !isdefined(ODEParameterEstimation, :multishot_parameter_estimation)

        @test_throws MethodError EstimationOptions(try_more_methods = true)
        @test_throws ErrorException ODEParameterEstimation.merge_options(base; try_more_methods = true)
        # terminal_fallback + FlowDirectOpt is redundant but not invalid (just warns)
        redundant_terminal_fallback = EstimationOptions(
            flow = FlowDirectOpt,
            terminal_fallback = :direct_opt,
        )
        @test ODEParameterEstimation.validate_options(redundant_terminal_fallback)

        invalid_seed_policy = EstimationOptions(t0_state_completion = :maybe)
        @test !ODEParameterEstimation.validate_options(invalid_seed_policy)

        invalid_uq_policy = EstimationOptions(uq_failure_policy = :maybe)
        @test !ODEParameterEstimation.validate_options(invalid_uq_policy)

        invalid_placeholder_policy = EstimationOptions(si_placeholder_fail_categories = [:not_a_real_category])
        @test !ODEParameterEstimation.validate_options(invalid_placeholder_policy)
        @test !ODEParameterEstimation.validate_options(EstimationOptions(branch_completion_max_anchors = 0))
        @test !ODEParameterEstimation.validate_options(EstimationOptions(branch_completion_residual_tol = -1.0))
        @test EstimationOptions().system_construction_policy == :noise_frontier
        frontier_opts = EstimationOptions(
            system_construction_policy = :noise_frontier,
            construction_candidate_limit = 8,
            construction_beam_width = 4,
            construction_compute_mixed_volume = false,
        )
        @test ODEParameterEstimation.validate_options(frontier_opts)
        @test !ODEParameterEstimation.validate_options(EstimationOptions(system_construction_policy = :unknown))
        @test !ODEParameterEstimation.validate_options(EstimationOptions(construction_candidate_limit = 0))
        @test !ODEParameterEstimation.validate_options(EstimationOptions(construction_beam_width = 0))

        @test_throws ErrorException ODEParameterEstimation.merge_options(base; definitely_not_an_option = true)

        # Names that were once exported with nothing defined behind them.
        public_names = names(ODEParameterEstimation; all = false, imported = false)
        @test :solve_with_monodromy ∉ public_names
        @test :aaad_in_testing ∉ public_names
        # `estimate` was a third, until it became the short entry point (2026-10-06).
        @test :estimate in public_names
        @test ODEParameterEstimation.estimate isa Function
    end

    @testset "Transcendental data variable evaluation" begin
        t0 = 1.054739652870494
        bare_sin = "_obs_trfn_cos_5_0_sin(t)"
        d1_cos = "Differential(t, 1)(_obs_trfn_cos_5_0_cos(t))"
        d4_sin = "Differential(t, 4)(_obs_trfn_cos_5_0_sin(t))"

        @test ODEParameterEstimation.evaluate_known_trfn_variable("_trfn_sin_0_5", 1.0) ≈ sin(0.5)
        @test ODEParameterEstimation.evaluate_known_trfn_variable("_trfn_cos_0_5", 1.0) ≈ cos(0.5)
        @test ODEParameterEstimation.evaluate_known_trfn_variable("_trfn_sin_0_5_1", 1.0) ≈ 0.5 * cos(0.5)
        @test ODEParameterEstimation.evaluate_known_trfn_variable("_obs_trfn_exp_0_5", 1.0) ≈ exp(0.5)
        @test ODEParameterEstimation.evaluate_known_trfn_variable(bare_sin, t0) ≈ sin(5.0 * t0)
        @test ODEParameterEstimation.evaluate_known_trfn_variable(d1_cos, t0) ≈ -5.0 * sin(5.0 * t0)
        @test ODEParameterEstimation.evaluate_known_trfn_variable(d4_sin, t0) ≈ 5.0^4 * sin(5.0 * t0)

        data_vars = Any[bare_sin, d1_cos, d4_sin]
        fake_DD = (obs_lhs = [data_vars],)
        vals = ODEParameterEstimation.evaluate_data_vars_at_point(
            Dict{Any, Any}(),
            data_vars,
            fake_DD,
            ModelingToolkit.Equation[],
            t0,
        )
        @test vals ≈ [
            sin(5.0 * t0),
            -5.0 * sin(5.0 * t0),
            5.0^4 * sin(5.0 * t0),
        ]
    end

    @testset "Template interpolation ignores unused SIAN jets" begin
        @independent_variables t
        @variables x(t) y(t) x0
        D = Differential(t)
        # Reproduce a parent ring supporting order 26 while the retained system
        # needs only y'. The interpolator's numeric backend supports <= 20.
        lhs = [Num[y]]
        for _ in 1:26
            push!(lhs, Symbolics.expand_derivatives.(D.(last(lhs))))
        end
        dd = (obs_lhs = lhs,)
        bookkeeping = Dict(only(level) => i - 1 for (i, level) in enumerate(lhs))
        data = OrderedDict{Any, Any}("t" => [0.0, 1.0, 2.0])
        interps = Dict{Any, Any}(Symbolics.diff2term(x) => (τ -> 2.0 * τ + 1.0))
        instantiate(equations) = ODEParameterEstimation.instantiate_si_template_equations(
            equations, [y ~ x], data, bookkeeping, dd;
            interpolants = interps, time_index = 2, prune_overdetermined = false)

        inst = instantiate(Num[x0 + only(lhs[2])])
        @test inst.source_indices == [1]
        @test isequal(only(inst.equations), x0 + 2.0)
        @test Set(Num.(inst.vars)) == Set([x0])

        # A genuinely required unsupported derivative must still fail visibly.
        @test_throws ODEParameterEstimation.UnsupportedDerivativeOrderError instantiate(
            Num[x0 + only(lhs[22])])
    end

    @testset "Transcendental template pruning preserves equation provenance" begin
        @independent_variables t
        @variables x(t) y(t) y0 y1 x0

        trfn_sin = Symbolics.variable(Symbol("_trfn_sin_0_5"))
        trfn_cos = Symbolics.variable(Symbol("_trfn_cos_0_5"))
        template_equations = Num[
            x0 + y0,
            trfn_sin^2 + trfn_cos^2 - 1,
            x0^2 + y1,
        ]
        measured_quantities = [y ~ x]
        data_sample = OrderedDict{Any, Any}("t" => [0.0, 1.0, 2.0])
        interpolants = Dict{Any, Any}(Symbolics.diff2term(x) => (τ -> 2.0 * τ + 1.0))
        template_DD = (obs_lhs = [[y0], [y1]],)
        derivative_dict = Dict{Any, Int}(y0 => 0, y1 => 1)

        inst = with_logger(NullLogger()) do
            ODEParameterEstimation.instantiate_si_template_equations(
                template_equations,
                measured_quantities,
                data_sample,
                derivative_dict,
                template_DD;
                interpolants = interpolants,
                time_index = 2,
                prune_overdetermined = false,
                substitute_trfn = true,
            )
        end

        @test length(inst.equations) == 2
        @test inst.source_indices == [1, 3]
        @test length(inst.trivial_residuals) == 1
        @test abs(only(inst.trivial_residuals)) < 1e-12

        combined_inst = Any[]
        combined_symb = Any[]
        source_indices = Int[]
        points = Int[]
        ODEParameterEstimation._noise_append_source_mapped_equations!(
            combined_inst,
            combined_symb,
            source_indices,
            points,
            inst.equations,
            template_equations,
            inst.source_indices,
            2;
            diagnostics = false,
            context = "test",
        )
        @test string.(combined_symb) == string.([template_equations[1], template_equations[3]])
        @test source_indices == [1, 3]
        @test points == [2, 2]
    end

    @testset "Surplus template equations are dropped from the end" begin
        @independent_variables t
        @variables x(t) y(t) y0 y1 x0

        # y = x = 2t + 1, so at t = 1: y0 = 3, y1 = 2, and every equation says x0 = 3.
        template_equations = Num[x0 - y0, 2 * x0 - 3 * y1, x0^2 - 3 * y0]
        inst = with_logger(NullLogger()) do
            ODEParameterEstimation.instantiate_si_template_equations(
                template_equations,
                [y ~ x],
                OrderedDict{Any, Any}("t" => [0.0, 1.0, 2.0]),
                Dict{Any, Int}(y0 => 0, y1 => 1),
                (obs_lhs = [[y0], [y1]],);
                interpolants = Dict{Any, Any}(Symbolics.diff2term(x) => (τ -> 2.0 * τ + 1.0)),
                time_index = 2,
                prune_overdetermined = true,
            )
        end

        @test inst.source_indices == [1]
        @test isequal(only(inst.equations), x0 - 3.0)
        @test Set(Num.(inst.vars)) == Set([x0])
    end

    @testset "Re-solve fallback: substitute single-unknown equations, then solve the rest" begin
        @variables x1 x2 x3
        cascade(equations, vars) = with_logger(NullLogger()) do
            ODEParameterEstimation._cascade_resolve(equations, vars, OrderedDict{Symbol, Float64}())
        end
        values_of(result, k) = Dict(zip(string.(result.state_vars), result.solutions[k]))

        # x1 alone; then x1 and x2 in turn.
        full = cascade(Num[x1 - 2, x1 * x2 - 6], Any[x1, x2])
        @test full.cascading_status == :full
        @test full.status == :cascade_full_solution
        @test values_of(full, 1)["x1"] ≈ 2.0
        @test values_of(full, 1)["x2"] ≈ 3.0

        # x1 = 2, and homotopy continuation solves the two left: x3 = -1 ± √5, x2 = x3 + 1.
        system = Num[x1 - 2, x1 * x2 + x3^2 - 6, x2 - x3 - 1]
        merged = cascade(system, Any[x1, x2, x3])
        @test merged.cascading_status == :merged
        @test merged.status == :cascade_hc_merged
        @test length(merged.solutions) == 2
        @test all(isempty, merged.missing_vars_per_solution)
        for k in 1:2
            v = values_of(merged, k)
            @test v["x1"] ≈ 2.0
            @test v["x2"] ≈ v["x3"] + 1
            @test v["x1"] * v["x2"] + v["x3"]^2 ≈ 6
        end
        @test sort([values_of(merged, k)["x3"] for k in 1:2]) ≈ [-1 - sqrt(5), -1 + sqrt(5)]

        # x1 = 2 leaves one equation in two unknowns: a partial answer naming the rest.
        partial = cascade(Num[x1 - 2, x1 * x2 + x3 - 5], Any[x1, x2, x3])
        @test partial.cascading_status == :partial
        @test partial.status == :cascade_partial_solution
        @test string.(partial.state_vars) == ["x1"]
        @test only(only(partial.solutions)) ≈ 2.0
        @test Set(string.(only(partial.missing_vars_per_solution))) == Set(["x2", "x3"])
        @test :remaining_system_underdetermined in partial.notes

        # No equation has a single unknown.
        none = cascade(Num[x1 * x2 - 1], Any[x1, x2])
        @test none.cascading_status == :failed
        @test isempty(none.solutions)
        @test :cascade_found_no_substitutions in none.notes
    end

    @testset "HC conversion accepts empty parameter lists" begin
        @variables x
        hc_system, hc_variables, hc_params = ODEParameterEstimation.convert_to_hc_format_with_params(
            [x^2 - 1],
            Any[x],
            Any[],
        )

        @test hc_system isa HomotopyContinuation.System
        @test eltype(hc_variables) == HomotopyContinuation.ModelKit.Variable
        @test eltype(hc_params) == HomotopyContinuation.ModelKit.Variable
        @test isempty(hc_params)

        results = ODEParameterEstimation.solve_with_hc_parameterized(
            [x^2 - 1],
            Any[x],
            Any[],
            [Float64[], Float64[]],
        )
        @test length(results) == 2
        for sols in results
            @test sort(first.(sols)) ≈ [-1.0, 1.0] atol = 1e-8
        end
    end

    @testset "Column scales reach HomotopyContinuation as parameters" begin
        # Unknowns are named by derivative order (`_0`, `_1`) and the data are derivatives of `y`,
        # so `x_1` is scaled by the size of the first derivative of the data.
        @independent_variables t
        @variables y(t) a_0 x_0 x_1
        dy = Differential(t)(y)
        equations = [x_0 - y, x_1 - dy, a_0 * x_0 - x_1]
        solutions_at(points) = ODEParameterEstimation.solve_with_hc_parameterized(
            equations, Any[a_0, x_0, x_1], Any[y, dy], points)

        @test only(only(solutions_at([[2.0, 6.0]]))) ≈ [3.0, 2.0, 6.0]

        # A second data set, with a different scale, is solved by the system already compiled.
        # With the scale written into the coefficients it was a system HomotopyContinuation had
        # not seen, and was compiled again. The table is HomotopyContinuation's own record.
        ModelKit = HomotopyContinuation.ModelKit
        counted = isdefined(ModelKit, :TSYSTEM_TABLE)
        compiled_systems() = counted ? sum(length, values(ModelKit.TSYSTEM_TABLE)) : 0
        before = compiled_systems()
        @test only(only(solutions_at([[2.0, 600.0]]))) ≈ [300.0, 2.0, 600.0]
        @test only(only(solutions_at([[-5.0, 0.25]]))) ≈ [-0.05, -5.0, 0.25]
        counted || @info "HomotopyContinuation's table of compiled systems was not found; the count is not checked"
        @test compiled_systems() == before
    end

    @testset "Noise-frontier construction probes" begin
        opts = EstimationOptions(
            datasize = 11,
            noise_level = 0.0,
            shooting_points = 3,
            nooutput = true,
            diagnostics = false,
            save_system = false,
            interpolators = [InterpolatorAAAD],
            polish_solver_solutions = false,
            polish_solutions = false,
        )
        pep = ODEParameterEstimation.sample_problem_data(ODEParameterEstimation.Examples.harmonic_oscillator(), opts)
        ident = ODEParameterEstimation.setup_identifiability(pep; max_num_points = 1, nooutput = true)
        ordered_model = pep.model
        si_template, _ = ODEParameterEstimation.prepare_si_template_with_structural_fix(
            ordered_model,
            pep.measured_quantities,
            pep.data_sample,
            ident.good_DD,
            false;
            states = ident.states,
            params = ident.params,
            infolevel = 0,
            placeholder_fail_categories = opts.si_placeholder_fail_categories,
        )
        interp_func = ODEParameterEstimation.get_interpolator_function(only(opts.interpolators))
        interpolants = ODEParameterEstimation.create_interpolants(
            pep.measured_quantities,
            pep.data_sample,
            ident.t_vector,
            interp_func,
        )
        setup = (
            good_deriv_level = ident.good_deriv_level,
            good_udict = ident.good_udict,
            good_varlist = ident.good_varlist,
            good_DD = ident.good_DD,
            interpolants = interpolants,
        )

        sp = ODEParameterEstimation.build_noise_frontier_system(
            pep,
            setup,
            si_template;
            n_points = 1,
            compute_mixed_volume = false,
            candidate_limit = 8,
            beam_width = 4,
            diagnostics = false,
        )
        @test sp isa ODEParameterEstimation.NoiseFrontierResult
        @test sp.n_points == 1
        @test !isnothing(sp.selected)
        @test sp.selected.rank_complete
        @test sp.selected.is_square
        @test sp.selected.max_observed_order == sp.minimal_max_observed_order
        @test sp.selected.mixed_volume === nothing
        @test length(sp.selected.equations) == length(sp.selected.solve_vars)
        @test isfinite(sp.selected.sigma_max)
        @test isfinite(sp.selected.sigma_min)
        @test isfinite(sp.selected.unfloored_svd_ratio)
        @test sp.selected.rank_atol == 1e-8
        frontier_rows = ODEParameterEstimation.noise_frontier_rows(sp)
        @test !isempty(frontier_rows)
        @test :sigma_max in propertynames(frontier_rows[1])
        @test :sigma_min in propertynames(frontier_rows[1])
        @test :unfloored_svd_ratio in propertynames(frontier_rows[1])
        @test :rank_atol in propertynames(frontier_rows[1])

        structural_setup = (
            good_deriv_level = ident.good_deriv_level,
            good_udict = ident.good_udict,
            good_varlist = ident.good_varlist,
            good_DD = ident.good_DD,
        )
        sp_structural = ODEParameterEstimation.build_noise_frontier_system(
            pep,
            structural_setup,
            si_template;
            n_points = 1,
            compute_mixed_volume = false,
            candidate_limit = 8,
            beam_width = 4,
            diagnostics = false,
        )
        @test sp_structural.selected.selected_equation_indices == sp.selected.selected_equation_indices

        inst_eqs, inst_vars = ODEParameterEstimation.instantiate_noise_frontier_candidate(
            sp.selected,
            interpolants,
            pep.data_sample,
            pep.measured_quantities,
            5,
        )
        @test length(inst_eqs) == length(inst_vars)
        data_values = ODEParameterEstimation.evaluate_noise_frontier_data_vars_at_point(
            interpolants,
            sp.selected.data_vars,
            pep.measured_quantities,
            pep.data_sample["t"][5],
        )
        validation = ODEParameterEstimation.validate_noise_frontier_candidate_at_values(sp.selected, data_values)
        @test validation.valid
        @test validation.rank == validation.target_rank
        @test isfinite(validation.sigma_max)
        @test isfinite(validation.sigma_min)
        @test isfinite(validation.unfloored_svd_ratio)
        @test isfinite(validation.condition_proxy)
        @test validation.rank_atol == 1e-8
        bad_validation = ODEParameterEstimation.validate_noise_frontier_candidate_at_values(
            sp.selected,
            fill(NaN, length(sp.selected.data_vars)),
        )
        @test !bad_validation.valid
        @test bad_validation.reason == :nonfinite_data
        @test isnan(bad_validation.sigma_max)
        @test bad_validation.condition_proxy == Inf
        @test bad_validation.rank_atol == 1e-8

        @variables z d
        deficient_validation = ODEParameterEstimation.validate_noise_frontier_instantiation(
            [0 * z],
            [z],
            [d],
            [1.0],
        )
        @test !deficient_validation.valid
        @test deficient_validation.reason == :rank_deficient
        @test isfinite(deficient_validation.sigma_max)
        @test deficient_validation.sigma_min == 0.0
        @test deficient_validation.unfloored_svd_ratio == Inf
        @test deficient_validation.condition_proxy == 0.0

        mp = ODEParameterEstimation.build_noise_frontier_system(
            pep,
            setup,
            si_template;
            n_points = 2,
            compute_mixed_volume = false,
            candidate_limit = 8,
            beam_width = 4,
            diagnostics = false,
        )
        @test mp.n_points == 2
        @test !isnothing(mp.selected)
        @test mp.selected.rank_complete
        @test mp.selected.is_square
        @test any(endswith(string(v), "_pt2") for v in vcat(mp.selected.solve_vars, mp.selected.data_vars))

        mpt = ODEParameterEstimation.build_noise_frontier_multipoint_template(
            pep,
            setup,
            si_template;
            n_points = 2,
            compute_mixed_volume = false,
            candidate_limit = 8,
            beam_width = 4,
            diagnostics = false,
        )
        @test mpt isa ODEParameterEstimation.MultiPointTemplate
        @test length(mpt.stripped_equations) == length(mpt.solve_vars)
        @test length(mpt.per_point_data_var_indices) == 2
        @test sort(vcat(mpt.per_point_data_var_indices...)) == collect(1:length(mpt.data_vars))
        mpe = ODEParameterEstimation.evaluate_multipoint_template(mpt, [3, 8], interpolants, pep.data_sample)
        @test all(isfinite, mpe.data_values)
        @test length(mpe.data_values) == length(mpt.data_vars)

        # data_var_meta: authoritative (obs, order, point, kind) aligned to data_vars
        @test length(mpt.data_var_meta) == length(mpt.data_vars)
        @test all(m -> m.kind in (:observable_jet, :transcendental), mpt.data_var_meta)
        for (i, m) in enumerate(mpt.data_var_meta)
            @test m.point == (endswith(string(mpt.data_vars[i]), "_pt2") ? 2 : 1)
            if m.kind == :observable_jet
                @test 1 <= m.obs_idx <= length(mpt.measured_quantities)
                @test m.order >= 0
            end
        end

        csv_path = joinpath(mktempdir(), "noise_frontier.csv")
        @test ODEParameterEstimation.write_noise_frontier_csv(csv_path, sp) == csv_path
        @test isfile(csv_path)
        @test occursin("mixed_volume", read(csv_path, String))
    end

    @testset "Noise-frontier hoist timing capture" begin
        opts = EstimationOptions(
            datasize = 15,
            noise_level = 0.0,
            shooting_points = 3,
            nooutput = true,
            diagnostics = false,
            save_system = false,
            flow = FlowStandard,
            use_si_template = true,
            use_parameter_homotopy = true,
            use_multipoint = true,
            multipoint_n_points = 2,
            multipoint_max_pairs = 1,
            system_construction_policy = :noise_frontier,
            construction_compute_mixed_volume = false,
            construction_candidate_limit = 8,
            construction_beam_width = 4,
            interpolators = [InterpolatorAAAD, InterpolatorChebyshevAICc],
            polish_solver_solutions = false,
            polish_solutions = false,
            synthesize_aggregate_candidates = false,
            branch_completion = false,
        )
        pep = ODEParameterEstimation.sample_problem_data(ODEParameterEstimation.Examples.harmonic_oscillator(), opts)
        (_, timing) = ODEParameterEstimation.with_estimation_timing() do
            ODEParameterEstimation.analyze_parameter_estimation_problem(pep, opts)
        end

        @test timing isa ODEParameterEstimation.TimingBreakdown
        details = timing.details
        @test details[:noise_frontier_sp_cache_misses] == 1
        @test details[:noise_frontier_sp_cache_hits] >= 1
        @test details[:sp_generic_start_cache_misses] == 1
        @test details[:sp_generic_start_cache_hits] >= 1
        @test haskey(details, :multipoint_template_seconds_by_source)
        @test haskey(details, :multipoint_solve_seconds_by_source)
        sp_validation_records = filter(
            rec -> get(rec, :category, nothing) == :single_point_noise_frontier_validation,
            get(details, :detailed_timing_records, NamedTuple[]),
        )
        @test !isempty(sp_validation_records)
        @test all(rec -> :sigma_max in propertynames(rec), sp_validation_records)
        @test all(rec -> :sigma_min in propertynames(rec), sp_validation_records)
        @test all(rec -> :unfloored_svd_ratio in propertynames(rec), sp_validation_records)
        @test all(rec -> :rank_atol in propertynames(rec), sp_validation_records)
        timing_dict = ODEParameterEstimation.timing_breakdown_to_dict(timing)
        @test timing_dict["details"]["sp_generic_start_cache_misses"] == 1
    end

    @testset "Polish coordinate transforms" begin
        # Per-variable :auto rule:
        #   lb>0 → :log; lb==0 → :shifted_log; lb<0 with |lb|≤ub/10 → :shifted_log
        #   ("positive, wiggle around 0", e.g. (-0.01,100)); comparable straddle / unbounded → :linear
        lb = Float64[1e-3, -0.01, -1.0, -Inf, 0.0]
        ub = Float64[100.0, 100.0, 1.0, Inf, 10.0]
        transforms, shifts = ODEParameterEstimation._choose_polish_transforms(lb, ub; policy = :auto)
        @test transforms == [:log, :shifted_log, :linear, :linear, :shifted_log]
        @test shifts[1] == 0.0
        @test 0.0 < shifts[2] < 1.0     # small "wiggle" shift (~|lb|), NOT the bound magnitude
        @test shifts[3] == 0.0          # :linear, symmetric-ish (-1,1)
        @test shifts[4] == 0.0          # :linear, unbounded
        @test shifts[5] > 0.0           # :shifted_log, lb==0

        # Forward / inverse round-trip in original scale (x in each transform's domain)
        x = Float64[2.5, 0.3, -0.5, -7.4, 4.7]
        y = ODEParameterEstimation._polish_external_to_internal(x, transforms, shifts)
        x_back = ODEParameterEstimation._polish_internal_to_external(y, transforms, shifts)
        @test x_back ≈ x rtol = 1e-12

        # `:shifted_log` keeps the internal lower bound finite even when external lb is just
        # barely negative — the formula `s = max(εM, -lb + εM)` prevents a near-zero log argument
        lb_marginal = Float64[-1e-10]
        ub_marginal = Float64[10.0]
        t_marg, s_marg = ODEParameterEstimation._choose_polish_transforms(lb_marginal, ub_marginal; policy = :auto)
        @test t_marg == [:shifted_log]
        ilb_marg, _ = ODEParameterEstimation._polish_coordinate_bounds(t_marg, s_marg, lb_marginal, ub_marginal)
        @test isfinite(ilb_marg[1]) && ilb_marg[1] > -100  # not absurdly negative

        # Policy overrides
        t_lin, _ = ODEParameterEstimation._choose_polish_transforms(lb, ub; policy = :linear)
        @test all(t_lin .== :linear)

        # `:log_only` is strict — errors when any variable has non-positive or non-finite bounds
        @test_throws ArgumentError ODEParameterEstimation._choose_polish_transforms(lb, ub; policy = :log_only)
        # but works fine when every bound permits `:log`
        lb_pos = Float64[1e-3, 0.5]
        ub_pos = Float64[100.0, 10.0]
        t_logpos, _ = ODEParameterEstimation._choose_polish_transforms(lb_pos, ub_pos; policy = :log_only)
        @test all(t_logpos .== :log)
    end

    @testset "Polish method dispatch" begin
        # New residual-mode methods are recognized
        @test ODEParameterEstimation.is_residual_polish_method(PolishLSOBoundedLog)
        @test ODEParameterEstimation.is_residual_polish_method(PolishFastLMBoundedLog)
        @test !ODEParameterEstimation.is_residual_polish_method(PolishNewtonTrust)
        @test !ODEParameterEstimation.is_residual_polish_method(PolishBFGS)

        # `get_polish_optimizer` returns a tagged tuple for residual methods, a plain
        # constructor for scalar methods
        token_lso = ODEParameterEstimation.get_polish_optimizer(PolishLSOBoundedLog)
        @test token_lso isa Tuple
        @test token_lso[1] === :lso_direct
        # Don't pull `LeastSquaresOptim` into Main scope; check the type name instead.
        instance_name = string(typeof(token_lso[2]()).name.module, ".", typeof(token_lso[2]()).name.name)
        @test instance_name == "LeastSquaresOptim.LevenbergMarquardt"

        token_fastlm = ODEParameterEstimation.get_polish_optimizer(PolishFastLMBoundedLog)
        @test token_fastlm isa Tuple
        @test token_fastlm[1] === :fastlm_direct

        # Scalar path still works (`NewtonTrustRegion` is exported from `Optim` by `using ODEParameterEstimation`)
        @test ODEParameterEstimation.get_polish_optimizer(PolishNewtonTrust) === ODEParameterEstimation.NewtonTrustRegion
    end

    @testset "Unsupported model-class validation" begin
        @independent_variables t
        @parameters a
        @variables x(t) y(t)
        D = Differential(t)

        trig_model, trig_mq = ODEParameterEstimation.create_ordered_ode_system(
            "unsupported_trig",
            [x],
            [a],
            [D(x) ~ -a * sin(x)],
            [y ~ x],
        )
        trig_pep = ODEParameterEstimation.ParameterEstimationProblem(
            "unsupported_trig",
            trig_model,
            trig_mq,
            nothing,
            [0.0, 1.0],
            nothing,
            OrderedDict(a => 1.0),
            OrderedDict(x => 0.1),
            0,
        )
        trig_err = try
            ODEParameterEstimation.validate_supported_model_class(trig_pep)
            nothing
        catch err
            err
        end
        @test trig_err isa ODEParameterEstimation.UnsupportedModelClassError
        @test trig_err.category == :state_trigonometric

        sqrt_model, sqrt_mq = ODEParameterEstimation.create_ordered_ode_system(
            "unsupported_sqrt",
            [x],
            [a],
            [D(x) ~ a - sqrt(x)],
            [y ~ x],
        )
        sqrt_pep = ODEParameterEstimation.ParameterEstimationProblem(
            "unsupported_sqrt",
            sqrt_model,
            sqrt_mq,
            nothing,
            [0.0, 1.0],
            nothing,
            OrderedDict(a => 1.0),
            OrderedDict(x => 1.0),
            0,
        )
        sqrt_err = try
            ODEParameterEstimation.validate_supported_model_class(sqrt_pep)
            nothing
        catch err
            err
        end
        @test sqrt_err isa ODEParameterEstimation.UnsupportedModelClassError
        @test sqrt_err.category == :sqrt_nonlinearity

        supported_time_trig_model, supported_time_trig_mq = ODEParameterEstimation.create_ordered_ode_system(
            "supported_time_trig",
            [x],
            [a],
            [D(x) ~ -a * x + sin(2.0 * t)],
            [y ~ x],
        )
        supported_time_trig_pep = ODEParameterEstimation.ParameterEstimationProblem(
            "supported_time_trig",
            supported_time_trig_model,
            supported_time_trig_mq,
            nothing,
            [0.0, 1.0],
            nothing,
            OrderedDict(a => 1.0),
            OrderedDict(x => 0.1),
            0,
        )
        @test isnothing(ODEParameterEstimation.validate_supported_model_class(supported_time_trig_pep))
    end

    @testset "Sampling validation" begin
        @independent_variables t
        @parameters a
        @variables x(t) y(t)
        D = Differential(t)

        unstable_model, unstable_mq = ODEParameterEstimation.create_ordered_ode_system(
            "unstable_sampling_case",
            [x],
            [a],
            [D(x) ~ x^2],
            [y ~ x],
        )

        unstable_err = try
            ODEParameterEstimation.sample_data(
                unstable_model.system,
                unstable_mq,
                [0.0, 2.0],
                OrderedDict(a => 1.0),
                OrderedDict(x => 1.0),
                41,
            )
            nothing
        catch err
            err
        end
        @test unstable_err isa ODEParameterEstimation.SamplingFailureError

        maglev = ODEParameterEstimation.Examples.magnetic_levitation()
        maglev_opts = EstimationOptions(
            datasize = 41,
            noise_level = 0.0,
            time_interval = maglev.recommended_time_interval,
            nooutput = true,
        )
        sampled_maglev = ODEParameterEstimation.sample_problem_data(maglev, maglev_opts)
        @test length(sampled_maglev.data_sample["t"]) == 41
        @test all(length(values) == 41 for (key, values) in sampled_maglev.data_sample if key != "t")
    end

    @testset "Derivative order guard" begin
        err = try
            ODEParameterEstimation.nth_deriv(sin, ODEParameterEstimation.TAYLORDIFF_MAX_DERIVATIVE_ORDER + 1, 0.0)
            nothing
        catch caught
            caught
        end
        @test err isa ODEParameterEstimation.UnsupportedDerivativeOrderError
        @test err.requested_order == ODEParameterEstimation.TAYLORDIFF_MAX_DERIVATIVE_ORDER + 1
        @test err.supported_order == ODEParameterEstimation.TAYLORDIFF_MAX_DERIVATIVE_ORDER
        @test err.backend == :taylordiff
        @test isnothing(err.context)
    end

	@testset "UQ policy helper" begin
		# Estimator-aware contract: `nothing` is reserved for UQ disabled and is
		# always a pass-through. Requested failures are typed `UQUnavailable`
		# outcomes (covered in test_estimator_aware_uq.jl).
		passthrough_opts = EstimationOptions(uq_failure_policy = :return_failed)
		throw_opts = EstimationOptions(uq_failure_policy = :throw)

		@test ODEParameterEstimation.apply_uq_failure_policy(nothing, passthrough_opts) === nothing
		@test ODEParameterEstimation.apply_uq_failure_policy(nothing, throw_opts) === nothing
	end

    @testset "SI placeholder policy helpers" begin
        @test !ODEParameterEstimation.should_fail_si_placeholder(:dd_observable_index_oob, Symbol[])
        @test ODEParameterEstimation.should_fail_si_placeholder(:dd_observable_index_oob, [:dd_observable_index_oob])
        @test ODEParameterEstimation.should_fail_si_placeholder(:observable_derivative_overflow, [:dd_derivative_unmapped])
        @test ODEParameterEstimation.should_fail_si_placeholder(:support_jet, [:state_or_input_jet])
        @test ODEParameterEstimation.should_fail_si_placeholder(:true_unknown_variable, [:unknown_variable])

        z_aux_classification = ODEParameterEstimation.classify_si_ring_variable("z_aux", Dict{String, Int}(), nothing)
        @test z_aux_classification.category == :sian_auxiliary

        role_context = (
            state_names = Set(["x1"]),
            param_names = Set(["b"]),
            measured_rhs_names = Set(["z2"]),
        )
        state_jet_classification = ODEParameterEstimation.classify_si_ring_variable("x1_0", Dict{String, Int}(), (obs_lhs = [[1]],), role_context)
        @test state_jet_classification.category == :state_jet
        param_support_classification = ODEParameterEstimation.classify_si_ring_variable("b_0", Dict{String, Int}(), (obs_lhs = [[1]],), role_context)
        @test param_support_classification.category == :parameter_or_ic_symbol
        measured_rhs_classification = ODEParameterEstimation.classify_si_ring_variable("z2_0", Dict{String, Int}(), (obs_lhs = [[1]],), role_context)
        @test measured_rhs_classification.category == :measured_rhs_jet
        trfn_support_classification = ODEParameterEstimation.classify_si_ring_variable("_trfn_u_0", Dict{String, Int}(), (obs_lhs = [[1]],), role_context)
        @test trfn_support_classification.category == :transformed_analytic_support

        unknown_classification = ODEParameterEstimation.classify_si_ring_variable("mystery_symbol", Dict{String, Int}(), nothing)
        @test unknown_classification.category == :true_unknown_variable

        R_used, gens_used = ODEParameterEstimation.Nemo.polynomial_ring(ODEParameterEstimation.Nemo.QQ, ["x", "y1_0", "y1_1"])
        used_vars = ODEParameterEstimation.collect_used_nemo_variables([gens_used[1] + gens_used[2]])
        @test Set(string.(used_vars)) == Set(["x", "y1_0"])

        pep_dd = ODEParameterEstimation.sample_problem_data(ODEParameterEstimation.Examples.harmonic_oscillator(), EstimationOptions(datasize = 11, noise_level = 0.0, nooutput = true))
        shallow_dd = ODEParameterEstimation.populate_derivatives(pep_dd.model.system, pep_dd.measured_quantities, 1, OrderedCollections.OrderedDict())
        extended_dd = ODEParameterEstimation.ensure_si_template_dd_support(
            pep_dd.model,
            pep_dd.measured_quantities,
            shallow_dd,
            Dict(:fake_y => 3),
        )
        @test length(extended_dd.obs_lhs) >= 4

        placeholder_stats = Dict{Symbol, Vector{String}}()
        placeholder_map = Dict{Any, Any}()
        @test_throws ErrorException ODEParameterEstimation._create_si_symbolic_placeholder!(
            placeholder_map,
            :fake_var,
            "fake_var",
            placeholder_stats,
            :dd_observable_index_oob;
            fail_categories = [:dd_observable_index_oob],
        )

        R, gens = ODEParameterEstimation.Nemo.polynomial_ring(ODEParameterEstimation.Nemo.QQ, ["late_x"])
        late_poly = gens[1]
        @test_throws ErrorException ODEParameterEstimation.nemo_to_symbolics(
            late_poly,
            Dict();
            fail_categories = [:late_map_miss],
        )

        R_aux, gens_aux = ODEParameterEstimation.Nemo.polynomial_ring(ODEParameterEstimation.Nemo.QQ, ["z_aux"])
        aux_poly = gens_aux[1]
        aux_sym = ODEParameterEstimation.nemo_to_symbolics(aux_poly, Dict())
        @test string(aux_sym) == "z_aux"
    end

    @testset "SEIR algebraic rescue resolves states at t0 and shooting time" begin
        base_pep = ODEParameterEstimation.sample_problem_data(ODEParameterEstimation.Examples.harmonic_oscillator(), EstimationOptions(datasize = 11, noise_level = 0.0, nooutput = true))
        failing_runner(args...) = error("synthetic advisory failure")

        fallback_ident = ODEParameterEstimation.setup_identifiability(
            base_pep;
            max_num_points = 1,
            nooutput = true,
            advisory_runner = failing_runner,
        )
        @test fallback_ident.numerical_advisory.status == :failed
        @test fallback_ident.numerical_advisory.failure_reason == "synthetic advisory failure"
        @test :heuristic_fallback in fallback_ident.numerical_advisory.notes
        @test fallback_ident.good_num_points == 1
        @test isempty(fallback_ident.good_udict)
        @test isempty(fallback_ident.all_unidentifiable)
        @test isempty(fallback_ident.good_DD.all_unidentifiable)

        # Seeded: this was the only unseeded noisy sampling in CI (postcampaign
        # review P1 / lane-4 flakiness audit).
        Random.seed!(20260610)
        opts = EstimationOptions(
            datasize = 21,
            noise_level = 1e-8,
            nooutput = true,
            diagnostics = false,
            flow = FlowStandard,
            use_si_template = true,
            use_parameter_homotopy = true,
            interpolators = [InterpolatorAAAD, InterpolatorAGPRobust],
            save_system = false,
            polish_solver_solutions = true,
            polish_solutions = false,
        )
        pep = ODEParameterEstimation.sample_problem_data(ODEParameterEstimation.Examples.seir(), opts)
        ident = ODEParameterEstimation.setup_identifiability(pep; max_num_points = 1, nooutput = true)
        ordered_model = isa(pep.model.system, ODEParameterEstimation.OrderedODESystem) ?
            pep.model.system :
            ODEParameterEstimation.OrderedODESystem(pep.model.system, ident.states, ident.params)
        si_template, _ = ODEParameterEstimation.prepare_si_template_with_structural_fix(
            ordered_model,
            pep.measured_quantities,
            pep.data_sample,
            ident.good_DD,
            false;
            states = ident.states,
            params = ident.params,
            infolevel = 0,
            placeholder_fail_categories = opts.si_placeholder_fail_categories,
        )
        interp_spec = first(ODEParameterEstimation.resolve_interpolator_list(opts))
        interp_func = ODEParameterEstimation.get_interpolator_function(interp_spec[1], interp_spec[2])
        interpolants = ODEParameterEstimation.create_interpolants(
            pep.measured_quantities,
            pep.data_sample,
            ident.t_vector,
            interp_func,
        )
        known_param_dict = OrderedDict{Any, Float64}(k => Float64(v) for (k, v) in pep.p_true)
        resolve_t0 = ODEParameterEstimation.resolve_states_with_fixed_params(
            pep.model.system,
            pep.measured_quantities,
            pep.data_sample,
            ident.good_deriv_level,
            ident.good_udict,
            ident.good_varlist,
            ident.good_DD,
            known_param_dict,
            interpolants;
            si_template = si_template,
            time_index = 1,
            diagnostics = false,
            placeholder_fail_categories = opts.si_placeholder_fail_categories,
        )
        resolve_shoot = ODEParameterEstimation.resolve_states_with_fixed_params(
            pep.model.system,
            pep.measured_quantities,
            pep.data_sample,
            ident.good_deriv_level,
            ident.good_udict,
            ident.good_varlist,
            ident.good_DD,
            known_param_dict,
            interpolants;
            si_template = si_template,
            time_index = 2,
            diagnostics = false,
            placeholder_fail_categories = opts.si_placeholder_fail_categories,
        )

        # SEIR's t0 resolve now completes (0 missing states); it previously left
        # states unresolved at t0 and only completed at a shooting point (the
        # original "retry at shooting time" scenario). Assert current behavior at
        # both points; the shooting-time rescue candidate is still built/checked below.
        @test ODEParameterEstimation._resolve_missing_state_count(resolve_t0) == 0
        @test ODEParameterEstimation._resolve_missing_state_count(resolve_shoot) == 0

        candidate = ODEParameterEstimation._build_algebraic_resolve_candidate(
            pep,
            known_param_dict,
            ident.good_udict,
            ident.all_unidentifiable,
            resolve_shoot,
            resolve_shoot.state_vars,
            first(resolve_shoot.solutions),
            2,
            2,
            :aaad,
            1,
            1.0,
            opts;
            rescue_path = :algebraic_resolve_shoot,
        )
        candidate.provenance = ODEParameterEstimation.copy_provenance(
            candidate.provenance;
            ODEParameterEstimation.si_template_lineage_kwargs(si_template)...,
        )
        ODEParameterEstimation.sync_result_contract!(candidate)

        @test candidate.return_code == :algebraic_resolve_shoot
        @test :resolved_at_shooting_time in candidate.provenance.notes
        @test candidate.provenance.practical_identifiability_status == :not_assessed
        @test all(isfinite, values(candidate.states))
    end

    @testset "Math helpers" begin
        @test ODEParameterEstimation.count_turns([1, 2, 3, 4, 5]) == 0
        @test ODEParameterEstimation.count_turns([1, 3, 2, 1]) == 1

        stats = ODEParameterEstimation.calculate_timeseries_stats([1.0, 3.0, 2.0, 4.0, 2.0])
        @test stats.mean ≈ 2.4
        @test stats.turns == 3

        valid_points, valid_params, dropped = ODEParameterEstimation.filter_finite_shooting_point_params(
            [1, 5, 9],
            [
                [1.0, 2.0],
                [NaN, 3.0],
                [4.0, Inf],
            ],
        )
        @test valid_points == [1]
        @test valid_params == [[1.0, 2.0]]
        @test dropped == [(5, 1), (9, 1)]
    end

    @testset "SI template shape guard" begin
        @parameters a

        fake_structure = (
            status = :residual_underdetermined,
            n_equations = 10,
            n_variables = 12,
            n_data_vars = 3,
            n_effective_eqs = 8,
            n_effective_vars = 10,
            dropped_equation_indices = [2, 5],
        )
        fake_roles = (
            suspicious_categories = Dict(:true_unknown_variable => 1),
        )

        err = try
            ODEParameterEstimation.throw_on_nonsquare_si_template(
                fake_structure,
                OrderedDict(a => 1.0),
                fake_roles,
            )
            nothing
        catch e
            e
        end

        @test err isa ODEParameterEstimation.SITemplateShapeError
        @test occursin("residual_underdetermined", sprint(showerror, err))
        @test occursin("effective system has 8 equations and 10 unknowns", sprint(showerror, err))
    end

    @testset "Result compatibility helpers" begin
        @parameters a
        @variables t x(t)

        result = ODEParameterEstimation.ParameterEstimationResult(
            OrderedDict(a => 1.0),
            OrderedDict(x => 2.0),
            0.0,
            1e-6,
            nothing,
            10,
            nothing,
            OrderedDict{Num, Float64}(),
            Set{Num}(),
            nothing,
        )
        result.provenance = ODEParameterEstimation.ResultProvenance(
            primary_method = :direct_opt,
            rescue_path = :direct_opt_fallback,
            interpolator_source = :aaad,
            source_shooting_index = 3,
            source_candidate_index = 7,
            polish_applied = true,
            structural_fix_set = OrderedDict(a => 1.0),
            template_status = :determined,
            practical_identifiability_status = :advisory_available,
            numerical_advisory = ODEParameterEstimation.NumericalIdentifiabilityAdvisory(
                status = :available,
                recommended_num_points = 1,
                recommended_deriv_level = Dict(1 => 3),
            ),
        )

        ODEParameterEstimation.sync_result_contract!(result)

        @test result.return_code == :direct_opt_fallback
        @test result.interpolator_source == :aaad
        @test ODEParameterEstimation.compatibility_return_code(result.provenance) == :direct_opt_fallback
        @test occursin("method=direct_opt", ODEParameterEstimation.lineage_summary(result))
        @test occursin("rescue=direct_opt_fallback", ODEParameterEstimation.lineage_summary(result))
        @test occursin("structural_fix=1", ODEParameterEstimation.lineage_summary(result))
        @test occursin("template=determined", ODEParameterEstimation.lineage_summary(result))
        @test occursin("practical=advisory_available", ODEParameterEstimation.lineage_summary(result))
        @test occursin("advisory=available", ODEParameterEstimation.lineage_summary(result))
    end

    @testset "Algebraic resolve failure stays candidate-local" begin
        candidates = [
            ODEParameterEstimation.ParameterEstimationResult(
                OrderedDict{Num, Float64}(),
                OrderedDict{Num, Float64}(),
                0.0,
                1.0,
                nothing,
                0,
                0.0,
                OrderedDict{Num, Float64}(),
                Set{Num}(),
                nothing,
                nothing,
                ODEParameterEstimation.ResultProvenance(),
            ),
            ODEParameterEstimation.ParameterEstimationResult(
                OrderedDict{Num, Float64}(),
                OrderedDict{Num, Float64}(),
                0.0,
                1.0,
                nothing,
                0,
                0.0,
                OrderedDict{Num, Float64}(),
                Set{Num}(),
                nothing,
                nothing,
                ODEParameterEstimation.ResultProvenance(),
            ),
        ]

        upstream_err = TaskFailedException(Task(() -> nothing))
        ODEParameterEstimation._note_algebraic_resolve_failure!(candidates, [1], upstream_err)
        @test :algebraic_resolve_failed in candidates[1].provenance.notes
        @test :algebraic_resolve_upstream_failure in candidates[1].provenance.notes
        @test isempty(candidates[2].provenance.notes)
        @test ODEParameterEstimation._algebraic_resolve_failure_notes(ErrorException("boom")) == [:algebraic_resolve_failed, :algebraic_resolve_exception]
    end

    @testset "Analysis and UQ helpers handle unscored candidates" begin
        @independent_variables t
        @parameters a
        @variables x(t) y(t)
        D = Differential(t)

        model, measured_quantities = ODEParameterEstimation.create_ordered_ode_system(
            "analysis_no_scores",
            [x],
            [a],
            [D(x) ~ a * x],
            [y ~ x],
        )
        pep = ODEParameterEstimation.ParameterEstimationProblem(
            "analysis_no_scores",
            model,
            measured_quantities,
            OrderedDict{Union{String, Num}, Vector{Float64}}(
                y => [1.0, 1.1],
                "t" => [0.0, 1.0],
            ),
            [0.0, 1.0],
            nothing,
            OrderedDict(a => 1.0),
            OrderedDict(x => 1.0),
            0,
        )

        unscored_results = [
            ODEParameterEstimation.ParameterEstimationResult(
                OrderedDict(a => 1.0),
                OrderedDict(x => 1.0),
                0.0,
                nothing,
                nothing,
                2,
                nothing,
                OrderedDict{Num, Float64}(),
                Set{Num}(),
                nothing,
            ),
            ODEParameterEstimation.ParameterEstimationResult(
                OrderedDict(a => 1.5),
                OrderedDict(x => 0.9),
                0.0,
                nothing,
                nothing,
                2,
                nothing,
                OrderedDict{Num, Float64}(),
                Set{Num}(),
                nothing,
            ),
        ]

        analysis = ODEParameterEstimation.analyze_estimation_result(pep, unscored_results; nooutput = true)
        @test analysis.returned_results == Any[]
        @test all(isinf, (analysis.besterror, analysis.best_min_error, analysis.best_mean_error,
                          analysis.best_median_error, analysis.best_max_error,
                          analysis.best_approximation_error, analysis.best_rms_error))
        @test analysis.algebraic_multiplicity === nothing  # opts default = nothing → unchanged

        uq_result = ODEParameterEstimation._compute_uq_result(
            pep,
            analysis,
            EstimationOptions(compute_uncertainty = true, nooutput = true),
        )
        @test uq_result isa ODEParameterEstimation.UQUnavailable
        @test uq_result.reason == :no_selected_result
        @test isnothing(ODEParameterEstimation._compute_uq_result(
            pep, analysis, EstimationOptions(compute_uncertainty = false, nooutput = true)))

        @test_throws ArgumentError ODEParameterEstimation.solve_parameter_estimation(pep, (;))
    end

    @testset "Branch-diverse M selection" begin
        @independent_variables tt
        @parameters a
        @variables x(tt)

        function branch_result(a_value, x_value, err)
            return ODEParameterEstimation.ParameterEstimationResult(
                OrderedDict(a => Float64(a_value)),
                OrderedDict(x => Float64(x_value)),
                0.0,
                Float64(err),
                nothing,
                2,
                nothing,
                OrderedDict{Num, Float64}(),
                Set{Num}(),
                nothing,
            )
        end

        best = branch_result(1.0, 1.0, 1.0)
        near_duplicate = branch_result(1.001, 1.001, 1.1)
        distinct = branch_result(2.0, 2.0, 10.0)

        opts = EstimationOptions(
            algebraic_multiplicity = 2,
            branch_diversity_selection = true,
            branch_diversity_eps = 0.01,
        )
        selected = ODEParameterEstimation.select_branch_diverse_reps(
            [best, near_duplicate, distinct],
            2,
            opts,
        )
        @test selected[1] === best
        @test selected[2] === distinct

        fallback = ODEParameterEstimation.select_branch_diverse_reps(
            [best, near_duplicate],
            2,
            opts,
        )
        @test fallback[1] === best
        @test fallback[2] === near_duplicate

        disabled = ODEParameterEstimation.select_branch_diverse_reps(
            [best, near_duplicate, distinct],
            2,
            EstimationOptions(algebraic_multiplicity = 2, branch_diversity_selection = false),
        )
        @test disabled[1] === best
        @test disabled[2] === near_duplicate

        multiplicity_one = ODEParameterEstimation.select_branch_diverse_reps(
            [best, near_duplicate, distinct],
            1,
            EstimationOptions(algebraic_multiplicity = 1),
        )
        @test length(multiplicity_one) == 1
        @test multiplicity_one[1] === best

        # The default-on branch_completion flag must not silently override the
        # configured output clustering method when completion was a no-op.
        ordinary_candidates = [
            branch_result(10.0, 10.0, 1.0),
            branch_result(10.01, 10.01, 1.1),
            branch_result(10.02, 10.02, 1.2),
        ]
        identifiable_opts = EstimationOptions(
            branch_completion = true,
            cluster_method = :identifiable_subspace,
            rough_cluster_eps = 1.0,
            subspace_cluster_eps = 0.05,
        )
        identifiable_clusters = ODEParameterEstimation._cluster_output_candidates(
            ordinary_candidates,
            identifiable_opts,
        )
        @test length(identifiable_clusters) == 1
        @test length(only(identifiable_clusters)) == 3

        bit_identical_clusters = ODEParameterEstimation._cluster_output_candidates(
            ordinary_candidates,
            EstimationOptions(branch_completion = true, cluster_method = :bit_identical),
        )
        @test length(bit_identical_clusters) == 3

        # A pool genuinely replaced by branch completion is the sole exception:
        # keep its observationally equivalent algebraic siblings distinct in
        # full coordinate space even under identifiable-subspace clustering.
        completed_candidates = [
            branch_result(10.0, 10.0, 1.0),
            branch_result(10.01, 10.01, 1.1),
            branch_result(10.02, 10.02, 1.2),
        ]
        for candidate in completed_candidates
            candidate.provenance.source_type = :branch_completed
        end
        completed_clusters = ODEParameterEstimation._cluster_output_candidates(
            completed_candidates,
            identifiable_opts,
        )
        @test length(completed_clusters) == 3

        # A RAW pool with detected/declared M >= 2 gets the same full-space
        # protection: siblings found organically by the solver must survive
        # even when branch completion was inactive or failed (its default
        # anchor budget is 1) — the protection keys on the math, not on the
        # completion feature's bookkeeping.
        m2_raw_opts = EstimationOptions(
            branch_completion = true,
            cluster_method = :identifiable_subspace,
            rough_cluster_eps = 1.0,
            subspace_cluster_eps = 0.05,
            algebraic_multiplicity = 2,
        )
        m2_raw_clusters = ODEParameterEstimation._cluster_output_candidates(
            ordinary_candidates,
            m2_raw_opts,
        )
        @test length(m2_raw_clusters) == 3

        # ...and M = 1 (globally identifiable) keeps the honored cluster_method.
        m1_opts = EstimationOptions(
            branch_completion = true,
            cluster_method = :identifiable_subspace,
            rough_cluster_eps = 1.0,
            subspace_cluster_eps = 0.05,
            algebraic_multiplicity = 1,
        )
        m1_clusters = ODEParameterEstimation._cluster_output_candidates(
            ordinary_candidates,
            m1_opts,
        )
        @test length(m1_clusters) == 1
    end

end
