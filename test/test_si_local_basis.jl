using ODEParameterEstimation, ModelingToolkit, Symbolics, OrderedCollections, LinearAlgebra, Random, Test
using ODEParameterEstimation: merge_options
using ModelingToolkit: t_nounits as t, D_nounits as D

include("estimation_helpers.jl")
const ODEPE = ODEParameterEstimation

@testset "Local SI coordinate basis" begin
    @parameters a b c
    @variables x(t) z(t) y(t)
    # Each invariant list is known analytically, independent of SI's algorithm.
    cases = [
        ("sum", [x], [a,b], [D(x) ~ (a+b)*x], [y ~ x], [a+b,x], 1, ["a","b"]),
        ("product", [x], [a,b], [D(x) ~ a*b*x], [y ~ x], [a*b,x], 1, ["a","b"]),
        ("two_freedoms", [x], [a,b,c], [D(x) ~ (a+b+c)*x], [y ~ x], [a+b+c,x], 2, ["a","b","c"]),
        ("parameter_ic", [x], [a,b], [D(x) ~ a*x], [y ~ b*x], [a,b*x], 1, ["b"]),
        ("state_freedom", [x,z], [a], [D(x) ~ a*x, D(z) ~ a*z], [y ~ x+z], [a,x+z], 1, ["x","z"]),
        ("finite_ambiguity", [x], [a], [D(x) ~ a^2*x], [y ~ x], [a^2,x], 0, String[]),
    ]
    for (name, states, params, equations, measured, invariants, freedom_count, allowed) in cases
        @testset "$name" begin
            model, mq = create_ordered_ode_system("basis_" * name, states, params, equations, measured)
            si_ode, _, _ = ODEPE.convert_to_si_ode(model, mq)
            previous_basis = nothing
            for seed in 20260914:20260916
                Random.seed!(seed)
                analysis = quiet_call() do
                    ODEPE.analyze_si_structure(model, mq, si_ode; strategy=:local_basis)
                end
                @test length(analysis.coordinate_basis) == freedom_count
                @test all(in(allowed), analysis.coordinate_basis)
                @test isnothing(analysis.identifiable_functions)
                @test Set(keys(analysis.timing)) == Set([:assess_local_identifiability])
                @test isnothing(previous_basis) || analysis.coordinate_basis == previous_basis
                previous_basis = analysis.coordinate_basis

                template = (unidentifiable=Set(Symbolics.variable(Symbol(k)) for (k,v) in analysis.classification if v == :nonidentifiable),
                    identifiable_funcs=nothing, structural_analysis=analysis)
                fixing = ODEPE.derive_structural_fix_set(template, false; states=states, params=params)
                @test length(fixing.reported) == freedom_count
                @test all(==(1.0), values(fixing.reported))
                coordinates = vcat(params, states)
                # The invariant map plus the selected coordinates must have full
                # differential rank. Fixing every nonidentifiable quantity would
                # fail the freedom-count assertion even though this rank is full.
                augmented = vcat(invariants, collect(keys(fixing.reported)))
                values_at_probe = Dict(Num(q) => 0.7 + 0.3i for (i,q) in enumerate(coordinates))
                jac = Symbolics.jacobian(augmented, coordinates)
                numeric_jac = map(q -> Float64(Symbolics.value(Symbolics.substitute(q, values_at_probe))), jac)
                @test rank(numeric_jac) == length(coordinates)
            end
            local_analysis, legacy = quiet_call() do
                (ODEPE.analyze_si_structure(model, mq, si_ode; strategy=:local_basis),
                 ODEPE.analyze_si_structure(model, mq, si_ode; strategy=:identifiable_functions))
            end
            nonid(analysis) = Set(k for (k,v) in analysis.classification if v == :nonidentifiable)
            @test nonid(local_analysis) == nonid(legacy)
            @test !isnothing(legacy.identifiable_functions)
            if name == "finite_ambiguity"
                @test local_analysis.classification["a"] == :locally_identifiable
                @test legacy.classification["a"] == :locally
            end
        end
    end
end

@testset "Analysis reuse and state-at-anchor substitutions" begin
    @parameters a
    @variables x(t) z(t) y(t)
    model, mq = create_ordered_ode_system("basis_state_template", [x,z], [a],
        [D(x) ~ a*x, D(z) ~ a*z], [y ~ x+z])
    data = OrderedDict{Union{Num,String},Vector{Float64}}("t" => [0.0,0.5,1.0],
        Num(x+z) => 3exp.(0.4 .* [0.0,0.5,1.0]))
    initial = quiet_call() do
        ODEPE.build_si_template_for_fixed_params(model, mq, data, nothing;
            si_fix_strategy=:local_basis, compute_multiplicity=false)
    end
    fixing = ODEPE.derive_structural_fix_set(initial, false; states=[x,z], params=[a])
    fixed = quiet_call() do
        ODEPE.build_si_template_for_fixed_params(model, mq, data, nothing;
            si_fix_strategy=:local_basis, structural_analysis=initial.structural_analysis,
            pre_fixed_params=fixing.pre_fixed, compute_multiplicity=false)
    end
    @test initial.structural_analysis === fixed.structural_analysis
    @test !initial.rank_trimming_metadata.structural_analysis_reused
    @test fixed.rank_trimming_metadata.structural_analysis_reused
    @test !haskey(fixed.rank_trimming_metadata.equation_builder_timing, :assess_local_identifiability)
    @test any(isequal(only(keys(fixing.reported))), (x,z))
    fixed_state_name = string(only(keys(fixing.pre_fixed)))
    fixed_jet = Symbolics.variable(Symbol(fixed_state_name * "_0"))
    all_variables = union(Symbolics.get_variables.(fixed.all_equations)...)
    @test !(fixed_jet in all_variables)
    # Independently substitute ONLY the anchor value in the original equations.
    # A constant-trajectory substitution would also erase the nonzero derivative.
    expected = Symbolics.substitute.(initial.all_equations, Ref(Dict(fixed_jet=>1.0)))
    expected = filter(q -> !isequal(Symbolics.simplify(q), 0), expected)
    @test isequal(expected, fixed.all_equations)
    @test isequal(ModelingToolkit.equations(model.system), [D(x) ~ a*x, D(z) ~ a*z])
    @test haskey(initial.structural_analysis.classification, fixed_state_name)
    @test initial.structural_analysis.classification[fixed_state_name] == :nonidentifiable
    @test !(fixed_state_name in string.(fixed.unidentifiable))

    @test_throws ArgumentError ODEPE.validate_si_analysis_reuse(initial.structural_analysis, model, mq, :identifiable_functions, 0.99)
    @test_throws ArgumentError ODEPE.validate_si_analysis_reuse(initial.structural_analysis, model, mq, :local_basis, 0.9)
    @test_throws ArgumentError ODEPE.validate_si_analysis_reuse(initial.structural_analysis, model, [y ~ x], :local_basis, 0.99)
    changed, changed_mq = create_ordered_ode_system("different_model", [x,z], [a],
        [D(x) ~ a^2*x, D(z) ~ a*z], mq)
    @test_throws ArgumentError ODEPE.validate_si_analysis_reuse(initial.structural_analysis, changed, changed_mq, :local_basis, 0.99)
end

@testset "Finite branches survive both SI strategies" begin
    @parameters a
    @variables x(t) y(t)
    model, mq = create_ordered_ode_system("basis_two_branches", [x], [a], [D(x) ~ a^2*x], [y ~ x])
    pep = ParameterEstimationProblem("basis_two_branches", model, mq, nothing, [-0.5,0.5], nothing,
        OrderedDict(a=>0.7), OrderedDict(x=>1.2), 0)
    for strategy in (:local_basis, :identifiable_functions)
        opts = merge_options(FAST_STANDARD_OPTS; si_fix_strategy=strategy, use_multipoint=false,
            auto_rescale=false, compute_uncertainty=false)
        sampled = ODEPE.sample_problem_data(pep, opts)
        raw, analysis, _ = quiet_call() do
            analyze_parameter_estimation_problem(sampled, opts)
        end
        @test !isempty(raw[1])
        estimates = first(analysis)
        @test any(r -> isapprox(r.parameters[a], 0.7; atol=1e-6), estimates)
        @test any(r -> isapprox(r.parameters[a], -0.7; atol=1e-6), estimates)
        @test all(r -> isempty(r.provenance.structural_fix_set), estimates)
        @test all(r -> isempty(r.all_unidentifiable), estimates)
    end
end

@testset "Recovery with parameter and state representative choices" begin
    @parameters a b
    @variables x(t) z(t) y(t)
    examples = [
        ("parameter_ic", [x], [a,b], [D(x) ~ a*x], [y ~ b*x],
            OrderedDict(a=>0.4,b=>2.0), OrderedDict(x=>1.5)),
        ("state_freedom", [x,z], [a], [D(x) ~ a*x,D(z) ~ a*z], [y ~ x+z],
            OrderedDict(a=>0.4), OrderedDict(x=>1.3,z=>1.7)),
    ]
    for (name, states, params, equations, measured, truth, ic) in examples
        model, mq = create_ordered_ode_system("basis_recovery_" * name, states, params, equations, measured)
        pep = ParameterEstimationProblem(name, model, mq, nothing, [0.0,1.0], nothing, truth, ic, 0)
        opts = merge_options(FAST_STANDARD_OPTS; si_fix_strategy=:local_basis,
            use_multipoint=false, auto_rescale=false, compute_uncertainty=false)
        sampled = ODEPE.sample_problem_data(pep, opts)
        raw, analysis, _ = quiet_call() do
            analyze_parameter_estimation_problem(sampled, opts)
        end
        @test !isempty(raw[1])
        best = best_cluster_solution(analysis)
        @test !isnothing(best)
        @test best.err < 1e-8
        @test best.parameters[a] ≈ 0.4 atol=1e-6
        @test length(best.provenance.structural_fix_set) == 1
        amplitude = name == "parameter_ic" ? best.parameters[b]*best.states[x] : best.states[x]+best.states[z]
        @test amplitude ≈ 3.0 atol=1e-6
        @test isequal(ModelingToolkit.equations(model.system), equations)
    end
end
