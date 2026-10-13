using ODEParameterEstimation, ModelingToolkit, Symbolics, OrderedCollections, Random, Test
using ModelingToolkit: t_nounits as t, D_nounits as D
include("estimation_helpers.jl")
const ODEPE = ODEParameterEstimation
const N = ODEPE.Nemo
const SIAN = ODEPE.SIANBackend

@testset "Exact template representative substitution" begin
    R, (k_1, k_10, x0, x1, y0) = N.polynomial_ring(N.QQ,
        ["k_1_0", "k_10_0", "x_0", "x_1", "y_0"])
    # The first row becomes exactly zero by cancellation after specialization.
    # A nonzero constant row must survive, and fixing x_0 must not fix x_1.
    original = [k_1*x0 - 2*x0, x1-k_1*x0, y0-k_10*x0, k_1-3]
    snapshot = deepcopy(original)
    fixes = OrderedDict(:k_1 => 2, :x_0 => 3//2)
    result = quiet_call() do
        ODEPE._substitute_si_template_polynomials(original, fixes)
    end
    @test result == [R(0), x1-3, y0-(3//2)*k_10, R(-1)]
    @test original == snapshot
    @test all(p -> parent(p) === R, result)
    @test findall(iszero, result) == [1]
    @test x1 in N.vars(result[2])
    @test k_10 in N.vars(result[3])

    # Preserve Float64's exact value; do not silently round 0.1 to 1//10.
    float_result = quiet_call() do
        ODEPE._substitute_si_template_polynomials([k_1*x0, k_1-1//10],
            OrderedDict(:k_1 => 0.1))
    end
    exact_tenth = N.QQ(Rational{BigInt}(0.1))
    @test float_result == [exact_tenth*x0, R(exact_tenth-1//10)]
    @test !iszero(float_result[2])

    # Check polynomial identities at independent points, including a zero state.
    for point in ([7,5,0,11,13], [17,19,23,29,31])
        unfixed_point = N.QQ.(point)
        fixed_point = copy(unfixed_point)
        fixed_point[1], fixed_point[3] = N.QQ(2), N.QQ(3//2)
        @test [N.evaluate(p, unfixed_point) for p in result] ==
            [N.evaluate(p, fixed_point) for p in original]
    end
end

@testset "Dependent rows do not imply coordinate identifiability" begin
    _, (a,b) = N.polynomial_ring(N.QQ, ["a", "b"])
    sample = N.QQ.([1,2])
    sum_equations = [a+b-3, 2*(a+b-3)]
    info = ODEPE._sian_local_coordinates(sum_equations, [a,b], [a,b], sample)
    @test info.jacobian_rank == 1
    @test isempty(info.locally_identifiable)
    # A finite sign ambiguity remains locally identifiable even with redundant rows.
    finite = ODEPE._sian_local_coordinates([a^2-1, 2*(a^2-1)], [a,b], [a,b], sample)
    @test finite.jacobian_rank == 1
    @test finite.locally_identifiable == [a]
end

@testset "Supplied or disabled M bypasses counting in estimation" begin
    base = (; (k => getfield(FAST_STANDARD_OPTS, k) for k in fieldnames(EstimationOptions))...)
    for (compute, supplied, expected) in ((false, nothing, nothing), (true, 7, 7), (true, nothing, 1))
        opts = EstimationOptions(; merge(base, (compute_algebraic_multiplicity=compute,
            algebraic_multiplicity=supplied, compute_uncertainty=false))...)
        sampled = ODEPE.sample_problem_data(ODEPE.Examples.harmonic_oscillator(), opts)
        context = ODEPE.RunContext(; capture_timing=true)
        Random.seed!(20260915)
        _, analysis, _ = quiet_call() do
            Base.ScopedValues.with(ODEPE.RUN_CONTEXT => context) do
                ODEPE.analyze_parameter_estimation_problem(sampled, opts)
            end
        end
        @test !isempty(first(analysis))
        @test first(first(analysis)).err < 1e-8
        @test analysis.algebraic_multiplicity === expected
        timing = context.timing.details[:si_template_algebraic_multiplicity_timing]
        if compute && isnothing(supplied)
            @test haskey(timing, :groebner_seconds)
            @test timing[:multiplicity] == 1
        else
            @test timing[:skipped]
            @test timing[:disabled_by_caller]
            @test !haskey(timing, :groebner_seconds)
        end
    end
end

function multiplicity_fixture(model, measured, fixes)
    ode, _, _ = ODEPE.convert_to_si_ode(model, measured)
    _, Q, x_eqs, y_eqs, xs, ys, us, mu, _, gens = SIAN.get_equations(ode)
    n,m,u = length(xs),length(ys),length(us)
    s = n+length(mu)
    X,X_eq = SIAN.get_x_eq(x_eqs,y_eqs,n,m,s,u,gens)
    Y,Y_eq = SIAN.get_y_eq(x_eqs,y_eqs,n,m,s,u,gens)
    params = vcat(gens[(end-length(mu)+1):end],gens[1:n])
    sample = SIAN.sample_point(big(101),xs,ys,N.QQMPolyRingElem[],params,X_eq,Y_eq,Q)
    # Full constraints through the first output derivative suffice for these
    # independently solvable fixtures. Include redundant observations as given.
    Et = vcat([X[i][1] for i in 1:n], [Y[i][j] for i in 1:m for j in 1:2])
    return ODEPE._prepare_sian_multiplicity_system(ode,Et,Q,X_eq,Y_eq,params,sample,big(101);
        pre_fixed_params=fixes), Et
end

@testset "Multiplicity after representative fixing" begin
    @parameters a b c
    @variables x(t) z(t) y(t) w(t)
    cases = [
        ("parameter_ic", [x], [a,b], [D(x)~a*x], [y~b*x], OrderedDict(b=>1.0), 1),
        ("finite_branches", [x], [a,b], [D(x)~a^2*x], [y~b*x], OrderedDict(b=>1.0), 2),
        ("state_at_anchor", [x,z], [a], [D(x)~a*x,D(z)~a*z], [y~x+z], OrderedDict(z=>1.0), 1),
        ("rational", [x], [a,b,c], [D(x)~a*x/(b+c)], [y~x], OrderedDict(b=>1.0,c=>2.0), 1),
        ("consistent_fixed_rate", [x], [a], [D(x)~a*x], [y~x], OrderedDict(a=>1.5), 1),
        ("redundant_output_constraints", [x], [a], [D(x)~a*x], [y~x^2,w~x], OrderedDict{Num,Float64}(), 1),
    ]
    for (name,xs,ps,eqs,obs,reported_fixes,expected_M) in cases
        @testset "$name" begin
            model,mq = create_ordered_ode_system("multiplicity_"*name,xs,ps,eqs,obs)
            # Structural fixing uses SI base names, not timed MTK state symbols.
            fixes = OrderedDict(Symbolics.variable(Symbol(replace(string(k),"(t)"=>"")))=>v for (k,v) in reported_fixes)
            Random.seed!(20260915)
            system,Et = quiet_call() do
                multiplicity_fixture(model,mq,fixes)
            end
            names = string.(N.gens(system.ring))
            @test all(k -> !(string(k)*"_0" in names),keys(fixes))
            @test all(p -> iszero(N.evaluate(p,system.point)),system.polynomials)
            @test length(system.polynomials) == length(Et)+1
            @test N.ngens(system.ring)-1-system.jacobian_rank == 0
            gb = ODEPE.Groebner.groebner(system.polynomials)
            @test ODEPE.Groebner.dimension(gb) == 0
            @test length(ODEPE.Groebner.quotient_basis(gb)) == expected_M
            if name == "state_at_anchor"
                @test "z_1" in names
                @test !iszero(system.point[findfirst(==("z_1"),names)])
                @test isequal(ModelingToolkit.equations(model.system),eqs)
            elseif name == "rational"
                # Substituting both denominator parameters must also simplify Q.
                @test length(N.vars(last(system.polynomials))) == 1
                @test N.total_degree(last(system.polynomials)) == 1
            end
            # Exercise the actual construction/caller, including fixed-ring M metadata.
            data = OrderedDict{Union{Num,String},Vector{Float64}}("t"=>[0.0,1.0])
            for equation in mq
                data[Num(equation.rhs)] = [1.0,2.0]
            end
            template = quiet_call() do
                ODEPE.build_si_template_for_fixed_params(model,mq,data,nothing;pre_fixed_params=fixes)
            end
            @test template.rank_trimming_metadata.algebraic_multiplicity == expected_M
            metadata = template.rank_trimming_metadata
            @test length(metadata.selected_equation_indices) == length(template.equations)
            @test isempty(intersect(metadata.selected_equation_indices, metadata.dropped_equation_indices))
            @test sort(vcat(metadata.selected_equation_indices, metadata.dropped_equation_indices)) ==
                collect(1:metadata.original_equation_count)
            @test all(k -> all(eq -> !(string(k)*"_0" in string.(Symbolics.get_variables(eq))),
                template.all_equations), keys(fixes))
            timing = template.rank_trimming_metadata.algebraic_multiplicity_timing
            @test timing[:sample_verified]
            @test timing[:linearized_dimension] == 0
            @test length(timing[:fixed_coordinates]) == length(fixes)
        end
    end

    model,mq = create_ordered_ode_system("multiplicity_pole",[x],[a,b],[D(x)~a*x/(b-1)],[y~x])
    @test_throws ArgumentError quiet_call(() -> multiplicity_fixture(model,mq,OrderedDict(b=>1.0)))
    @test_throws ArgumentError quiet_call(() -> multiplicity_fixture(model,mq,OrderedDict(c=>1.0)))
end
