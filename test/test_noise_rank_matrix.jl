using ODEParameterEstimation
using ODEParameterEstimation: package_wide_default_ode_solver
using LinearAlgebra
using Symbolics
using Test
using ModelingToolkit, OrderedCollections
using ModelingToolkit: t_nounits as t, D_nounits as D

@testset "Noise-frontier Jacobian and rank" begin
    @variables x[1:17]
    variables = collect(x)

    # More variables than a typical AD chunk, with an independent, exact
    # Jacobian. Redundant equations must not change the column-rank decision.
    A = zeros(19, 17)
    A[1:17, :] = diagm(Float64.(1:17))
    A[18, :] = A[1, :] + A[3, :]
    A[19, :] = 2 * A[17, :]
    J, observed_rank = ODEParameterEstimation._noise_rank_matrix(A * variables, variables)
    @test J == A
    @test observed_rank == 17

    A[:, 17] = 2 * A[:, 16]
    J, observed_rank = ODEParameterEstimation._noise_rank_matrix(A * variables, variables)
    @test J == A
    @test observed_rank == 16

    # Nonlinear equations with an exact row dependence at every probe point.
    equations = [x[1]^2 + x[2], 2 * x[1]^2 + 2 * x[2], x[3]]
    J, observed_rank = ODEParameterEstimation._noise_rank_matrix(equations, variables[1:3])
    @test observed_rank == 2
    @test J[2, :] == 2 * J[1, :]
    @test J[1, 2:3] == [1, 0]
    @test J[3, :] == [0, 0, 1]
end

@testset "Shared multipoint parameters without reference values" begin
    @parameters k
    @variables x(t) y(t)
    model, measured = create_ordered_ode_system("unlabelled_decay", [x], [k],
        [D(x) ~ -k*x], [y ~ x])
    pep = ParameterEstimationProblem("unlabelled_decay", model, measured,
        OrderedDict{Union{Num, String}, Vector{Float64}}(x=>[1.0, 0.5, 0.25], "t"=>[0.0, 1.0, 2.0]),
        [0.0, 2.0], package_wide_default_ode_solver,
        OrderedDict{Num, Float64}(), OrderedDict{Num, Float64}(), 0)
    @variables k_0 x_0 y_1
    eqs = [k_0*x_0 + y_1]
    variables = [k_0, x_0, y_1]
    ODEPE = ODEParameterEstimation
    renamed = ODEPE._noise_rename_symbolic_equations(eqs, variables,
        Dict(v=>string(v) for v in variables), ODEPE._noise_model_param_set(pep), 2)
    @test Set(string.(Symbolics.get_variables(only(renamed)))) == Set(["k_0", "x_0_pt2", "y_1_pt2"])
    @test ODEPE._classify_polynomial_variables(["k", "k_0", "x_0", "y_1"], pep) ==
        Dict("k"=>:parameter, "k_0"=>:parameter, "x_0"=>:state_ic, "y_1"=>:data_derivative)
end
