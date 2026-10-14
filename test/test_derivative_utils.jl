using ODEParameterEstimation
using ODEParameterEstimation: calculate_higher_derivative_terms,
    calculate_higher_derivatives
using ModelingToolkit: @parameters
using Test
using Symbolics

@testset "Derivative utilities" begin
    t = ODEParameterEstimation.t
    @variables x(t) y(t)
    @parameters a b
    derivative = Differential(t)
    jet(value, order) = order == 0 ? value : Symbolics.diff2term((derivative^order)(value))
    same_expression(left, right) = isequal(Symbolics.simplify(Num(left - right)), Num(0))

    equations = [derivative(x) ~ a*x + b*y, derivative(y) ~ -b*x + a*y]
    original_equations = copy(equations)
    levels = calculate_higher_derivatives(equations, 2)
    @test length(levels) == 3
    @test isequal(equations, original_equations)
    for order in 0:2
        @test length(levels[order + 1]) == 2
        @test isequal(levels[order + 1][1].lhs, jet(x, order + 1))
        @test isequal(levels[order + 1][2].lhs, jet(y, order + 1))
        @test same_expression(levels[order + 1][1].rhs, a*jet(x, order) + b*jet(y, order))
        @test same_expression(levels[order + 1][2].rhs, -b*jet(x, order) + a*jet(y, order))
    end

    lhs, rhs = [x, y], [a*x + b*y, -b*x + a*y]
    original_lhs, original_rhs = copy(lhs), copy(rhs)
    lhs_levels, rhs_levels = calculate_higher_derivative_terms(lhs, rhs, 2)
    @test isequal(lhs, original_lhs)
    @test isequal(rhs, original_rhs)
    @test length(lhs_levels) == length(rhs_levels) == 3
    for order in 0:2
        @test isequal(lhs_levels[order + 1], [jet(x, order), jet(y, order)])
        @test same_expression(rhs_levels[order + 1][1], a*jet(x, order) + b*jet(y, order))
        @test same_expression(rhs_levels[order + 1][2], -b*jet(x, order) + a*jet(y, order))
    end

    @variables s z(s)
    custom_lhs, custom_rhs = calculate_higher_derivative_terms([z], [s^3], 2; independent_variable=s)
    @test isequal(custom_lhs[2][1], Symbolics.diff2term(Differential(s)(z)))
    @test same_expression(custom_rhs[2][1], 3s^2)
    @test same_expression(custom_rhs[3][1], 6s)
    @test calculate_higher_derivatives(Equation[], 2) == [Equation[], Equation[], Equation[]]
    @test_throws ArgumentError calculate_higher_derivatives(equations, -1)
    @test_throws DimensionMismatch calculate_higher_derivative_terms([x], [x, y], 1)
end
