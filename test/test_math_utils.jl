using ODEParameterEstimation
using ODEParameterEstimation: calculate_error_stats, clear_denoms, hmcs
using Test
using ModelingToolkit
using Symbolics
using HomotopyContinuation
using Statistics

@testset "Math Utils" begin
    @testset "clear_denoms" begin
        # Create symbolic variables and equations with fractions
        @variables x y z w
        
        # Test a simple fraction equation: x/y = z
        eq1 = x/y ~ z
        cleared_eq1 = ODEParameterEstimation.clear_denoms(eq1)
        @test isequal(cleared_eq1.lhs, x)
        @test isequal(cleared_eq1.rhs, y*z)
        
        # Test a complex fraction: (x+y)/(z-w) = y
        eq2 = (x+y)/(z-w) ~ y
        cleared_eq2 = ODEParameterEstimation.clear_denoms(eq2)
        @test isequal(cleared_eq2.lhs, (x+y))
        @test isequal(cleared_eq2.rhs, y*(z-w))
        
        # Test a non-fraction equation: x = y
        eq3 = x ~ y
        cleared_eq3 = ODEParameterEstimation.clear_denoms(eq3)
        @test isequal(cleared_eq3, eq3)  # Should remain unchanged

        # Both denominators must disappear, including a fraction only on RHS.
        cleared_both = ODEParameterEstimation.clear_denoms(x/y ~ z/w)
        @test isequal(cleared_both.lhs, x*w)
        @test isequal(cleared_both.rhs, z*y)
        cleared_rhs = ODEParameterEstimation.clear_denoms(x ~ y/z)
        @test isequal(cleared_rhs.lhs, x*z)
        @test isequal(cleared_rhs.rhs, y)
        @test isequal(ODEParameterEstimation.clear_denoms((x+y)/z), x+y)
    end
    
    @testset "hmcs" begin
        # Test conversion of strings to HomotopyContinuation variables
        var1 = ODEParameterEstimation.hmcs("x")
        @test typeof(var1) == HomotopyContinuation.ModelKit.Variable
        @test string(var1) == "x"
        
        var2 = ODEParameterEstimation.hmcs("parameter_1")
        @test typeof(var2) == HomotopyContinuation.ModelKit.Variable
        @test string(var2) == "parameter_1"
    end
    
    @testset "count_turns" begin
        # Test with no turns
        series1 = [1, 2, 3, 4, 5]
        @test ODEParameterEstimation.count_turns(series1) == 0
        
        # Test with one turn (increasing then decreasing)
        series2 = [1, 2, 3, 2, 1]
        @test ODEParameterEstimation.count_turns(series2) == 1
        
        # Test with two turns (up, down, up)
        series3 = [1, 3, 2, 1, 2, 4]
        @test ODEParameterEstimation.count_turns(series3) == 2
        
        # Test with short series (fewer than 3 points)
        series4 = [1, 2]
        @test ODEParameterEstimation.count_turns(series4) == 0
    end
    
    @testset "calculate_timeseries_stats" begin
        # Test with a simple series
        series = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = ODEParameterEstimation.calculate_timeseries_stats(series)
        
        @test stats.mean ≈ 3.0
        @test stats.std ≈ std(series)
        @test stats.min == 1.0
        @test stats.max == 5.0
        @test stats.range == 4.0
        @test stats.turns == 0
        
        # Test with a series containing turns
        series2 = [1.0, 3.0, 2.0, 4.0, 2.0]
        stats2 = ODEParameterEstimation.calculate_timeseries_stats(series2)
        
        @test stats2.mean ≈ 2.4
        # Local extrema occur at 3.0, 2.0, and 4.0.
        @test stats2.turns == 3
    end
    
    @testset "calculate_error_stats" begin
        # Test with known predicted and actual values
        predicted = [1.1, 2.2, 3.3, 4.4, 5.5]
        actual = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        error_stats = ODEParameterEstimation.calculate_error_stats(predicted, actual)
        
        # Check absolute error statistics
        @test error_stats.absolute.mean ≈ 0.1 * mean(1:5)
        @test error_stats.absolute.min ≈ 0.1
        @test error_stats.absolute.max ≈ 0.5
        
        # Check relative error statistics
        @test error_stats.relative.mean ≈ mean([0.1/1.0, 0.2/2.0, 0.3/3.0, 0.4/4.0, 0.5/5.0])
        
        # Test with some zero values in actual
        predicted2 = [0.1, 1.1, 2.1, 3.1, 4.1]
        actual2 = [0.0, 1.0, 2.0, 3.0, 4.0]
        
        error_stats2 = ODEParameterEstimation.calculate_error_stats(predicted2, actual2)
        # This should not throw a division by zero error due to the small constant added
        @test !isnan(error_stats2.relative.mean)
    end
end
