using ODEParameterEstimation
using ODEParameterEstimation: tag_symbol, unpack_ODE
using Test
using ModelingToolkit
using ModelingToolkit: System
using OrderedCollections

@testset "Model Utils" begin
    @testset "unpack_ODE" begin
        # Create a simple test ODE model
        @independent_variables t
        @parameters a b
        @variables x1(t) x2(t)
        D = Differential(t)
        
        eqs = [
            D(x1) ~ -a * x2,
            D(x2) ~ b * x1
        ]
        states = [x1, x2]
        params = [a, b]
        
        @named test_model = System(eqs, t, states, params)
        
        # Test the unpack_ODE function
        time_var, equations, state_vars, parameters = ODEParameterEstimation.unpack_ODE(test_model)
        
        # Verify the results
        @test isequal(time_var, t)
        @test length(equations) == 2
        @test isequal(state_vars, states)
        @test isequal(parameters, params)
    end
    
    @testset "tag_symbol" begin
        # Create test symbols
        @variables t x(t) y(t) z
        
        # Test tagging time-dependent variables
        tagged_x = ODEParameterEstimation.tag_symbol(x, "pre_", "_post")
        @test string(tagged_x) == "pre_x_t_post"
        
        # Test tagging another time-dependent variable
        tagged_y = ODEParameterEstimation.tag_symbol(y, "tag_", "_end")
        @test string(tagged_y) == "tag_y_t_end"
        
        # Test tagging a non-time-dependent variable
        tagged_z = ODEParameterEstimation.tag_symbol(z, "start_", "_finish")
        @test string(tagged_z) == "start_z_finish"
    end
    
    @testset "create_ordered_ode_system" begin
        # Create a simple ODE system
        @independent_variables t
        @parameters a b
        @variables x1(t) x2(t) y1(t) y2(t)
        D = Differential(t)
        
        eqs = [
            D(x1) ~ -a * x2,
            D(x2) ~ b * x1
        ]
        states = [x1, x2]
        params = [a, b]
        measured_quantities = [y1 ~ x1, y2 ~ x2]
        
        # Create ordered ODE system
        ordered_system, mq = ODEParameterEstimation.create_ordered_ode_system(
            "TestSystem", states, params, eqs, measured_quantities)
        
        # Verify the results
        @test typeof(ordered_system) == ODEParameterEstimation.OrderedODESystem
        @test nameof(ordered_system.system) == :TestSystem
        @test isequal(ordered_system.original_parameters, params)
        @test isequal(ordered_system.original_states, states)
        @test isequal(mq, measured_quantities)
    end
    
    @testset "unident_subst!" begin
        # Create equations and measured quantities
        @parameters a b c
        @variables t x1(t) x2(t) y1(t)
        D = Differential(t)
        
        eqs = [
            D(x1) ~ a * x1 + b * x2,
            D(x2) ~ c * x1 - b * x2
        ]
        
        measured_quantities = [y1 ~ a * x1 + c * x2]
        
        # Create unidentifiable parameters dictionary
        unident_dict = Dict(b => 2.0)
        
        # Make a deep copy for testing
        eqs_copy = deepcopy(eqs)
        mq_copy = deepcopy(measured_quantities)
        
        # Apply substitution
        ODEParameterEstimation.unident_subst!(eqs_copy, mq_copy, unident_dict)
        
        # Verify the results - b should be replaced with 2.0
        @test !any(contains.(string.(eqs_copy[1].rhs), "b"))
        @test !any(contains.(string.(eqs_copy[2].rhs), "b"))
        @test !any(contains.(string.(mq_copy[1].rhs), "b"))
        
        # Test that a and c are still present
        @test any(contains.(string.(eqs_copy[1].rhs), "a"))
        @test any(contains.(string.(eqs_copy[2].rhs), "c"))
        @test any(contains.(string.(mq_copy[1].rhs), "a"))
        @test any(contains.(string.(mq_copy[1].rhs), "c"))
    end
end