using ODEParameterEstimation
using ODEParameterEstimation: DerivativeData, OrderedODESystem
using Test
using ModelingToolkit
using ModelingToolkit: System
using OrderedCollections
using Symbolics

@testset "Core Types" begin
    @testset "OrderedODESystem" begin
        # Create a simple test ODE system
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
        
        @named model = System(eqs, t, states, params)
        
        # Create OrderedODESystem
        ordered_sys = ODEParameterEstimation.OrderedODESystem(model, params, states)
        
        # Test type and property access
        @test typeof(ordered_sys) == ODEParameterEstimation.OrderedODESystem
        @test ordered_sys.system === model
        @test isequal(ordered_sys.original_parameters, params)
        @test isequal(ordered_sys.original_states, states)
    end
    
    @testset "ParameterEstimationProblem" begin
        # Create a simple test ODE system
        @independent_variables t
        @parameters a b
        @variables x1(t) x2(t) y1(t)
        D = Differential(t)
        
        eqs = [
            D(x1) ~ -a * x2,
            D(x2) ~ b * x1
        ]
        states = [x1, x2]
        params = [a, b]
        measured_quantities = [y1 ~ x1]
        
        @named model = System(eqs, t, states, params)
        ordered_sys = ODEParameterEstimation.OrderedODESystem(model, params, states)
        
        # Basic PEP with no data
        pep = ODEParameterEstimation.ParameterEstimationProblem(
            "TestModel",
            ordered_sys,
            measured_quantities,
            nothing,
            nothing,
            ODEParameterEstimation.package_wide_default_ode_solver,
            OrderedDict(a => 0.5, b => 1.0),  # p_true
            OrderedDict(x1 => 1.0, x2 => 2.0),  # ic
            0            # unident_count
        )
        
        # Check properties
        @test pep.name == "TestModel"
        @test pep.model === ordered_sys
        @test pep.measured_quantities == measured_quantities
        @test isnothing(pep.data_sample)
        @test isnothing(pep.recommended_time_interval)
        @test isequal(pep.p_true, OrderedDict(a => 0.5, b => 1.0))
        @test isequal(pep.ic, OrderedDict(x1 => 1.0, x2 => 2.0))
        @test pep.unident_count == 0
        
        # PEP with data sample. Build at the field's concrete key type
        # (Union{String, Num}) so the constructor stores the object without a
        # converting copy — otherwise the `===` check below compares a copy.
        data_dict = OrderedDict{Union{String, Symbolics.Num}, Vector{Float64}}(
            "t" => [0.0, 0.1, 0.2],
            x1 => [1.0, 1.1, 1.2]
        )
        
        pep2 = ODEParameterEstimation.ParameterEstimationProblem(
            "TestModel2",
            ordered_sys,
            measured_quantities,
            data_dict,
            [-0.5, 0.5],
            ODEParameterEstimation.package_wide_default_ode_solver,
            OrderedDict(a => 0.5, b => 1.0),
            OrderedDict(x1 => 1.0, x2 => 2.0),
            0
        )
        
        @test pep2.name == "TestModel2"
        @test pep2.data_sample === data_dict
        @test pep2.recommended_time_interval == [-0.5, 0.5]
    end
    
    @testset "DerivativeData" begin
        # Build with real Symbolics variables at the struct's concrete field
        # types (Vector{Vector{Num}} / Set{Num}) so the stored objects are
        # identity-preserved. (The old Symbolics.wrap(Symbol(...)) produced
        # Vector{Vector{Symbol}}, which does not convert into the Num fields.)
        @variables x1 y1 obs1 val1 param1
        states_lhs = [[x1]]
        states_rhs = [[y1]]
        obs_lhs = [[obs1]]
        obs_rhs = [[val1]]
        all_unident = Set{Num}([param1])

        # Create DerivativeData object
        dd = ODEParameterEstimation.DerivativeData(
            states_lhs, states_rhs, obs_lhs, obs_rhs,
            states_lhs, states_rhs, obs_lhs, obs_rhs,
            all_unident
        )
        
        # Check properties
        @test dd.states_lhs_cleared === states_lhs
        @test dd.states_rhs_cleared === states_rhs
        @test dd.obs_lhs_cleared === obs_lhs
        @test dd.obs_rhs_cleared === obs_rhs
        @test dd.states_lhs === states_lhs
        @test dd.states_rhs === states_rhs
        @test dd.obs_lhs === obs_lhs
        @test dd.obs_rhs === obs_rhs
        @test dd.all_unidentifiable === all_unident
    end
    
    @testset "ParameterEstimationResult" begin
        # Create test data
        @parameters a b
        @variables t x1(t) x2(t)
        
        params_dict = OrderedDict([a => 0.5, b => 1.0])
        states_dict = OrderedDict([x1 => 1.0, x2 => 2.0])
        
        # Create result object
        result = ODEParameterEstimation.ParameterEstimationResult(
            params_dict,
            states_dict,
            0.0,            # at_time
            1e-5,           # err
            :Success,       # return_code
            10,             # datasize
            nothing,        # report_time
            Dict(b => 1.0), # unident_dict
            Set([b]),       # all_unidentifiable
            nothing         # solution
        )
        
        # Check properties
        @test result.parameters === params_dict
        @test result.states === states_dict
        @test result.at_time == 0.0
        @test result.err == 1e-5
        @test result.return_code == :Success
        @test result.datasize == 10
        @test isnothing(result.report_time)
        @test haskey(result.unident_dict, b)
        @test b in result.all_unidentifiable
        @test isnothing(result.solution)
        @test result.provenance isa ODEParameterEstimation.ResultProvenance
        @test result.provenance.primary_method == :algebraic
        @test result.provenance.rescue_path == :none
        @test !result.provenance.polish_applied
        @test isempty(result.provenance.representative_assignments)

        @testset "Mapping conversion across constructor forms" begin
            # Raw MTK keys and integer values require conversion. Unordered
            # dictionaries must work without relying on Base.convert, which
            # OrderedCollections 2 deliberately rejects for this operation.
            plain_params = Dict(Symbolics.value(a) => 2, Symbolics.value(b) => 3)
            plain_states = Dict(Symbolics.value(x1) => 4, Symbolics.value(x2) => 5)
            plain_unident = Dict(Symbolics.value(b) => 3)
            args = (plain_params, plain_states, 0, nothing, nothing, 10,
                    nothing, plain_unident, Set([b]), nothing)
            provenance = ODEParameterEstimation.ResultProvenance()
            constructed = (
                ODEParameterEstimation.ParameterEstimationResult(args...),
                ODEParameterEstimation.ParameterEstimationResult(args..., :aaa, provenance),
                ODEParameterEstimation.ParameterEstimationResult(args..., :aaa, provenance, 2),
            )
            for converted in constructed
                @test converted.parameters isa OrderedDict{Num, Float64}
                @test converted.states isa OrderedDict{Num, Float64}
                @test converted.unident_dict isa OrderedDict{Num, Float64}
                @test isequal(converted.parameters, OrderedDict(a => 2.0, b => 3.0))
                @test isequal(converted.states, OrderedDict(x1 => 4.0, x2 => 5.0))
                @test converted.unident_dict[b] == 3.0
            end
            @test constructed[1].branch_size == 1
            @test constructed[2].provenance === provenance
            @test constructed[3].branch_size == 2

            ordered_params = OrderedDict{Num, Float64}(b => 3.0, a => 2.0)
            ordered_states = OrderedDict{Num, Float64}(x2 => 5.0, x1 => 4.0)
            ordered_unident = OrderedDict{Num, Float64}(b => 3.0)
            for unident in (nothing, ordered_unident)
                preserved = @inferred ODEParameterEstimation.ParameterEstimationResult(
                    ordered_params, ordered_states, 0.0, nothing, nothing, 10,
                    nothing, unident, Set{Num}([b]), nothing, :aaa, provenance, 1,
                )
                @test preserved.parameters === ordered_params
                @test preserved.states === ordered_states
                @test preserved.unident_dict === unident
                @test isequal(collect(keys(preserved.parameters)), [b, a])
                @test isequal(collect(keys(preserved.states)), [x2, x1])
            end
        end
    end
    
    @testset "Constants" begin
        # Test that constants are defined and have expected types
        @test typeof(ODEParameterEstimation.CLUSTERING_THRESHOLD) <: AbstractFloat
        @test typeof(ODEParameterEstimation.MAX_ERROR_THRESHOLD) <: AbstractFloat
        @test typeof(ODEParameterEstimation.IMAG_THRESHOLD) <: AbstractFloat
        @test typeof(ODEParameterEstimation.MAX_SOLUTIONS) <: Integer
    end
end
