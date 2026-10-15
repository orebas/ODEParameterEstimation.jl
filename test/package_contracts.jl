using ODEParameterEstimation
using Test

@testset "Exported bindings are defined" begin
    undefined = filter(name -> !isdefined(ODEParameterEstimation, name),
                       names(ODEParameterEstimation))
    @test isempty(undefined)
end

# What 1.x promises. Removing a name from either list is a breaking change;
# adding one is not, but should be a decision, not a side effect.
@testset "Exported and public names are the intended ones" begin
    exported = Set([
        :ParameterEstimationProblem, :create_ordered_ode_system, :sample_problem_data,
        :ObservationData, :ObservationSeries, :observation_times,
        :estimate, :analyze_parameter_estimation_problem, :EstimationOptions,
        :SolverHC, :SolverTrustRegion, :SolverLevenbergMarquardt,
        :FlowStandard, :FlowDirectOpt,
        :PolishLeastSquares, :PolishNewtonTrust, :PolishBFGS, :PolishLBFGS,
        :InterpolatorAAA, :InterpolatorGP, :InterpolatorGPUncertainty, :InterpolatorCustom,
        :ParameterEstimationResult,
        :UncertaintyReport, :UncertaintyUnavailable, :print_uncertainty_results,
        :UnsupportedModelClassError, :UnsupportedDerivativeOrderError, :SamplingFailureError,
    ])
    public_only = Set([
        :Examples,
        :SystemSolverMethod, :EstimationFlow, :PolishMethod, :InterpolatorMethod,
        :OrderedODESystem, :ResultProvenance,
        :AbstractUQOutcome, :UncertaintyError, :uq_reliability, :UQReliabilityAssessment,
        :merge_options, :validate_options, :print_options,
    ])
    M = ODEParameterEstimation
    listed = filter(!=(:ODEParameterEstimation), names(M))
    @test Set(filter(n -> Base.isexported(M, n), listed)) == exported
    @test Set(filter(n -> !Base.isexported(M, n) && Base.ispublic(M, n), listed)) == public_only
end

# A name exported by two imported packages cannot be used unqualified: it stays
# undefined in this module and fails only when that line runs. Lowered method
# bodies record every global they reference, so all of them can be checked up
# front. This reads runtime internals; it returns `nothing` where they are
# unavailable (Julia versions before the single global method table).
function ambiguous_import_references(pkg::Module)
    isdefined(Core, :methodtable) || return nothing
    inpkg(m::Module) = m === pkg || (parentmodule(m) !== m && inpkg(parentmodule(m)))
    referenced = Set{Tuple{Module, Symbol}}()
    function walk(x)
        if x isa GlobalRef
            push!(referenced, (x.mod, x.name))
        elseif x isa Expr
            foreach(walk, x.args)
        end
        return nothing
    end
    Base.visit(Core.methodtable) do method
        inpkg(method.module) || return nothing
        code = try
            Base.uncompressed_ast(method).code
        catch
            return nothing  # no stored source, e.g. generated functions
        end
        foreach(walk, code)
        return nothing
    end
    imports(mod) = ccall(:jl_module_usings, Any, (Any,), mod)
    return sort!([
        string(mod, ".", name) for (mod, name) in referenced
        if inpkg(mod) && !isdefined(mod, name) &&
           any(used -> Base.isexported(used, name), imports(mod))
    ])
end

@testset "Unqualified names are not ambiguous imports" begin
    ambiguous = try
        ambiguous_import_references(ODEParameterEstimation)
    catch err
        err isa InterruptException && rethrow()
        nothing
    end
    ambiguous === nothing && @info "Ambiguous-import check skipped: method bodies are not inspectable on this Julia version"
    @test ambiguous === nothing || isempty(ambiguous)
end

# The citation files name the version by hand. A release that changes
# `Project.toml` has to change them too.
@testset "Citation files name the current version" begin
    root = pkgdir(ODEParameterEstimation)
    version = string(pkgversion(ODEParameterEstimation))
    @test occursin("\nversion: $version\n", read(joinpath(root, "CITATION.cff"), String))
    @test occursin("version $version}", read(joinpath(root, "CITATION.bib"), String))
end
