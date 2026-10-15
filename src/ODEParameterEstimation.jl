"""
    ODEParameterEstimation

Estimate the parameters and initial conditions of an ODE model from time-series
data.

Build a [`ParameterEstimationProblem`](@ref) from a ModelingToolkit system and
your data, call [`estimate`](@ref), and read the
[`ParameterEstimationResult`](@ref)s it returns. The manual is at
<https://orebas.github.io/ODEParameterEstimation.jl/dev/>.
"""
module ODEParameterEstimation

using ModelingToolkit
import StructuralIdentifiability: ODE
using AbstractAlgebra
using BaryRational
using Dates
using ForwardDiff
using KernelFunctions
using AbstractGPs
using Groebner
using HomotopyContinuation
using LinearAlgebra
using Logging
using StructuralIdentifiability
using Nemo#using GLPK
using NonlinearSolve
using Optim, LineSearches
using Optimization, OptimizationOptimJL
using LeastSquaresOptim
using FastLevenbergMarquardt
using OrderedCollections
using OrdinaryDiffEq
using PrecompileTools
using Printf
using Random
using Statistics
using Symbolics
using TaylorDiff


using NonlinearSolve, Symbolics, ForwardDiff, FiniteDiff, LinearAlgebra
using NLopt, Optim, NLSolversBase
using SymbolicUtils

include("core/dependency_compat.jl")

include("internal/sian/SIANBackend.jl")
include("internal/gp/GPBackend.jl")

#using CSV
#using DataFrames
#using DelimitedFiles
#using Oscar
#using Plots
#using Singular
#using SymbolicIndexingInterface


const t = ModelingToolkit.t_nounits
const D = ModelingToolkit.D_nounits
const package_wide_default_ode_solver = AutoVern9(Rodas5P())
#const package_wide_default_ode_solver = Vern9()


# Include core types first
include("types/observation_data.jl")
include("types/core_types.jl")
include("types/estimation_options.jl")  # New options struct

# Include utility modules
include("core/logging_utils.jl")
include("core/run_context.jl")  # per-run scoped state (RunContext; replaces run-state globals)
include("core/math_utils.jl")
include("core/model_utils.jl")
include("core/analysis_utils.jl")
include("core/derivative_utils.jl")

# Include core functionality
include("core/si_equation_builder.jl")  # StructuralIdentifiability integration
include("core/transcendental_utils.jl")  # Transcendental function handling (sin/cos/exp) — after si_equation_builder for parse_derivative_variable_name
include("core/problem_rescaling.jl")  # Opt-in power-of-2 problem rescaling (states/params/observables/data → O(1))
include("core/si_template_integration.jl")  # Template-based SI.jl integration
include("core/homotopy_continuation.jl")
include("core/robust_system.jl")  # Reusable polynomial residual/Jacobian kernels
include("core/solve_with_robust.jl")  # Robust solver with multiple fallbacks

include("core/parameter_estimation_helpers.jl")
include("core/parameter_estimation.jl")
include("core/nested_ad.jl")  # the outer tag for derivatives of an ODE solve
include("core/polish_residual.jl")
include("core/branch_completion.jl")
include("core/optimized_multishot_estimation.jl")  # New optimized workflow
include("core/multipoint_template.jl")  # Multi-point polynomial template system
include("core/noise_frontier_construction.jl")  # Probe-only noise-first system construction
include("core/derivatives.jl")
include("core/sigma_d.jl")  # Per-(observable, order) derivative uncertainty σ_d
include("core/sensitivity_seeds.jl")  # σ_d-aware seed generation (Layer 2)
include("core/synthesize_aggregates.jl")  # Per-component median/mean/trim25 synthesis of SP+MP aggregates
include("core/uncertainty_quantification.jl")  # UQ via GP derivative covariances and IFT
include("core/sampling.jl")
# Taylor coefficients, sensitivities and the uncertainty reports
include("core/diagnostics/taylor_oracle.jl")
include("core/diagnostics/feasibility_sensitivity.jl")
include("core/diagnostics/uq_and_reports.jl")
include("core/diagnostics/estimator_aware_uq.jl")
include("core/estimate.jl")  # keyword problem constructor, `estimate`, result display
include("examples/Examples.jl")  # the example models, a submodule

# The names a user writes. Everything else is reached as
# `ODEParameterEstimation.name`; the `public` names below are supported that way,
# the rest are internal and may change.

# Setting up a problem
export ParameterEstimationProblem, create_ordered_ode_system, sample_problem_data
export ObservationData, ObservationSeries, observation_times

# Estimating
export estimate, analyze_parameter_estimation_problem, EstimationOptions

# Option values
export SolverHC, SolverTrustRegion, SolverLevenbergMarquardt
export FlowStandard, FlowDirectOpt
export PolishLeastSquares, PolishNewtonTrust, PolishBFGS, PolishLBFGS
export InterpolatorAAA, InterpolatorGP, InterpolatorGPUncertainty, InterpolatorCustom

# Results, uncertainty and errors
export ParameterEstimationResult
export UncertaintyReport, UncertaintyUnavailable, print_uncertainty_results
export UnsupportedModelClassError, UnsupportedDerivativeOrderError, SamplingFailureError

# Supported, but written with the module name
public Examples
public SystemSolverMethod, EstimationFlow, PolishMethod, InterpolatorMethod
public OrderedODESystem, ResultProvenance
public AbstractUQOutcome, UncertaintyError, uq_reliability, UQReliabilityAssessment
public merge_options, validate_options, print_options

include("precompile_workload.jl")  # the estimation that runs while the package is precompiled

end # module
