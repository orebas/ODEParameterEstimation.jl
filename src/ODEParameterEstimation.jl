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
include("core/pointpicker.jl")

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
include("core/svg_plots.jl")
# diagnostics.jl (5,443 lines) split 2026-06-10 along its section seams (Phase F2);
# original include order preserved.
include("core/diagnostics/taylor_oracle.jl")
include("core/diagnostics/feasibility_sensitivity.jl")
include("core/diagnostics/multipoint_sensitivity.jl")  # estimate-conditioned S over multipoint templates
include("core/diagnostics/error_budget.jl")
include("core/diagnostics/orchestrators.jl")
include("core/diagnostics/html_report.jl")
include("core/diagnostics/uq_and_reports.jl")
include("core/diagnostics/estimator_aware_uq.jl")
include("core/estimate.jl")  # keyword problem constructor, `estimate`, result display
include("examples/Examples.jl")  # the example models, a submodule

# Export types
export OrderedODESystem, ParameterEstimationProblem, ParameterEstimationResult, ResultProvenance, EstimatorIdentity, NumericalIdentifiabilityAdvisory, DerivativeData, UnsupportedModelClassError, SamplingFailureError, UnsupportedDerivativeOrderError, TAYLORDIFF_MAX_DERIVATIVE_ORDER
export provenance_metadata_dict, uq_metadata_dict
export ObservationSeries, ObservationData, observation_times

# Export constants
export package_wide_default_ode_solver, CLUSTERING_THRESHOLD, MAX_ERROR_THRESHOLD, IMAG_THRESHOLD, MAX_SOLUTIONS

# Export core functions
export solve_with_hc
export optimized_multishot_parameter_estimation, solve_with_robust
export direct_optimization_parameter_estimation

# Export utility functions
export unpack_ODE, tag_symbol, create_ordered_ode_system
export add_relative_noise, add_additive_noise, add_synthetic_noise, sample_problem_data, calculate_error_stats
export analyze_estimation_result, cluster_solutions
export clear_denoms, hmcs, analyze_parameter_estimation_problem, estimate
export aaad, aaad_old_reliable, AAADapprox, GPRapprox, FHDapprox, nth_deriv, nth_deriv_at, aaad_gpr_pivot, fhdn
export ChebyshevApprox, chebyshev_aicc, chebyshev_bic
export InterpolatorMethod, InterpolatorAAAD, InterpolatorAAADGPR, InterpolatorAAADOld, InterpolatorFHD
export InterpolatorAGPRobust, InterpolatorAGPRobustRQ, InterpolatorAGPRobustSEpRQ, InterpolatorAGPRobustSExRQ, InterpolatorAGPRobustMatern52
export InterpolatorS2AAAMLE, InterpolatorS3AdaptSE, InterpolatorS3AdaptRQ, InterpolatorS3AdaptSEpRQ, InterpolatorS3AdaptSExRQ, InterpolatorS3AdaptMatern52
export InterpolatorS3BICSE, InterpolatorS3BICRQ, InterpolatorS3BICSEpRQ, InterpolatorS3BICSExRQ, InterpolatorS3BICMatern52
export InterpolatorChebyshevAICc, InterpolatorChebyshevBIC, InterpolatorAGPUQ, InterpolatorCustom
export AGPInterpolator, agp_gpr_robust, mean_and_var
export create_interpolants, AbstractInterpolator, solve_with_nlopt, solve_with_fast_nlopt
export solve_with_hc_parameterized, convert_to_hc_format_with_params, extract_data_variables_from_DD, evaluate_data_vars_at_point
export MultiPointTemplate, MultiPointEvaluation, build_multipoint_template, evaluate_multipoint_template, solve_multipoint_direct, solve_multipoint_parameterized
export solve_multipoint_overdetermined
export NoiseFrontierCandidate, NoiseFrontierResult, build_noise_frontier_system, build_noise_frontier_multipoint_template, evaluate_noise_frontier_data_vars_at_point, instantiate_noise_frontier_candidate, validate_noise_frontier_instantiation, validate_noise_frontier_candidate_at_values, noise_frontier_rows, write_noise_frontier_csv

# Export derivative utilities
export calculate_higher_derivatives, calculate_higher_derivative_terms

# Export transcendental handling
export detect_transcendentals, transform_pep_for_estimation, TranscendentalInfo

# Export problem rescaling
export rescale_pep, unrescale_results, ScaleInfo, choose_scales

# Export diagnostic functions and types
export diagnose, diagnose_derivative_accuracy, diagnose_polynomial_system, diagnose_sensitivity
export PerfectInterpolant, DiagnosticReport, ComprehensiveDiagnosticReport, DerivativeAccuracyReport, PolynomialFeasibilityReport, SensitivityReport, MultipointDiagnosticAnalysis
export EstimationResultsReport, BacksolveUQReport, AbstractUQOutcome, UncertaintyReport, UQUnavailable, UQComputationError, UQTargetSnapshot, UQLinearizationDiagnostics, UQReliabilityAssessment, uq_reliability, JetInfluenceEstimate, StackedJetInfluenceEstimate, PracticalIdentifiabilityIndex, LocalUQSnapshot, UQBacksolveTransform
export DerivativeUncertaintyEstimate, compute_sigma_d, get_sigma_d
export SensitivitySeedReport, generate_sensitivity_seeds, seed_vectors_to_candidates
export ErrorBudgetEntry, ErrorBudgetReport, compute_error_budget, compute_multipoint_error_budget
export ParameterSpreadEntry, CrossSolutionSpread
export build_perfect_interpolants, compute_oracle_taylor_coefficients, compute_observable_taylor_coefficients
export TimingPhaseEntry, TimingBreakdown, with_estimation_timing, timing_breakdown_to_dict

# Export UQ (Uncertainty Quantification) functions
export AGPInterpolatorUQ, agp_gpr_uq, gp_factorization_diagnostics
export se_kernel_derivative, se_kernel_prior_covariance_matrix, se_kernel_cross_time_covariance_matrix
export joint_derivative_covariance, build_observation_covariance
export gp_derivative_influence_matrix, joint_derivative_estimator_covariance, stacked_jet_index, learned_observation_noise_variance, compute_practical_identifiability_index, physicalize_uncertainty_report, unrescale_uncertainty_report
export print_uncertainty_results

# The example models are reached as `ODEParameterEstimation.Examples`.
public Examples


# Export the main types and functions
export EstimationOptions, SystemSolverMethod, PolishMethod, EstimationFlow
export FlowStandard, FlowDirectOpt
export SolverHC, SolverNLOpt, SolverFastNLOpt, SolverRobust
export InterpolatorS3SE, InterpolatorS3RQ, InterpolatorS3SEpRQ, InterpolatorS3SExRQ, InterpolatorS3Matern52
export PolishNewtonTrust, PolishBFGS, PolishLBFGS,
       PolishLSOBoundedLog, PolishFastLMBoundedLog
export get_solver_function, get_interpolator_function, get_polish_optimizer, get_ad_backend
export interpolator_method_to_symbol, resolve_interpolator_list, setup_identifiability, compute_shooting_indices
export s3_refine_gp, s3_refine_gp_adaptive, s3_refine_gp_bic
export merge_options, validate_options, print_options
export compatibility_return_code, sync_result_contract!, lineage_summary

include("precompile_workload.jl")  # the estimation that runs while the package is precompiled

end # module
