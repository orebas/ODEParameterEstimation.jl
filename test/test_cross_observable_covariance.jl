"""
Educational test: Cross-observable covariance and linear combinations of GP posteriors.

Three ODEs:
  X'(t) = 2X,  X(0) = 1   → X(t) = e^{2t}
  Y'(t) = 2Y,  Y(0) = -1  → Y(t) = -e^{2t}
  Z'(t) = 2Z,  Z(0) = 3   → Z(t) = 3e^{2t}

Key relationships:
  Y = -X  (perfectly anticorrelated)
  Z = 3X  (perfectly correlated)
  X + Y = 0  (exact cancellation)
  Z - 3X = 0  (exact cancellation)

For linear combination a = c'·[X,Y,Z]:
  Var(a) = c' Σ c  where Σ is the JOINT covariance of [X,Y,Z]

If we treat X,Y,Z as independent GPs:
  Var_indep(a) = c₁²Var(X) + c₂²Var(Y) + c₃²Var(Z)

The gap: Var_indep(X+Y) = Var(X) + Var(Y) > 0
          Var_true(X+Y)  = 0  (since Y = -X exactly)
"""

using Test
using ODEParameterEstimation
using ODEParameterEstimation: AGPInterpolatorUQ, agp_gpr_uq, build_observation_covariance,
    joint_derivative_covariance
using LinearAlgebra
using Statistics
using Random
using OrderedCollections

Random.seed!(123)

@testset "Cross-Observable Covariance Education" begin

    # ================================================================
    # Step 1: Generate data from 3 correlated exponentials
    # ================================================================
    ts = collect(0.0:0.05:2.0)
    n = length(ts)

    # True signals
    X_true = exp.(2.0 .* ts)
    Y_true = -exp.(2.0 .* ts)    # Y = -X exactly
    Z_true = 3.0 .* exp.(2.0 .* ts)  # Z = 3X exactly

    # Add noise (same σ for all)
    σ_noise = 0.1
    X_noisy = X_true .+ σ_noise .* randn(n)
    Y_noisy = Y_true .+ σ_noise .* randn(n)
    Z_noisy = Z_true .+ σ_noise .* randn(n)

    println("\n" * "="^70)
    println("EDUCATIONAL TEST: Cross-Observable GP Covariance")
    println("="^70)
    println("\nTrue signals: X = e^{2t}, Y = -e^{2t}, Z = 3e^{2t}")
    println("Noise level: σ = $σ_noise")
    println("Time points: $(length(ts)) from $(ts[1]) to $(ts[end])")

    # ================================================================
    # Step 2: Fit independent GPs to each observable
    # ================================================================
    gp_X = agp_gpr_uq(ts, X_noisy)
    gp_Y = agp_gpr_uq(ts, Y_noisy)
    gp_Z = agp_gpr_uq(ts, Z_noisy)

    println("\n--- GP Hyperparameters ---")
    println("X: ℓ=$(round(gp_X.lengthscale, digits=4)), σ²=$(round(gp_X.signal_var, digits=4)), σₙ²=$(round(gp_X.noise_var, digits=6))")
    println("Y: ℓ=$(round(gp_Y.lengthscale, digits=4)), σ²=$(round(gp_Y.signal_var, digits=4)), σₙ²=$(round(gp_Y.noise_var, digits=6))")
    println("Z: ℓ=$(round(gp_Z.lengthscale, digits=4)), σ²=$(round(gp_Z.signal_var, digits=4)), σₙ²=$(round(gp_Z.noise_var, digits=6))")

    # ================================================================
    # Step 3: Evaluate at a mid-range point
    # ================================================================
    t_eval = 1.0
    println("\n--- Evaluating at t = $t_eval ---")
    println("True values: X=$(exp(2*t_eval)), Y=$(-exp(2*t_eval)), Z=$(3*exp(2*t_eval))")

    # Get mean and variance for each GP (order 0 only)
    μ_X, Σ_X = joint_derivative_covariance(gp_X, t_eval, 0)
    μ_Y, Σ_Y = joint_derivative_covariance(gp_Y, t_eval, 0)
    μ_Z, Σ_Z = joint_derivative_covariance(gp_Z, t_eval, 0)

    var_X = Σ_X[1, 1]
    var_Y = Σ_Y[1, 1]
    var_Z = Σ_Z[1, 1]

    println("\nIndividual GP posteriors:")
    println("  X: mean=$(round(μ_X[1], digits=4)), std=$(round(sqrt(var_X), digits=6))")
    println("  Y: mean=$(round(μ_Y[1], digits=4)), std=$(round(sqrt(var_Y), digits=6))")
    println("  Z: mean=$(round(μ_Z[1], digits=4)), std=$(round(sqrt(var_Z), digits=6))")

    @test μ_X[1] ≈ exp(2 * t_eval) atol=0.5   # GP mean should be close to truth
    @test μ_Y[1] ≈ -exp(2 * t_eval) atol=0.5
    @test μ_Z[1] ≈ 3 * exp(2 * t_eval) atol=1.5

    # ================================================================
    # Step 4: Linear combinations — what our framework gives vs truth
    # ================================================================
    println("\n" * "="^70)
    println("LINEAR COMBINATIONS OF OBSERVABLES")
    println("="^70)

    # Build the "independent GP" covariance matrix (block-diagonal)
    Σ_indep = Diagonal([var_X, var_Y, var_Z])

    # What the TRUE covariance should be (if we knew Y=-X, Z=3X):
    # The noise is independent across sensors, so:
    #   Cov_data(X,Y) = 0  (noise is independent)
    #   Cov_data(X,Z) = 0  (noise is independent)
    # BUT the GP posterior incorporates the smoothness assumption.
    # For the POSTERIOR, the correlation comes from the fact that
    # the underlying functions are deterministically related.
    #
    # Monte Carlo approach: sample from each GP, compute empirical covariance
    println("\n--- Monte Carlo estimate of true joint covariance ---")

    n_mc = 5000
    # Sample from each GP posterior at t_eval
    # GP posterior is N(μ, Σ), so sample = μ + sqrt(Σ) * z
    samples_X = μ_X[1] .+ sqrt(var_X) .* randn(n_mc)
    samples_Y = μ_Y[1] .+ sqrt(var_Y) .* randn(n_mc)
    samples_Z = μ_Z[1] .+ sqrt(var_Z) .* randn(n_mc)

    # These are INDEPENDENT samples because we fit independent GPs!
    # The empirical cross-covariance will be ~0 (sampling noise only)
    empirical_cov_XY = cov(samples_X, samples_Y)
    empirical_cov_XZ = cov(samples_X, samples_Z)
    println("Empirical Cov(X,Y) from independent GP samples: $(round(empirical_cov_XY, digits=8))")
    println("Empirical Cov(X,Z) from independent GP samples: $(round(empirical_cov_XZ, digits=8))")
    println("(These are ~0 because the GPs are independent)")

    # ================================================================
    # Step 5: The key comparisons
    # ================================================================
    println("\n" * "-"^70)
    println("COMBINATION         | INDEP. GP Var | TRUE Var | TRUTH (if known)")
    println("-"^70)

    combos = [
        ("X + Y",     [1.0, 1.0, 0.0],  "= 0 exactly (Y = -X)"),
        ("X - Y",     [1.0, -1.0, 0.0],  "= 2X = 2e^{2t}"),
        ("Z - 3X",    [-3.0, 0.0, 1.0],  "= 0 exactly (Z = 3X)"),
        ("Z + 3Y",    [0.0, 3.0, 1.0],   "= 0 exactly (Z = -3Y)"),
        ("X + Z",     [1.0, 0.0, 1.0],   "= 4X = 4e^{2t}"),
        ("X + Y + Z", [1.0, 1.0, 1.0],   "= 3X = 3e^{2t}"),
    ]

    μ_vec = [μ_X[1], μ_Y[1], μ_Z[1]]

    for (name, c, truth_str) in combos
        c_vec = c

        # Mean of combination
        mean_combo = dot(c_vec, μ_vec)

        # Variance assuming independent GPs (our framework)
        var_indep = dot(c_vec, Σ_indep * c_vec)

        # What the variance SHOULD be if we knew the exact relationships:
        # For "X + Y": true answer is 0 since X + Y = 0 + noise
        # But GP posterior smooths through the noise, so the posterior
        # uncertainty about X+Y is NOT zero — it depends on the GP's
        # ability to resolve the cancellation.

        # Monte Carlo with independent GP samples
        combo_samples = c_vec[1] .* samples_X .+ c_vec[2] .* samples_Y .+ c_vec[3] .* samples_Z
        var_mc = var(combo_samples)

        println("$(rpad(name, 12)) mean=$(rpad(round(mean_combo, digits=3), 8)) | "
              * "σ_indep=$(rpad(round(sqrt(var_indep), digits=5), 8)) | "
              * "σ_mc=$(rpad(round(sqrt(var_mc), digits=5), 8)) | $truth_str")
    end

    # ================================================================
    # Step 6: The "direct GP on the combination" approach
    # ================================================================
    println("\n" * "="^70)
    println("ALTERNATIVE: Fit GP directly to X+Y data")
    println("="^70)

    # If we KNOW we want X+Y, we can fit a GP directly to the sum
    XpY_noisy = X_noisy .+ Y_noisy   # This is 0 + noise_X + noise_Y
    XpY_true = X_true .+ Y_true      # This is exactly 0

    gp_XpY = agp_gpr_uq(ts, XpY_noisy)
    μ_XpY, Σ_XpY = joint_derivative_covariance(gp_XpY, t_eval, 0)

    println("\nX+Y data is noise around 0 (since X = -Y exactly)")
    println("  Direct GP on X+Y: mean=$(round(μ_XpY[1], digits=4)), std=$(round(sqrt(Σ_XpY[1,1]), digits=6))")
    println("  Independent GP:   mean=$(round(μ_X[1] + μ_Y[1], digits=4)), std=$(round(sqrt(var_X + var_Y), digits=6))")
    println("  True X+Y at t=$t_eval: $(XpY_true[findfirst(==(t_eval), ts)])")

    # Key comparison: the direct GP on X+Y data (which is pure noise) gives a
    # DIFFERENT uncertainty than the independent GP sum.
    # - Independent GP: σ(X+Y) = sqrt(Var(X) + Var(Y)) — small because each GP
    #   is confident about its smooth exponential. Missing the Cov(X,Y) < 0 term.
    # - Direct GP: sees pure noise data → larger uncertainty (can't predict noise).
    # Neither is "right" — the independent GP underestimates (ignores cancellation),
    # the direct GP correctly reflects the noise floor.
    ratio = sqrt(Σ_XpY[1, 1]) / sqrt(var_X + var_Y)
    println("\n  → Direct GP uncertainty is $(round(ratio * 100, digits=1))% of independent GP uncertainty")
    @test isfinite(ratio)

    # ================================================================
    # Step 7: Similarly for Z - 3X
    # ================================================================
    Zm3X_noisy = Z_noisy .- 3.0 .* X_noisy  # = 0 + noise_Z - 3*noise_X
    gp_Zm3X = agp_gpr_uq(ts, Zm3X_noisy)
    μ_Zm3X, Σ_Zm3X = joint_derivative_covariance(gp_Zm3X, t_eval, 0)

    var_Zm3X_indep = 9 * var_X + var_Z  # (-3)²Var(X) + 1²Var(Z)
    println("\nZ - 3X data is noise around 0 (since Z = 3X exactly)")
    println("  Direct GP on Z-3X: mean=$(round(μ_Zm3X[1], digits=4)), std=$(round(sqrt(Σ_Zm3X[1,1]), digits=6))")
    println("  Independent GP:    mean=$(round(μ_Z[1] - 3*μ_X[1], digits=4)), std=$(round(sqrt(var_Zm3X_indep), digits=6))")

    ratio2 = sqrt(Σ_Zm3X[1, 1]) / sqrt(var_Zm3X_indep)
    println("\n  → Direct GP uncertainty is $(round(ratio2 * 100, digits=1))% of independent GP uncertainty")
    @test isfinite(ratio2)

    # ================================================================
    # Step 8: What does build_observation_covariance actually produce?
    # ================================================================
    println("\n" * "="^70)
    println("WHAT build_observation_covariance ACTUALLY GIVES")
    println("="^70)

    gp_dict = Dict{String, AGPInterpolatorUQ}("X" => gp_X, "Y" => gp_Y, "Z" => gp_Z)
    μ_z, Σ_z, labels = build_observation_covariance(gp_dict, [t_eval], 0)

    println("\nΣ_z at t=$t_eval (max_deriv=0, 3 observables):")
    println("  Shape: $(size(Σ_z))")
    println("  Labels: $labels")
    println("  Matrix:")
    for i in 1:3
        row = [round(Σ_z[i, j], sigdigits=4) for j in 1:3]
        println("    $(labels[i]): $row")
    end

    println("\n  Diagonal (variances): $(round.(diag(Σ_z), sigdigits=4))")
    println("  Off-diagonal (cross-obs): Σ[1,2]=$(Σ_z[1,2]), Σ[1,3]=$(Σ_z[1,3]), Σ[2,3]=$(Σ_z[2,3])")
    println("  → All cross-observable entries are exactly 0 (independent GP assumption)")

    @test Σ_z[1, 2] == 0.0  # X-Y cross-covariance is zero
    @test Σ_z[1, 3] == 0.0  # X-Z cross-covariance is zero
    @test Σ_z[2, 3] == 0.0  # Y-Z cross-covariance is zero

    # ================================================================
    # Step 9: Summary
    # ================================================================
    println("\n" * "="^70)
    println("SUMMARY")
    println("="^70)
    println("""
    Our framework fits INDEPENDENT GPs to each observable, so:
      Cov(X(t), Y(t)) = 0  always  (even though Y = -X)
      Cov(X(t), Z(t)) = 0  always  (even though Z = 3X)

    Consequence for linear combinations:
      Var_framework(X + Y) = Var(X) + Var(Y) > 0
      Var_truth(X + Y)     = 0  (the signals cancel exactly)

    The framework OVERESTIMATES uncertainty for combinations where
    observables are correlated through the dynamics.

    However, for PARAMETER uncertainty via the IFT:
      Cov(θ) = S · Σ_z · S'  where S = pinv(J_θ)
    The coupling between observables enters through J_θ (the ODE dynamics
    Jacobian), not through Σ_z. So the parameter uncertainty CAN still
    capture correlations — but the observation-space uncertainty is inflated.

    To fix this, one would need a multi-output GP (ICM/LMC kernel)
    that models cross-observable correlations directly.
    """)
end
