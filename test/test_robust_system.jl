using ODEParameterEstimation, Test, LinearAlgebra
using ODEParameterEstimation: solve_multipoint_overdetermined, solve_with_robust
using ODEParameterEstimation.Symbolics: @variables, Num, substitute
const RobustODEPE = ODEParameterEstimation

@testset "Prepared polynomial kernels preserve data and unknown ordering" begin
	@variables a b d1 d2
	equations = [d1*a^2 + b - d2, a*b + d2*b^2 - d1]
	vars, data_vars = [a,b], [d1,d2]
	x = [0.7,1.2]
	for mode in (:symbolic, :forwarddiff, :finitediff)
		system = RobustODEPE.prepare_robust_system(equations, vars;
			data_vars, jacobian = mode)
		for data in ([2.0,3.0], [5.0,0.25], [2.0,3.0])
			r = zeros(2)
			system.residual!(r,x,data)
			@test r ≈ [data[1]*x[1]^2+x[2]-data[2], x[1]*x[2]+data[2]*x[2]^2-data[1]]
			residual! = (r,u) -> system.residual!(r,u,data)
			jac! = RobustODEPE._bind_robust_jacobian(system,residual!,data)
			J = zeros(2,2)
			jac!(J,x)
			expected = [2data[1]*x[1] 1; x[2] x[1]+2data[2]*x[2]]
			@test J ≈ expected rtol = (mode == :finitediff ? 1e-6 : 1e-13)
		end
	end
	@test_throws ArgumentError RobustODEPE.prepare_robust_system(equations,vars; data_vars=[a,d2])
	@test_throws ArgumentError RobustODEPE.prepare_robust_system(equations,vars; jacobian=:typo)
	@test_throws ArgumentError RobustODEPE.prepare_robust_system(equations,vars; forwarddiff_chunk_size=-1)

	cache = Dict{Tuple,RobustODEPE.PreparedRobustSystem}()
	opts = EstimationOptions()
	p = RobustODEPE._cached_robust_system!(cache,equations,vars,data_vars,opts)
	@test RobustODEPE._cached_robust_system!(cache,copy(equations),copy(vars),copy(data_vars),opts) === p
	@test RobustODEPE._cached_robust_system!(cache,equations,reverse(vars),data_vars,opts) !== p
	@test RobustODEPE._cached_robust_system!(cache,equations,vars,reverse(data_vars),opts) !== p
	@test RobustODEPE._cached_robust_system!(cache,[equations[1]+a,equations[2]],vars,data_vars,opts) !== p
	@test RobustODEPE._cached_robust_system!(cache,equations,vars,data_vars,
		EstimationOptions(polish_solver_jacobian=:symbolic)) !== p
	@test length(cache) == 5
	@test_throws DimensionMismatch solve_with_robust(equations,vars; prepared_system=p,data_values=[1.0])
	@test_throws ArgumentError solve_with_robust(equations,reverse(vars); prepared_system=p,data_values=[1.0,2.0])
	@test_throws ArgumentError solve_with_robust(equations,vars; prepared_system=p,data_values=[1.0,2.0],options=Dict(:jacobian=>:symbolic))
end

@testset "Estimator reuses one kernel across points and interpolators" begin
	for parameter_homotopy in (false,true)
		opts = EstimationOptions(datasize=31,shooting_points=3,
			interpolators=[InterpolatorAAAD,InterpolatorChebyshevAICc],
			use_parameter_homotopy=parameter_homotopy,use_multipoint=false,
			polish_solver_solutions=true,polish_solutions=false,
			compute_uncertainty=false,branch_completion=false,terminal_fallback=:none,
			construction_compute_mixed_volume=false,construction_candidate_limit=8,
			construction_beam_width=4,nooutput=true,diagnostics=false,save_system=false)
		pep = RobustODEPE.sample_problem_data(RobustODEPE.Examples.harmonic_oscillator(),opts)
		(answer,timing) = RobustODEPE.with_estimation_timing() do
			RobustODEPE.analyze_parameter_estimation_problem(pep,opts)
		end
		rows = timing.details[:detailed_timing_records]
		preparations = filter(r->r.category==:prepare_robust_system,rows)
		polishes = filter(r->r.category==:solve_with_robust,rows)
		@test count(r->!r.cache_hit,preparations)==1
		@test count(r->r.cache_hit,preparations)>=1
		@test !isempty(polishes)
		@test all(r->r.used_prepared_system && r.data_value_count>0,polishes)
		ranked = first(answer[2])
		@test !isempty(ranked)
		@test maximum(abs(first(ranked).parameters[p]-v)/max(abs(v),1e-12) for (p,v) in pep.p_true)<1e-4
	end
end

@testset "Prepared polynomial solves follow new data and return both branches" begin
	@variables x y d
	equations = [x^2-d, y-x-1]
	vars = [x,y]
	for mode in (:symbolic, :forwarddiff, :finitediff)
		system = RobustODEPE.prepare_robust_system(equations,vars;data_vars=[d],jacobian=mode)
		for value in (4.0,9.0,4.0), sign in (-1,1)
			root = [sign*sqrt(value),sign*sqrt(value)+1]
			instantiated = substitute.(equations,Ref(Dict(d=>value)))
			solutions, _, _, _ = solve_with_robust(instantiated,vars;
				prepared_system=system,data_values=[value],start_point=root.+[0.05,-0.03],
				polish_only=true,options=Dict(:abstol=>1e-12,:reltol=>1e-12))
			@test !isempty(solutions)
			@test first(solutions) ≈ root atol=1e-9 rtol=1e-9
		end
	end
	# Rectangular least-squares checks input/output buffer lengths and JᵀF.
	rectangular = [x-d,2x-2d,3x-3d]
	for mode in (:symbolic,:forwarddiff,:finitediff), algorithm in (:trustregion,:bfgs,:levenberg)
		system = RobustODEPE.prepare_robust_system(rectangular,[x];data_vars=[d],jacobian=mode)
		instantiated = substitute.(rectangular,Ref(Dict(d=>2.0)))
		solutions, _, _, _ = solve_with_robust(instantiated,[x];
			prepared_system=system,data_values=[2.0],start_point=[1.8],polish_only=true,
			options=Dict(:algorithm=>algorithm,:abstol=>1e-10,:reltol=>1e-10))
		@test !isempty(solutions)
		@test first(solutions) ≈ [2.0] atol=1e-7
	end
	# Existing unprepared entry point still constructs and solves its equations.
	solutions, _, _, _ = solve_with_robust([x^2-4],[x];start_point=[2.1],polish_only=true)
	@test first(solutions) ≈ [2.0] atol=1e-5
	# With no real solution the result has the same shape, with no solutions.
	none, returned_vars, _, _ = solve_with_robust([x^2+1],[x];start_point=[0.5],polish_only=true)
	@test isempty(none)
	@test isequal(returned_vars, [x])
end

@testset "Overdetermined multipoint solve applies its least-squares refinement" begin
	@variables x d
	metadata = NamedTuple{(:point,:is_data,:order),Tuple{Int,Bool,Int}}[]
	template = RobustODEPE.MultiPointTemplate(1, nothing, Num[], Num[x-d], Any[x], Any[d],
		Int[], String[], metadata, 0, Int[], Int[], [Int[]], nothing,
		RobustODEPE.ModelingToolkit.Equation[], RobustODEPE.DataVarMeta[])
	evaluation = RobustODEPE.MultiPointEvaluation(template, [1], [0.0], [1.0])
	# The square core gives x = 1. Against [x - 1, x - 3] the least-squares
	# point is x = 2, so an unrefined answer is distinguishable from a refined one.
	refined = solve_multipoint_overdetermined(template, evaluation, [x-d, x-3d], [x])
	@test length(refined) == 1
	@test only(refined) ≈ [2.0] atol=1e-8
end

