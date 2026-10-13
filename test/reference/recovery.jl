# Run once with the frozen baseline environment, then with the internalized
# checkout, supplying the same input TOML. Does not require upstream packages.
using ODEParameterEstimation, Random, TOML, Statistics, Logging
# The frozen baseline exports the models from the package itself.
isdefined(ODEParameterEstimation, :Examples) && @eval using ODEParameterEstimation.Examples

input_path, output_path = ARGS
constructors = (lotka_volterra, vanderpol, fitzhugh_nagumo, biohydrogenation)
if !isfile(input_path)
    draws = Dict{String,Any}()
    for (index,ctor) in enumerate(constructors)
        Random.seed!(20261010+index)
        sampled = sample_problem_data(ctor(),EstimationOptions(datasize=81,noise_level=1e-6))
        draws[string(ctor)] = Dict(string(k)=>v for (k,v) in sampled.data_sample)
    end
    open(input_path,"w") do io
        TOML.print(io,draws)
    end
end
draws = TOML.parsefile(input_path)
records = []
for ctor in constructors, route in ("gp", "pool")
    println("START ",ctor," ",route); flush(stdout)
    pep = sample_problem_data(ctor(),EstimationOptions(datasize=81,noise_level=0.0))
    for key in keys(pep.data_sample)
        pep.data_sample[key] .= draws[string(ctor)][string(key)]
    end
    base = EstimationOptions(datasize=81,noise_level=1e-6,nooutput=true,
        diagnostics=false,save_system=false,compute_uncertainty=false,
        shooting_points=0,polish_solutions=true)
    opts = route == "gp" ? merge_options(base;interpolators=[InterpolatorAAADGPR]) : base
    Random.seed!(20261020)
    seconds = @elapsed _,analysis,_ = analyze_parameter_estimation_problem(pep,opts)
    estimates = first(analysis)
    errors = [maximum(abs(r.parameters[p]-truth)/max(abs(truth),1e-12)
        for (p,truth) in pep.p_true) for r in estimates]
    push!(records,Dict("model"=>string(ctor),"route"=>route,"seconds"=>seconds,
        "count"=>length(estimates),"rank_one_error"=>isempty(errors) ? Inf : first(errors),
        "best_branch_error"=>isempty(errors) ? Inf : minimum(errors),
        "fit_error"=>isempty(estimates) ? Inf : first(estimates).err))
    open(output_path,"w") do io
        TOML.print(io,Dict("julia"=>string(VERSION),"cases"=>records))
    end
    println("RESULT ",last(records));flush(stdout)
end
