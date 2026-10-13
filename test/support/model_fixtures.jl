# Models that tests need and that are not examples for users.

using ODEParameterEstimation
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrderedCollections: OrderedDict

function _fixture_problem(name, states, parameters, equations, measured_quantities, p_true, ic_true)
	model, mq = ODEParameterEstimation.create_ordered_ode_system(name, states, parameters, equations, measured_quantities)
	return ParameterEstimationProblem(name, model, mq, nothing, nothing, nothing,
		OrderedDict(parameters .=> p_true), OrderedDict(states .=> ic_true), 0)
end

# `beta` enters only through the product a^2 * b^2 * beta.
function substr_test()
	parameters = @parameters a b beta
	states = @variables x1(t) x2(t) x3(t)
	@variables y1(t) y2(t) y3(t)
	equations = [
		D(x1) ~ -a * x2,
		D(x2) ~ b * (x1),
		D(x3) ~ a * b * beta * b * a * x3,
	]
	return _fixture_problem("substr_test", states, parameters, equations, [y1 ~ x1, y2 ~ x2, y3 ~ x3],
		[0.1, 0.2, 0.3], [2.0, 3.0, 4.0])
end

# Three states, each measured through its cube.
function threesp_cubed()
	parameters = @parameters a b c
	states = @variables x1(t) x2(t) x3(t)
	@variables y1(t) y2(t) y3(t)
	equations = [
		D(x1) ~ -a * x2,
		D(x2) ~ -b * x1,
		D(x3) ~ -c * x1,
	]
	return _fixture_problem("threesp_cubed", states, parameters, equations,
		[y1 ~ x1 * x1 * x1, y2 ~ x2 * x2 * x2, y3 ~ x3 * x3 * x3], [0.1, 0.2, 0.3], [2.0, 3.0, 4.0])
end
