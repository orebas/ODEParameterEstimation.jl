# Derivatives of functions that solve an ODE.
#
# A stiff ODE solver differentiates the right-hand side with ForwardDiff. When
# we differentiate the whole solve with ForwardDiff as well, the two layers of
# dual numbers have to know which one is the outer layer. ForwardDiff decides
# that by a counter: each tag gets the counter's value when the tag is first
# compiled, and a smaller number means "outer".
#
# That number is saved with precompiled code. Our tag is compiled while the
# package is precompiled (the precompile workload runs the trajectory polish),
# so it comes back with a large number. The solver's tag is compiled later, in
# the user's session, where the counter has started again from zero, and gets
# a small one. The order is then reversed, every dual-number solve throws, and
# the polish returns its starting point. This happened in every default
# estimation from the day the workload was added until this file.
#
# So derivatives of an ODE solve use a tag of our own whose place in the order
# is written down here and does not depend on any counter.

"""
	_OuterADTag

Marks the dual numbers of a derivative that the package takes of a whole ODE
solve. Its tag is the outer one relative to every other tag: a tag made
inside the function being differentiated depends on this one.
"""
struct _OuterADTag end

ForwardDiff.:≺(::Type{ForwardDiff.Tag{_OuterADTag, V1}}, ::Type{ForwardDiff.Tag{F, V2}}) where {V1, F, V2} = true
ForwardDiff.:≺(::Type{ForwardDiff.Tag{F, V1}}, ::Type{ForwardDiff.Tag{_OuterADTag, V2}}) where {F, V1, V2} = false
# Our own tag against itself, which would otherwise match both methods above.
ForwardDiff.:≺(::Type{ForwardDiff.Tag{_OuterADTag, V1}}, ::Type{ForwardDiff.Tag{_OuterADTag, V2}}) where {V1, V2} = false

_outer_ad_tag(x::AbstractArray) = ForwardDiff.Tag{_OuterADTag, eltype(x)}()

"""
	_outer_jacobian!(J, f, x) -> J

`ForwardDiff.jacobian!` of `f` at `x`, for an `f` that solves an ODE. The dual
numbers carry `_OuterADTag`, so they nest correctly with the solver's own.
"""
function _outer_jacobian!(J::AbstractMatrix, f, x::AbstractVector)
	config = ForwardDiff.JacobianConfig(nothing, x, ForwardDiff.Chunk(x), _outer_ad_tag(x))
	# The tag is not derived from `f`, so ForwardDiff's check that it is must be off.
	ForwardDiff.jacobian!(J, f, x, config, Val{false}())
	return J
end

"""
	_outer_jacobian(f, x) -> Matrix

`ForwardDiff.jacobian` of `f` at `x`, for an `f` that solves an ODE.
"""
function _outer_jacobian(f, x::AbstractVector)
	config = ForwardDiff.JacobianConfig(nothing, x, ForwardDiff.Chunk(x), _outer_ad_tag(x))
	return ForwardDiff.jacobian(f, x, config, Val{false}())
end
