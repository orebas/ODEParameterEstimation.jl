#
# Point selection is intentionally simple: the package uses the pick_points path
# below (point-hint + equidistant shooting indices). The older derivative-scoring
# machinery was archived 2026-06-09 to deprecated/pointpicker_scoring.jl
# (not part of the build) to cut core complexity.
#

function pick_points(vec, n, interpolants, point_hint = 0.5)
	len = length(vec)
	at(fraction) = clamp(round(Int, fraction * len), 1, len)
	if n == 1
		return at(point_hint)
	else
		if n == 2
			first_index = at(point_hint)
			second_index = at(point_hint + 1 / 3)
			# Near the end of the interval the second point would land on the
			# first; take it a third of the way back instead.
			second_index == first_index && (second_index = at(point_hint - 1 / 3))
			second_index == first_index && (second_index = first_index == 1 ? 2 : first_index - 1)
			return [first_index, second_index]
		else
			# For n>2 points, use equidistant shooting indices (warp=false to preserve legacy behavior)
			return compute_shooting_indices(n, length(vec); warp = false)
		end
	end
end
