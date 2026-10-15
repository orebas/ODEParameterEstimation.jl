using ODEParameterEstimation
# Characterization tests for the three label/name parsers.
#
# These LOCK the parsers' CURRENT behavior, measured empirically, including their
# distinct failure contracts:
#   _parse_data_label        -> ("", 0)   on failure   (diagnostics.jl)
#   parse_sensitivity_label  -> nothing   on failure   (sigma_d.jl)
#   _multipoint_deriv_order  -> 0         on failure   (multipoint_template.jl; order only)
#
# Known warts locked AS-IS (candidates for a deliberate fix, never to be changed
# silently):
#   * Only the comma form "Differential(t, N)(var(t))" parses; the order-1 form
#     "Differential(t)(var(t))" and nested forms fail.
#   * A bare underscore-digit name parses its suffix as an order: "k_1" -> ("k", 1).
#   * _trfn_ names with frequency suffixes misparse: "_trfn_sin_0_5" -> ("_trfn_sin_0", 5).
#   * Only _multipoint_deriv_order strips _ptK suffixes.

using Test

@testset "Label parser characterization" begin
	PDL = ODEParameterEstimation._parse_data_label
	PSL = ODEParameterEstimation.parse_sensitivity_label
	MDO = ODEParameterEstimation._multipoint_deriv_order

	@testset "_parse_data_label" begin
		@test PDL("y1_2") == ("y1", 2)
		@test PDL("y1_0") == ("y1", 0)
		@test PDL("k_1_0") == ("k_1", 0)
		@test PDL("x_1_2") == ("x_1", 2)
		@test PDL("y1(t)") == ("y1", 0)
		@test PDL("x_1(t)") == ("x_1", 0)
		@test PDL("Differential(t, 2)(y1(t))") == ("y1", 2)
		@test PDL("Differential(t,2)(y1(t))") == ("y1", 2)
		@test PDL("Differential(t, 1)(y1(t))") == ("y1", 1)
		# Failure contract: ("", 0)
		@test PDL("y1") == ("", 0)
		@test PDL("z_aux") == ("", 0)
		@test PDL("") == ("", 0)
		@test PDL("Differential(t)(y1(t))") == ("", 0)              # order-1 non-comma form unhandled
		@test PDL("Differential(t)(Differential(t)(y1(t)))") == ("", 0)
		@test PDL("y1_2_pt3") == ("", 0)                            # no _ptK stripping here
		# Locked warts:
		@test PDL("k_1") == ("k", 1)
		@test PDL("_trfn_sin_0_5_1") == ("_trfn_sin_0_5", 1)
		@test PDL("_trfn_sin_0_5") == ("_trfn_sin_0", 5)
	end

	@testset "parse_sensitivity_label" begin
		@test PSL("y1_2") == ("y1", 2)
		@test PSL("y1_0") == ("y1", 0)
		@test PSL("k_1_0") == ("k_1", 0)
		@test PSL("x_1_2") == ("x_1", 2)
		@test PSL("y1(t)") == ("y1", 0)
		@test PSL("x_1(t)") == ("x_1", 0)
		@test PSL("Differential(t, 2)(y1(t))") == ("y1", 2)
		@test PSL("Differential(t,2)(y1(t))") == ("y1", 2)
		@test PSL("Differential(t, 1)(y1(t))") == ("y1", 1)
		# Failure contract: nothing
		@test PSL("y1") === nothing
		@test PSL("z_aux") === nothing
		@test PSL("") === nothing
		@test PSL("Differential(t)(y1(t))") === nothing
		@test PSL("Differential(t)(Differential(t)(y1(t)))") === nothing
		@test PSL("y1_2_pt3") === nothing
		# Locked warts:
		@test PSL("k_1") == ("k", 1)
		@test PSL("_trfn_sin_0_5_1") == ("_trfn_sin_0_5", 1)
		@test PSL("_trfn_sin_0_5") == ("_trfn_sin_0", 5)
	end

	@testset "_multipoint_deriv_order" begin
		@test MDO("y1_2") == 2
		@test MDO("y1_0") == 0
		@test MDO("k_1_0") == 0
		@test MDO("x_1_2") == 2
		# _ptK stripping is unique to this parser:
		@test MDO("y1_2_pt3") == 2
		@test MDO("x_1_0_pt2") == 0
		@test MDO("k_1_0_pt2") == 0
		# Failure contract: 0
		@test MDO("y1") == 0
		@test MDO("z_aux") == 0
		@test MDO("") == 0
		@test MDO("Differential(t, 2)(y1(t))") == 0                 # no Differential handling here
		# Locked warts:
		@test MDO("k_1") == 1
		@test MDO("_trfn_sin_0_5") == 5
	end
end
