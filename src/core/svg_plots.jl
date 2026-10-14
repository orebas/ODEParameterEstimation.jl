"""
Inline SVG plot generation for HTML diagnostic reports.

No external dependencies — produces SVG markup strings that embed directly
into HTML.  Uses viewBox-based coordinate mapping for responsive sizing.
"""

# ─── Coordinate helpers ────────────────────────────────────────────────

"""
    _nice_tick_step(range_span, target_ticks=5)

Compute a "nice" tick spacing for axis labels using Heckbert's algorithm.
Returns a step size that is 1, 2, or 5 times a power of 10.
"""
function _nice_tick_step(range_span::Float64, target_ticks::Int = 5)
    range_span <= 0 && return 1.0
    raw_step = range_span / max(target_ticks, 1)
    magnitude = 10.0^floor(log10(raw_step))
    residual = raw_step / magnitude
    nice = if residual <= 1.5
        1.0
    elseif residual <= 3.5
        2.0
    elseif residual <= 7.5
        5.0
    else
        10.0
    end
    return nice * magnitude
end

"""
    _generate_ticks(vmin, vmax; target_ticks=5)

Generate a vector of "nice" tick values spanning [vmin, vmax].
"""
function _generate_ticks(vmin::Float64, vmax::Float64; target_ticks::Int = 5)
    span = vmax - vmin
    span <= 0 && return [vmin]
    step = _nice_tick_step(span, target_ticks)
    first_tick = ceil(vmin / step) * step
    ticks = Float64[]
    t = first_tick
    while t <= vmax + step * 0.001  # tiny tolerance
        push!(ticks, t)
        t += step
    end
    return ticks
end

"""
    _format_tick(val)

Format a tick value compactly: no trailing zeros, scientific for extreme values.
"""
function _format_tick(val::Float64)
    if val == 0.0
        return "0"
    elseif abs(val) >= 1e5 || (abs(val) < 0.01 && abs(val) > 0)
        return @sprintf("%.1e", val)
    else
        s = @sprintf("%.4g", val)
        return s
    end
end

# ─── Layout constants ──────────────────────────────────────────────────

const _SVG_MARGIN_LEFT = 65
const _SVG_MARGIN_RIGHT = 20
const _SVG_MARGIN_TOP = 35
const _SVG_MARGIN_BOTTOM = 45
const _SVG_DEFAULT_WIDTH = 600
const _SVG_DEFAULT_HEIGHT = 300

# ─── Color palette ─────────────────────────────────────────────────────

const _SVG_COLORS = [
    "#0969da",  # blue
    "#cf222e",  # red
    "#1a7f37",  # green
    "#8250df",  # purple
    "#bf8700",  # amber
    "#0550ae",  # navy
    "#953800",  # brown
    "#0a3069",  # dark blue
]

const _SVG_DOT_COLOR = "#656d76"  # gray for scatter dots

# ─── SVG building blocks ──────────────────────────────────────────────

"""
    _svg_open(width, height)

SVG opening tag with viewBox for responsive sizing.
"""
function _svg_open(width::Int, height::Int)
    return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 $width $height" width="100%" style="max-width:$(width)px;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;font-size:11px;">"""
end

"""
    _svg_axes(xmin, xmax, ymin, ymax, width, height)

Draw x and y axes with tick marks and labels.
Returns SVG markup string.
"""
function _svg_axes(xmin::Float64, xmax::Float64, ymin::Float64, ymax::Float64,
    width::Int, height::Int)

    plot_left = _SVG_MARGIN_LEFT
    plot_right = width - _SVG_MARGIN_RIGHT
    plot_top = _SVG_MARGIN_TOP
    plot_bottom = height - _SVG_MARGIN_BOTTOM
    plot_w = plot_right - plot_left
    plot_h = plot_bottom - plot_top

    buf = IOBuffer()

    # Axis lines
    println(buf, """<line x1="$plot_left" y1="$plot_bottom" x2="$plot_right" y2="$plot_bottom" stroke="#d0d7de" stroke-width="1"/>""")
    println(buf, """<line x1="$plot_left" y1="$plot_top" x2="$plot_left" y2="$plot_bottom" stroke="#d0d7de" stroke-width="1"/>""")

    # X ticks
    x_ticks = _generate_ticks(xmin, xmax)
    for xt in x_ticks
        px = plot_left + (xt - xmin) / max(xmax - xmin, 1e-30) * plot_w
        (px < plot_left - 1 || px > plot_right + 1) && continue
        pxi = round(Int, px)
        println(buf, """<line x1="$pxi" y1="$plot_bottom" x2="$pxi" y2="$(plot_bottom+4)" stroke="#d0d7de" stroke-width="1"/>""")
        println(buf, """<text x="$pxi" y="$(plot_bottom+16)" text-anchor="middle" fill="#656d76">$(_format_tick(xt))</text>""")
    end

    # Y ticks
    y_ticks = _generate_ticks(ymin, ymax)
    for yt in y_ticks
        py = plot_bottom - (yt - ymin) / max(ymax - ymin, 1e-30) * plot_h
        (py < plot_top - 1 || py > plot_bottom + 1) && continue
        pyi = round(Int, py)
        println(buf, """<line x1="$(plot_left-4)" y1="$pyi" x2="$plot_left" y2="$pyi" stroke="#d0d7de" stroke-width="1"/>""")
        # Gridline
        println(buf, """<line x1="$plot_left" y1="$pyi" x2="$plot_right" y2="$pyi" stroke="#f0f0f0" stroke-width="0.5"/>""")
        println(buf, """<text x="$(plot_left-8)" y="$(pyi+4)" text-anchor="end" fill="#656d76">$(_format_tick(yt))</text>""")
    end

    return String(take!(buf))
end

"""
    _svg_title(title, width)

Centered title text at the top of the plot.
"""
function _svg_title(title::String, width::Int)
    cx = width ÷ 2
    return """<text x="$cx" y="18" text-anchor="middle" font-weight="600" font-size="13" fill="#1f2328">$(title)</text>"""
end

"""
    _svg_xlabel(label, width, height)

Centered x-axis label at the bottom.
"""
function _svg_xlabel(label::String, width::Int, height::Int)
    cx = width ÷ 2
    return """<text x="$cx" y="$(height-4)" text-anchor="middle" fill="#656d76" font-size="11">$(label)</text>"""
end

"""
    _svg_ylabel(label, height)

Rotated y-axis label on the left.
"""
function _svg_ylabel(label::String, height::Int)
    cy = height ÷ 2
    return """<text x="12" y="$cy" text-anchor="middle" fill="#656d76" font-size="11" transform="rotate(-90,12,$cy)">$(label)</text>"""
end

# ─── Coordinate mapping ───────────────────────────────────────────────

"""
Map data coordinates to pixel coordinates.
"""
function _data_to_px(x::Float64, y::Float64,
    xmin::Float64, xmax::Float64, ymin::Float64, ymax::Float64,
    width::Int, height::Int)

    plot_left = _SVG_MARGIN_LEFT
    plot_right = width - _SVG_MARGIN_RIGHT
    plot_top = _SVG_MARGIN_TOP
    plot_bottom = height - _SVG_MARGIN_BOTTOM

    x_frac = (x - xmin) / max(xmax - xmin, 1e-30)
    y_frac = (y - ymin) / max(ymax - ymin, 1e-30)

    px = plot_left + x_frac * (plot_right - plot_left)
    py = plot_bottom - y_frac * (plot_bottom - plot_top)

    return (px, py)
end

# ─── Polyline renderer ─────────────────────────────────────────────────

"""
    _svg_polyline(x, y, xmin, xmax, ymin, ymax, width, height; color, stroke_width, dash)

Render a data series as an SVG polyline.
"""
function _svg_polyline(x::Vector{Float64}, y::Vector{Float64},
    xmin::Float64, xmax::Float64, ymin::Float64, ymax::Float64,
    width::Int, height::Int;
    color::String = _SVG_COLORS[1],
    stroke_width::Float64 = 1.5,
    dash::String = "")

    length(x) == 0 && return ""

    buf = IOBuffer()
    print(buf, """<polyline fill="none" stroke="$color" stroke-width="$stroke_width" """)
    !isempty(dash) && print(buf, """stroke-dasharray="$dash" """)
    print(buf, """points=\"""")

    for i in eachindex(x)
        (isnan(x[i]) || isnan(y[i]) || isinf(y[i])) && continue
        px, py = _data_to_px(x[i], y[i], xmin, xmax, ymin, ymax, width, height)
        i > 1 && print(buf, " ")
        @printf(buf, "%.1f,%.1f", px, py)
    end

    println(buf, "\"/>")
    return String(take!(buf))
end

# ─── Shaded band renderer ─────────────────────────────────────────────

"""
    _svg_shaded_band(x, y_upper, y_lower, xmin, xmax, ymin, ymax, width, height; color, opacity)

Render a filled polygon between upper and lower bounds (e.g. confidence bands).
"""
function _svg_shaded_band(x::Vector{Float64}, y_upper::Vector{Float64}, y_lower::Vector{Float64},
    xmin::Float64, xmax::Float64, ymin::Float64, ymax::Float64,
    width::Int, height::Int;
    color::String = _SVG_COLORS[1],
    opacity::Float64 = 0.15)

    length(x) == 0 && return ""

    buf = IOBuffer()
    print(buf, """<polygon fill="$color" opacity="$opacity" points=\"""")

    # Forward pass: upper bound
    first_point = true
    for i in eachindex(x)
        (isnan(x[i]) || isnan(y_upper[i]) || isinf(y_upper[i])) && continue
        px, py = _data_to_px(x[i], y_upper[i], xmin, xmax, ymin, ymax, width, height)
        !first_point && print(buf, " ")
        @printf(buf, "%.1f,%.1f", px, py)
        first_point = false
    end

    # Reverse pass: lower bound (closes the polygon)
    for i in length(x):-1:1
        (isnan(x[i]) || isnan(y_lower[i]) || isinf(y_lower[i])) && continue
        px, py = _data_to_px(x[i], y_lower[i], xmin, xmax, ymin, ymax, width, height)
        print(buf, " ")
        @printf(buf, "%.1f,%.1f", px, py)
    end

    println(buf, "\"/>")
    return String(take!(buf))
end

# ─── Scatter dots renderer ─────────────────────────────────────────────

"""
    _svg_scatter(x, y, xmin, xmax, ymin, ymax, width, height; color, radius)

Render data points as SVG circles.
"""
function _svg_scatter(x::Vector{Float64}, y::Vector{Float64},
    xmin::Float64, xmax::Float64, ymin::Float64, ymax::Float64,
    width::Int, height::Int;
    color::String = _SVG_DOT_COLOR,
    radius::Float64 = 2.5)

    length(x) == 0 && return ""

    buf = IOBuffer()
    for i in eachindex(x)
        (isnan(x[i]) || isnan(y[i]) || isinf(y[i])) && continue
        px, py = _data_to_px(x[i], y[i], xmin, xmax, ymin, ymax, width, height)
        @printf(buf, """<circle cx="%.1f" cy="%.1f" r="%.1f" fill="%s" opacity="0.6"/>\n""", px, py, radius, color)
    end
    return String(take!(buf))
end

# ─── Legend ─────────────────────────────────────────────────────────────

"""
    _svg_legend(labels, colors, width)

A compact legend inside the plot area (top-right corner).
"""
function _svg_legend(labels::Vector{String}, colors::Vector{String}, width::Int;
    is_dots::Vector{Bool} = fill(false, length(labels)))

    length(labels) == 0 && return ""
    x0 = width - _SVG_MARGIN_RIGHT - 10
    y0 = _SVG_MARGIN_TOP + 8

    buf = IOBuffer()
    for (i, label) in enumerate(labels)
        y = y0 + (i - 1) * 16
        c = colors[min(i, length(colors))]
        if i <= length(is_dots) && is_dots[i]
            # Dot marker
            @printf(buf, """<circle cx="%d" cy="%d" r="3" fill="%s"/>\n""", x0 - 75, y, c)
        else
            # Line marker
            @printf(buf, """<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="2"/>\n""",
                x0 - 85, y, x0 - 70, y, c)
        end
        esc_label = replace(label, "&" => "&amp;", "<" => "&lt;", ">" => "&gt;")
        @printf(buf, """<text x="%d" y="%d" text-anchor="end" fill="#1f2328" font-size="10">%s</text>\n""",
            x0, y + 4, esc_label)
    end
    return String(take!(buf))
end

# ─── Public API: single-series line plot ───────────────────────────────


# ─── Public API: multi-series line plot with optional scatter ──────────

"""
    _svg_multi_line_plot(x, ys, labels; kwargs...) → String

Generate a multi-series SVG plot.  Each entry in `ys` is a y-vector
plotted against the shared `x` vector.

# Keyword arguments
- `scatter_x`, `scatter_y`: optional scatter overlay (data points)
- `scatter_label`: label for scatter series
- `width`, `height`, `title`, `xlabel`, `ylabel`: size in pixels and the labels
"""
function _svg_multi_line_plot(x::Vector{Float64}, ys::Vector{Vector{Float64}},
    labels::Vector{String};
    scatter_x::Vector{Float64} = Float64[],
    scatter_y::Vector{Float64} = Float64[],
    scatter_label::String = "data",
    width::Int = _SVG_DEFAULT_WIDTH,
    height::Int = _SVG_DEFAULT_HEIGHT,
    title::String = "",
    xlabel::String = "",
    ylabel::String = "")

    # Compute global range across all series + scatter
    all_x = copy(x)
    all_y = Float64[]
    for yv in ys
        append!(all_y, yv)
    end
    if !isempty(scatter_x)
        append!(all_x, scatter_x)
        append!(all_y, scatter_y)
    end

    # Filter finite
    valid_x = filter(isfinite, all_x)
    valid_y = filter(isfinite, all_y)
    (isempty(valid_x) || isempty(valid_y)) && return "<!-- no valid data -->"

    xmin, xmax = extrema(valid_x)
    ymin, ymax = extrema(valid_y)

    x_pad = max((xmax - xmin) * 0.05, 1e-10)
    y_pad = max((ymax - ymin) * 0.05, abs(ymin) * 0.05 + 1e-10)
    xmin -= x_pad; xmax += x_pad
    ymin -= y_pad; ymax += y_pad

    buf = IOBuffer()
    println(buf, _svg_open(width, height))
    println(buf, """<rect width="$width" height="$height" fill="white" rx="4"/>""")
    !isempty(title) && println(buf, _svg_title(title, width))
    println(buf, _svg_axes(xmin, xmax, ymin, ymax, width, height))

    # Scatter points (draw first so lines overlay)
    if !isempty(scatter_x)
        println(buf, _svg_scatter(scatter_x, scatter_y, xmin, xmax, ymin, ymax, width, height))
    end

    # Line series
    legend_labels = String[]
    legend_colors = String[]
    legend_dots = Bool[]

    for (i, yv) in enumerate(ys)
        c = _SVG_COLORS[mod1(i, length(_SVG_COLORS))]
        println(buf, _svg_polyline(x, yv, xmin, xmax, ymin, ymax, width, height; color = c))
        push!(legend_labels, labels[min(i, length(labels))])
        push!(legend_colors, c)
        push!(legend_dots, false)
    end

    if !isempty(scatter_x)
        push!(legend_labels, scatter_label)
        push!(legend_colors, _SVG_DOT_COLOR)
        push!(legend_dots, true)
    end

    println(buf, _svg_legend(legend_labels, legend_colors, width; is_dots = legend_dots))

    !isempty(xlabel) && println(buf, _svg_xlabel(xlabel, width, height))
    !isempty(ylabel) && println(buf, _svg_ylabel(ylabel, height))
    println(buf, "</svg>")

    return String(take!(buf))
end

# ─── Convenience: observable plot (data dots + ODE curve) ──────────────

"""
    _svg_observable_plot(t_data, y_data, t_ode, y_ode, obs_name; band_upper, band_lower) → String

Plot an observable: scatter for data sample, line for true ODE trajectory.
Optionally render ±2σ GP confidence bands.
"""
function _svg_observable_plot(t_data::Vector{Float64}, y_data::Vector{Float64},
    t_ode::Vector{Float64}, y_ode::Vector{Float64},
    obs_name::String;
    band_upper::Vector{Float64} = Float64[],
    band_lower::Vector{Float64} = Float64[],
    t_band::Vector{Float64} = Float64[],
    t_est::Vector{Float64} = Float64[],
    y_est::Vector{Float64} = Float64[])

    has_band = !isempty(band_upper)
    has_est = !isempty(t_est) && !isempty(y_est)
    # Use dedicated t_band if provided, otherwise fall back to t_ode
    t_band_actual = !isempty(t_band) ? t_band : t_ode

    if !has_band && !has_est
        return _svg_multi_line_plot(t_ode, [y_ode], ["True ODE"];
            scatter_x = t_data, scatter_y = y_data, scatter_label = "data",
            title = "Observable: $obs_name",
            xlabel = "t", ylabel = obs_name)
    end

    # Custom rendering with band and/or estimated trajectory
    all_x = copy(t_ode)
    all_y = copy(y_ode)
    if has_band
        append!(all_x, t_band_actual)
        append!(all_y, band_upper)
        append!(all_y, band_lower)
    end
    append!(all_x, t_data)
    append!(all_y, y_data)
    if has_est
        append!(all_x, t_est)
        append!(all_y, y_est)
    end

    valid_x = filter(isfinite, all_x)
    valid_y = filter(isfinite, all_y)
    (isempty(valid_x) || isempty(valid_y)) && return "<!-- no valid data -->"

    xmin, xmax = extrema(valid_x)
    ymin, ymax = extrema(valid_y)
    x_pad = max((xmax - xmin) * 0.05, 1e-10)
    y_pad = max((ymax - ymin) * 0.05, abs(ymin) * 0.05 + 1e-10)
    xmin -= x_pad; xmax += x_pad; ymin -= y_pad; ymax += y_pad

    w = _SVG_DEFAULT_WIDTH; h = _SVG_DEFAULT_HEIGHT
    buf = IOBuffer()
    println(buf, _svg_open(w, h))
    println(buf, """<rect width="$w" height="$h" fill="white" rx="4"/>""")
    println(buf, _svg_title("Observable: $obs_name", w))
    println(buf, _svg_axes(xmin, xmax, ymin, ymax, w, h))
    # Band first (behind everything)
    if has_band
        println(buf, _svg_shaded_band(t_band_actual, band_upper, band_lower,
            xmin, xmax, ymin, ymax, w, h; color = _SVG_COLORS[1], opacity = 0.12))
    end
    # ODE line (true)
    println(buf, _svg_polyline(t_ode, y_ode, xmin, xmax, ymin, ymax, w, h; color = _SVG_COLORS[1]))
    # Estimated trajectory (dashed red)
    if has_est
        println(buf, _svg_polyline(t_est, y_est, xmin, xmax, ymin, ymax, w, h;
            color = _SVG_COLORS[2], dash = "6,3"))
    end
    # Data scatter ON TOP of lines and bands so dots are always visible
    println(buf, _svg_scatter(t_data, y_data, xmin, xmax, ymin, ymax, w, h))
    # Legend
    legend_labels = ["True ODE"]
    legend_colors = [_SVG_COLORS[1]]
    legend_dots = [false]
    if has_est
        push!(legend_labels, "Estimated")
        push!(legend_colors, _SVG_COLORS[2])
        push!(legend_dots, false)
    end
    if has_band
        push!(legend_labels, "±2σ GP")
        push!(legend_colors, _SVG_COLORS[1])
        push!(legend_dots, false)
    end
    push!(legend_labels, "data")
    push!(legend_colors, _SVG_DOT_COLOR)
    push!(legend_dots, true)
    println(buf, _svg_legend(legend_labels, legend_colors, w; is_dots = legend_dots))
    println(buf, _svg_xlabel("t", w, h))
    println(buf, _svg_ylabel(obs_name, h))
    println(buf, "</svg>")
    return String(take!(buf))
end

"""
    _svg_state_plot(t_ode, y_ode, state_name; dash="", band_upper, band_lower) → String

Plot a single state variable trajectory.  Pass `dash="6,3"` for dashed lines
(used for latent/unobservable states). Optionally render ±2σ GP confidence bands.
"""
function _svg_state_plot(t_ode::Vector{Float64}, y_ode::Vector{Float64},
    state_name::String; dash::String = "",
    band_upper::Vector{Float64} = Float64[],
    band_lower::Vector{Float64} = Float64[],
    t_band::Vector{Float64} = Float64[],
    t_est::Vector{Float64} = Float64[],
    y_est::Vector{Float64} = Float64[])

    has_est = !isempty(t_est) && !isempty(y_est)
    has_band = !isempty(band_upper)
    # Use dedicated t_band if provided, otherwise fall back to t_ode
    t_band_actual = !isempty(t_band) ? t_band : t_ode

    all_x = copy(t_ode)
    all_y = copy(y_ode)
    if has_band
        append!(all_x, t_band_actual)
        append!(all_y, band_upper)
        append!(all_y, band_lower)
    end
    if has_est
        append!(all_x, t_est)
        append!(all_y, y_est)
    end

    valid = [(t_ode[i], y_ode[i]) for i in eachindex(t_ode) if isfinite(t_ode[i]) && isfinite(y_ode[i])]
    isempty(valid) && return "<!-- no valid data -->"

    all_finite_x = filter(isfinite, all_x)
    all_finite_y = filter(isfinite, all_y)
    isempty(all_finite_y) && return "<!-- no valid data -->"

    xmin, xmax = extrema(all_finite_x)
    ymin = minimum(all_finite_y)
    ymax = maximum(all_finite_y)
    x_pad = max((xmax - xmin) * 0.05, 1e-10)
    y_pad = max((ymax - ymin) * 0.05, abs(ymin) * 0.05 + 1e-10)
    xmin -= x_pad; xmax += x_pad; ymin -= y_pad; ymax += y_pad

    w = _SVG_DEFAULT_WIDTH; h = _SVG_DEFAULT_HEIGHT
    buf = IOBuffer()
    println(buf, _svg_open(w, h))
    println(buf, """<rect width="$w" height="$h" fill="white" rx="4"/>""")
    println(buf, _svg_title("State: $state_name", w))
    println(buf, _svg_axes(xmin, xmax, ymin, ymax, w, h))
    # Band (behind line)
    if has_band
        println(buf, _svg_shaded_band(t_band_actual, band_upper, band_lower,
            xmin, xmax, ymin, ymax, w, h; color = _SVG_COLORS[2], opacity = 0.12))
    end
    println(buf, _svg_polyline(t_ode, y_ode, xmin, xmax, ymin, ymax, w, h;
        color = _SVG_COLORS[2], dash = dash))
    # Estimated trajectory (dashed, darker)
    if has_est
        println(buf, _svg_polyline(t_est, y_est, xmin, xmax, ymin, ymax, w, h;
            color = _SVG_COLORS[4], dash = "6,3"))
        # Add legend
        legend_labels = ["True", "Estimated"]
        legend_colors = [_SVG_COLORS[2], _SVG_COLORS[4]]
        legend_dots = [false, false]
        println(buf, _svg_legend(legend_labels, legend_colors, w; is_dots = legend_dots))
    end
    println(buf, _svg_xlabel("t", w, h))
    println(buf, _svg_ylabel(state_name, h))
    println(buf, "</svg>")
    return String(take!(buf))
end

# ─── Generate all trajectory plots for a PEP ──────────────────────────

"""
    _generate_trajectory_plots(pep; uq_interpolants=nothing) → Vector{Tuple{String, String, Bool}}

Solve the ODE at true parameters, generate SVG plots for each observable
and each state variable.  Returns vector of (section_title, svg_string, is_observable)
triples.  Observable states use solid lines; latent states use dashed lines.

When `uq_interpolants` is provided (a `Dict{String, AGPInterpolatorUQ}`), GP
posterior ±2σ bands are overlaid on observable plots.
"""
function _generate_trajectory_plots(pep; uq_interpolants = nothing,
    estimated_result::Union{Nothing, ParameterEstimationResult} = nothing)
    plots = Tuple{String, String, Bool}[]

    # Solve ODE at true parameters
    sol = _solve_ode_at_true_params(pep)
    isnothing(sol) && return plots

    t_ode = collect(range(sol.t[1], sol.t[end]; length = 500))
    t_data = pep.data_sample["t"]

    # Solve ODE at estimated parameters (if available)
    est_sol = nothing
    if !isnothing(estimated_result) && !isnothing(estimated_result.solution)
        est_sol = estimated_result.solution
    elseif !isnothing(estimated_result)
        est_sol = _solve_ode_at_estimated_params(pep, estimated_result)
    end
    t_est = !isnothing(est_sol) ? t_ode : Float64[]

    # Build set of state names that appear directly as observable RHS
    # (e.g.  y1 ~ x(t)  means "x" is an observable state)
    # Also build a mapping: state_name → observable_name (for UQ band lookup)
    observable_state_names = Set{String}()
    state_to_obs_name = Dict{String, String}()  # state clean name → observable LHS clean name
    for mq in pep.measured_quantities
        rhs_str = replace(string(mq.rhs), "(t)" => "")
        obs_name = replace(string(mq.lhs), r"\(.*\)" => "")
        # Only flag as observable if RHS is a bare state (no operators)
        # A bare state string contains only word characters (letters, digits, _)
        if occursin(r"^\w+$", rhs_str)
            push!(observable_state_names, rhs_str)
            state_to_obs_name[rhs_str] = obs_name
        end
    end

    # Extend time range 10% beyond data for GP bands to show endpoint uncertainty growth
    dt = t_data[end] - t_data[1]
    t_band = collect(range(t_data[1] - 0.1 * dt, t_data[end] + 0.1 * dt; length = 500))

    # Observable plots (always marked is_observable=true)
    for mq in pep.measured_quantities
        obs_name = string(mq.lhs)
        # Skip _trfn_ auxiliaries
        if startswith(replace(obs_name, r"\(.*\)" => ""), "_obs_trfn_")
            continue
        end

        obs_rhs = Symbolics.diff2term(mq.rhs)

        # Get data sample for this observable
        y_data = _get_observable_data(pep, obs_rhs)
        isnothing(y_data) && continue

        # Evaluate observable along ODE solution
        y_ode = _evaluate_observable_on_solution(sol, mq, t_ode)
        isnothing(y_ode) && continue

        clean_name = replace(obs_name, r"\(.*\)" => "")

        # Evaluate observable on estimated solution
        y_est_obs = Float64[]
        if !isnothing(est_sol)
            y_est_obs_raw = _evaluate_observable_on_solution(est_sol, mq, t_ode)
            if !isnothing(y_est_obs_raw)
                y_est_obs = Float64.(y_est_obs_raw)
            end
        end

        # Build GP uncertainty bands if available (using extended t_band for endpoint growth)
        band_upper = Float64[]
        band_lower = Float64[]
        obs_t_band = Float64[]
        if !isnothing(uq_interpolants) && haskey(uq_interpolants, clean_name)
            try
                interp_uq = uq_interpolants[clean_name]
                band_upper = Float64[]
                band_lower = Float64[]
                for ti in t_band
                    μ, Σ = joint_derivative_covariance(interp_uq, ti, 0)
                    σ = sqrt(max(Σ[1, 1], 0.0))
                    push!(band_upper, μ[1] + 2σ)
                    push!(band_lower, μ[1] - 2σ)
                end
                obs_t_band = t_band
            catch e
                @debug "[SVG] GP band computation failed for $clean_name: $e"
                band_upper = Float64[]
                band_lower = Float64[]
                obs_t_band = Float64[]
            end
        end

        svg = _svg_observable_plot(t_data, y_data, t_ode, y_ode, clean_name;
            band_upper = band_upper, band_lower = band_lower,
            t_band = obs_t_band,
            t_est = isempty(y_est_obs) ? Float64[] : t_ode,
            y_est = y_est_obs)
        push!(plots, ("Observable: $clean_name", svg, true))
    end

    # State variable plots — dashed for latent states
    states = pep.model.original_states
    for (i, s) in enumerate(states)
        s_name = replace(string(s), "(t)" => "")
        # Skip _trfn_ auxiliary states
        startswith(s_name, "_trfn_") && continue

        y_ode = try
            [sol(ti; idxs = i) for ti in t_ode]
        catch
            try
                [sol(ti)[i] for ti in t_ode]
            catch
                nothing
            end
        end
        isnothing(y_ode) && continue

        is_obs = s_name in observable_state_names
        dash_style = is_obs ? "" : "6,3"
        title_prefix = is_obs ? "Observable" : "Latent State"

        # Estimated state trajectory
        y_est_state = Float64[]
        if !isnothing(est_sol)
            y_est_state = try
                Float64[est_sol(ti; idxs = i) for ti in t_ode]
            catch
                try
                    Float64[est_sol(ti)[i] for ti in t_ode]
                catch
                    Float64[]
                end
            end
        end

        # Build GP bands for observable states (using extended t_band)
        state_band_upper = Float64[]
        state_band_lower = Float64[]
        state_t_band = Float64[]
        if is_obs && !isnothing(uq_interpolants)
            # Look up UQ interp by observable name (e.g. state "r" → obs "y1")
            obs_key = get(state_to_obs_name, s_name, s_name)
            if haskey(uq_interpolants, obs_key)
                try
                    interp_uq = uq_interpolants[obs_key]
                    for ti in t_band
                        μ, Σ = joint_derivative_covariance(interp_uq, ti, 0)
                        σ = sqrt(max(Σ[1, 1], 0.0))
                        push!(state_band_upper, μ[1] + 2σ)
                        push!(state_band_lower, μ[1] - 2σ)
                    end
                    state_t_band = t_band
                catch e
                    @debug "[SVG] GP band failed for state $s_name: $e"
                    state_band_upper = Float64[]
                    state_band_lower = Float64[]
                    state_t_band = Float64[]
                end
            end
        end

        svg = _svg_state_plot(t_ode, Float64.(y_ode), s_name; dash = dash_style,
            band_upper = state_band_upper, band_lower = state_band_lower,
            t_band = state_t_band,
            t_est = isempty(y_est_state) ? Float64[] : t_ode,
            y_est = y_est_state)
        push!(plots, ("$title_prefix: $s_name", svg, is_obs))
    end

    return plots
end

"""
Solve the PEP's ODE system at true parameter values and initial conditions.
Returns the ODE solution or nothing on failure.
"""
function _solve_ode_at_true_params(pep)
    try
        sys = pep.model.system
        params = pep.model.original_parameters
        states = pep.model.original_states

        # Build Dict-based u0/p for modern MTK interface
        u0_dict = Dict(states .=> [pep.ic[s] for s in states])
        p_dict = Dict(params .=> [pep.p_true[p] for p in params])

        t_data = pep.data_sample["t"]
        tspan = (t_data[1], t_data[end])

        prob = ODEProblem(sys, merge(u0_dict, p_dict), tspan)
        sol = OrdinaryDiffEq.solve(prob, AutoVern9(Rodas5P()); abstol = 1e-12, reltol = 1e-12, saveat = Float64[])
        return sol.retcode == SciMLBase.ReturnCode.Success ? sol : nothing
    catch e
        @warn "[SVG] ODE solve at true params failed: $e"
        return nothing
    end
end

"""
Solve the PEP's ODE system at estimated parameter values and initial conditions.
Returns the ODE solution or nothing on failure.
"""
function _solve_ode_at_estimated_params(pep, est_result::ParameterEstimationResult)
    try
        sys = pep.model.system
        params = pep.model.original_parameters
        states = pep.model.original_states

        # Build u0 from estimated states
        u0_dict = Dict{Any, Any}()
        for s in states
            s_name = replace(string(s), "(t)" => "")
            for (es, ev) in est_result.states
                if replace(string(es), "(t)" => "") == s_name
                    u0_dict[s] = ev
                    break
                end
            end
        end
        # Build p from estimated parameters
        p_dict = Dict{Any, Any}()
        for p in params
            p_name = replace(string(p), "(t)" => "")
            for (ep, ev) in est_result.parameters
                if replace(string(ep), "(t)" => "") == p_name
                    p_dict[p] = ev
                    break
                end
            end
        end

        t_data = pep.data_sample["t"]
        tspan = (t_data[1], t_data[end])

        prob = ODEProblem(sys, merge(u0_dict, p_dict), tspan)
        sol = OrdinaryDiffEq.solve(prob, AutoVern9(Rodas5P()); abstol = 1e-12, reltol = 1e-12, saveat = Float64[])
        return sol.retcode == SciMLBase.ReturnCode.Success ? sol : nothing
    catch e
        @debug "[SVG] ODE solve at estimated params failed: $e"
        return nothing
    end
end

"""
Get observable data from the PEP data_sample, matching by observable RHS key.
"""
function _get_observable_data(pep, obs_rhs)
    # Try direct key lookup
    if haskey(pep.data_sample, obs_rhs)
        return pep.data_sample[obs_rhs]
    end
    # Try string key fallback
    obs_str = string(obs_rhs)
    for (k, v) in pep.data_sample
        k == "t" && continue
        if string(k) == obs_str
            return v
        end
    end
    return nothing
end

"""
Evaluate a measured quantity expression along an ODE solution at given time points.
Uses symbolic substitution of state values and parameters into the observable RHS.
"""
function _evaluate_observable_on_solution(sol, mq, t_points)
    try
        rhs = mq.rhs

        # Get the system states and parameters from the ODE problem
        sys = sol.prob.f.sys
        states = ModelingToolkit.unknowns(sys)
        params = ModelingToolkit.parameters(sys)
        t_var = ModelingToolkit.get_iv(sys)

        # Extract parameter values using symbolic lookup (MTKParameters doesn't support integer indexing)
        p_val_dict = Dict{Any, Any}()
        for p in params
            p_val_dict[p] = try
                sol.prob.ps[p]
            catch
                try
                    sol[p]
                catch
                    NaN
                end
            end
        end

        y = Float64[]
        for ti in t_points
            sol_at_t = sol(ti)
            subst_dict = Dict{Any, Any}()
            for (i, s) in enumerate(states)
                subst_dict[s] = sol_at_t[i]
            end
            merge!(subst_dict, p_val_dict)
            # Also substitute the independent variable
            subst_dict[t_var] = ti
            val = try
                Float64(Symbolics.value(Symbolics.substitute(rhs, subst_dict)))
            catch
                NaN
            end
            push!(y, val)
        end

        any(isfinite, y) && return y
        return nothing
    catch
        return nothing
    end
end
