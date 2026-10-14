# ── Cancellation discipline ──────────────────────────────────────────────────
# Broad production catches must never swallow Ctrl-C. Drop this as the FIRST
# statement of any `catch` that handles expected numerical failure, so an
# InterruptException always propagates (2026-07 review finding: 0/64 catches
# filtered it, turning Ctrl-C into "log, return empty, keep trying fallbacks").

"""
    _rethrow_if_interrupt(err)

Rethrow `err` if it is an `InterruptException` (cancellation must propagate);
otherwise do nothing. Use as the first statement of broad `catch` blocks.
"""
@inline _rethrow_if_interrupt(err) = err isa InterruptException ? rethrow(err) : nothing

# ── Phase heartbeats ──────────────────────────────────────────────────────────
# Live, flushed, timestamped phase markers so a silent log still localizes the
# running phase (2026-07 hang post-mortem: buffered output made a stalled run
# unlocatable — see repro/hc_threading_mwe_2026_07_22/PROVENANCE_*.md). Coarse
# by design: once-per-run phases plus one line per interpolator — never inside
# per-point/per-combo inner loops.

"""
    _heartbeat(opts, phase; kind=:start, extra="")

Emit a flushed, timestamped `[HH:MM:SS] ▶/✓ phase` marker. Printed when
`opts.progress` is true, or when `opts.heartbeat` is true and `opts.nooutput`
is false. In the second case the line starts with `[HB HH:MM:SS]`, so that it
can be found among the rest of the development output. `kind` is `:start`,
`:done`, or `:note`; `extra` is appended verbatim.
"""
function _heartbeat(opts, phase::AbstractString; kind::Symbol = :start, extra::AbstractString = "")
    among_other_output = opts.heartbeat && !opts.nooutput
    (opts.progress || among_other_output) || return nothing
    mark = kind === :start ? "▶" : kind === :done ? "✓" : "•"
    stamp = Dates.format(Dates.now(), "HH:MM:SS")
    opening = among_other_output ? "[HB " : "["
    if isempty(extra)
        println(opening, stamp, "] ", mark, " ", phase)
    else
        println(opening, stamp, "] ", mark, " ", phase, " ", extra)
    end
    flush(stdout)
    flush(stderr)
    return nothing
end

"""
    _heartbeat_run_header(opts, model_name)

One-line self-identifying run header (model, pid, julia/ODEPE/HC versions,
thread counts) so any log fragment is attributable without external context.
"""
function _heartbeat_run_header(opts, model_name::AbstractString)
    (opts.heartbeat && !opts.nooutput) || return nothing
    hc_ver = try
        string(pkgversion(HomotopyContinuation))
    catch
        "?"
    end
    odepe_ver = try
        string(pkgversion(@__MODULE__))
    catch
        "?"
    end
    _heartbeat(opts, "run";
        extra = string("model=", model_name, " pid=", Base.Libc.getpid(),
            " julia=", VERSION, " odepe=", odepe_ver, " hc=", hc_ver,
            " threads=", Threads.nthreads(), " gcthreads=", Threads.ngcthreads()))
end

