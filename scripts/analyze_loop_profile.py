#!/usr/bin/env python3
"""
Offline analysis for control-loop timing profiles captured by LoopProfiler.

Reads one or more ``.npz`` (or ``.csv``) profile files produced by
``mini_bdx_runtime.loop_profiler`` and reports the baseline metrics:

  * loop period  -> mean -> actual achieved frequency
  * jitter       -> std of the period (headline metric)
  * tail latency -> p50 / p99 / p99.9 / max of per-iteration compute time
  * per-section  -> read (sensors) / infer (policy) / write (servos) + glue
  * overrun      -> fraction of iterations whose compute exceeds the target
                    period; informs the max stable frequency
  * GC           -> whether latency spikes line up with garbage-collection events

Pass a single file for a full report (metrics + plots). Pass several files to
get a side-by-side comparison table and overlaid histograms -- this is how the
later C++ run gets compared against this Python baseline apples-to-apples
("cut jitter from X to Y, raised stable frequency from W to Z Hz").

Run this on a desktop, not the Pi -- it only needs numpy (+ matplotlib for the
plots). The capture happens on the robot; the analysis happens anywhere.

Usage:
    python analyze_loop_profile.py run.npz
    python analyze_loop_profile.py gc_on.npz gc_off.npz --labels gc_on,gc_off
    python analyze_loop_profile.py python.npz cpp.npz --labels python,cpp --outdir report/
"""
import argparse
import json
import os
import sys

import numpy as np

NS_PER_US = 1e3
NS_PER_MS = 1e6
NS_PER_S = 1e9

PERCENTILES = [50.0, 99.0, 99.9]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _scalar(v):
    """Unwrap a 0-d numpy array / bytes into a plain Python value."""
    a = np.asarray(v)
    if a.ndim == 0:
        a = a.item()
    if isinstance(a, bytes):
        return a.decode()
    return a


def load_profile(path, target_hz=None):
    """Load a profile file into a normalized dict of numpy arrays + metadata."""
    if path.endswith(".csv"):
        return _load_csv(path, target_hz)
    return _load_npz(path, target_hz)


def _load_npz(path, target_hz):
    d = np.load(path, allow_pickle=False)
    keys = set(d.files)
    section_names = (
        [str(s) for s in d["section_names"]] if "section_names" in keys
        else ["read", "infer", "write"]
    )
    prof = {
        "path": path,
        "iter_start_ns": d["iter_start_ns"].astype(np.int64),
        "busy_ns": d["busy_ns"].astype(np.int64),
        "section_names": section_names,
        "sections": {
            name: d[f"{name}_ns"].astype(np.int64)
            for name in section_names if f"{name}_ns" in keys
        },
        "gc_event_ns": d["gc_event_ns"].astype(np.int64) if "gc_event_ns" in keys else np.array([], np.int64),
        "gc_event_phase": d["gc_event_phase"].astype(np.int8) if "gc_event_phase" in keys else np.array([], np.int8),
        "impl": str(_scalar(d["impl"])) if "impl" in keys else "unknown",
        "gc_enabled": int(_scalar(d["gc_enabled"])) if "gc_enabled" in keys else 1,
        "sched_fifo": int(_scalar(d["sched_fifo"])) if "sched_fifo" in keys else 0,
        "platform": str(_scalar(d["platform"])) if "platform" in keys else "",
        "hostname": str(_scalar(d["hostname"])) if "hostname" in keys else "",
        "notes": str(_scalar(d["notes"])) if "notes" in keys else "",
    }
    if target_hz is not None:
        prof["target_period_ns"] = int(round(NS_PER_S / target_hz))
        prof["target_freq_hz"] = float(target_hz)
    else:
        prof["target_period_ns"] = int(_scalar(d["target_period_ns"])) if "target_period_ns" in keys else 0
        prof["target_freq_hz"] = float(_scalar(d["target_freq_hz"])) if "target_freq_hz" in keys else 0.0
    return prof


def _load_csv(path, target_hz):
    """Minimal CSV loader so a future C++ build can emit the same columns.

    Expects a header row with at least iter_start_ns,busy_ns and optionally
    read_ns,infer_ns,write_ns. GC events / metadata are not carried in CSV;
    pass --target-hz so period overruns can still be computed.
    """
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.int64)
    names = list(data.dtype.names)
    section_names = [n[:-3] for n in names if n.endswith("_ns") and n not in ("iter_start_ns", "busy_ns")]
    prof = {
        "path": path,
        "iter_start_ns": np.asarray(data["iter_start_ns"], np.int64),
        "busy_ns": np.asarray(data["busy_ns"], np.int64),
        "section_names": section_names,
        "sections": {n: np.asarray(data[f"{n}_ns"], np.int64) for n in section_names},
        "gc_event_ns": np.array([], np.int64),
        "gc_event_phase": np.array([], np.int8),
        "impl": os.path.basename(path),
        "gc_enabled": 1, "sched_fifo": 0, "platform": "", "hostname": "", "notes": "",
    }
    hz = target_hz or 50.0
    prof["target_period_ns"] = int(round(NS_PER_S / hz))
    prof["target_freq_hz"] = float(hz)
    return prof


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _pct(arr, p):
    return float(np.percentile(arr, p)) if arr.size else float("nan")


def _stats_us(arr):
    """Common stats for an array of nanosecond durations, reported in µs."""
    if arr.size == 0:
        return {k: float("nan") for k in ("mean", "std", "p50", "p99", "p99_9", "max", "min")}
    a = arr.astype(np.float64) / NS_PER_US
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "p50": _pct(a, 50.0),
        "p99": _pct(a, 99.0),
        "p99_9": _pct(a, 99.9),
        "max": float(a.max()),
        "min": float(a.min()),
    }


def gc_intervals(prof):
    """Reconstruct [start_ns, stop_ns] GC intervals from the event stream."""
    ev_ns = prof["gc_event_ns"]
    ev_ph = prof["gc_event_phase"]
    intervals = []
    open_start = None
    for t, ph in zip(ev_ns, ev_ph):
        if ph == 0:  # start
            open_start = t
        elif ph == 1 and open_start is not None:  # stop
            intervals.append((open_start, t))
            open_start = None
    return intervals


def iters_overlapping_gc(prof):
    """Boolean mask: which iterations' compute window overlaps a GC interval."""
    iter_start = prof["iter_start_ns"]
    busy = prof["busy_ns"]
    n = busy.size
    overlaps = np.zeros(n, dtype=bool)
    intervals = gc_intervals(prof)
    if not intervals or n == 0:
        return overlaps
    starts = iter_start
    ends = iter_start + busy
    gc_starts = np.array([a for a, _ in intervals], np.int64)
    gc_ends = np.array([b for _, b in intervals], np.int64)
    # A GC (gs, ge) overlaps iteration (s, e) iff gs <= e and ge >= s. Find, per
    # iteration, the last GC starting at/before its end, then check that GC's end.
    idx = np.searchsorted(gc_starts, ends, side="right") - 1
    valid = idx >= 0
    cand = idx[valid]
    overlaps[valid] = gc_ends[cand] >= starts[valid]
    return overlaps


def compute_metrics(prof):
    iter_start = prof["iter_start_ns"]
    busy = prof["busy_ns"]
    n = busy.size
    target = prof["target_period_ns"]

    # period = start-to-start spacing (includes the trailing sleep)
    period = np.diff(iter_start) if n > 1 else np.array([], np.int64)

    m = {
        "path": prof["path"],
        "impl": prof["impl"],
        "n_iters": int(n),
        "target_freq_hz": prof["target_freq_hz"],
        "target_period_us": target / NS_PER_US,
        "gc_enabled": bool(prof["gc_enabled"]),
        "sched_fifo": bool(prof["sched_fifo"]),
        "platform": prof["platform"],
        "hostname": prof["hostname"],
        "notes": prof["notes"],
        "duration_s": float((iter_start[-1] - iter_start[0]) / NS_PER_S) if n > 1 else 0.0,
    }

    # --- period / achieved frequency / jitter ---
    if period.size:
        period_us = period.astype(np.float64) / NS_PER_US
        mean_period_ns = float(period.mean())
        m["period"] = _stats_us(period)
        m["achieved_hz"] = NS_PER_S / mean_period_ns if mean_period_ns else float("nan")
        # jitter = std of the period; also RMS deviation from the target period
        m["jitter_std_us"] = float(period_us.std())
        if target:
            m["jitter_rms_from_target_us"] = float(
                np.sqrt(np.mean((period_us - target / NS_PER_US) ** 2))
            )
    else:
        m["period"] = _stats_us(np.array([], np.int64))
        m["achieved_hz"] = float("nan")
        m["jitter_std_us"] = float("nan")

    # --- tail latency of per-iteration compute (busy) ---
    m["busy"] = _stats_us(busy)

    # --- per-section breakdown ---
    m["sections"] = {name: _stats_us(arr) for name, arr in prof["sections"].items()}
    if prof["sections"]:
        section_sum = np.zeros(n, np.int64)
        for arr in prof["sections"].values():
            section_sum += arr
        glue = np.clip(busy - section_sum, 0, None)  # Python bookkeeping remainder
        m["sections"]["glue"] = _stats_us(glue)

    # --- overruns / headroom (informs max stable frequency) ---
    if target:
        overruns = int(np.count_nonzero(busy > target))
        m["overrun_fraction"] = overruns / n if n else float("nan")
        # headroom: how much budget is left at the 99th-pct compute time
        m["headroom_p99_us"] = (target - _pct(busy.astype(np.float64), 99.0)) / 1.0 / NS_PER_US
        # crude estimate of the highest period the p99 compute can sustain
        p99_busy_ns = _pct(busy.astype(np.float64), 99.0)
        m["max_stable_hz_p99_estimate"] = NS_PER_S / p99_busy_ns if p99_busy_ns else float("nan")

    # --- GC correlation ---
    m["gc"] = _gc_correlation(prof, busy)
    return m


def _gc_correlation(prof, busy):
    intervals = gc_intervals(prof)
    n = busy.size
    out = {
        "n_gc_events": int(prof["gc_event_ns"].size),
        "n_gc_collections": len(intervals),
    }
    if not intervals or n == 0:
        out["iters_overlapping_gc_fraction"] = 0.0
        return out

    # Flag iterations whose compute window [start, start+busy] overlaps a GC.
    overlaps = iters_overlapping_gc(prof)
    n_overlap = int(np.count_nonzero(overlaps))
    out["iters_overlapping_gc_fraction"] = n_overlap / n
    if n_overlap and n_overlap < n:
        out["busy_us_with_gc_mean"] = float(busy[overlaps].mean() / NS_PER_US)
        out["busy_us_without_gc_mean"] = float(busy[~overlaps].mean() / NS_PER_US)
        out["busy_us_with_gc_p99"] = _pct(busy[overlaps].astype(np.float64) / NS_PER_US, 99.0)

    # Of the worst 0.1% slowest iterations, how many coincide with a GC?
    if n >= 1000:
        thresh = np.percentile(busy, 99.9)
        worst = busy >= thresh
        n_worst = int(np.count_nonzero(worst))
        if n_worst:
            out["worst_0p1pct_count"] = n_worst
            out["worst_0p1pct_gc_coincidence_fraction"] = float(
                np.count_nonzero(worst & overlaps) / n_worst
            )
    return out


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt(x, nd=1):
    if isinstance(x, float) and (x != x):  # nan
        return "—"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def print_report(metrics_list):
    for m in metrics_list:
        print("=" * 72)
        print(f"  {m['impl']}   ({os.path.basename(m['path'])})")
        print("=" * 72)
        print(f"  iterations      : {m['n_iters']}  over {_fmt(m['duration_s'])} s")
        print(f"  host / platform : {m['hostname']}  {m['platform']}")
        print(f"  target          : {_fmt(m['target_freq_hz'])} Hz "
              f"({_fmt(m['target_period_us'])} µs/iter)")
        print(f"  GC              : {'enabled' if m['gc_enabled'] else 'DISABLED'}"
              f"   SCHED_FIFO: {'yes' if m['sched_fifo'] else 'no'}")
        print("  ---- loop period ----")
        print(f"  achieved freq   : {_fmt(m['achieved_hz'], 2)} Hz")
        print(f"  jitter (std)    : {_fmt(m['jitter_std_us'])} µs   <-- headline")
        if "jitter_rms_from_target_us" in m:
            print(f"  jitter (RMS dev): {_fmt(m['jitter_rms_from_target_us'])} µs from target")
        p = m["period"]
        print(f"  period p50/p99/p99.9/max : "
              f"{_fmt(p['p50'])}/{_fmt(p['p99'])}/{_fmt(p['p99_9'])}/{_fmt(p['max'])} µs")
        print("  ---- per-iteration compute (busy) ----")
        b = m["busy"]
        print(f"  mean            : {_fmt(b['mean'])} µs")
        print(f"  p50/p99/p99.9/max: "
              f"{_fmt(b['p50'])}/{_fmt(b['p99'])}/{_fmt(b['p99_9'])}/{_fmt(b['max'])} µs")
        if "overrun_fraction" in m:
            print(f"  overruns        : {_fmt(m['overrun_fraction']*100, 3)} % of iters "
                  f"exceed the {_fmt(m['target_period_us'])} µs budget")
            print(f"  headroom @p99   : {_fmt(m['headroom_p99_us'])} µs")
            print(f"  max stable Hz   : ~{_fmt(m.get('max_stable_hz_p99_estimate', float('nan')), 1)} "
                  f"(1e9 / busy p99; sweep --control_freq to confirm)")
        print("  ---- per-section (mean / p99 / max, µs) ----")
        for name, s in m["sections"].items():
            print(f"    {name:6s}: {_fmt(s['mean']):>9} / {_fmt(s['p99']):>9} / {_fmt(s['max']):>9}")
        g = m["gc"]
        print("  ---- garbage collection ----")
        print(f"  collections     : {g['n_gc_collections']}  ({g['n_gc_events']} events)")
        if "busy_us_with_gc_mean" in g:
            print(f"  busy w/ GC      : mean {_fmt(g['busy_us_with_gc_mean'])} µs "
                  f"(vs {_fmt(g['busy_us_without_gc_mean'])} µs without)")
        if "worst_0p1pct_gc_coincidence_fraction" in g:
            print(f"  of worst 0.1%   : {_fmt(g['worst_0p1pct_gc_coincidence_fraction']*100)} % "
                  f"coincide with a GC event")
        print()


def comparison_table_md(metrics_list, labels):
    cols = ["metric"] + labels
    rows = [
        ("impl", [m["impl"] for m in metrics_list]),
        ("iterations", [m["n_iters"] for m in metrics_list]),
        ("achieved Hz", [_fmt(m["achieved_hz"], 2) for m in metrics_list]),
        ("jitter std (µs)", [_fmt(m["jitter_std_us"]) for m in metrics_list]),
        ("period p99 (µs)", [_fmt(m["period"]["p99"]) for m in metrics_list]),
        ("period max (µs)", [_fmt(m["period"]["max"]) for m in metrics_list]),
        ("busy mean (µs)", [_fmt(m["busy"]["mean"]) for m in metrics_list]),
        ("busy p99 (µs)", [_fmt(m["busy"]["p99"]) for m in metrics_list]),
        ("busy p99.9 (µs)", [_fmt(m["busy"]["p99_9"]) for m in metrics_list]),
        ("busy max (µs)", [_fmt(m["busy"]["max"]) for m in metrics_list]),
        ("overrun %", [_fmt(m.get("overrun_fraction", float("nan")) * 100, 3) for m in metrics_list]),
        ("GC enabled", [_fmt(m["gc_enabled"]) for m in metrics_list]),
    ]
    for sec in ("read", "infer", "write", "glue"):
        rows.append((f"{sec} mean (µs)",
                     [_fmt(m["sections"].get(sec, {}).get("mean", float("nan"))) for m in metrics_list]))
    out = ["| " + " | ".join(cols) + " |",
           "| " + " | ".join(["---"] * len(cols)) + " |"]
    for name, vals in rows:
        out.append("| " + " | ".join([name] + [str(v) for v in vals]) + " |")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def make_plots(profiles, metrics_list, labels, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # matplotlib optional (e.g. on the Pi)
        print(f"[plots] matplotlib unavailable ({e}); skipping plots.", file=sys.stderr)
        return []

    os.makedirs(outdir, exist_ok=True)
    written = []

    # 1) Period histogram (overlaid for comparison).
    fig, ax = plt.subplots(figsize=(9, 5))
    for prof, lab in zip(profiles, labels):
        period_us = np.diff(prof["iter_start_ns"]).astype(np.float64) / NS_PER_US
        if period_us.size == 0:
            continue
        hi = np.percentile(period_us, 99.9)
        ax.hist(period_us, bins=200, range=(period_us.min(), hi), alpha=0.55, label=lab)
    target_us = profiles[0]["target_period_ns"] / NS_PER_US
    if target_us:
        ax.axvline(target_us, color="k", ls="--", lw=1, label=f"target {target_us:.0f} µs")
    ax.set_xlabel("loop period (µs)")
    ax.set_ylabel("count")
    ax.set_title("Loop period distribution")
    ax.legend()
    p = os.path.join(outdir, "period_histogram.png")
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig); written.append(p)

    # 2) Latency time-series with GC markers (one panel per profile).
    fig, axes = plt.subplots(len(profiles), 1, figsize=(11, 3.2 * len(profiles)),
                             squeeze=False, sharex=False)
    for k, (prof, lab) in enumerate(zip(profiles, labels)):
        ax = axes[k][0]
        t0 = prof["iter_start_ns"][0] if prof["iter_start_ns"].size else 0
        t_s = (prof["iter_start_ns"] - t0).astype(np.float64) / NS_PER_S
        busy_us = prof["busy_ns"].astype(np.float64) / NS_PER_US
        ax.plot(t_s, busy_us, lw=0.4, color="C0", zorder=1)
        target_us = prof["target_period_ns"] / NS_PER_US
        if target_us:
            ax.axhline(target_us, color="k", ls="--", lw=0.8, zorder=3)
        # Mark iterations whose compute coincided with a GC as red dots, so the
        # eye can see whether the spikes line up with garbage collection.
        gc_mask = iters_overlapping_gc(prof)
        n_gc = int(np.count_nonzero(gc_mask))
        if n_gc:
            ax.scatter(t_s[gc_mask], busy_us[gc_mask], s=6, color="red",
                       zorder=2, label=f"GC iters (n={n_gc})")
            ax.legend(loc="upper right", fontsize=8)
        ax.set_ylabel("busy (µs)")
        ax.set_title(f"{lab}: per-iteration compute (red = coincides with GC)")
        ax.set_ylim(0, np.percentile(busy_us, 99.9) * 1.3 if busy_us.size else 1)
    axes[-1][0].set_xlabel("time (s)")
    p = os.path.join(outdir, "latency_timeseries.png")
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig); written.append(p)

    # 3) Per-section breakdown (mean stacked bar, one bar per profile).
    section_order = ["read", "infer", "write", "glue"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bottoms = np.zeros(len(metrics_list))
    x = np.arange(len(metrics_list))
    for sec in section_order:
        vals = np.array([m["sections"].get(sec, {}).get("mean", 0.0) for m in metrics_list])
        vals = np.nan_to_num(vals)
        ax.bar(x, vals, bottom=bottoms, label=sec)
        bottoms += vals
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("mean time per iteration (µs)")
    ax.set_title("Per-section breakdown (mean)")
    ax.legend()
    p = os.path.join(outdir, "section_breakdown.png")
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig); written.append(p)

    return written


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("profiles", nargs="+", help="one or more .npz/.csv profile files")
    ap.add_argument("--labels", default=None, help="comma-separated labels (default: impl names)")
    ap.add_argument("--target-hz", type=float, default=None,
                    help="override target frequency (needed for .csv)")
    ap.add_argument("--outdir", default="loop_profile_report", help="output directory")
    ap.add_argument("--no-plots", action="store_true", help="skip plot generation")
    args = ap.parse_args()

    profiles = [load_profile(p, args.target_hz) for p in args.profiles]
    metrics_list = [compute_metrics(p) for p in profiles]
    if args.labels:
        labels = args.labels.split(",")
    else:
        labels = [m["impl"] for m in metrics_list]
    if len(labels) != len(profiles):
        ap.error("number of --labels must match number of profiles")

    print_report(metrics_list)

    os.makedirs(args.outdir, exist_ok=True)
    # metrics.json
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump({lab: m for lab, m in zip(labels, metrics_list)}, f, indent=2)
    # summary.md
    md = ["# Control-loop timing baseline\n", comparison_table_md(metrics_list, labels), ""]
    with open(os.path.join(args.outdir, "summary.md"), "w") as f:
        f.write("\n".join(md))

    if not args.no_plots:
        written = make_plots(profiles, metrics_list, labels, args.outdir)
        for w in written:
            print(f"[plot] {w}")
    print(f"\nWrote metrics.json and summary.md to {args.outdir}/")
    if len(profiles) > 1:
        print("\n" + comparison_table_md(metrics_list, labels))


if __name__ == "__main__":
    main()
