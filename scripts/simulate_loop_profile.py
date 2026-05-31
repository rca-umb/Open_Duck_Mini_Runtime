#!/usr/bin/env python3
"""
Generate synthetic control-loop profiles for testing the analysis pipeline
*without* the robot, and for previewing what the baseline report will look like.

Two modes:

  fabricate  (default) -- write a realistic synthetic .npz in the LoopProfiler
             schema: ~50 Hz loop dominated by the half-duplex serial read/write,
             gaussian jitter, plus injected tail spikes and (optionally) GC
             collections that coincide with some of those spikes. Use this to
             sanity-check analyze_loop_profile.py and the plots.

  drive      -- actually exercise the real LoopProfiler hot-path methods
             (begin/lap/skip/end + gc.callbacks + save) at full speed with no
             real sleeps. Confirms the harness records and round-trips
             correctly; timings are tiny/synthetic, not realistic.

This file is a test/demo aid -- the real numbers come from running the
instrumented robot (see docs/control_loop_baseline_report.md).
"""
import argparse

import numpy as np

US = 1_000  # ns per µs


def fabricate(n=30000, target_hz=50.0, gc_enabled=True, seed=0):
    """Fabricate a realistic per-iteration dataset (all times in ns)."""
    rng = np.random.default_rng(seed)
    target_ns = int(round(1e9 / target_hz))

    # Per-section nominal costs (µs) -- serial bus dominates on a Pi Zero.
    read = np.clip(rng.normal(6500, 600, n), 3000, None) * US   # 14-servo serial read
    infer = np.clip(rng.normal(450, 60, n), 150, None) * US     # NumPy MLP forward
    write = np.clip(rng.normal(3200, 350, n), 1500, None) * US  # serial write
    glue = np.clip(rng.normal(250, 50, n), 50, None) * US       # Python bookkeeping
    busy = (read + infer + write + glue).astype(np.int64)

    # Always-present scheduler/OS jitter spikes (single-core Pi can't escape these).
    n_sched = max(1, int(n * 0.002))
    sched_idx = rng.choice(n, size=n_sched, replace=False)
    busy[sched_idx] += rng.integers(1500, 6000, n_sched) * US

    # GC-correlated spikes (only when GC is enabled).
    gc_intervals = []  # (iter_index, dur_ns)
    if gc_enabled:
        duration_s = n / target_hz
        n_gc = int(duration_s * 1.5)  # ~1.5 collections / second
        gc_idx = rng.choice(n, size=min(n_gc, n), replace=False)
        # Most collections are cheap gen0; a few are expensive gen2.
        durs = rng.choice(
            [800, 1500, 3000, 8000, 14000],
            size=gc_idx.size,
            p=[0.45, 0.30, 0.15, 0.07, 0.03],
        ) * US
        for k, dur in zip(gc_idx, durs):
            busy[k] += int(dur)
            gc_intervals.append((int(k), int(dur)))

    # Period = sleep-to-target, except when compute overran the budget.
    sleep_jitter = np.abs(rng.normal(40, 60, n)).astype(np.int64) * US
    period = np.maximum(target_ns, busy) + sleep_jitter
    iter_start = np.empty(n, np.int64)
    iter_start[0] = 1_000_000_000  # arbitrary perf_counter origin
    iter_start[1:] = iter_start[0] + np.cumsum(period[:-1])

    # Build GC events positioned *inside* the iteration window they affected.
    gc_ns, gc_phase, gc_gen = [], [], []
    for k, dur in sorted(gc_intervals):
        start = iter_start[k] + (busy[k] - dur) // 2
        gen = 2 if dur >= 8000 * US else (1 if dur >= 3000 * US else 0)
        gc_ns += [start, start + dur]
        gc_phase += [0, 1]
        gc_gen += [gen, gen]

    return {
        "schema_version": np.int64(1),
        "iter_start_ns": iter_start,
        "busy_ns": busy,
        "read_ns": read.astype(np.int64),
        "infer_ns": infer.astype(np.int64),
        "write_ns": write.astype(np.int64),
        "gc_event_ns": np.array(gc_ns, np.int64),
        "gc_event_phase": np.array(gc_phase, np.int8),
        "gc_event_gen": np.array(gc_gen, np.int8),
        "impl": np.array("python-numpy-SIM"),
        "target_freq_hz": np.float64(target_hz),
        "target_period_ns": np.int64(target_ns),
        "gc_enabled": np.int64(1 if gc_enabled else 0),
        "sched_fifo": np.int64(0),
        "n_iters": np.int64(n),
        "n_gc_events": np.int64(len(gc_ns)),
        "t_unix_start": np.float64(0.0),
        "platform": np.array("synthetic"),
        "hostname": np.array("sim"),
        "notes": np.array("fabricated by simulate_loop_profile.py"),
        "section_names": np.array(["read", "infer", "write"]),
    }


def drive(n=20000, target_hz=50.0, gc_disable=False, out="drive_profile.npz"):
    """Exercise the real LoopProfiler at full speed (no real sleeps)."""
    import gc as _gc
    from mini_bdx_runtime.loop_profiler import (
        LoopProfiler, SEC_READ, SEC_INFER, SEC_WRITE,
    )

    prof = LoopProfiler(
        enabled=True, capacity=n, target_freq_hz=target_hz,
        gc_disable=gc_disable, impl="python-numpy-DRIVE", out_path=out,
    )
    junk = []
    for i in range(n):
        prof.begin()
        _busy_ns(200_000)              # ~0.2 ms "read"
        prof.lap(SEC_READ)
        junk.append([0] * 50)          # create garbage to provoke real GC
        if len(junk) > 2000:
            junk.clear()
        prof.skip()
        _busy_ns(50_000)               # ~0.05 ms "infer"
        prof.lap(SEC_INFER)
        prof.skip()
        _busy_ns(120_000)              # ~0.12 ms "write"
        prof.lap(SEC_WRITE)
        prof.end()
        if prof.done:
            break
    path = prof.save()
    prof.close()
    print(f"[drive] recorded {prof.n_recorded} iters, {prof._gc_idx} gc events -> {path}")
    return path


def _busy_ns(ns):
    import time
    end = time.perf_counter_ns() + ns
    while time.perf_counter_ns() < end:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["fabricate", "drive"], default="fabricate")
    ap.add_argument("--out", default="sim_profile.npz")
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--target-hz", type=float, default=50.0)
    ap.add_argument("--gc-disable", action="store_true",
                    help="fabricate/drive a GC-disabled run")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.mode == "drive":
        drive(args.n, args.target_hz, args.gc_disable, args.out)
        return

    data = fabricate(args.n, args.target_hz, gc_enabled=not args.gc_disable, seed=args.seed)
    with open(args.out, "wb") as f:
        np.savez(f, **data)
    print(f"[fabricate] wrote {args.n} iters (gc={'off' if args.gc_disable else 'on'}) -> {args.out}")


if __name__ == "__main__":
    main()
