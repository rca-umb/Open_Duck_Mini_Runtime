"""
Non-invasive timing harness for the real-time control loop.

Goal: establish a trustworthy performance baseline of the Python control loop
(period, jitter, tail latency, per-section breakdown) *without* the measurement
itself introducing the jitter we are trying to measure. This is the prerequisite
for deciding whether/where a C++ rewrite of the inner loop is justified.

Design rules that keep the instrumentation honest:

  * ``time.perf_counter_ns()`` only -- monotonic, integer nanoseconds, no float
    rounding, not affected by wall-clock adjustments.
  * Everything is written into **preallocated** fixed-size numpy arrays by index.
    There is **no** ``list.append`` and **no** allocation in the hot path, so we
    don't trigger reallocations / GC churn mid-measurement.
  * **No** ``print`` / logging inside the loop. We buffer in RAM and dump to disk
    once, after the run.
  * When profiling is disabled, every hot-path method is a single attribute load
    plus a branch and an immediate ``return`` -- constant, negligible, and (most
    importantly) the *same* on every iteration.

On-disk schema (``.npz``)
-------------------------
The saved file is the contract that makes a later Python-vs-C++ comparison
trivial: the C++ version only has to emit the same column names (as ``.npz`` or
CSV) and ``analyze_loop_profile.py`` will compare them apples-to-apples.

Per-iteration arrays (all int64 nanoseconds, length = number of recorded iters):

  iter_start_ns   perf_counter_ns at the top of each iteration.
                  period[i] = iter_start_ns[i] - iter_start_ns[i-1]
  busy_ns         compute time per iteration (top of loop -> last section end),
                  i.e. everything except the trailing sleep.
  read_ns         section (a): read state / sensors  (get_obs)
  infer_ns        section (b): policy inference
  write_ns        section (c): serial write to the 14 servos
                  glue_ns (Python bookkeeping) is recoverable offline as
                  busy_ns - (read_ns + infer_ns + write_ns).

GC event arrays (length = number of gc callbacks fired):

  gc_event_ns     perf_counter_ns when a gc.callbacks hook fired.
  gc_event_phase  0 = "start", 1 = "stop".
  gc_event_gen    generation being collected (-1 if unknown).

Scalar metadata (stored as 0-d arrays):

  schema_version, impl, target_freq_hz, target_period_ns, gc_enabled,
  sched_fifo, n_iters, n_gc_events, t_unix_start, platform, hostname, notes,
  section_names.
"""

import gc
import os
import socket
import platform as _platform
import time

import numpy as np

SCHEMA_VERSION = 1

# Hot-path section indices. Kept as module constants so the call sites in the
# control loop read clearly and we never do a string lookup per iteration.
SEC_READ = 0
SEC_INFER = 1
SEC_WRITE = 2
DEFAULT_SECTION_NAMES = ("read", "infer", "write")

# gc phase encoding
_PHASE_START = 0
_PHASE_STOP = 1


class LoopProfiler:
    """Preallocated, allocation-free per-iteration timing recorder.

    Typical hot-loop usage (no-ops entirely when ``enabled`` is False)::

        prof.begin()                 # top of iteration
        obs = get_obs()
        prof.lap(SEC_READ)           # record (a)
        ...glue (e.g. phase calc)...
        prof.skip()                  # discard glue from the next section
        action = policy.infer(obs)
        prof.lap(SEC_INFER)          # record (b)
        ...glue (action bookkeeping)...
        prof.skip()
        hwi.set_position_all(...)
        prof.lap(SEC_WRITE)          # record (c)
        prof.end()                   # finalize iteration, advance index

        if prof.done:                # buffer full -> already saved to disk
            break
    """

    def __init__(
        self,
        enabled=False,
        capacity=60000,
        target_freq_hz=50.0,
        section_names=DEFAULT_SECTION_NAMES,
        gc_trace=True,
        gc_disable=False,
        gc_capacity=200000,
        sched_fifo_priority=0,
        impl="python",
        out_path=None,
        notes="",
        stop_when_full=True,
    ):
        self.enabled = bool(enabled)
        # ``active`` is the single hot-path flag. It starts equal to enabled and
        # flips to False once the buffer is full so recording freezes cleanly.
        self.active = self.enabled
        self.done = False

        self.capacity = int(capacity)
        self.n_sections = len(section_names)
        self.section_names = tuple(section_names)
        self.target_freq_hz = float(target_freq_hz)
        self.target_period_ns = int(round(1e9 / self.target_freq_hz)) if target_freq_hz else 0
        self.stop_when_full = bool(stop_when_full)
        self.impl = str(impl)
        self.notes = str(notes)
        self.out_path = out_path

        # --- preallocate everything up front -------------------------------
        # int64 nanoseconds. These allocations happen once, before the loop.
        self._iter_start = np.zeros(self.capacity, dtype=np.int64)
        self._busy = np.zeros(self.capacity, dtype=np.int64)
        # one row per section -> shape (n_sections, capacity), indexed [sec][i]
        self._sections = np.zeros((self.n_sections, self.capacity), dtype=np.int64)
        self._idx = 0

        # gc event ring -- preallocated as well.
        self._gc_trace = bool(gc_trace) and self.enabled
        self.gc_capacity = int(gc_capacity)
        self._gc_ns = np.zeros(self.gc_capacity, dtype=np.int64)
        self._gc_phase = np.zeros(self.gc_capacity, dtype=np.int8)
        self._gc_gen = np.full(self.gc_capacity, -1, dtype=np.int8)
        self._gc_idx = 0
        self._gc_cb = None

        # transient per-iteration scratch (plain Python ints, no allocation)
        self._t0 = 0
        self._t_last = 0

        # --- environment knobs we only touch when actually profiling -------
        self.gc_disabled = False
        self.sched_fifo = False
        self.sched_fifo_priority = int(sched_fifo_priority)

        if self.enabled:
            if gc_disable:
                gc.disable()
                self.gc_disabled = True
            if self._gc_trace:
                # Register a lightweight callback. It only fires during GC, and
                # all it does is index-assign into preallocated arrays.
                self._gc_cb = self._on_gc
                gc.callbacks.append(self._gc_cb)
            if self.sched_fifo_priority > 0:
                self.sched_fifo = try_set_sched_fifo(self.sched_fifo_priority)

        self.t_unix_start = time.time()

    @property
    def n_recorded(self):
        """Number of fully recorded iterations so far."""
        return self._idx

    # ---- hot path ---------------------------------------------------------
    # Each of these is intentionally tiny and branch-first so the disabled
    # case costs the same on every iteration.

    def begin(self):
        if not self.active:
            return
        t = time.perf_counter_ns()
        self._t0 = t
        self._t_last = t

    def lap(self, section_idx):
        """Record the time since the last lap/skip/begin into ``section_idx``."""
        if not self.active:
            return
        t = time.perf_counter_ns()
        self._sections[section_idx, self._idx] = t - self._t_last
        self._t_last = t

    def skip(self):
        """Advance the section baseline without recording (drops glue time)."""
        if not self.active:
            return
        self._t_last = time.perf_counter_ns()

    def end(self):
        if not self.active:
            return
        i = self._idx
        self._iter_start[i] = self._t0
        self._busy[i] = self._t_last - self._t0
        i += 1
        self._idx = i
        if i >= self.capacity:
            # Buffer full: freeze recording. The caller checks ``self.done`` to
            # break out of the loop and then saves once (typically in a finally).
            self.active = False
            self.done = True

    # ---- gc callback ------------------------------------------------------
    def _on_gc(self, phase, info):
        j = self._gc_idx
        if j >= self.gc_capacity:
            return
        self._gc_ns[j] = time.perf_counter_ns()
        self._gc_phase[j] = _PHASE_START if phase == "start" else _PHASE_STOP
        if info:
            self._gc_gen[j] = info.get("generation", -1)
        self._gc_idx = j + 1

    # ---- teardown / persistence ------------------------------------------
    def close(self):
        """Deregister the gc callback and re-enable gc if we disabled it."""
        if self._gc_cb is not None and self._gc_cb in gc.callbacks:
            gc.callbacks.remove(self._gc_cb)
            self._gc_cb = None
        if self.gc_disabled:
            gc.enable()
            self.gc_disabled = False

    def save(self, out_path=None):
        """Dump recorded data to ``.npz``. Safe to call once after the run."""
        path = out_path or self.out_path
        if path is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = f"loop_profile_{self.impl}_{int(self.target_freq_hz)}hz_{ts}.npz"
        n = self._idx
        m = self._gc_idx

        arrays = {
            "schema_version": np.int64(SCHEMA_VERSION),
            "iter_start_ns": self._iter_start[:n].copy(),
            "busy_ns": self._busy[:n].copy(),
            "gc_event_ns": self._gc_ns[:m].copy(),
            "gc_event_phase": self._gc_phase[:m].copy(),
            "gc_event_gen": self._gc_gen[:m].copy(),
            # metadata
            "impl": np.array(self.impl),
            "target_freq_hz": np.float64(self.target_freq_hz),
            "target_period_ns": np.int64(self.target_period_ns),
            "gc_enabled": np.int64(0 if self.gc_disabled else 1),
            "sched_fifo": np.int64(1 if self.sched_fifo else 0),
            "n_iters": np.int64(n),
            "n_gc_events": np.int64(m),
            "t_unix_start": np.float64(self.t_unix_start),
            "platform": np.array(_platform.platform()),
            "hostname": np.array(socket.gethostname()),
            "notes": np.array(self.notes),
            "section_names": np.array(list(self.section_names)),
        }
        for s, name in enumerate(self.section_names):
            arrays[f"{name}_ns"] = self._sections[s, :n].copy()

        # Write to a temp file then atomically rename. Passing an open file
        # handle to np.savez stops it from auto-appending ".npz" to our name.
        tmp = f"{path}.tmp"
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)
        os.replace(tmp, path)
        self.out_path = path
        return path


def try_set_sched_fifo(priority):
    """Best-effort SCHED_FIFO real-time scheduling (Linux only).

    Returns True on success. Purely informational for the baseline -- on the
    single-core Pi Zero we cannot pin to a dedicated core, but SCHED_FIFO can
    still change how often the OS scheduler preempts us, which shows up in the
    tail latency. Never raises (e.g. on macOS, or without CAP_SYS_NICE).
    """
    if not hasattr(os, "sched_setscheduler") or not hasattr(os, "SCHED_FIFO"):
        return False
    try:
        param = os.sched_param(priority)
        os.sched_setscheduler(0, os.SCHED_FIFO, param)
        return True
    except (OSError, PermissionError, AttributeError):
        return False


def from_env(target_freq_hz, section_names=DEFAULT_SECTION_NAMES):
    """Construct a LoopProfiler from BDX_PROFILE_* environment variables.

    Off by default: if BDX_PROFILE is unset/0 you get a disabled profiler whose
    hot-path methods are no-ops, so normal operation is unaffected.

    Env vars:
      BDX_PROFILE              "1" to enable (default off)
      BDX_PROFILE_ITERS        capacity / iterations to record (default 60000)
      BDX_PROFILE_OUT          output .npz path (default auto-named)
      BDX_PROFILE_GC_DISABLE   "1" -> gc.disable() for the run (GC on/off experiment)
      BDX_PROFILE_NO_GC_TRACE  "1" -> do not register the gc.callbacks hook
      BDX_PROFILE_SCHED_FIFO   integer priority -> attempt SCHED_FIFO (0 = off)
      BDX_PROFILE_IMPL         label for this implementation (default "python-numpy")
      BDX_PROFILE_NOTES        free-text note saved into the file
      BDX_PROFILE_NO_STOP      "1" -> keep looping after buffer full (default: stop)
    """
    enabled = os.environ.get("BDX_PROFILE", "0") not in ("0", "", "false", "False")
    return LoopProfiler(
        enabled=enabled,
        capacity=int(os.environ.get("BDX_PROFILE_ITERS", "60000")),
        target_freq_hz=target_freq_hz,
        section_names=section_names,
        gc_trace=os.environ.get("BDX_PROFILE_NO_GC_TRACE", "0") in ("0", "", "false", "False"),
        gc_disable=os.environ.get("BDX_PROFILE_GC_DISABLE", "0") not in ("0", "", "false", "False"),
        sched_fifo_priority=int(os.environ.get("BDX_PROFILE_SCHED_FIFO", "0")),
        impl=os.environ.get("BDX_PROFILE_IMPL", "python-numpy"),
        out_path=os.environ.get("BDX_PROFILE_OUT") or None,
        notes=os.environ.get("BDX_PROFILE_NOTES", ""),
        stop_when_full=os.environ.get("BDX_PROFILE_NO_STOP", "0") in ("0", "", "false", "False"),
    )
