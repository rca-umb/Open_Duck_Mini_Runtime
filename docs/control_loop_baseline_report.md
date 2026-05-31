# Control-loop performance baseline

**Goal.** Establish a clean, trustworthy performance baseline of the current
Python control loop *before* any C++ rewrite, so the rewrite is justified by data
and we have before/after numbers. **This is measurement only — no C++ here.**

**Hypothesis (to test, not assume).** Python's interpreter overhead, the GIL, and
GC pauses add *jitter* — nondeterministic per-iteration timing — which hurts a
balance controller that assumes a constant `dt`. C++ would give deterministic,
low-latency execution. But we don't yet know where the time and jitter come from
(serial comms vs. inference vs. Python glue), so the rewrite target is unknown
until we profile. In particular **do not assume NumPy inference is the
bottleneck** — its matmuls already call compiled C, so the likely C++ win is
removing interpreter overhead and timing nondeterminism, not the math.

---

## 1. What was instrumented

The real-time loop is `RLWalk.run()` in
[`scripts/v2_rl_walk_mujoco.py`](../scripts/v2_rl_walk_mujoco.py). Each iteration:

| section | code | what it costs |
|---|---|---|
| (a) read state / sensors | `get_obs()` | half-duplex **serial read of 14 servo positions** (dominant), IMU (non-blocking queue read — the I²C read runs in a background thread), feet-contact GPIO, `np.concatenate` |
| (b) policy inference | `policy.infer(obs)` | NumPy MLP forward pass (`NumpyInfer`) |
| (c) serial write | `hwi.set_position_all()` | one `write_goal_position` to all 14 servos over the bus |

The instrumentation lives in
[`mini_bdx_runtime/loop_profiler.py`](../mini_bdx_runtime/mini_bdx_runtime/loop_profiler.py)
(`LoopProfiler`) and is wired into the loop with `begin / lap / skip / end`
calls. Python "glue" between sections (imitation-phase math, action bookkeeping,
`make_action_dict`) is excluded from the three sections via `skip()` and recovered
offline as `glue = busy − (read + infer + write)`.

> **Note on the inference backend.** The committed loop hard-coded `OnnxInfer`,
> but the robot runs the NumPy reimplementation (no working ARMv6 ONNX Runtime).
> The loop now auto-selects the backend from the model file extension
> (`.npz → NumpyInfer`, `.onnx → OnnxInfer`), so profiling targets the
> configuration that actually runs.

### Why it doesn't corrupt the measurement

- `time.perf_counter_ns()` only — monotonic, integer nanoseconds.
- All metrics are written by index into **preallocated** NumPy arrays. No
  `list.append`, no allocation, no formatting in the hot path.
- **No `print`/logging inside the loop.** The buffer is dumped to disk once, after
  the run (in a `finally`).
- When profiling is off (default), every `prof.*` call is a single attribute load
  + branch + `return` — constant cost, identical every iteration. Verified:
  disabled calls are no-ops; normal operation is unaffected.

---

## 2. How to run (on the robot)

Profiling is **off by default**. Enable it with environment variables; nothing
about normal operation changes when `BDX_PROFILE` is unset.

```bash
# Baseline: 50 Hz target, record 30k iterations (~10 min), GC on, trace GC events.
# Auto-stops and saves when the buffer fills.
BDX_PROFILE=1 \
BDX_PROFILE_ITERS=30000 \
BDX_PROFILE_IMPL=python-numpy \
BDX_PROFILE_OUT=baseline_gc_on.npz \
  python scripts/keyboard_walk.py --control_freq 50
# (or scripts/v2_rl_walk_mujoco.py --onnx_model_path ~/BEST_WALK.npz ...)
```

**Experiment 2 — GC on vs. off** (isolates GC as a jitter source):

```bash
# Same run, but disable the garbage collector.
BDX_PROFILE=1 BDX_PROFILE_ITERS=30000 \
BDX_PROFILE_GC_DISABLE=1 \
BDX_PROFILE_OUT=baseline_gc_off.npz \
  python scripts/keyboard_walk.py --control_freq 50
```

**Optional — SCHED_FIFO** (informational; single-core Pi can't be pinned):

```bash
sudo -E BDX_PROFILE=1 BDX_PROFILE_ITERS=30000 \
BDX_PROFILE_SCHED_FIFO=80 BDX_PROFILE_OUT=baseline_fifo.npz \
  python scripts/keyboard_walk.py --control_freq 50
```

### All env knobs

| var | default | meaning |
|---|---|---|
| `BDX_PROFILE` | `0` | `1` to enable profiling |
| `BDX_PROFILE_ITERS` | `60000` | iterations to record (buffer capacity) |
| `BDX_PROFILE_OUT` | auto-named | output `.npz` path |
| `BDX_PROFILE_GC_DISABLE` | `0` | `1` → `gc.disable()` for the run |
| `BDX_PROFILE_NO_GC_TRACE` | `0` | `1` → don't register the `gc.callbacks` hook |
| `BDX_PROFILE_SCHED_FIFO` | `0` | priority int → attempt `SCHED_FIFO` (needs root) |
| `BDX_PROFILE_IMPL` | `python-numpy` | label stored in the file (set to `cpp` later) |
| `BDX_PROFILE_NOTES` | `""` | free-text note saved into the file |
| `BDX_PROFILE_NO_STOP` | `0` | `1` → keep running after the buffer fills |

> The instrumented modules must be importable on the Pi. If `mini_bdx_runtime`
> was installed non-editable, reinstall (`pip install -e .`) so `loop_profiler.py`
> is picked up.
>
> The loop's pre-existing "Policy control budget exceeded" `print` fires on
> overruns and can itself perturb the tail. For the cleanest tail measurement,
> comment it out during a profiling run.

## 3. How to analyze (on a desktop)

Copy the `.npz` files off the Pi and run
[`scripts/analyze_loop_profile.py`](../scripts/analyze_loop_profile.py) (needs
`numpy`, plus `matplotlib` for plots):

```bash
# Single run -> full report + plots
python scripts/analyze_loop_profile.py baseline_gc_on.npz --outdir report/

# GC on vs off -> comparison table + overlaid histograms
python scripts/analyze_loop_profile.py baseline_gc_on.npz baseline_gc_off.npz \
    --labels gc_on,gc_off --outdir report/
```

It prints a metrics report, writes `report/metrics.json` and `report/summary.md`,
and saves three plots: `period_histogram.png`, `latency_timeseries.png` (with
GC-coinciding iterations marked in red), and `section_breakdown.png`.

---

## 4. Metrics (definitions)

- **Loop period** — `iter_start[i] − iter_start[i-1]` (start-to-start; includes the
  trailing `sleep`). Mean → **achieved frequency**.
- **Jitter** — std of the loop period (headline). Also reported: RMS deviation
  from the target period.
- **Tail latency** — p50 / p99 / p99.9 / max of per-iteration **compute** (`busy` =
  everything except the sleep).
- **Per-section** — mean / p99 / max of read, infer, write, and glue.
- **Overruns** — fraction of iterations whose `busy` exceeds the target period;
  informs the **max stable frequency** (see §6).

---

## 5. Results

> Fill these in from the hardware run (`report/summary.md` and the printed report
> drop straight in here).

### 5.1 Baseline (GC on)

| metric | value |
|---|---|
| iterations / duration | _TODO_ |
| achieved frequency | _TODO_ Hz |
| **jitter (std of period)** | _TODO_ µs |
| period p50 / p99 / p99.9 / max | _TODO_ µs |
| busy mean | _TODO_ µs |
| busy p50 / p99 / p99.9 / max | _TODO_ µs |
| overrun fraction | _TODO_ % |
| read mean / p99 / max | _TODO_ µs |
| infer mean / p99 / max | _TODO_ µs |
| write mean / p99 / max | _TODO_ µs |
| glue mean / p99 / max | _TODO_ µs |

Plots: `report/period_histogram.png`, `report/latency_timeseries.png`,
`report/section_breakdown.png`.

### 5.2 GC on vs. off

| metric | GC on | GC off |
|---|---|---|
| jitter (std) | _TODO_ | _TODO_ |
| period max | _TODO_ | _TODO_ |
| busy p99.9 / max | _TODO_ | _TODO_ |
| overrun % | _TODO_ | _TODO_ |

**GC correlation:** of the worst 0.1% slowest iterations, _TODO_% coincided with a
garbage-collection event. (If this is high and jitter drops with GC off, that is a
clean, citable result and direct evidence for one thing the C++ rewrite removes
for free.)

> **Illustrative format only — synthetic numbers, not the robot.** Generated with
> `scripts/simulate_loop_profile.py` to show what filled-in results look like:
> jitter 125 µs (GC on) → 43 µs (GC off); period max 25.5 ms → 20.3 ms; **100% of
> the worst-0.1% iterations coincided with GC**; per-section means read ≈ 6.5 ms,
> write ≈ 3.2 ms, infer ≈ 0.45 ms, glue ≈ 0.3 ms — i.e. the **serial bus
> dominates and inference is a sliver**. The real Pi-Zero-W numbers replace these.

---

## 6. Max stable frequency

The highest target rate the loop hits without consistently overrunning. Two ways
to read it from the data:

1. **From a single run:** the `busy` p99 implies a ceiling — the analyzer prints
   `max_stable_hz_p99_estimate = 1e9 / busy_p99_ns`. If `busy` p99 ≈ 11 ms, the
   loop can't reliably sustain much above ~90 Hz regardless of language.
2. **By sweeping** (authoritative): run the loop at increasing `--control_freq`
   (e.g. 40, 50, 60, 80, 100 Hz) and pick the highest rate whose **overrun
   fraction stays near zero**.

```bash
for hz in 40 50 60 80 100; do
  BDX_PROFILE=1 BDX_PROFILE_ITERS=10000 \
  BDX_PROFILE_OUT=sweep_${hz}hz.npz \
    python scripts/keyboard_walk.py --control_freq $hz
done
python scripts/analyze_loop_profile.py sweep_*.npz \
    --labels 40,50,60,80,100 --outdir sweep_report/
```

Record the result: **max stable frequency = _TODO_ Hz** (highest rate with
overrun ≈ 0).

---

## 7. Platform caveats

- **Single core (ARMv6).** We can't pin the loop to a dedicated core, so some tail
  latency is the OS scheduler preempting the process. Treat that as part of the
  baseline, not a bug. `SCHED_FIFO` (§2) may shift the tail; report it if measured.
- The Pi Zero W is weaker than the Zero 2 W the upstream project targets, so
  timing margins are tight by design.

---

## 8. Enabling the C++ before/after comparison

The on-disk schema (documented in `loop_profiler.py`) is the contract. The C++
inner loop only has to emit the **same column names** — as `.npz` or CSV
(`iter_start_ns, busy_ns, read_ns, infer_ns, write_ns`, plus optional
`gc_event_*`) — and the *same* `analyze_loop_profile.py` compares them:

```bash
python scripts/analyze_loop_profile.py python-numpy.npz cpp.npz \
    --labels python,cpp --outdir before_after/
```

The headline we're aiming for: *"moved the inner control loop from Python to C++
and cut loop jitter from X µs to Y µs, raising the stable control frequency from
W Hz to Z Hz."* This harness is built so that sentence is a one-command diff.
