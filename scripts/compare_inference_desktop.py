#!/usr/bin/env python3
"""
Runs on your Windows desktop (normal x86 onnxruntime).
Loads pi_inference.json, re-runs the SAME ONNX model on the same observations,
and compares the desktop actions to the Pi actions element-wise.

Decision rule:
- If max abs diff is ~0 (<1e-5), ARMv6 ONNX Runtime is producing identical
  outputs — the policy itself is fine and the problem is in the observations
  the Pi is feeding it (sensors, calibration, IMU).
- If max abs diff is non-trivial (>1e-3), the ARMv6 build is producing
  different numerical results — time to rebuild or replace it.

Usage:
    python scripts/compare_inference_desktop.py \
        --pi pi_inference.json \
        --model path/to/BEST_WALK_ONNX.onnx
"""
import argparse
import json

import numpy as np
import onnxruntime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pi", default="pi_inference.json")
    parser.add_argument("--model", required=True, help="path to the SAME BEST_WALK_ONNX.onnx used on the Pi")
    args = parser.parse_args()

    with open(args.pi, "r") as f:
        payload = json.load(f)

    records = payload["records"]
    print(f"Loaded {len(records)} Pi (obs, action) pairs")
    print(f"Pi was using model: {payload.get('model_path')}")
    print(f"Desktop is using model: {args.model}")
    print()

    sess_options = onnxruntime.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    session = onnxruntime.InferenceSession(args.model, sess_options=sess_options)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    diffs = []
    for i, rec in enumerate(records):
        obs = np.asarray(rec["obs"], dtype=np.float32)
        pi_action = np.asarray(rec["action"], dtype=np.float32)

        desktop_action = session.run([output_name], {input_name: obs[None, :]})[0]
        desktop_action = np.asarray(desktop_action, dtype=np.float32).reshape(-1)

        diff = desktop_action - pi_action
        diffs.append(diff)

        if i < 3:
            print(f"obs[{i}] first 5 values: {obs[:5]}")
            print(f"  Pi action     : {pi_action}")
            print(f"  Desktop action: {desktop_action}")
            print(f"  Diff (max abs): {np.max(np.abs(diff)):.6e}")
            print()

    diffs = np.array(diffs)
    abs_diffs = np.abs(diffs)
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Mean abs diff:  {abs_diffs.mean():.6e}")
    print(f"Max abs diff:   {abs_diffs.max():.6e}")
    print(f"Median abs diff:{np.median(abs_diffs):.6e}")
    print()

    max_abs = abs_diffs.max()
    if max_abs < 1e-5:
        print("VERDICT: ARMv6 onnxruntime is producing IDENTICAL outputs to desktop.")
        print("  --> The ONNX build is not the bug.")
        print("  --> The problem is in the OBSERVATIONS the Pi is feeding the policy")
        print("      (sensor calibration, IMU axis/bias, joint offsets, feet contacts).")
    elif max_abs < 1e-3:
        print("VERDICT: Tiny numerical differences (expected FP-level rounding).")
        print("  --> ONNX build is fine. Focus on observations.")
    else:
        print("VERDICT: NON-TRIVIAL differences between Pi and desktop inference.")
        print("  --> ARMv6 ONNX Runtime build is likely producing wrong outputs.")
        print("  --> Worst joints:")
        per_joint_max = abs_diffs.max(axis=0)
        for j, d in enumerate(per_joint_max):
            print(f"     joint {j:2d}: max abs diff {d:.4e}")


if __name__ == "__main__":
    main()
