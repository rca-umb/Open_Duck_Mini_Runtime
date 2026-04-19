#!/usr/bin/env python3
"""
Runs on the Pi. Loads robot_saved_obs.pkl, runs ONNX inference on each obs
locally (via the ARMv6 onnxruntime build), and writes obs+action pairs to
pi_inference.json — a plain-text file easy to transfer to a desktop.

Usage:
    python3 scripts/dump_inference_pi.py \
        --obs robot_saved_obs.pkl \
        --model ~/BEST_WALK_ONNX.onnx \
        --n 50 \
        --out pi_inference.json
"""
import argparse
import json
import os
import pickle

import numpy as np
from mini_bdx_runtime.onnx_infer import OnnxInfer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs", default="robot_saved_obs.pkl")
    parser.add_argument("--model", default=os.path.expanduser("~/BEST_WALK_ONNX.onnx"))
    parser.add_argument("--n", type=int, default=50, help="how many obs to dump")
    parser.add_argument("--out", default="pi_inference.json")
    args = parser.parse_args()

    with open(args.obs, "rb") as f:
        saved = pickle.load(f)

    print(f"Loaded {len(saved)} observations from {args.obs}")
    print(f"First obs dtype: {np.asarray(saved[0]).dtype}, shape: {np.asarray(saved[0]).shape}")

    policy = OnnxInfer(args.model, awd=True)

    n = min(args.n, len(saved))
    records = []
    for i in range(n):
        obs = np.asarray(saved[i], dtype=np.float32)
        action = policy.infer(obs)
        records.append({
            "obs": obs.tolist(),
            "action": np.asarray(action, dtype=np.float32).tolist(),
        })

    payload = {
        "model_path": args.model,
        "n": n,
        "records": records,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f)

    print(f"Wrote {n} (obs, action) pairs to {args.out}")
    print(f"File size: {os.path.getsize(args.out)} bytes")
    print()
    print("Transfer options if scp is broken:")
    print(f"  cat {args.out}")
    print("  # copy terminal output, paste into a file on your desktop")
