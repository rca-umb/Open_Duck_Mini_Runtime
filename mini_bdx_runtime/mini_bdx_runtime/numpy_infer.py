"""
Pure numpy inference for the Open Duck Mini walking policy.

Drop-in replacement for OnnxInfer — sidesteps the broken ARMv6 ONNX Runtime
build by reimplementing the model's forward pass directly. The model is a
4-layer MLP with Swish activations, ~220K params total, so this runs in well
under a millisecond even on a Pi Zero 1.

The model graph (extracted from BEST_WALK_ONNX_2.onnx):

    obs[101]
      -> (obs - norm_mean) * norm_recip       (elementwise normalization)
      -> Gemm(101 -> 512) -> Swish
      -> Gemm(512 -> 256) -> Swish
      -> Gemm(256 -> 128) -> Swish
      -> Gemm(128 -> 28)  -> Split[:14] -> Tanh
      -> action[14]

Weights are loaded from a .npz file produced by scripts/export_onnx_to_npz.py.
"""
import numpy as np


def _swish(x):
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0))))


class NumpyInfer:
    def __init__(self, npz_path: str, awd: bool = True):
        data = np.load(npz_path)
        self.norm_mean = data["norm_mean"].astype(np.float32)
        self.norm_recip = data["norm_recip"].astype(np.float32)
        self.W0 = data["W0"].astype(np.float32)
        self.b0 = data["b0"].astype(np.float32)
        self.W1 = data["W1"].astype(np.float32)
        self.b1 = data["b1"].astype(np.float32)
        self.W2 = data["W2"].astype(np.float32)
        self.b2 = data["b2"].astype(np.float32)
        self.W3 = data["W3"].astype(np.float32)
        self.b3 = data["b3"].astype(np.float32)

    def infer(self, obs):
        x = np.asarray(obs, dtype=np.float32).reshape(-1)
        x = (x - self.norm_mean) * self.norm_recip
        x = _swish(x @ self.W0 + self.b0)
        x = _swish(x @ self.W1 + self.b1)
        x = _swish(x @ self.W2 + self.b2)
        x = x @ self.W3 + self.b3
        mean = x[:14]
        return np.tanh(mean)
