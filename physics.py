"""Simulated trajectory and regression features."""

import numpy as np


def simulate(seed=42, noise_std=0.08):
    """Time (s), noisy height (m), and ideal height (m)."""
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    time = np.linspace(0.0, 1.8, 61)
    true_height = 1.5 + 8.0 * time - 0.5 * 9.81 * time**2
    rng = np.random.default_rng(seed)
    height = true_height + rng.normal(0.0, noise_std, size=len(time))
    return time, height, true_height


def features(time, quadratic=False):
    """Build [t] or [t, t²] features."""
    time = np.asarray(time, dtype=float)
    if time.ndim != 1:
        raise ValueError("time must be a one-dimensional sequence")
    if not np.isfinite(time).all() or (time < 0).any():
        raise ValueError("time must contain finite non-negative values")
    if quadratic:
        return np.column_stack((time, time**2))
    return time.reshape(-1, 1)
