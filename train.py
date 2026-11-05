"""Fit and compare linear and quadratic trajectory models."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from physics import features, simulate


def run(output="artifacts", seed=42, noise_std=0.08):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    time, measured, true_height = simulate(seed=seed, noise_std=noise_std)

    # Test on later times to measure extrapolation.
    split = 40
    early, later = slice(0, split), slice(split, None)
    straight = LinearRegression().fit(features(time[early]), measured[early])
    curved = LinearRegression().fit(features(time[early], quadratic=True), measured[early])
    linear_pred = straight.predict(features(time[later]))
    physics_pred = curved.predict(features(time[later], quadratic=True))

    # c = -g/2 in h(t) = a + b*t + c*t².
    initial_height = float(curved.intercept_)
    initial_velocity = float(curved.coef_[0])
    estimated_g = float(-2 * curved.coef_[1])
    result = {
        "data": "one synthetic trajectory, metres and seconds",
        "assumptions": "constant g; no air resistance",
        "seed": seed, "noise_std_metres": noise_std,
        "n_train": split, "n_test": len(time) - split,
        "last_train_time_seconds": float(time[split - 1]),
        "first_test_time_seconds": float(time[split]),
        "linear_test_mae_metres": float(mean_absolute_error(measured[later], linear_pred)),
        "quadratic_test_mae_metres": float(mean_absolute_error(measured[later], physics_pred)),
        "quadratic_vs_true_test_mae_metres": float(mean_absolute_error(true_height[later], physics_pred)),
        "estimated_initial_height_metres": initial_height,
        "estimated_initial_velocity_m_s": initial_velocity,
        "estimated_g_m_s2": estimated_g,
        "simulated_g_m_s2": 9.81,
        "sklearn_version": sklearn.__version__,
    }
    (output / "model.json").write_text(json.dumps({
        "intercept": initial_height,
        "coefficient_t": initial_velocity,
        "coefficient_t_squared": float(curved.coef_[1]),
        "last_train_time_seconds": float(time[split - 1]),
    }, indent=2) + "\n", encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    grid = np.linspace(0, float(time[-1]), 200)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.scatter(time[early], measured[early], label="train (noisy)", s=18)
    ax.scatter(time[later], measured[later], label="test (noisy)", s=18)
    ax.plot(grid, 1.5 + 8 * grid - 0.5 * 9.81 * grid**2, label="true simulated path")
    ax.plot(grid, straight.predict(features(grid)), label="linear fit")
    ax.plot(grid, curved.predict(features(grid, quadratic=True)), label="quadratic fit")
    ax.axvline(float(time[split - 1]), color="gray", linestyle="--", linewidth=1)
    ax.set(xlabel="Time, s", ylabel="Height, m", title="Projectile trajectory: later times held out")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "trajectory.png", dpi=150)
    plt.close(fig)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-std", type=float, default=0.08, help="Measurement noise in metres")
    args = parser.parse_args()
    run(args.output, args.seed, args.noise_std)
