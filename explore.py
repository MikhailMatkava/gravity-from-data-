"""Check how the gravity estimate varies with simulated measurement noise."""
import argparse
import json

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from physics import features, simulate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, default=[7, 21, 42])
    parser.add_argument('--noise-std', type=float, default=0.08)
    args = parser.parse_args()

    results = []
    for seed in args.seeds:
        time, measured, _ = simulate(seed=seed, noise_std=args.noise_std)
        model = LinearRegression().fit(features(time[:40], quadratic=True), measured[:40])
        predicted = model.predict(features(time[40:], quadratic=True))
        results.append({
            'seed': seed,
            'estimated_g_m_s2': round(float(-2 * model.coef_[1]), 4),
            'later_mae_metres': round(float(mean_absolute_error(measured[40:], predicted)), 4),
        })
    print(json.dumps({'noise_std_metres': args.noise_std, 'results': results}, indent=2))


if __name__ == '__main__':
    main()
