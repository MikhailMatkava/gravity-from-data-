import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

from physics import features, simulate
from predict import predict
from train import run


class PhysicsTests(unittest.TestCase):
    def test_simulation_matches_known_equation_without_noise(self):
        time, measured, ideal = simulate(noise_std=0)
        self.assertEqual(time.shape, (61,))
        self.assertAlmostEqual(time[0], 0)
        self.assertAlmostEqual(time[-1], 1.8)
        np.testing.assert_array_equal(measured, ideal)
        np.testing.assert_allclose(ideal, 1.5 + 8 * time - 4.905 * time**2)

    def test_quadratic_recovers_exact_gravity_without_noise(self):
        time, measured, _ = simulate(noise_std=0)
        model = LinearRegression().fit(features(time[:40], quadratic=True), measured[:40])
        self.assertAlmostEqual(-2 * model.coef_[1], 9.81, places=10)
        self.assertAlmostEqual(model.intercept_, 1.5, places=10)
        self.assertAlmostEqual(model.coef_[0], 8, places=10)

    def test_features_keep_samples_as_rows(self):
        np.testing.assert_array_equal(features([0, 2], quadratic=True), [[0, 0], [2, 4]])
        self.assertEqual(features([0, 2]).shape, (2, 1))

    def test_seed_repeats_noise_and_bad_input_rejected(self):
        np.testing.assert_array_equal(simulate(seed=8)[1], simulate(seed=8)[1])
        with self.assertRaises(ValueError):
            simulate(noise_std=-1)
        for value in [[[1, 2]], [float("nan")], [-1]]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                features(value)

    def test_json_prediction_and_range_flag(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            path.write_text('{"intercept":1.5,"coefficient_t":8,"coefficient_t_squared":-4.905,"last_train_time_seconds":1}')
            self.assertAlmostEqual(predict(2, path)["height_metres"], -2.12)
            self.assertTrue(predict(2, path)["beyond_training_range"])
            self.assertFalse(predict(0.5, path)["beyond_training_range"])
            with self.assertRaises(ValueError):
                predict(-1, path)

    def test_prediction_rejects_non_finite_time(self):
        for time in (float('nan'), float('inf'), float('-inf')):
            with self.subTest(time=time), self.assertRaises(ValueError):
                predict(time)

    def test_noiseless_training_saves_consistent_model(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            result = run(output=output, noise_std=0)
            metrics = json.loads((output / 'metrics.json').read_text())
            model = json.loads((output / 'model.json').read_text())
            self.assertTrue((output / 'trajectory.png').is_file())
            self.assertEqual(metrics, result)
            self.assertEqual((result['n_train'], result['n_test']), (40, 21))
            self.assertAlmostEqual(result['estimated_g_m_s2'], 9.81, places=10)
            self.assertLess(result['quadratic_test_mae_metres'], 1e-10)
            self.assertAlmostEqual(model['coefficient_t_squared'], -4.905, places=10)
            self.assertAlmostEqual(predict(0, output / 'model.json')['height_metres'], 1.5, places=10)
            self.assertTrue(predict(1.2, output / 'model.json')['beyond_training_range'])


if __name__ == "__main__":
    unittest.main()
