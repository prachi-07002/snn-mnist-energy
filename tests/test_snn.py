"""Unit tests for snn-mnist-energy.

Run:
    python3 -m pytest tests/ -v
or:
    python3 -m unittest discover -s tests -v
"""

import unittest

import numpy as np

from snn_mnist_energy import (
    make_dataset, render_digit, rate_encode, LIFLayer,
    ANNClassifier, SNNClassifier,
)
from snn_mnist_energy.energy import snn_energy_pj, ann_energy_pj


class TestDigits(unittest.TestCase):
    def test_dataset_shapes(self):
        X_train, y_train, X_test, y_test = make_dataset(
            n_train_per_class=10, n_test_per_class=5, seed=0)
        self.assertEqual(X_train.shape, (100, 256))
        self.assertEqual(y_train.shape, (100,))
        self.assertEqual(X_test.shape, (50, 256))
        self.assertEqual(y_test.shape, (50,))
        self.assertTrue(((X_train >= 0) & (X_train <= 1)).all())
        self.assertEqual(set(y_train.tolist()), set(range(10)))

    def test_digits_differ(self):
        rng = np.random.default_rng(0)
        imgs = [render_digit(d, rng) for d in range(10)]
        # No two digits render identically.
        for i in range(10):
            for j in range(i + 1, 10):
                self.assertFalse(np.array_equal(imgs[i], imgs[j]))


class TestEncoding(unittest.TestCase):
    def test_brighter_pixels_spike_more(self):
        """A bright pixel must fire far more often than a dark one."""
        rng = np.random.default_rng(0)
        bright = np.ones((1, 4))
        dark = np.full((1, 4), 0.05)
        spikes_bright = rate_encode(bright, n_steps=200, max_rate_hz=100.0, rng=rng)
        spikes_dark = rate_encode(dark, n_steps=200, max_rate_hz=100.0, rng=rng)
        self.assertGreater(spikes_bright.sum(), 5 * spikes_dark.sum())
        self.assertGreater(spikes_bright.mean(), 0.05)

    def test_zero_input_never_spikes(self):
        rng = np.random.default_rng(0)
        spikes = rate_encode(np.zeros((2, 8)), n_steps=50, rng=rng)
        self.assertEqual(int(spikes.sum()), 0)

    def test_encoding_shape(self):
        rng = np.random.default_rng(0)
        spikes = rate_encode(np.random.default_rng(1).random((3, 256)),
                             n_steps=50, rng=rng)
        self.assertEqual(spikes.shape, (3, 50, 256))
        self.assertEqual(spikes.dtype, bool)


class TestLIF(unittest.TestCase):
    def test_no_input_no_spikes(self):
        layer = LIFLayer(n_in=4, n_out=3, seed=0)
        counts, _ = layer.run(np.zeros((2, 20, 4), dtype=bool))
        np.testing.assert_array_equal(counts, np.zeros((2, 3), dtype=np.int64))

    def test_strong_constant_input_spikes(self):
        # Every input spikes every step with positive weights: the
        # membrane must cross threshold and the neuron must fire.
        layer = LIFLayer(n_in=4, n_out=3, v_th=0.5, seed=0)
        layer.W = np.full((4, 3), 0.5)  # each step I = 2.0 > v_th
        counts, _ = layer.run(np.ones((1, 20, 4), dtype=bool))
        self.assertTrue((counts > 0).all())

    def test_leak_decays_membrane(self):
        # A single input spike, then silence: voltage must decay and
        # the neuron must not keep firing.
        layer = LIFLayer(n_in=1, n_out=1, tau_m_ms=5.0, v_th=10.0, seed=0)
        train = np.zeros((1, 50, 1), dtype=bool)
        train[0, 0, 0] = True
        counts, voltages = layer.run_single(train[0])
        self.assertEqual(int(counts[0]), 0)
        self.assertLess(voltages[-1, 0], voltages[1, 0])

    def test_reset_after_spike(self):
        # After firing, the membrane is reset to v_reset (0.0).
        layer = LIFLayer(n_in=1, n_out=1, v_th=0.1, v_reset=0.0, seed=0)
        train = np.ones((10, 1), dtype=bool)
        _, voltages = layer.run_single(train)
        # Voltages right after a spike step are at reset (0) or below
        # threshold; no runaway accumulation.
        self.assertTrue((voltages < layer.v_th).all())

    def test_counts_shape(self):
        layer = LIFLayer(n_in=8, n_out=5, seed=0)
        rng = np.random.default_rng(0)
        counts, out = layer.run(rng.random((2, 10, 8)) < 0.5)
        self.assertEqual(counts.shape, (2, 5))
        self.assertEqual(out.shape, (2, 10, 5))


class TestClassifiers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X_train, cls.y_train, cls.X_test, cls.y_test = make_dataset(
            n_train_per_class=40, n_test_per_class=20, seed=0)

    def test_ann_beats_chance(self):
        ann = ANNClassifier(seed=0)
        ann.fit(self.X_train, self.y_train, epochs=200, lr=0.5)
        acc = ann.accuracy(self.X_test, self.y_test)
        self.assertGreater(acc, 0.50, f"ANN accuracy {acc:.2f} too low")

    def test_snn_beats_chance(self):
        snn = SNNClassifier(n_hidden=256, n_steps=50, max_rate_hz=200.0,
                            tau_m_ms=10.0, weight_scale=4.0, seed=0)
        snn.fit(self.X_train, self.y_train, epochs=1000, lr=1.0, seed=0)
        acc = snn.accuracy(self.X_test, self.y_test, seed=1)
        self.assertGreater(acc, 0.50, f"SNN accuracy {acc:.2f} too low")

    def test_snn_produces_spikes(self):
        snn = SNNClassifier(n_hidden=256, n_steps=50, max_rate_hz=200.0,
                            tau_m_ms=10.0, weight_scale=4.0, seed=0)
        mean_spikes = snn.mean_spikes_per_sample(self.X_test[:20], seed=1)
        self.assertGreater(mean_spikes, 10)

    def test_predict_shapes(self):
        ann = ANNClassifier(seed=0)
        ann.fit(self.X_train, self.y_train, epochs=5, lr=0.5)
        preds = ann.predict(self.X_test)
        self.assertEqual(preds.shape, (len(self.X_test),))
        self.assertTrue(((preds >= 0) & (preds <= 9)).all())


class TestEnergy(unittest.TestCase):
    def test_energy_increases_with_spikes(self):
        e_low = snn_energy_pj(100, 50, fanout_in=32, fanout_out=10,
                              readout_mac=320)
        e_high = snn_energy_pj(500, 200, fanout_in=32, fanout_out=10,
                               readout_mac=320)
        self.assertGreater(e_high, e_low)

    def test_energy_nonnegative_and_scales(self):
        e = snn_energy_pj(0, 0, fanout_in=32, fanout_out=10, readout_mac=0)
        self.assertEqual(e, 0.0)
        e2 = snn_energy_pj(10, 0, fanout_in=1, fanout_out=1, readout_mac=0)
        self.assertGreater(e2, 0.0)
        # ANN energy scales linearly with MAC count.
        self.assertAlmostEqual(ann_energy_pj(2000), 2 * ann_energy_pj(1000))


if __name__ == "__main__":
    unittest.main()
