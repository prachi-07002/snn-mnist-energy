"""Classifiers: a plain ANN baseline and the spiking network.

ANN baseline
------------
Multinomial logistic regression (softmax + cross-entropy, full-batch
gradient descent), implemented from scratch in NumPy. It sees the raw
pixel intensities directly.

SNN classifier
--------------
Rate-coded input spikes -> LIF hidden layer (fixed random weights) ->
spike counts over the window -> the same softmax readout, trained on
the spike counts. So the SNN's only information about the image is
what survives spike encoding and LIF dynamics; the comparison with
the ANN measures the cost of going event-based.

Both classifiers share the same `fit`/`predict` interface and report
test accuracy in the demo.
"""

import numpy as np

from .encoding import rate_encode
from .lif import LIFLayer


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class SoftmaxReadout:
    """Multinomial logistic regression trained by gradient descent."""

    def __init__(self, n_in, n_classes=10, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0.0, 0.01, size=(n_in, n_classes))
        self.b = np.zeros(n_classes)
        self.n_classes = n_classes

    def fit(self, X, y, epochs=300, lr=0.5, verbose=False):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        n = X.shape[0]
        one_hot = np.zeros((n, self.n_classes))
        one_hot[np.arange(n), y] = 1.0
        for ep in range(epochs):
            probs = _softmax(X @ self.W + self.b)
            grad = (probs - one_hot) / n
            self.W -= lr * (X.T @ grad)
            self.b -= lr * grad.sum(axis=0)
            if verbose and (ep % 50 == 0 or ep == epochs - 1):
                acc = (probs.argmax(axis=1) == y).mean()
                print(f"  epoch {ep:4d}  train acc {acc:.3f}")
        return self

    def predict(self, X):
        return _softmax(np.asarray(X, dtype=np.float64) @ self.W + self.b).argmax(axis=1)

    def accuracy(self, X, y):
        return float((self.predict(X) == np.asarray(y)).mean())


class ANNClassifier:
    """ANN baseline: softmax logistic regression on raw pixels."""

    def __init__(self, n_features=256, n_classes=10, seed=0):
        self.readout = SoftmaxReadout(n_features, n_classes, seed=seed)

    def fit(self, X, y, epochs=300, lr=0.5):
        return self.readout.fit(X, y, epochs=epochs, lr=lr)

    def predict(self, X):
        return self.readout.predict(X)

    def accuracy(self, X, y):
        return self.readout.accuracy(X, y)

    @property
    def n_mac_per_inference(self):
        """Multiply-accumulates per forward pass (weights x inputs)."""
        return int(self.readout.W.size)


class SNNClassifier:
    """Spiking network: rate encoding -> LIF layer -> softmax readout.

    The readout is trained on per-neuron spike counts, normalised by
    the number of timesteps so features stay in a stable range.
    """

    def __init__(self, n_features=256, n_hidden=256, n_classes=10,
                 n_steps=50, max_rate_hz=200.0, tau_m_ms=10.0,
                 v_th=1.0, weight_scale=4.0, seed=0):
        self.n_steps = n_steps
        self.max_rate_hz = max_rate_hz
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.lif = LIFLayer(n_features, n_hidden, tau_m_ms=tau_m_ms,
                            v_th=v_th, weight_scale=weight_scale,
                            seed=seed)
        self.readout = SoftmaxReadout(n_hidden, n_classes, seed=seed + 1)
        self._last_stats = {}

    def _spike_counts(self, X, rng):
        """Encode X to spikes, run the LIF layer, return counts + stats."""
        spikes = rate_encode(X, n_steps=self.n_steps,
                             max_rate_hz=self.max_rate_hz, rng=rng)
        counts, out_spikes = self.lif.run(spikes)
        self._last_stats = {
            "input_spikes": int(spikes.sum()),
            "hidden_spikes": int(out_spikes.sum()),
            "n_samples": int(X.shape[0]),
        }
        return counts

    def fit(self, X, y, epochs=1000, lr=1.0, seed=0):
        rng = np.random.default_rng(seed)
        counts = self._spike_counts(X, rng)
        # Normalise counts to [0, ~1] so the readout trains stably.
        self.readout.fit(counts / self.n_steps, y, epochs=epochs, lr=lr)
        return self

    def predict(self, X, seed=1):
        rng = np.random.default_rng(seed)
        counts = self._spike_counts(X, rng)
        return self.readout.predict(counts / self.n_steps)

    def accuracy(self, X, y, seed=1):
        return float((self.predict(X, seed=seed) == np.asarray(y)).mean())

    def mean_spikes_per_sample(self, X, seed=1):
        """Mean total spikes (input + hidden) per sample on X."""
        rng = np.random.default_rng(seed)
        self._spike_counts(X, rng)
        s = self._last_stats
        return (s["input_spikes"] + s["hidden_spikes"]) / s["n_samples"]

    @property
    def last_stats(self):
        return dict(self._last_stats)
