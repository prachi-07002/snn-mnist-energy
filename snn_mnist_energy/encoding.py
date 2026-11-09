"""Rate (Poisson) spike encoding.

Pixel intensity is mapped to a firing rate: a bright pixel fires often,
a dark pixel rarely. Each input pixel becomes an independent Poisson
spike train over T timesteps.
"""

import numpy as np


def rate_encode(images, n_steps=50, max_rate_hz=100.0, dt_ms=1.0, rng=None):
    """Encode images (n_samples, n_pixels) in [0, 1] to spike trains.

    Returns a boolean array of shape (n_samples, n_steps, n_pixels)
    where entry [i, t, j] is True if input neuron j fired at step t
    for sample i. The per-step firing probability is
    p = intensity * max_rate_hz * dt_ms / 1000, clipped to [0, 1].

    A pixel with intensity 1.0 at 100 Hz over 50 steps fires on
    average 5 times; a pixel with intensity 0.1 fires ~0.5 times.
    """
    rng = np.random.default_rng() if rng is None else rng
    images = np.asarray(images, dtype=np.float64)
    p = np.clip(images * max_rate_hz * dt_ms / 1000.0, 0.0, 1.0)
    # (n_samples, 1, n_pixels) compared against (n_steps,) uniform draws.
    draws = rng.random((images.shape[0], n_steps, images.shape[1]))
    return draws < p[:, None, :]
