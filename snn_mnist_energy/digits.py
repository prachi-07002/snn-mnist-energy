"""Synthetic digit dataset.

Digits are drawn as stroke patterns on a 16x16 grid from hand-defined
polylines (a seven-segment-style stroke vocabulary, one per digit),
then jittered (shift / scale / thickness / background noise) so samples
vary like handwriting without needing any downloaded dataset.

This is a SYNTHETIC dataset, not the real MNIST download: it is small,
fully reproducible, and generated locally in milliseconds. The stroke
patterns are deliberately simple so that the classification task is a
clean test-bed for the spiking network rather than a benchmark claim.
"""

import numpy as np

GRID = 16

# Polylines in (row, col) coordinates on a 16x16 grid, roughly a 12-row
# tall, 8-col wide digit box starting at row 2, col 4.
_SEGMENTS = {
    # (r0, c0) -> (r1, c1) named strokes
    "top": ((2, 4), (2, 11)),
    "upper_left": ((2, 4), (7, 4)),
    "upper_right": ((2, 11), (7, 11)),
    "middle": ((7, 4), (7, 11)),
    "lower_left": ((7, 4), (13, 4)),
    "lower_right": ((7, 11), (13, 11)),
    "bottom": ((13, 4), (13, 11)),
    "diag_down": ((2, 11), (13, 4)),   # diagonal for 7 / 1 variants
}

_DIGIT_SEGMENTS = {
    0: ["top", "upper_left", "upper_right", "lower_left", "lower_right", "bottom"],
    1: ["upper_right", "lower_right"],
    2: ["top", "upper_right", "middle", "lower_left", "bottom"],
    3: ["top", "upper_right", "middle", "lower_right", "bottom"],
    4: ["upper_left", "upper_right", "middle", "lower_right"],
    5: ["top", "upper_left", "middle", "lower_right", "bottom"],
    6: ["top", "upper_left", "middle", "lower_left", "lower_right", "bottom"],
    7: ["top", "upper_right", "lower_right"],
    8: ["top", "upper_left", "upper_right", "middle",
        "lower_left", "lower_right", "bottom"],
    9: ["top", "upper_left", "upper_right", "middle", "lower_right", "bottom"],
}


def _draw_line(img, p0, p1, value=1.0):
    """Draw a 1-pixel-wide line between two (row, col) points."""
    r0, c0 = p0
    r1, c1 = p1
    steps = int(max(abs(r1 - r0), abs(c1 - c0))) + 1
    for i in range(steps):
        t = i / max(steps - 1, 1)
        r = int(round(r0 + (r1 - r0) * t))
        c = int(round(c0 + (c1 - c0) * t))
        if 0 <= r < GRID and 0 <= c < GRID:
            img[r, c] = value


def _base_digit(digit):
    img = np.zeros((GRID, GRID), dtype=np.float64)
    for name in _DIGIT_SEGMENTS[digit]:
        _draw_line(img, *_SEGMENTS[name])
    return img


def render_digit(digit, rng, shift=1, noise=0.05, thicken=True):
    """Render one jittered sample of `digit` as a 16x16 float image.

    Jitter: random integer shift of up to +/-`shift` px in each axis,
    random per-pixel intensity in [0.7, 1.0] on strokes, optional
    thickening (a second offset line), and uniform background noise.
    Pixel values are clipped to [0, 1].
    """
    img = _base_digit(digit)
    # Thicken: overlay the same strokes shifted by one pixel down-right.
    if thicken and rng.random() < 0.8:
        img = np.maximum(img, np.roll(np.roll(img, 1, axis=0), 1, axis=1))
    # Random shift.
    dr = int(rng.integers(-shift, shift + 1))
    dc = int(rng.integers(-shift, shift + 1))
    img = np.roll(np.roll(img, dr, axis=0), dc, axis=1)
    # Vary stroke intensity.
    mask = img > 0
    img[mask] = rng.uniform(0.7, 1.0, size=int(mask.sum()))
    # Background noise.
    if noise > 0:
        img = img + rng.uniform(0.0, noise, size=img.shape)
    return np.clip(img, 0.0, 1.0)


def make_dataset(n_train_per_class=60, n_test_per_class=30, seed=0):
    """Return (X_train, y_train, X_test, y_test).

    X are float arrays of shape (n_samples, 256) with values in [0, 1];
    y are integer labels 0-9. Deterministic for a given seed.
    """
    rng = np.random.default_rng(seed)
    X_train, y_train, X_test, y_test = [], [], [], []
    for digit in range(10):
        for _ in range(n_train_per_class):
            X_train.append(render_digit(digit, rng).reshape(-1))
            y_train.append(digit)
        for _ in range(n_test_per_class):
            X_test.append(render_digit(digit, rng).reshape(-1))
            y_test.append(digit)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    # Shuffle (keeping X/y aligned).
    perm = rng.permutation(len(X_train))
    X_train, y_train = X_train[perm], y_train[perm]
    perm = rng.permutation(len(X_test))
    X_test, y_test = X_test[perm], y_test[perm]
    return X_train, y_train, X_test, y_test


def ascii_digit(img):
    """Pretty-print a 16x16 (or 256-vector) digit as ASCII art."""
    img = np.asarray(img).reshape(GRID, GRID)
    chars = " .:-=+*#%@"
    out = []
    for row in img:
        out.append("".join(chars[min(int(v * len(chars)), len(chars) - 1)]
                           for v in row))
    return "\n".join(out)
