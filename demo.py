"""Demo / experiment: ANN baseline vs spiking network on synthetic digits.

Prints test accuracy for both models, mean spikes per sample for the
SNN, and estimated energy per inference for both (model, not measured
hardware — see snn_mnist_energy/energy.py for the assumptions).

Usage:
    python3 demo.py
"""

import time

import numpy as np

from snn_mnist_energy import make_dataset, ANNClassifier, SNNClassifier
from snn_mnist_energy.energy import snn_energy_pj, ann_energy_pj, pj_to_nj


def main():
    t0 = time.time()
    print("=== snn-mnist-energy demo ===")
    print("Dataset: synthetic 16x16 stroke digits (NOT real MNIST).")
    X_train, y_train, X_test, y_test = make_dataset(
        n_train_per_class=60, n_test_per_class=30, seed=0)
    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples, "
          f"features: {X_train.shape[1]}, classes: 10\n")

    # ---- ANN baseline: logistic regression on raw pixels ----
    print("Training ANN baseline (softmax logistic regression)...")
    ann = ANNClassifier(n_features=256, n_classes=10, seed=0)
    ann.fit(X_train, y_train, epochs=300, lr=0.5)
    ann_acc = ann.accuracy(X_test, y_test)
    ann_energy = ann_energy_pj(ann.n_mac_per_inference)
    print(f"  ANN test accuracy:      {ann_acc * 100:.1f}%")
    print(f"  ANN MACs / inference:   {ann.n_mac_per_inference}")
    print(f"  ANN est. energy / inf:  {pj_to_nj(ann_energy):.2f} nJ "
          f"({ann_energy:.0f} pJ)\n")

    # ---- SNN: rate encoding -> LIF hidden layer -> softmax readout ----
    print("Training SNN (rate encoding + LIF layer + softmax readout)...")
    snn = SNNClassifier(n_features=256, n_hidden=256, n_classes=10,
                        n_steps=50, max_rate_hz=200.0, tau_m_ms=10.0,
                        v_th=1.0, weight_scale=4.0, seed=0)
    snn.fit(X_train, y_train, epochs=1000, lr=1.0, seed=0)
    snn_acc = snn.accuracy(X_test, y_test, seed=1)
    mean_spikes = snn.mean_spikes_per_sample(X_test, seed=1)
    stats = snn.last_stats
    n_in = stats["input_spikes"] / stats["n_samples"]
    n_hid = stats["hidden_spikes"] / stats["n_samples"]
    readout_mac = 256 * 10  # hidden -> readout weights, computed once/sample
    snn_energy = snn_energy_pj(n_in, n_hid, fanout_in=256, fanout_out=10,
                               readout_mac=readout_mac)
    print(f"  SNN test accuracy:      {snn_acc * 100:.1f}%")
    print(f"  Mean spikes / sample:   {mean_spikes:.1f} "
          f"(input {n_in:.1f} + hidden {n_hid:.1f})")
    print(f"  SNN est. energy / inf:  {pj_to_nj(snn_energy):.2f} nJ "
          f"({snn_energy:.0f} pJ)\n")

    # ---- Headline comparison ----
    print("=== Headline ===")
    print(f"ANN accuracy: {ann_acc * 100:.1f}%  |  SNN accuracy: {snn_acc * 100:.1f}%")
    print(f"Mean spikes / sample (SNN): {mean_spikes:.1f}")
    print(f"Est. energy / inference: ANN {pj_to_nj(ann_energy):.2f} nJ, "
          f"SNN {pj_to_nj(snn_energy):.2f} nJ "
          f"({pj_to_nj(snn_energy) / pj_to_nj(ann_energy):.2f}x ANN)")
    print(f"Chance level (10 classes): 10.0%")
    print(f"\nDone in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
