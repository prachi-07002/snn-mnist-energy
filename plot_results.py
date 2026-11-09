"""Optional plot: accuracy vs estimated energy for ANN and SNN.

Re-runs the demo experiment and saves a bar chart to /tmp
(the repo stays text-only, so the figure is not committed).

Usage:
    python3 plot_results.py
"""

import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/tmp/snn-mnist-energy-plot.png"


def main():
    # Reuse the demo's numbers by importing and re-running its pieces.
    from snn_mnist_energy import make_dataset, ANNClassifier, SNNClassifier
    from snn_mnist_energy.energy import snn_energy_pj, ann_energy_pj, pj_to_nj

    X_train, y_train, X_test, y_test = make_dataset(60, 30, seed=0)
    ann = ANNClassifier(seed=0)
    ann.fit(X_train, y_train, epochs=300, lr=0.5)
    ann_acc = ann.accuracy(X_test, y_test) * 100
    ann_e = pj_to_nj(ann_energy_pj(ann.n_mac_per_inference))

    snn = SNNClassifier(n_hidden=256, n_steps=50, max_rate_hz=200.0,
                        tau_m_ms=10.0, v_th=1.0, weight_scale=4.0, seed=0)
    snn.fit(X_train, y_train, epochs=1000, lr=1.0, seed=0)
    snn_acc = snn.accuracy(X_test, y_test, seed=1) * 100
    mean_spikes = snn.mean_spikes_per_sample(X_test, seed=1)
    stats = snn.last_stats
    n_in = stats["input_spikes"] / stats["n_samples"]
    n_hid = stats["hidden_spikes"] / stats["n_samples"]
    snn_e = pj_to_nj(snn_energy_pj(n_in, n_hid, fanout_in=256,
                                   fanout_out=10, readout_mac=2560))

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].bar(["ANN", "SNN"], [ann_acc, snn_acc], color=["#4a90d9", "#e26d9c"])
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_ylim(0, 100)
    axes[0].set_title("Accuracy (chance = 10%)")
    for i, v in enumerate([ann_acc, snn_acc]):
        axes[0].text(i, v + 2, f"{v:.1f}%", ha="center")
    axes[1].bar(["ANN", "SNN"], [ann_e, snn_e], color=["#4a90d9", "#e26d9c"])
    axes[1].set_ylabel("Est. energy / inference (nJ)")
    axes[1].set_title(f"Modelled energy (SNN {mean_spikes:.0f} spikes/sample)")
    for i, v in enumerate([ann_e, snn_e]):
        axes[1].text(i, v + 4, f"{v:.1f} nJ", ha="center")
    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    print(f"Saved {OUT}")
    print(f"ANN {ann_acc:.1f}% / {ann_e:.2f} nJ | SNN {snn_acc:.1f}% / "
          f"{snn_e:.2f} nJ | spikes/sample {mean_spikes:.1f}")


if __name__ == "__main__":
    main()
