# snn-mnist-energy

A small spiking neural network (SNN) for digit classification, compared
head-to-head with a regular ANN baseline on **accuracy vs. estimated
energy per inference**. Built as a software prototype for a Spring 2027
Honors College research project on a **spiking artificial neuron**
(Prof. Walter Daniel Leon-Salas): a low-power analog board emulating a
spiking neuron, coin-cell powered, with potentiometers to adjust weights
manually, driving a small network to control a robot.

The code models the same ideas the hardware will embody — rate-coded
spikes, leaky integrate-and-fire (LIF) dynamics, and event-driven
energy accounting — so the trade-offs measured here (accuracy lost to
spike noise, energy spent per spike and per synaptic event) are the
trade-offs that will shape the board design.

## Headline results (measured by running `demo.py`)

| Model | Test accuracy | Est. energy / inference | Notes |
|---|---|---|---|
| ANN baseline (softmax logistic regression, NumPy) | **98.7%** | **11.78 nJ** | 2,560 MACs / inference |
| SNN (rate coding + 256 LIF neurons + softmax readout) | **93.3%** | **176.64 nJ** | 1,800 spikes / sample |

- Dataset: 600 train / 300 test synthetic 16×16 digits, 10 classes
  (chance = 10%).
- Mean spikes per sample (SNN): **1,800.3** (523.8 input + 1,276.4 hidden).
- SNN energy is **15.0× the ANN** in this model — see "What the energy
  numbers mean" below. This is an honest, measured outcome, not a
  failure of the experiment: it is the finding.
- Demo runtime: ~7 s. Tests: **16 passed** (`python3 -m unittest discover -s tests`).

## What the energy numbers mean

The energy figures are a **model, not measured hardware**. Assumptions
(see `snn_mnist_energy/energy.py`):

- 1 pJ per synaptic operation (one spike arriving at one synapse),
- 10 pJ per spike emitted (firing + routing),
- 4.6 pJ per 32-bit float MAC for the ANN (Horowitz 2014, 45 nm).

Why does the SNN cost 15× more here? Two reasons, both real:

1. **Dense rate coding is expensive.** Every input spike fans out to all
   256 hidden neurons, so ~524 input spikes become ~134k synaptic
   operations. The ANN, by contrast, is only 2,560 MACs — this toy
   problem is tiny, so the ANN has almost nothing to beat.
2. **Poisson rate coding throws information away.** Pixel intensity
   becomes a noisy spike count, which is why the SNN gives up ~5 points
   of accuracy (93.3% vs 98.7%).

So the lesson is not "SNNs are worse" — it is **where SNNs win**: sparse,
event-driven, temporal data and larger networks, where the ANN's MAC
count explodes but a spiking network only pays for the spikes that
actually fire. That is exactly the regime the analog neuron board is
designed for (sparse sensor events driving a robot), and it is what the
next iteration should test: sparser coding (e.g. time-to-first-spike),
sparse connectivity, and temporal input.

## The dataset: synthetic digits, NOT real MNIST

No dataset is downloaded. Digits are drawn as **stroke patterns on a
16×16 grid** from hand-defined polylines (a seven-segment-style stroke
vocabulary, one per digit), then jittered with random shifts, stroke
thickness, intensity variation, and background noise
(`snn_mnist_energy/digits.py`). This keeps the project fully
reproducible, dependency-free, and fast — and it makes the task a clean
test-bed for the spiking network rather than a benchmark claim. Do not
compare these accuracies against MNIST leaderboard numbers; the digits
are simpler than handwriting.

## How it works

```
16x16 digit image (256 pixels, values in [0, 1])
        |
        v  rate encoding: pixel intensity -> Poisson spike train
        |  (200 Hz max rate, 50 timesteps of 1 ms)
        v
LIF hidden layer: 256 leaky integrate-and-fire neurons,
        fixed random weights (std 4/sqrt(256)), tau_m = 10 ms,
        threshold 1.0, reset to 0 on spike
        |
        v  spike counts over the 50 ms window, normalised by T
softmax readout (logistic regression, trained by gradient descent)
        |
        v
predicted digit 0-9
```

The ANN baseline is the same softmax readout trained directly on the
raw pixel intensities, so the comparison isolates the cost of going
event-based: same classifier, same training, only the representation
(spikes vs. pixels) changes.

## How to run

```bash
python3 demo.py            # full experiment: accuracy + spikes + energy (~7 s)
python3 -m unittest discover -s tests -v   # 16 unit tests
python3 plot_results.py    # optional: bar chart of accuracy vs energy,
                           # saved to /tmp (repo stays text-only)
```

Requirements: Python 3.10+, NumPy. Matplotlib only for the optional plot.

## Files

```
README.md
requirements.txt
demo.py                        # the experiment: trains both models, prints results
plot_results.py                # optional accuracy/energy chart (writes to /tmp)
snn_mnist_energy/
    __init__.py
    digits.py                  # synthetic 16x16 stroke-digit generator
    encoding.py                # Poisson rate encoding (intensity -> spikes)
    lif.py                     # leaky integrate-and-fire layer, from scratch
    models.py                  # ANN baseline + SNN classifier + softmax readout
    energy.py                  # energy model (stated pJ assumptions)
tests/
    test_snn.py                # 16 unit tests
```

## Tests

16 tests, all passing, covering:

- **Encoding:** brighter pixels produce far more spikes than dark ones;
  zero input never spikes; output shapes are correct.
- **LIF neuron:** no input → no spikes; strong constant input → spikes;
  membrane leaks (voltage decays after a single spike); membrane resets
  to `v_reset` after firing; output shapes are correct.
- **Classifiers:** ANN and SNN both train and beat chance by a wide
  margin on 10 classes (ANN 98.7%, SNN 93.3% on the full test set);
  SNN produces spikes; prediction shapes and label ranges are correct.
- **Energy model:** energy increases with spike count; zero activity →
  zero energy; ANN energy scales linearly with MAC count.
- **Dataset:** train/test shapes, pixel range [0, 1], all 10 classes
  present, no two digits render identically.

## Application talking point (one line)

> Built a spiking neural network from scratch in NumPy (Poisson rate
> coding + 256 LIF neurons) that classifies synthetic digits at 93.3%
> vs. a 98.7% ANN baseline, and showed with an explicit pJ-level energy
> model that dense rate coding costs 15× the ANN on this toy problem —
> the quantitative case for the sparse, event-driven coding the analog
> neuron board will use.

## Limitations / next steps

- Synthetic digits only; validate on real sensor/event data when the
  board produces it.
- Readout is trained off-chip on spike counts; on-chip learning (e.g.
  STDP) and the analog weight potentiometers are the hardware side.
- Energy is a model with stated assumptions, not silicon measurement.
- Next experiments: time-to-first-spike coding, sparse connectivity,
  and a temporal task where the ANN baseline cannot just count MACs.
