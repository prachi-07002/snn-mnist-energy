"""Leaky integrate-and-fire (LIF) neuron layer, written from scratch.

Membrane dynamics (discrete time, Euler step):

    V[t+1] = V[t] * exp(-dt / tau_m) + I[t] * R
    if V[t+1] >= V_th: spike, then V <- V_reset

Spikes are counted per neuron over the whole simulation window; the
spike-count vector is what the readout classifier consumes.
"""

import numpy as np


class LIFLayer:
    """A layer of leaky integrate-and-fire neurons.

    Parameters
    ----------
    n_in : int
        Number of input neurons (size of the incoming spike vector).
    n_out : int
        Number of LIF neurons in this layer.
    tau_m_ms : float
        Membrane time constant in ms (leak speed).
    v_th : float
        Spike threshold (membrane potential units).
    v_reset : float
        Membrane potential after a spike.
    dt_ms : float
        Simulation timestep in ms.
    weight_scale : float
        Multiplier on the input weights. The default random weights
        (std 1/sqrt(n_in)) are small; with sparse rate-coded input a
        scale of ~4 is needed for the membrane to reach threshold at
        a healthy rate. Tune this if the layer barely fires or fires
        on every step.
    seed : int
        Random seed for the input weight matrix.
    """

    def __init__(self, n_in, n_out, tau_m_ms=20.0, v_th=1.0,
                 v_reset=0.0, dt_ms=1.0, weight_scale=1.0, seed=0):
        self.n_in = n_in
        self.n_out = n_out
        self.tau_m_ms = tau_m_ms
        self.v_th = v_th
        self.v_reset = v_reset
        self.dt_ms = dt_ms
        # Fixed random input weights, scaled so a typical input volley
        # brings the membrane near threshold without saturating it.
        rng = np.random.default_rng(seed)
        self.W = (rng.normal(0.0, 1.0 / np.sqrt(n_in), size=(n_in, n_out))
                  * weight_scale)
        self.decay = float(np.exp(-dt_ms / tau_m_ms))

    def run(self, spike_trains):
        """Simulate the layer over a spike train.

        Parameters
        ----------
        spike_trains : bool array, shape (n_samples, n_steps, n_in)

        Returns
        -------
        counts : int array, shape (n_samples, n_out)
            Total spikes emitted by each neuron per sample.
        out_spikes : bool array, shape (n_samples, n_steps, n_out)
            Full output spike trains (kept for analysis / energy count).
        """
        spike_trains = np.asarray(spike_trains)
        n_samples, n_steps, _ = spike_trains.shape
        V = np.zeros((n_samples, self.n_out), dtype=np.float64)
        counts = np.zeros((n_samples, self.n_out), dtype=np.int64)
        out_spikes = np.zeros((n_samples, n_steps, self.n_out), dtype=bool)
        for t in range(n_steps):
            # Synaptic current: each incoming spike contributes W.
            I = spike_trains[:, t, :].astype(np.float64) @ self.W
            V = V * self.decay + I
            fired = V >= self.v_th
            V = np.where(fired, self.v_reset, V)
            counts += fired.astype(np.int64)
            out_spikes[:, t, :] = fired
        return counts, out_spikes

    def run_single(self, spike_train):
        """Simulate one sample; returns (counts, voltage trace).

        spike_train : bool array, shape (n_steps, n_in)
        counts : int array, shape (n_out,)
        voltages : float array, shape (n_steps, n_out)
        """
        spike_train = np.asarray(spike_train)
        n_steps = spike_train.shape[0]
        V = np.zeros(self.n_out, dtype=np.float64)
        counts = np.zeros(self.n_out, dtype=np.int64)
        voltages = np.zeros((n_steps, self.n_out), dtype=np.float64)
        for t in range(n_steps):
            I = spike_train[t].astype(np.float64) @ self.W
            V = V * self.decay + I
            fired = V >= self.v_th
            V = np.where(fired, self.v_reset, V)
            counts += fired.astype(np.int64)
            voltages[t] = V
        return counts, voltages
