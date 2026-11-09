"""Energy model (a MODEL, not measured hardware).

Everything here is an estimate built from stated per-operation costs.
Real silicon numbers depend on the process node, circuit style and
activity; these are order-of-magnitude assumptions chosen to be
transparent and easy to swap out.

Assumptions
-----------
- One synaptic operation (a spike arriving at one synapse, i.e. one
  accumulate) costs E_SYNOP = 1 pJ. This is in the range reported for
  digital neuromorphic chips (sub-pJ to a few pJ per synaptic event).
- Emitting one spike (the neuron firing + routing it) costs
  E_SPIKE = 10 pJ, again a typical order-of-magnitude figure for
  event routing in neuromorphic hardware.
- One ANN multiply-accumulate costs E_MAC = 4.6 pJ, the widely cited
  45 nm figure for a 32-bit floating-point MAC (Horowitz, 2014).
  A fixed-point / 8-bit MAC would be cheaper; 4.6 pJ is the
  conservative, commonly quoted reference point.

SNN energy per inference
------------------------
    E = n_input_spikes * fanout_in * E_SYNOP
      + n_hidden_spikes * fanout_out * E_SYNOP
      + (n_input_spikes + n_hidden_spikes) * E_SPIKE

where fanout_in is the number of hidden neurons each input spike
reaches, and fanout_out is the number of readout neurons each hidden
spike reaches. The readout itself is counted as ANN-style MACs
(hidden x classes) since in this prototype the readout is computed
off-chip once per sample; this is stated explicitly so the comparison
stays honest.

ANN energy per inference
------------------------
    E = n_mac * E_MAC
"""

E_SYNOP_PJ = 1.0    # pJ per synaptic operation (accumulate on spike arrival)
E_SPIKE_PJ = 10.0   # pJ per spike emitted (firing + routing)
E_MAC_PJ = 4.6      # pJ per 32-bit float MAC (Horowitz 2014, 45 nm)


def snn_energy_pj(n_input_spikes, n_hidden_spikes,
                  fanout_in, fanout_out, readout_mac):
    """Estimated SNN energy per inference, in picojoules."""
    synops = (n_input_spikes * fanout_in
              + n_hidden_spikes * fanout_out)
    spikes = n_input_spikes + n_hidden_spikes
    return (synops * E_SYNOP_PJ
            + spikes * E_SPIKE_PJ
            + readout_mac * E_MAC_PJ)


def ann_energy_pj(n_mac):
    """Estimated ANN energy per inference, in picojoules."""
    return n_mac * E_MAC_PJ


def pj_to_nj(pj):
    return pj / 1000.0
