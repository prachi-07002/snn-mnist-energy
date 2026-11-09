"""snn-mnist-energy: a small spiking neural network for digit
classification, with an ANN baseline and an energy model.
"""

from .digits import make_dataset, render_digit, ascii_digit
from .encoding import rate_encode
from .lif import LIFLayer
from .models import ANNClassifier, SNNClassifier, SoftmaxReadout
from .energy import snn_energy_pj, ann_energy_pj, pj_to_nj

__all__ = [
    "make_dataset", "render_digit", "ascii_digit",
    "rate_encode", "LIFLayer",
    "ANNClassifier", "SNNClassifier", "SoftmaxReadout",
    "snn_energy_pj", "ann_energy_pj", "pj_to_nj",
]
