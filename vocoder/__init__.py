"""HiFi-GAN vocoding for SSL-GMM voice conversion.

Turns acoustic features (e.g. the WavLM-space targets produced by the joint
GMMs in :mod:`gmm`) back into a waveform. The prematched WavLM HiFi-GAN is
pulled from ``torch.hub`` (``bshall/knn-vc``)::

    from vocoder import Vocoder

    voc = Vocoder()                          # CUDA; raises if unavailable
    wav = voc.vocode(features, output_file="out.wav")
"""

from .hifigan import Vocoder

__all__ = [
    "Vocoder",
]
