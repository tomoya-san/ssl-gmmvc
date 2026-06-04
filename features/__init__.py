"""Self-supervised feature extraction for SSL-GMM voice conversion.

Turns 16 kHz audio into WavLM feature sequences -- the source/target
representations the joint GMMs in :mod:`gmm` are fitted on::

    from features import FeatureExtractor

    extractor = FeatureExtractor(layer_num=6)        # auto CPU/GPU
    feats = extractor.get_features_from_filepath(path, vad=True)
"""

from .extractor import FeatureExtractor, resolve_device, WAVLM_SAMPLE_RATE

__all__ = [
    "FeatureExtractor",
    "resolve_device",
    "WAVLM_SAMPLE_RATE",
]
