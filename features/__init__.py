"""Self-supervised feature extraction for SSL-GMM voice conversion.

Turns 16 kHz audio into WavLM feature sequences -- the source/target
representations the joint GMMs in :mod:`gmm` are fitted on::

    from features import FeatureExtractor

    extractor = FeatureExtractor(layer_num=6)        # auto CPU/GPU
    feats = extractor.get_features_from_filepath(path, vad=True)

Source and target feature sequences are then paired with :mod:`matching`
before being concatenated and handed to the joint GMMs::

    from features import symmetric_matching

    src_matched, tgt_matched = symmetric_matching(src_feats, tgt_feats, k=4)
"""

from .extractor import FeatureExtractor, resolve_device, WAVLM_SAMPLE_RATE
from .matching import symmetric_matching

__all__ = [
    "FeatureExtractor",
    "resolve_device",
    "WAVLM_SAMPLE_RATE",
    "symmetric_matching",
]
