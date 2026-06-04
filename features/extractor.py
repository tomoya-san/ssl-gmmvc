"""WavLM feature extraction for the SSL-GMM voice conversion pipeline.

:class:`FeatureExtractor` wraps the pretrained ``wavlm_large`` model (pulled via
``torch.hub`` from ``bshall/knn-vc``) and turns 16 kHz audio files into
self-supervised feature sequences -- the source/target representations the
joint GMMs in :mod:`gmm` are fitted on.

The extractor runs on either CPU or GPU. Pass an explicit ``device`` (e.g.
``"cpu"``, ``"cuda"``, ``"cuda:0"``) or leave it as ``None`` to auto-select CUDA
when available and fall back to CPU otherwise.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as AF

# WavLM (and the knn-vc checkpoint) are trained on 16 kHz audio.
WAVLM_SAMPLE_RATE = 16000


def resolve_device(device: str | None = None) -> torch.device:
    """Resolve a device spec to a concrete :class:`torch.device`.

    ``None`` auto-selects CUDA when available and falls back to CPU, so the same
    code path serves both backends.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


class FeatureExtractor:
    """Extract WavLM features from 16 kHz audio on CPU or GPU.

    Parameters
    ----------
    layer_num : int
        WavLM transformer layer to read the features from.
    device : str or torch.device or None
        Compute device. ``None`` auto-selects CUDA when available, else CPU.
    verbose : int
        0 silent, 1 progress prints (matches the :mod:`gmm` convention).
    """

    def __init__(self, layer_num: int, device: str | None = None, verbose: int = 1):
        self.device = resolve_device(device)
        self.layer_num = layer_num
        self.verbose = verbose

        self.wavlm = torch.hub.load(
            "bshall/knn-vc", "wavlm_large", trust_repo=True, device=str(self.device)
        )
        self.wavlm = self.wavlm.to(self.device).eval()

    @torch.inference_mode()
    def get_features_from_filepath(self, filepath: Path, vad: bool) -> torch.Tensor:
        """Extract features from a single audio file.

        Parameters
        ----------
        filepath : Path
            Path to the audio file (must be 16 kHz sampling rate).
        vad : bool
            Whether to apply voice activity detection to trim leading and
            trailing silence.

        Returns
        -------
        features : torch.Tensor
            Feature tensor with shape ``(n_frames, feature_dim)``, on
            ``self.device``.
        """
        wav, sr = torchaudio.load(filepath)

        if sr != WAVLM_SAMPLE_RATE:
            raise ValueError(
                f"{filepath}: expected {WAVLM_SAMPLE_RATE} Hz audio, got {sr} Hz."
            )

        # Downmix to a single (1, n_samples) channel; WavLM expects mono.
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if vad:
            # VAD operates on the raw waveform and is cheap, so run it on CPU
            # (where the audio is loaded) before moving to the compute device.
            wav = AF.vad(wav, sample_rate=WAVLM_SAMPLE_RATE)  # trim leading silence
            wav = torch.flip(wav, dims=[-1])                  # reverse the audio
            wav = AF.vad(wav, sample_rate=WAVLM_SAMPLE_RATE)  # trim trailing silence
            wav = torch.flip(wav, dims=[-1])                  # reverse back

        wav = wav.to(self.device)

        features, _ = self.wavlm.extract_features(wav, output_layer=self.layer_num)
        # extract_features returns (batch=1, n_frames, feature_dim); drop the
        # batch axis only so single-frame clips keep their 2-D shape.
        return features.squeeze(0)

    @torch.inference_mode()
    def get_features_from_file_list(
        self, file_list: list[Path], vad: bool
    ) -> torch.Tensor:
        """Extract and stack features from many audio files.

        Parameters
        ----------
        file_list : list[Path]
            Paths to audio files (must be 16 kHz sampling rate).
        vad : bool
            Whether to apply voice activity detection to trim leading and
            trailing silence.

        Returns
        -------
        features_block : torch.Tensor
            Stacked features from all files with shape
            ``(total_frames, feature_dim)``, on ``self.device``.
        """
        features_block = []
        for i, filepath in enumerate(file_list):
            features_block.append(self.get_features_from_filepath(filepath, vad))
            if self.verbose > 1 and (i + 1) % 50 == 0:
                print(f"  Extracted features from {i + 1}/{len(file_list)} files")

        features_block = torch.vstack(features_block)

        if self.verbose > 0:
            print(
                f"Extracted features from {len(file_list)} files "
                f"-> {tuple(features_block.shape)} on {self.device}"
            )

        return features_block
