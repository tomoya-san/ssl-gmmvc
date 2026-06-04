"""HiFi-GAN vocoder for the SSL-GMM voice conversion pipeline.

:class:`Vocoder` loads the prematched WavLM HiFi-GAN generator from
``bshall/knn-vc`` via ``torch.hub`` and turns acoustic features (the WavLM-space
targets produced by the joint GMMs in :mod:`gmm`) back into a 16 kHz waveform.
The model is pulled from the hub, so this module no longer depends on a local
copy of the ``hifigan`` source.

Device policy (chosen to avoid silently running on CPU when a GPU was expected):
``device=None`` *requires* CUDA and raises ``RuntimeError`` if none is
available; pass ``device="cpu"`` to opt into CPU explicitly. This mirrors the
loud-error behaviour of the :mod:`gmm` GPU presets while keeping a CPU escape
hatch. Note it is intentionally the opposite of :mod:`features`, whose
``FeatureExtractor`` silently auto-falls-back to CPU.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torchaudio

from common.device import resolve_device

# knn-vc's WavLM HiFi-GAN operates at 16 kHz.
VOCODER_SAMPLE_RATE = 16000


class Vocoder:
    """Prematched WavLM HiFi-GAN vocoder loaded from ``torch.hub``.

    Parameters
    ----------
    device : str or torch.device or None
        ``None`` requires CUDA and raises ``RuntimeError`` if unavailable;
        pass ``"cpu"`` to run on CPU.
    prematched : bool
        Load the prematched checkpoint (the knn-vc default for conversion).
    progress : bool
        Show the hub download progress bar.
    """

    def __init__(self, device: str | None = None, prematched: bool = True, progress: bool = True):
        self.device = resolve_device(device, require_cuda=True)
        self.sample_rate = VOCODER_SAMPLE_RATE

        hifigan, _ = torch.hub.load(
            "bshall/knn-vc",
            "hifigan_wavlm",
            trust_repo=True,
            prematched=prematched,
            progress=progress,
            device=str(self.device),
        )
        self.vocoder_model = hifigan.eval()

    @torch.inference_mode()
    def vocode(self, input_feature: torch.Tensor, output_file: str | Path | None = None) -> torch.Tensor:
        """Synthesize a waveform from acoustic features.

        Parameters
        ----------
        input_feature : torch.Tensor
            Features of shape ``(n_frames, feature_dim)`` or
            ``(batch, n_frames, feature_dim)`` -- the last axis is the feature
            dimension, matching ``FeatureExtractor`` output and the HiFi-GAN
            generator's expected ``(bs, seq_len, dim)``. Moved to the vocoder's
            device.
        output_file : str or Path or None
            If given, the waveform is also saved there (16 kHz WAV).

        Returns
        -------
        waveform : torch.Tensor
            Synthesized audio of shape ``(1, n_samples)`` on CPU, sampled at
            ``self.sample_rate``.
        """
        feats = input_feature.float().to(self.device)
        if feats.dim() == 2:  # add the batch axis: (1, n_frames, feature_dim)
            feats = feats.unsqueeze(0)

        wav_hat = self.vocoder_model(feats).squeeze(0)   # (1, n_samples)
        waveform = wav_hat.detach().cpu().squeeze()      # (n_samples,)
        if waveform.dim() == 1:                          # back to (channels, samples)
            waveform = waveform.unsqueeze(0)

        if output_file is not None:
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            torchaudio.save(output_file, waveform, self.sample_rate)
            print(f"Vocoder output saved to {output_file}")

        return waveform
