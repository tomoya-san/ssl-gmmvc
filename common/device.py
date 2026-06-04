"""Shared device resolution for the SSL-GMM packages.

A single :func:`resolve_device` serves the two device policies used across the
codebase, selected by ``require_cuda``:

* ``require_cuda=False`` (default) -- *auto-fallback*: ``None`` picks CUDA when
  available and CPU otherwise, and nothing is raised. Used by inference that
  should run anywhere (see :mod:`features`).
* ``require_cuda=True`` -- *strict*: ``None`` means CUDA and any CUDA request
  raises :class:`RuntimeError` when no CUDA device is present (no silent CPU
  fallback). Pass ``device="cpu"`` to opt into CPU. Used where running on the
  wrong device should be a loud error (see :mod:`vocoder` and :mod:`gmm`'s
  ``TorchBackend``).
"""

from __future__ import annotations

import torch


def resolve_device(device: str | torch.device | None = None, *, require_cuda: bool = False) -> torch.device:
    """Resolve a device spec to a concrete :class:`torch.device`.

    Parameters
    ----------
    device : str or torch.device or None
        Explicit device (e.g. ``"cpu"``, ``"cuda"``, ``"cuda:0"``), or ``None``
        to choose automatically.
    require_cuda : bool
        If ``False`` (default), ``None`` auto-selects CUDA when available and
        falls back to CPU; no error is raised. If ``True``, ``None`` means CUDA
        and any CUDA request raises :class:`RuntimeError` when CUDA is
        unavailable -- pass ``device="cpu"`` to run on CPU.
    """
    if device is None:
        device = "cuda" if require_cuda else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    if require_cuda and device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"A CUDA device ({device}) was requested but none is available. "
            "Pass device='cpu' to run on CPU."
        )
    return device
