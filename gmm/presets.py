"""The four public GMM classes as thin presets over :class:`JointGMM`.

Each class is just a choice of (covariance structure x backend); all behaviour
lives in :mod:`gmm.estimator`, :mod:`gmm.covariance` and :mod:`gmm.backends`.
The naming and constructor signatures mirror the historical API so existing
imports (``from gmm import FullJointGMMGPU`` etc.) keep working.
"""

from __future__ import annotations

from .backends import NUMPY, TorchBackend
from .covariance import CrossDiagCovariance, FullCovariance
from .estimator import JointGMM


class FullJointGMMGPU(JointGMM):
    """Full joint covariance, PyTorch backend (CUDA or CPU tensors)."""

    def __init__(self, n_components, device="cuda", verbose=1):
        super().__init__(FullCovariance(n_components), TorchBackend(device), verbose)


class FullJointGMMCPU(JointGMM):
    """Full joint covariance, NumPy backend.

    Note
    ----
    This is now a hand-written NumPy EM (the same algorithm as the other three
    presets), not a :class:`sklearn.mixture.GaussianMixture` subclass. The old
    sklearn-only constructor arguments (``n_init``, ``init_params``,
    ``precisions_init``, ``warm_start``, ``random_state`` ...) no longer apply.
    """

    def __init__(self, n_components=1, verbose=1, feature_dim=None):
        # ``feature_dim`` is inferred from the data in ``fit``; the argument is
        # accepted for backward compatibility and otherwise ignored.
        super().__init__(FullCovariance(n_components), NUMPY, verbose)


class CrossDiagJointGMMGPU(JointGMM):
    """Cross-diagonal joint covariance, PyTorch backend (CUDA or CPU tensors)."""

    def __init__(self, n_components=1, device="cuda", verbose=1):
        super().__init__(CrossDiagCovariance(n_components), TorchBackend(device), verbose)


class CrossDiagJointGMMCPU(JointGMM):
    """Cross-diagonal joint covariance, NumPy backend."""

    def __init__(self, n_components=1, verbose=1):
        super().__init__(CrossDiagCovariance(n_components), NUMPY, verbose)
