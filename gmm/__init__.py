"""Joint Gaussian Mixture Models for voice conversion.

Four presets span two orthogonal axes -- covariance structure (full vs
cross-diagonal) and array backend (NumPy vs PyTorch)::

    from gmm import (
        FullJointGMMGPU, FullJointGMMCPU,
        CrossDiagJointGMMGPU, CrossDiagJointGMMCPU,
    )

    model = CrossDiagJointGMMGPU(n_components=64, device="cuda")
    model.fit(XY)                     # XY = concatenated [source | target]
    Y_hat = model.convert(X)          # map source -> target
    resp  = model.predict_responsibilities(X)
    model.save_model("model.npz")

For custom combinations, compose the pieces directly::

    from gmm import JointGMM, CrossDiagCovariance, TorchBackend
    model = JointGMM(CrossDiagCovariance(64), TorchBackend("cuda"))
"""

from .backends import Backend, NumpyBackend, TorchBackend, NUMPY
from .covariance import CovarianceModel, CrossDiagCovariance, FullCovariance
from .estimator import JointGMM
from .presets import (
    CrossDiagJointGMMCPU,
    CrossDiagJointGMMGPU,
    FullJointGMMCPU,
    FullJointGMMGPU,
)

__all__ = [
    # public presets
    "FullJointGMMGPU",
    "FullJointGMMCPU",
    "CrossDiagJointGMMGPU",
    "CrossDiagJointGMMCPU",
    # composable building blocks
    "JointGMM",
    "CovarianceModel",
    "FullCovariance",
    "CrossDiagCovariance",
    "Backend",
    "NumpyBackend",
    "TorchBackend",
    "NUMPY",
]
