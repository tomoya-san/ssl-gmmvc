"""Joint Gaussian Mixture Models for voice conversion.

Six presets span two orthogonal axes -- covariance structure (full,
cross-diagonal or shared/tied) and array backend (NumPy vs PyTorch)::

    from gmm import (
        FullJointGMMGPU, FullJointGMMCPU,
        CrossDiagJointGMMGPU, CrossDiagJointGMMCPU,
        SharedJointGMMGPU, SharedJointGMMCPU,
    )

    model = CrossDiagJointGMMGPU(n_components=64)   # CUDA; raises if unavailable
    model.fit(XY)                     # XY = concatenated [source | target]
    model.fit(XY, chunk_size=10_000)  # same result, memory bounded by the chunk
    Y_hat = model.convert(X)          # map source -> target
    resp  = model.predict_responsibilities(X)
    model.save_model("model.npz")

For custom combinations, compose the pieces directly::

    from gmm import JointGMM, CrossDiagCovariance, TorchBackend
    model = JointGMM(CrossDiagCovariance(64), TorchBackend("cuda"))
"""

from .backends import Backend, NumpyBackend, TorchBackend, NUMPY
from .covariance import (
    CovarianceModel,
    CrossDiagCovariance,
    FullCovariance,
    SharedCovariance,
)
from .estimator import JointGMM
from .presets import (
    CrossDiagJointGMMCPU,
    CrossDiagJointGMMGPU,
    FullJointGMMCPU,
    FullJointGMMGPU,
    SharedJointGMMCPU,
    SharedJointGMMGPU,
)

__all__ = [
    # public presets
    "FullJointGMMGPU",
    "FullJointGMMCPU",
    "CrossDiagJointGMMGPU",
    "CrossDiagJointGMMCPU",
    "SharedJointGMMGPU",
    "SharedJointGMMCPU",
    # composable building blocks
    "JointGMM",
    "CovarianceModel",
    "FullCovariance",
    "CrossDiagCovariance",
    "SharedCovariance",
    "Backend",
    "NumpyBackend",
    "TorchBackend",
    "NUMPY",
]
