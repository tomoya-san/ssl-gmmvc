"""Array backends for the joint GMMs.

A :class:`Backend` wraps everything that differs between NumPy and PyTorch: the
elementwise ops, reductions, linear algebra, and conversion to/from numpy for
persistence. The GMM math (see :mod:`gmm.covariance`) and the EM driver (see
:mod:`gmm.estimator`) are written *once* against this interface, so the only
place that mentions ``np`` vs ``torch`` is here.

The two concrete backends are stateless apart from ``device`` (Torch only), so
the module exposes a ready-made :data:`NUMPY` singleton and a
:func:`torch_backend` factory.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

EPSILON = 1e-6


class Backend:
    """Abstract array backend. Subclasses implement the primitives below.

    Conventions
    -----------
    * ``array`` is whatever native type the backend uses (ndarray or Tensor).
    * Reductions take an ``axis`` keyword (NumPy naming) for consistency.
    """

    name = "abstract"

    # -- ingest / export ---------------------------------------------------
    def asarray(self, x):
        """Coerce ``x`` (ndarray or tensor) into this backend's native array."""
        raise NotImplementedError

    def to_numpy(self, x):
        """Return a NumPy copy of a native array (for ``save_model``)."""
        raise NotImplementedError

    # -- elementwise / reductions ------------------------------------------
    def log(self, x):
        raise NotImplementedError

    def exp(self, x):
        raise NotImplementedError

    def clamp_min(self, x, lo=EPSILON):
        """Lower-bound ``x`` elementwise (variance / probability flooring)."""
        raise NotImplementedError

    def logsumexp(self, x, axis):
        raise NotImplementedError

    def concat(self, parts, axis=-1):
        raise NotImplementedError

    def isfinite_all(self, x):
        """Return ``True`` if ``x`` has no NaN or Inf entries."""
        raise NotImplementedError

    # -- linear algebra ----------------------------------------------------
    def eye(self, dim):
        """Identity matrix of shape ``(dim, dim)``."""
        raise NotImplementedError

    def inv(self, a):
        """Batched matrix inverse over the last two dims."""
        raise NotImplementedError

    def slogdet(self, a):
        """Batched log|det| over the last two dims (sign assumed positive)."""
        raise NotImplementedError

    def cholesky_solve_mahalanobis(self, cov, diff):
        """Return ``(mahalanobis, log_det)`` for ``diff`` under ``cov``.

        ``cov``  : (k, d, d) positive-definite matrices.
        ``diff`` : (n, k, d) deviations from the mean.
        Returns ``mahal`` (n, k) and ``log_det`` (k,). Uses a Cholesky solve
        when available for numerical stability.
        """
        raise NotImplementedError

    def einsum(self, subscripts, *operands):
        raise NotImplementedError

    # -- initialization ----------------------------------------------------
    def kmeans_plusplus(self, points, n_clusters, verbose=0):
        """Pick ``n_clusters`` seed centers with the k-means++ heuristic."""
        raise NotImplementedError

    def cluster_assignments(self, points, centers):
        """Return the index of the nearest center for each row of ``points``."""
        raise NotImplementedError

    def group_sum(self, values, labels, n_groups):
        """Sum rows of ``values`` grouped by integer ``labels`` -> (n_groups, d)."""
        raise NotImplementedError

    def group_outer_sum(self, values, labels, n_groups):
        """Per-group sum of outer products ``sum_n z_n z_n^T`` -> (n_groups, d, d)."""
        raise NotImplementedError

    def bincount(self, labels, n_groups):
        """Per-group row counts as a float array of shape ``(n_groups,)``."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
class NumpyBackend(Backend):
    name = "numpy"

    def asarray(self, x):
        return np.asarray(x, dtype=np.float64)

    def to_numpy(self, x):
        return np.asarray(x)

    def log(self, x):
        return np.log(x)

    def exp(self, x):
        return np.exp(x)

    def clamp_min(self, x, lo=EPSILON):
        return np.maximum(x, lo)

    def logsumexp(self, x, axis):
        return np.logaddexp.reduce(x, axis=axis)

    def concat(self, parts, axis=-1):
        return np.concatenate(parts, axis=axis)

    def isfinite_all(self, x):
        return bool(np.isfinite(x).all())

    def eye(self, dim):
        return np.eye(dim)

    def inv(self, a):
        return np.linalg.inv(a)

    def slogdet(self, a):
        return np.linalg.slogdet(a)[1]

    def cholesky_solve_mahalanobis(self, cov, diff):
        inv_cov = np.linalg.inv(cov)
        log_det = np.linalg.slogdet(cov)[1]
        temp = np.einsum("nkf,kfg->nkg", diff, inv_cov)
        mahal = np.einsum("nkf,nkf->nk", temp, diff)
        return mahal, log_det

    def einsum(self, subscripts, *operands):
        return np.einsum(subscripts, *operands)

    def kmeans_plusplus(self, points, n_clusters, verbose=0):
        n_samples = points.shape[0]
        rng = np.random.default_rng()
        centers = [points[rng.integers(n_samples)]]
        closest_sq = np.full(n_samples, np.inf)

        for i in range(1, n_clusters):
            d_sq = ((points - centers[-1]) ** 2).sum(axis=1)
            closest_sq = np.minimum(closest_sq, d_sq)
            probs = closest_sq / closest_sq.sum()
            centers.append(points[rng.choice(n_samples, p=probs)])
            if verbose > 1 and (i + 1) % 10 == 0:
                print(f"  Selected center {i + 1}/{n_clusters}")
        return np.stack(centers)

    def cluster_assignments(self, points, centers):
        dists = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        return np.argmin(dists, axis=1)

    def group_sum(self, values, labels, n_groups):
        sums = np.zeros((n_groups, values.shape[1]))
        np.add.at(sums, labels, values)
        return sums

    def group_outer_sum(self, values, labels, n_groups):
        d = values.shape[1]
        sums = np.zeros((n_groups, d, d))
        outer = values[:, :, None] * values[:, None, :]   # (n, d, d)
        np.add.at(sums, labels, outer)
        return sums

    def bincount(self, labels, n_groups):
        return np.bincount(labels, minlength=n_groups).astype(float)


# ---------------------------------------------------------------------------
class TorchBackend(Backend):
    name = "torch"

    def __init__(self, device="cuda"):
        self.device = device

    def asarray(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        return x.to(self.device).float()

    def to_numpy(self, x):
        return x.detach().cpu().numpy()

    def log(self, x):
        return torch.log(x)

    def exp(self, x):
        return torch.exp(x)

    def clamp_min(self, x, lo=EPSILON):
        return torch.clamp(x, min=lo)

    def logsumexp(self, x, axis):
        return torch.logsumexp(x, dim=axis)

    def concat(self, parts, axis=-1):
        return torch.cat(parts, dim=axis)

    def isfinite_all(self, x):
        return bool(torch.isfinite(x).all())

    def eye(self, dim):
        return torch.eye(dim, device=self.device)

    def inv(self, a):
        return torch.linalg.inv(a)

    def slogdet(self, a):
        return torch.linalg.slogdet(a)[1]

    def cholesky_solve_mahalanobis(self, cov, diff):
        try:
            L = torch.linalg.cholesky(cov)
            log_det = 2 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)
            inv_cov = torch.cholesky_inverse(L)
        except Exception:
            inv_cov = torch.linalg.inv(cov)
            log_det = torch.linalg.slogdet(cov)[1]
        temp = torch.einsum("nkf,kfg->nkg", diff, inv_cov)
        mahal = torch.einsum("nkf,nkf->nk", temp, diff)
        return mahal, log_det

    def einsum(self, subscripts, *operands):
        return torch.einsum(subscripts, *operands)

    def kmeans_plusplus(self, points, n_clusters, verbose=0):
        n_samples, n_dims = points.shape
        centers = torch.empty(n_clusters, n_dims, device=self.device, dtype=points.dtype)
        centers[0] = points[torch.randint(0, n_samples, (1,), device=self.device).item()]
        closest_sq = torch.full((n_samples,), float("inf"), device=self.device)

        for i in range(1, n_clusters):
            d_sq = torch.cdist(points, centers[i - 1:i], p=2.0).squeeze(1) ** 2
            closest_sq = torch.minimum(closest_sq, d_sq)
            probs = closest_sq / closest_sq.sum()
            centers[i] = points[torch.multinomial(probs, 1).item()]
            if verbose > 1 and (i + 1) % 10 == 0:
                print(f"  Selected center {i + 1}/{n_clusters}")
        return centers

    def cluster_assignments(self, points, centers):
        return torch.argmin(torch.cdist(points, centers), dim=1)

    def group_sum(self, values, labels, n_groups):
        one_hot = F.one_hot(labels, num_classes=n_groups).to(values.dtype)
        return one_hot.T @ values

    def group_outer_sum(self, values, labels, n_groups):
        one_hot = F.one_hot(labels, num_classes=n_groups).to(values.dtype)
        # sum_n one_hot[n, k] * values[n, i] * values[n, j] -> (k, i, j)
        return torch.einsum("nk,ni,nj->kij", one_hot, values, values)

    def bincount(self, labels, n_groups):
        return torch.bincount(labels, minlength=n_groups).float()


NUMPY = NumpyBackend()


def torch_backend(device="cuda"):
    """Return a :class:`TorchBackend` bound to ``device``."""
    return TorchBackend(device)
