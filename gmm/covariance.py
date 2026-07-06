"""Covariance models: the GMM math, written once per structure.

A :class:`CovarianceModel` owns the distribution parameters (means, weights and
the structure-specific covariance terms) and implements the math that depends on
the covariance structure:

* ``initialize``        : k-means init of all parameters from the joint data.
* ``e_step``            : responsibilities + total log-likelihood.
* ``m_step``            : closed-form parameter update.
* ``source_log_prob``   : log p(x | k) under the source marginal (for p(k | x)).
* ``conditional_mean``  : E[y | x, k], the per-component linear map.
* ``named_params`` / ``load_named_params`` : persistence + validation.

Everything is expressed through a :class:`~gmm.backends.Backend`, so each model
runs unchanged on NumPy or PyTorch. The EM control flow lives in
:class:`~gmm.estimator.JointGMM`; this module is pure mathematics.
"""

from __future__ import annotations

import numpy as np

from .backends import EPSILON


class CovarianceModel:
    """Base class holding shared parameter state and the init scaffold.

    Concrete subclasses set ``means_`` / ``weights_`` and their covariance terms
    in :meth:`initialize` and implement the four math methods. ``PARAM_NAMES``
    lists the persisted/validated parameter attributes (without the trailing
    underscore) and drives both ``named_params`` and ``load_named_params``.
    """

    PARAM_NAMES = ()

    def __init__(self, n_components):
        self.n_components = n_components
        self.feature_dim = None
        self.means_ = None
        self.weights_ = None

    # -- shared k-means initialization -------------------------------------
    def initialize(self, b, joint, verbose=0):
        """Initialize all parameters from a single-pass k-means++ clustering.

        Means, weights and the structure-specific covariance terms are set to
        the moment estimates of the resulting clusters. Empty clusters fall back
        to the global statistics so no component starts degenerate.
        """
        n_samples, n_joint = joint.shape
        self.feature_dim = n_joint // 2

        centers = b.kmeans_plusplus(joint, self.n_components, verbose)
        labels = b.cluster_assignments(joint, centers)
        Nk = b.bincount(labels, self.n_components)
        Nk_safe = b.clamp_min(Nk)

        self.weights_ = Nk / n_samples
        self.means_ = b.group_sum(joint, labels, self.n_components) / Nk_safe[:, None]

        # Structure-specific covariance moments from the same clustering.
        self._init_covariances(b, joint, labels, Nk_safe)

        empty = [k for k in range(self.n_components) if float(Nk[k]) == 0]
        if empty:
            if verbose > 0:
                print(f"Warning: Found {len(empty)} empty clusters. Re-initializing them.")
            self._reinit_clusters(b, joint, empty)

    def _init_covariances(self, b, joint, labels, Nk_safe):
        """Set the structure-specific covariance terms from cluster moments."""
        raise NotImplementedError

    def _reinit_clusters(self, b, joint, empty):
        """Reset the given empty components to the global statistics."""
        raise NotImplementedError

    # -- helpers shared by both structures ---------------------------------
    def _split(self, vec):
        """Split a joint-dim array into its source and target halves."""
        d = self.feature_dim
        return vec[..., :d], vec[..., d:]

    # -- math interface (implemented by subclasses) ------------------------
    def e_step(self, b, joint):
        raise NotImplementedError

    def m_step(self, b, joint, resp):
        raise NotImplementedError

    def source_log_prob(self, b, X):
        """log p(x | k) for each component, shape (n_samples, k)."""
        raise NotImplementedError

    def conditional_mean(self, b, X):
        """E[y | x, k] for each component, shape (n_samples, k, feature_dim)."""
        raise NotImplementedError

    def named_params(self):
        """Dict of {name: array} for save/load and NaN/Inf validation."""
        return {name: getattr(self, name + "_") for name in self.PARAM_NAMES}

    def load_named_params(self, params):
        """Set parameter arrays from a {name: array} dict."""
        for name in self.PARAM_NAMES:
            setattr(self, name + "_", params[name])

    @classmethod
    def param_names(cls):
        return cls.PARAM_NAMES

    # -- conversion-facing math shared by all structures -------------------
    def responsibilities(self, b, X):
        """Posterior p(k | x) using the source marginal, shape (n_samples, k)."""
        log_prob = self.source_log_prob(b, X)
        log_prob = log_prob + b.log(self.weights_ + EPSILON)[None, :]
        log_prob = log_prob - b.logsumexp(log_prob, axis=1)[:, None]
        return b.exp(log_prob)

    def convert(self, b, X):
        """Map source ``X`` to target ``Y`` via responsibility-weighted means."""
        resp = self.responsibilities(b, X)                  # (n, k)
        cond = self.conditional_mean(b, X)                  # (n, k, d)
        return (resp[:, :, None] * cond).sum(axis=1)


# ---------------------------------------------------------------------------
class FullCovariance(CovarianceModel):
    """Dense joint covariance over the concatenated ``[x, y]`` vector.

    Stores ``means_`` (k, 2d), ``weights_`` (k,) and ``covariances_`` (k, 2d, 2d).
    The conditional map is ``E[y | x, k] = mu_y + Sigma_yx Sigma_xx^{-1} (x - mu_x)``.
    """

    PARAM_NAMES = ("means", "weights", "covariances")

    def _init_covariances(self, b, joint, labels, Nk_safe):
        # Full sample covariance per cluster: E[z z^T | k] - mu_k mu_k^T,
        # regularized to stay positive-definite.
        sum_outer = b.group_outer_sum(joint, labels, self.n_components)   # (k, 2d, 2d)
        second_moment = sum_outer / Nk_safe[:, None, None]
        outer_mean = self.means_[:, :, None] * self.means_[:, None, :]    # (k, 2d, 2d)
        cov = second_moment - outer_mean
        self.covariances_ = cov + self._reg(b, cov.shape[-1])

    def _reinit_clusters(self, b, joint, empty):
        global_mean = joint.mean(axis=0)
        centered = joint - global_mean
        global_cov = b.einsum("ni,nj->ij", centered, centered) / joint.shape[0]
        global_cov = global_cov + self._reg(b, global_cov.shape[-1])[0]
        for k in empty:
            self.means_[k] = global_mean
            self.covariances_[k] = global_cov

    def _reg(self, b, dim):
        return EPSILON * b.eye(dim)[None, :, :]

    def e_step(self, b, joint):
        n, d = joint.shape
        diff = joint[:, None, :] - self.means_[None, :, :]            # (n, k, 2d)
        cov = self.covariances_ + self._reg(b, d)
        mahal, log_det = b.cholesky_solve_mahalanobis(cov, diff)      # (n, k), (k,)

        log_prob = -0.5 * (mahal + log_det[None, :] + d * np.log(2 * np.pi))
        weighted = log_prob + b.log(self.weights_ + EPSILON)[None, :]

        ll = b.logsumexp(weighted, axis=1)                           # (n,)
        resp = b.exp(weighted - ll[:, None])                         # (n, k)
        return resp, ll.sum()

    def m_step(self, b, joint, resp):
        n = joint.shape[0]
        Nk = resp.sum(axis=0) + EPSILON                              # (k,)

        self.weights_ = Nk / n
        self.means_ = b.einsum("nk,nf->kf", resp, joint) / Nk[:, None]

        diff = joint[:, None, :] - self.means_[None, :, :]           # (n, k, 2d)
        weighted_diff = diff * (resp ** 0.5)[:, :, None]
        covs = b.einsum("nkf,nkg->kfg", weighted_diff, weighted_diff) / Nk[:, None, None]
        self.covariances_ = covs + self._reg(b, covs.shape[-1])

    def source_log_prob(self, b, X):
        mu_x, _ = self._split(self.means_)
        cov_xx = self.covariances_[:, :self.feature_dim, :self.feature_dim]
        reg_cov = cov_xx + self._reg(b, self.feature_dim)
        inv_cov = b.inv(reg_cov)
        log_det = b.slogdet(reg_cov)

        diff = X[:, None, :] - mu_x[None, :, :]
        temp = b.einsum("nkf,kfg->nkg", diff, inv_cov)
        mahal = b.einsum("nkf,nkf->nk", temp, diff)
        return -0.5 * (mahal + log_det[None, :] + self.feature_dim * np.log(2 * np.pi))

    def conditional_mean(self, b, X):
        mu_x, mu_y = self._split(self.means_)
        cov_xx = self.covariances_[:, :self.feature_dim, :self.feature_dim]
        cov_yx = self.covariances_[:, self.feature_dim:, :self.feature_dim]
        inv_cov = b.inv(cov_xx + self._reg(b, self.feature_dim))

        A = cov_yx @ inv_cov                                        # (k, d, d)
        diff = X[:, None, :] - mu_x[None, :, :]                     # (n, k, d)
        return mu_y[None, :, :] + b.einsum("nkf,kgf->nkg", diff, A)


# ---------------------------------------------------------------------------
class CrossDiagCovariance(CovarianceModel):
    """Joint covariance with diagonal blocks ``[[Sigma_xx, Sigma_xy], [.., Sigma_yy]]``.

    Every block is diagonal, so they are stored as length-``d`` vectors:
    ``diagonal_covariances_`` (k, 2d) holds Sigma_xx and Sigma_yy stacked;
    ``cross_covariances_`` (k, d) holds the diagonal Sigma_xy. This is far
    cheaper than a full covariance while still modelling the per-dimension
    source/target correlation that drives conversion.

    Schur complements let the precision and log-determinant be computed
    elementwise::

        S_A = Sigma_xx - Sigma_xy^2 / Sigma_yy
        S_D = Sigma_yy - Sigma_xy^2 / Sigma_xx
        log|Sigma| = log|Sigma_xx| + log|S_D|
    """

    PARAM_NAMES = ("means", "weights", "diagonal_covariances", "cross_covariances")

    def _init_covariances(self, b, joint, labels, Nk_safe):
        # Diagonal blocks: per-dimension variance E[z^2 | k] - mu_k^2.
        sum_sq = b.group_sum(joint ** 2, labels, self.n_components)
        diag = (sum_sq / Nk_safe[:, None]) - self.means_ ** 2
        self.diagonal_covariances_ = b.clamp_min(diag)

        # Cross block: per-dimension covariance E[x y | k] - mu_x mu_y.
        X_feat, Y_feat = self._split(joint)
        sum_xy = b.group_sum(X_feat * Y_feat, labels, self.n_components)
        mu_x, mu_y = self._split(self.means_)
        self.cross_covariances_ = (sum_xy / Nk_safe[:, None]) - mu_x * mu_y

    def _reinit_clusters(self, b, joint, empty):
        X_feat, Y_feat = self._split(joint)
        global_mean = joint.mean(axis=0)
        global_diag = ((joint - global_mean) ** 2).mean(axis=0)
        gm_x, gm_y = self._split(global_mean)
        global_cross = ((X_feat - gm_x) * (Y_feat - gm_y)).mean(axis=0)
        for k in empty:
            self.means_[k] = global_mean
            self.diagonal_covariances_[k] = global_diag
            self.cross_covariances_[k] = global_cross

    def _precision(self, b):
        """Return ``(var_x, var_y, prec_x, prec_y, prec_xy, schur_D)`` (all (k, d))."""
        var_x, var_y = self._split(self.diagonal_covariances_)
        cov_xy = self.cross_covariances_
        schur_A = b.clamp_min(var_x - cov_xy ** 2 / var_y)
        schur_D = b.clamp_min(var_y - cov_xy ** 2 / var_x)
        prec_x = 1.0 / schur_A
        prec_y = 1.0 / schur_D
        prec_xy = -cov_xy / (var_x * schur_D)
        return var_x, var_y, prec_x, prec_y, prec_xy, schur_D

    def e_step(self, b, joint):
        n, d = joint.shape
        var_x, _, prec_x, prec_y, prec_xy, schur_D = self._precision(b)

        X_x, X_y = self._split(joint)
        mu_x, mu_y = self._split(self.means_)
        diff_x = X_x[:, None, :] - mu_x[None, :, :]                 # (n, k, d)
        diff_y = X_y[:, None, :] - mu_y[None, :, :]

        mahal = (diff_x ** 2 * prec_x[None, :, :]).sum(axis=2)
        mahal = mahal + (diff_y ** 2 * prec_y[None, :, :]).sum(axis=2)
        mahal = mahal + 2 * (diff_x * diff_y * prec_xy[None, :, :]).sum(axis=2)

        log_det = b.log(var_x).sum(axis=1) + b.log(schur_D).sum(axis=1)   # (k,)
        log_prob = -0.5 * (mahal + d * np.log(2 * np.pi) + log_det[None, :])
        weighted = log_prob + b.log(self.weights_ + EPSILON)[None, :]

        ll = b.logsumexp(weighted, axis=1)
        resp = b.exp(weighted - ll[:, None])
        return resp, ll.sum()

    def m_step(self, b, joint, resp):
        n = joint.shape[0]
        Nk = resp.sum(axis=0)
        Nk_safe = b.clamp_min(Nk)

        self.weights_ = Nk / n
        self.means_ = b.einsum("nk,nf->kf", resp, joint) / Nk_safe[:, None]

        X_x, X_y = self._split(joint)
        mu_x, mu_y = self._split(self.means_)
        diff_x = X_x[:, None, :] - mu_x[None, :, :]
        diff_y = X_y[:, None, :] - mu_y[None, :, :]

        var_x = (resp[:, :, None] * diff_x ** 2).sum(axis=0) / Nk_safe[:, None]
        var_y = (resp[:, :, None] * diff_y ** 2).sum(axis=0) / Nk_safe[:, None]
        self.diagonal_covariances_ = b.clamp_min(b.concat([var_x, var_y], axis=1))
        self.cross_covariances_ = (resp[:, :, None] * diff_x * diff_y).sum(axis=0) / Nk_safe[:, None]

    def source_log_prob(self, b, X):
        mu_x, _ = self._split(self.means_)
        var_x, _ = self._split(self.diagonal_covariances_)
        inv_var_x = 1.0 / b.clamp_min(var_x)

        diff = X[:, None, :] - mu_x[None, :, :]
        log_prob = -0.5 * (diff ** 2 * inv_var_x[None, :, :]).sum(axis=2)
        log_prob = log_prob - 0.5 * b.log(2 * np.pi * var_x).sum(axis=1)[None, :]
        return log_prob

    def conditional_mean(self, b, X):
        mu_x, mu_y = self._split(self.means_)
        var_x, _ = self._split(self.diagonal_covariances_)
        gain = self.cross_covariances_ / b.clamp_min(var_x)        # (k, d)
        diff = X[:, None, :] - mu_x[None, :, :]
        return mu_y[None, :, :] + gain[None, :, :] * diff


# ---------------------------------------------------------------------------
class SharedCovariance(CovarianceModel):
    """Dense joint covariance *tied* (shared) across all components.

    Stores ``means_`` (k, 2d) and ``weights_`` (k,) per component, but a single
    ``covariance_`` (2d, 2d) shared by every component. The conditional map
    ``A = Sigma_yx Sigma_xx^{-1}`` is therefore one global matrix::

        E[y | x, k] = mu_y_k + A (x - mu_x_k)

    so conversion is a global affine map modulated only by the per-component
    mean offsets. This is the classic tied-covariance EV-GMM: far fewer
    parameters than :class:`FullCovariance` (one 2d x 2d matrix instead of k),
    which helps when per-speaker adaptation data is scarce. Inverting and taking
    the log-determinant of that one matrix once per step also makes it cheaper
    than the k-batched Cholesky solve :class:`FullCovariance` runs.
    """

    PARAM_NAMES = ("means", "weights", "covariance")

    def _reg(self, b, dim):
        return EPSILON * b.eye(dim)

    def _init_covariances(self, b, joint, labels, Nk_safe):
        # Pooled within-cluster scatter of the k-means init, tied over clusters:
        # (sum_n z z^T - sum_k Nk mu_k mu_k^T) / N. PSD by construction, then
        # regularized to stay positive-definite.
        n = joint.shape[0]
        gram = b.einsum("ni,nj->ij", joint, joint)                   # (2d, 2d)
        mean_outer = b.einsum("k,ki,kj->ij", Nk_safe, self.means_, self.means_)
        cov = (gram - mean_outer) / n
        self.covariance_ = cov + self._reg(b, cov.shape[-1])

    def _reinit_clusters(self, b, joint, empty):
        # Only the means are per-component; the covariance is shared, so empty
        # clusters just take the global mean.
        global_mean = joint.mean(axis=0)
        for k in empty:
            self.means_[k] = global_mean

    def e_step(self, b, joint):
        n, d = joint.shape
        cov = self.covariance_ + self._reg(b, d)
        inv_cov = b.inv(cov)                                         # (2d, 2d)
        log_det = b.slogdet(cov)                                     # scalar

        diff = joint[:, None, :] - self.means_[None, :, :]           # (n, k, 2d)
        temp = b.einsum("nkf,fg->nkg", diff, inv_cov)
        mahal = b.einsum("nkf,nkf->nk", temp, diff)                  # (n, k)

        log_prob = -0.5 * (mahal + log_det + d * np.log(2 * np.pi))
        weighted = log_prob + b.log(self.weights_ + EPSILON)[None, :]

        ll = b.logsumexp(weighted, axis=1)
        resp = b.exp(weighted - ll[:, None])
        return resp, ll.sum()

    def m_step(self, b, joint, resp):
        n = joint.shape[0]
        Nk = resp.sum(axis=0) + EPSILON                             # (k,)

        self.weights_ = Nk / n
        self.means_ = b.einsum("nk,nf->kf", resp, joint) / Nk[:, None]

        # Tied ML covariance: (1/N) sum_n sum_k resp[n,k] (z-mu_k)(z-mu_k)^T.
        diff = joint[:, None, :] - self.means_[None, :, :]          # (n, k, 2d)
        weighted_diff = diff * (resp ** 0.5)[:, :, None]
        cov = b.einsum("nkf,nkg->fg", weighted_diff, weighted_diff) / n
        self.covariance_ = cov + self._reg(b, cov.shape[-1])

    def source_log_prob(self, b, X):
        d = self.feature_dim
        mu_x, _ = self._split(self.means_)
        reg_cov = self.covariance_[:d, :d] + self._reg(b, d)
        inv_cov = b.inv(reg_cov)
        log_det = b.slogdet(reg_cov)

        diff = X[:, None, :] - mu_x[None, :, :]                      # (n, k, d)
        temp = b.einsum("nkf,fg->nkg", diff, inv_cov)
        mahal = b.einsum("nkf,nkf->nk", temp, diff)
        return -0.5 * (mahal + log_det + d * np.log(2 * np.pi))

    def conditional_mean(self, b, X):
        d = self.feature_dim
        mu_x, mu_y = self._split(self.means_)
        cov_xx = self.covariance_[:d, :d]
        cov_yx = self.covariance_[d:, :d]
        A = cov_yx @ b.inv(cov_xx + self._reg(b, d))                 # (d, d), global

        diff = X[:, None, :] - mu_x[None, :, :]                      # (n, k, d)
        return mu_y[None, :, :] + b.einsum("nkf,gf->nkg", diff, A)
