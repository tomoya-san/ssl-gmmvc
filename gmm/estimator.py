"""The EM estimator that drives any backend + covariance model.

:class:`JointGMM` is backend- and structure-agnostic. It owns the parts of a
joint GMM that don't depend on the covariance structure:

* the EM control loop (convergence on the log-likelihood rise rate),
* progress / final reporting and the optional log-likelihood CSV export,
* parameter validation (NaN / Inf / out-of-range weights),
* the voice-conversion interface (``convert`` / ``predict_responsibilities``),
* persistence (``save_model`` / ``load_model``).

The structure-specific math is delegated to a
:class:`~gmm.covariance.CovarianceModel`; the array ops to a
:class:`~gmm.backends.Backend`. The four public classes in :mod:`gmm.presets`
are just constructor presets over this class.
"""

from __future__ import annotations

import os
import warnings

import numpy as np


class JointGMM:
    """Joint-density GMM for source -> target feature conversion.

    Parameters
    ----------
    covariance : CovarianceModel
        The structure-specific math (full or cross-diagonal).
    backend : Backend
        The array backend (numpy or torch).
    verbose : int
        0 silent, 1 progress, 2 detailed.
    """

    def __init__(self, covariance, backend, verbose=1):
        self.cov = covariance
        self.b = backend
        self.verbose = verbose

    # convenience pass-throughs ------------------------------------------------
    @property
    def n_components(self):
        return self.cov.n_components

    @property
    def feature_dim(self):
        return self.cov.feature_dim

    @property
    def weights_(self):
        return self.cov.weights_

    @property
    def means_(self):
        return self.cov.means_

    # -- EM driver ---------------------------------------------------------
    def fit(self, XY, max_iter=100, tol=1e-6, log_likelihood_file=None):
        """Fit with EM on the concatenated source/target data.

        Parameters
        ----------
        XY : array-like of shape (n_samples, 2 * feature_dim)
            Concatenated ``[source | target]`` features (numpy or tensor).
        max_iter, tol : EM iteration cap and relative-rise convergence tolerance.
        log_likelihood_file : optional path for the per-sample LL history (CSV).

        Returns
        -------
        self
        """
        b = self.b
        joint = b.asarray(XY)
        n_samples = joint.shape[0]

        if self.verbose > 0:
            print(f"Starting {type(self).__name__} fitting with {self.n_components} components")
            print(f"Data shape: {n_samples} samples, {joint.shape[1]} features")
            print(f"Backend: {b.name}")
            print("-" * 50)

        self.cov.initialize(b, joint, self.verbose)

        prev_ll = float("-inf")
        ll_history = []
        converged = invalid = False
        rise_rate = None
        ll_per_sample = float("nan")

        for it in range(max_iter):
            try:
                resp, log_likelihood = self.cov.e_step(b, joint)
                if not b.isfinite_all(resp):
                    invalid = self._warn("responsibilities", it)
                    break

                self.cov.m_step(b, joint, resp)
                if self._invalid_parameters():
                    invalid = True
                    break

                log_likelihood = float(log_likelihood)
                if not np.isfinite(log_likelihood):
                    invalid = self._warn("log-likelihood", it)
                    break

                ll_per_sample = log_likelihood / n_samples
                ll_history.append(ll_per_sample)

                if prev_ll > float("-inf"):
                    rise_rate = abs((log_likelihood - prev_ll) / prev_ll) if prev_ll else 0.0
                    if log_likelihood < prev_ll - 1e-10 and self.verbose > 0:
                        warnings.warn(f"Log-likelihood decreased at iteration {it + 1}!")
                    if rise_rate < tol:
                        converged = True
                        if self.verbose > 0:
                            print(f"\n✓ Converged at iteration {it + 1} "
                                  f"(rise rate {rise_rate:.2e} < {tol:.2e})")
                        break

                self._report_iteration(it, ll_per_sample, rise_rate)
                prev_ll = log_likelihood

            except Exception as exc:
                if self.verbose > 0:
                    print(f"\n❌ Error at iteration {it + 1}: {exc}")
                invalid = True
                break

        self._report_final(invalid, converged, max_iter, rise_rate, ll_per_sample)
        if invalid:
            raise RuntimeError("GMM training failed due to invalid learning")

        self._save_log_likelihood_history(ll_history, log_likelihood_file)
        return self

    # -- conversion interface ----------------------------------------------
    def convert(self, X):
        """Map source features ``X`` to target features (forward X -> Y)."""
        return self.cov.convert(self.b, self.b.asarray(X))

    def predict_responsibilities(self, X):
        """Posterior p(k | x) from the source marginal, shape (n_samples, k)."""
        return self.cov.responsibilities(self.b, self.b.asarray(X))

    # -- persistence -------------------------------------------------------
    def save_model(self, filepath):
        """Save parameters (as numpy) to ``filepath`` (.npz)."""
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
            if self.verbose > 0:
                print(f"Directory '{dir_path}' ensured.")

        arrays = {name: self.b.to_numpy(v) for name, v in self.cov.named_params().items()}
        np.savez_compressed(
            filepath,
            n_components=self.n_components,
            feature_dim=self.feature_dim,
            **arrays,
        )
        if self.verbose > 0:
            print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load parameters previously written by :meth:`save_model`."""
        data = np.load(filepath)
        self.cov.n_components = int(data["n_components"])
        self.cov.feature_dim = int(data["feature_dim"])

        params = {name: self.b.asarray(data[name]) for name in self.cov.param_names()}
        self.cov.load_named_params(params)
        if self.verbose > 0:
            print(f"Model loaded from {filepath}")

    # -- validation & reporting --------------------------------------------
    def _invalid_parameters(self):
        for name, tensor in self.cov.named_params().items():
            if not self.b.isfinite_all(tensor):
                self._warn(name, None)
                return True
        w = self.b.to_numpy(self.weights_)
        if (w < 0).any() or (w > 1).any():
            if self.verbose > 0:
                print("\n⚠️  Invalid weight values detected (outside [0, 1])")
            return True
        return False

    def _warn(self, what, iteration):
        if self.verbose > 0:
            where = f" at iteration {iteration + 1}" if iteration is not None else ""
            print(f"\n⚠️  Invalid {what} detected{where}")
        return True

    def _report_iteration(self, it, ll_per_sample, rise_rate):
        if self.verbose <= 0:
            return
        if it != 0 and (it + 1) % 10 != 0 and self.verbose <= 1:
            return
        rate = f"{rise_rate:12.8f}" if rise_rate is not None else "N/A (first iteration)"
        print(f"Iteration {it + 1:3d} | Log-likelihood/sample: {ll_per_sample:12.6f} | Rise rate: {rate}")

    def _report_final(self, invalid, converged, max_iter, rise_rate, ll_per_sample):
        if self.verbose <= 0:
            return
        print("-" * 50)
        if invalid:
            print("❌ Training failed due to invalid learning!")
        elif not converged:
            print(f"⚠️  Did not converge within {max_iter} iterations")
            if rise_rate is not None:
                print(f"   Final rise rate: {rise_rate:.2e}")
        else:
            print("✓ Training completed successfully!")
            print(f"   Final log-likelihood/sample: {ll_per_sample:.6f}")
        if self.verbose > 1 and not invalid:
            print(f"\nComponent weights: {self.b.to_numpy(self.weights_)}")

    def _save_log_likelihood_history(self, ll_history, log_likelihood_file):
        if not log_likelihood_file:
            return
        try:
            output_dir = os.path.dirname(log_likelihood_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            np.savetxt(log_likelihood_file, np.array(ll_history), delimiter=",",
                       header="log_likelihood", comments="")
            if self.verbose > 0:
                print(f"Log-likelihood history saved to {log_likelihood_file}")
        except IOError as exc:
            print(f"Could not save log-likelihood file: {exc}")
