import os
import numpy as np
import torch

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from scipy.stats import multivariate_normal

EPSILON = 1e-6

import torch
import torch.nn.functional as F
import os
import numpy as np
import warnings

import torch
import torch.nn.functional as F
import os
import numpy as np
import warnings

class FullJointGMMGPU:
    def __init__(self, n_components, device='cuda', verbose=1):
        self.n_components = n_components
        self.device = device
        self.verbose = verbose  # 0: silent, 1: progress, 2: detailed
        
    def fit(self, X, max_iter=100, tol=1e-6):
        X = X.to(self.device)
        n_samples, n_features = X.shape
        self.feature_dim = n_features // 2  # Store feature dimension for convert function
        
        if self.verbose > 0:
            print(f"Starting GMM fitting with {self.n_components} components")
            print(f"Data shape: {n_samples} samples, {n_features} features")
            print(f"Device: {self.device}")
            print("-" * 50)
        
        # Initialize with k-means++
        self.means_ = self._kmeans_plusplus(X, self.n_components)
        # Initialize with identity matrices
        self.covariances_ = torch.eye(n_features, device=self.device).unsqueeze(0).repeat(self.n_components, 1, 1)
        self.weights_ = torch.ones(self.n_components, device=self.device) / self.n_components
        
        log_likelihood_old = None
        converged = False
        invalid_learning = False
        rise_rate = None
        
        for iteration in range(max_iter):
            try:
                # E-step
                resp = self._e_step(X)
                
                # Check for invalid responsibilities
                if self._check_invalid_values(resp, "responsibilities"):
                    invalid_learning = True
                    break
                
                # M-step
                self._m_step(X, resp)
                
                # Check for invalid parameters after M-step
                if self._check_invalid_parameters():
                    invalid_learning = True
                    break
                
                # Check convergence
                log_likelihood = self._compute_log_likelihood(X)
                
                # Check for invalid log-likelihood
                if torch.isnan(log_likelihood) or torch.isinf(log_likelihood):
                    if self.verbose > 0:
                        print(f"\n⚠️  Invalid log-likelihood detected at iteration {iteration + 1}")
                    invalid_learning = True
                    break
                
                # Calculate rise rate
                if log_likelihood_old is not None:
                    # Only calculate rise rate after first iteration
                    rise_rate = abs((log_likelihood - log_likelihood_old) / abs(log_likelihood_old) if log_likelihood_old != 0 else 0)
                    
                    # Check for decreasing likelihood (shouldn't happen in EM)
                    if log_likelihood < log_likelihood_old - 1e-10:
                        if self.verbose > 0:
                            warnings.warn(f"Log-likelihood decreased at iteration {iteration + 1}! "
                                        f"({log_likelihood:.6f} < {log_likelihood_old:.6f})")
                            if self.verbose > 1:
                                print("   Possible causes: numerical instability, component collapse, or regularization effects")
                    
                    # Check convergence
                    if rise_rate < tol:
                        converged = True
                        if self.verbose > 0:
                            print(f"\n✓ Converged at iteration {iteration + 1} (rise rate {rise_rate:.2e} < {tol:.2e})")
                        break
                else:
                    # First iteration - no rise rate yet
                    rise_rate = None
                
                # Verbose output
                if self.verbose > 0:
                    if iteration == 0 or (iteration + 1) % 10 == 0 or self.verbose > 1:
                        if rise_rate is not None:
                            print(f"Iteration {iteration + 1:3d} | Log-likelihood: {log_likelihood:12.6f} | Rise rate: {rise_rate:12.8f}")
                        else:
                            print(f"Iteration {iteration + 1:3d} | Log-likelihood: {log_likelihood:12.6f} | Rise rate: N/A (first iteration)")
                    
                log_likelihood_old = log_likelihood
                
            except Exception as e:
                if self.verbose > 0:
                    print(f"\n❌ Error at iteration {iteration + 1}: {str(e)}")
                invalid_learning = True
                break
        
        # Final status report
        if self.verbose > 0:
            print("-" * 50)
            if invalid_learning:
                print("❌ Training failed due to invalid learning!")
                print("   Possible causes:")
                print("   - Singular covariance matrices")
                print("   - Numerical instability")
                print("   - Data contains NaN or Inf values")
                print("   - Too many components for the data")
            elif not converged:
                print(f"⚠️  Did not converge within {max_iter} iterations")
                if rise_rate is not None:
                    print(f"   Final rise rate: {rise_rate:.2e}")
            else:
                print(f"✓ Training completed successfully!")
                print(f"   Final log-likelihood: {log_likelihood:.6f}")
            
            # Print component weights if verbose > 1
            if self.verbose > 1 and not invalid_learning:
                print(f"\nComponent weights: {self.weights_.cpu().numpy()}")
                
        if invalid_learning:
            raise RuntimeError("GMM training failed due to invalid learning")
            
        return self
    
    def _check_invalid_values(self, tensor, name="tensor"):
        """Check for NaN or Inf values in a tensor"""
        if torch.isnan(tensor).any():
            if self.verbose > 0:
                print(f"\n⚠️  NaN values detected in {name}")
            return True
        if torch.isinf(tensor).any():
            if self.verbose > 0:
                print(f"\n⚠️  Inf values detected in {name}")
            return True
        return False
    
    def _check_invalid_parameters(self):
        """Check if model parameters are valid"""
        # Check means
        if self._check_invalid_values(self.means_, "means"):
            return True
        
        # Check weights
        if self._check_invalid_values(self.weights_, "weights"):
            return True
        
        # Check if weights are valid probabilities
        if (self.weights_ < 0).any() or (self.weights_ > 1).any():
            if self.verbose > 0:
                print(f"\n⚠️  Invalid weight values detected (outside [0,1])")
            return True
        
        # Check covariances
        if self._check_invalid_values(self.covariances_, "covariances"):
            return True
        
        # Check for singular covariances
        for k in range(self.n_components):
            try:
                # Check if covariance is positive definite
                eigenvalues = torch.linalg.eigvalsh(self.covariances_[k])
                min_abs_eigenvalue = abs(eigenvalues).min()
                
                if min_abs_eigenvalue < 1e-10:
                    if self.verbose > 1:
                        print(f"\n⚠️  Component {k} has near-singular covariance (min eigenvalue: {min_abs_eigenvalue:.2e})")
                    # Don't fail immediately, regularization might fix it
                    
            except Exception as e:
                if self.verbose > 0:
                    print(f"\n⚠️  Error checking covariance matrix for component {k}: {str(e)}")
                return True
        
        return False
    
    def _kmeans_plusplus(self, X, k):
        """Initialize means using k-means++ (optimized)"""
        n_samples = X.shape[0]
        centers = [X[torch.randint(n_samples, (1,))].squeeze()]
        
        if self.verbose > 1:
            print("Initializing with k-means++...")
        
        for i in range(1, k):
            # Vectorized distance computation to all centers at once
            centers_tensor = torch.stack(centers)  # (num_centers, n_features)
            # Compute distances from all points to all centers
            dists = torch.cdist(X.unsqueeze(0), centers_tensor.unsqueeze(0)).squeeze(0)  # (n_samples, num_centers)
            # Get minimum distance to nearest center for each point
            min_dists = dists.min(dim=1)[0]
            
            probs = min_dists / min_dists.sum()
            cumprobs = probs.cumsum(0)
            r = torch.rand(1, device=self.device)
            idx = (cumprobs >= r).nonzero()[0]
            centers.append(X[idx].squeeze())
            
            if self.verbose > 1:
                print(f"  Selected center {i+1}/{k}")
        
        return torch.stack(centers)
    
    def _e_step(self, X):
        """Compute responsibilities using full covariance (vectorized)"""
        n_samples = X.shape[0]
        
        # Compute differences for all components at once
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)  # (n_samples, n_components, n_features)
        
        # Add regularization for numerical stability
        reg_covariances = self.covariances_ + 1e-6 * torch.eye(self.covariances_.shape[-1], device=self.device).unsqueeze(0)
        
        # Batch compute inverse and log determinant
        try:
            # Try Cholesky decomposition first (more stable)
            L = torch.linalg.cholesky(reg_covariances)
            log_dets = 2 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)
            
            # Use Cholesky to solve (more stable than inverse)
            # We still compute inverse for the Mahalanobis distance
            inv_covs = torch.cholesky_inverse(L)
        except:
            # Fall back to standard inverse if Cholesky fails
            inv_covs = torch.linalg.inv(reg_covariances)
            log_dets = torch.linalg.slogdet(reg_covariances)[1]
            
            if self.verbose > 0:
                print(f"\n⚠️  Cholesky decomposition failed, using standard inverse")
        
        # Check for invalid log determinants
        if (log_dets < -1e10).any():
            if self.verbose > 1:
                print(f"\n⚠️  Very small determinants detected: min={log_dets.min():.2e}")
        
        # Vectorized Mahalanobis distance computation
        # diff: (n_samples, n_components, n_features)
        # inv_covs: (n_components, n_features, n_features)
        temp = torch.einsum('nkf,kfg->nkg', diff, inv_covs)  # (n_samples, n_components, n_features)
        mahal = torch.einsum('nkf,nkf->nk', temp, diff)  # (n_samples, n_components)
        
        # Compute log responsibilities
        log_resp = -0.5 * mahal - 0.5 * log_dets.unsqueeze(0)
        log_resp += self.weights_.log().unsqueeze(0)
        
        # Normalize
        log_resp -= torch.logsumexp(log_resp, dim=1, keepdim=True)
        return log_resp.exp()
    
    def _m_step(self, X, resp):
        """Update parameters with full covariance (vectorized)"""
        Nk = resp.sum(0) + 1e-6  # (n_components,)
        
        # Check for empty clusters
        if self.verbose > 1:
            min_Nk = Nk.min().item()
            if min_Nk < 1.0:
                print(f"  ⚠️  Component with very few points: min Nk = {min_Nk:.2e}")
        
        self.weights_ = Nk / X.shape[0]
        
        # Update means
        self.means_ = torch.einsum('nk,nf->kf', resp, X) / Nk.unsqueeze(1)
        
        # Vectorized covariance computation
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)  # (n_samples, n_components, n_features)
        
        # Weight differences by sqrt of responsibilities
        weighted_diff = diff * resp.unsqueeze(-1).sqrt()  # (n_samples, n_components, n_features)
        
        # Compute covariances for all components at once
        # Using einsum for batch matrix multiplication
        covs = torch.einsum('nkf,nkg->kfg', weighted_diff, weighted_diff) / Nk.unsqueeze(-1).unsqueeze(-1)
        
        # Add regularization
        reg_term = 1e-6 * torch.eye(covs.shape[-1], device=self.device).unsqueeze(0)
        self.covariances_ = covs + reg_term
    
    def _compute_log_likelihood(self, X):
        """Compute log likelihood for convergence check"""
        n_samples = X.shape[0]
        
        # Compute log probabilities more stably
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)
        
        # Use the same regularization as in E-step
        reg_covariances = self.covariances_ + 1e-6 * torch.eye(self.covariances_.shape[-1], device=self.device).unsqueeze(0)
        
        try:
            # Compute Cholesky decomposition for numerical stability
            L = torch.linalg.cholesky(reg_covariances)
            
            # Solve L @ L^T @ x = diff for x using forward substitution
            # This is more stable than computing inverse
            temp = torch.triangular_solve(diff.transpose(-2, -1), L, upper=False)[0]
            mahal = (temp ** 2).sum(dim=1).transpose(-2, -1)
            
            # Log determinant via Cholesky
            log_dets = 2 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)
            
        except:
            # Fall back to standard method if Cholesky fails
            inv_covs = torch.linalg.inv(reg_covariances)
            log_dets = torch.linalg.slogdet(reg_covariances)[1]
            temp = torch.einsum('nkf,kfg->nkg', diff, inv_covs)
            mahal = torch.einsum('nkf,nkf->nk', temp, diff)
        
        # Compute log probabilities
        log_prob = -0.5 * (mahal + log_dets.unsqueeze(0) + X.shape[1] * np.log(2 * np.pi))
        log_prob += self.weights_.log().unsqueeze(0)
        
        # Use log-sum-exp for numerical stability
        log_likelihood = torch.logsumexp(log_prob, dim=1).sum()
        
        return log_likelihood
    
    def convert(self, X):
        """
        Convert features using the learned GMM mapping (vectorized).
        For voice conversion: X->Y (forward conversion only)
        """
        X = X.to(self.device)
        n_samples = X.shape[0]
        
        # Split means and covariances into source and target parts
        mu_x = self.means_[:, :self.feature_dim]  # (n_components, feature_dim)
        mu_y = self.means_[:, self.feature_dim:]  # (n_components, feature_dim)
        
        # Split covariance matrices into blocks
        cov_xx = self.covariances_[:, :self.feature_dim, :self.feature_dim]  # (n_components, feature_dim, feature_dim)
        cov_yx = self.covariances_[:, self.feature_dim:, :self.feature_dim]  # (n_components, feature_dim, feature_dim)
        
        # Compute responsibilities based on source features (vectorized)
        diff_x = X.unsqueeze(1) - mu_x.unsqueeze(0)  # (n_samples, n_components, feature_dim)
        
        # Add regularization and compute inverses for all components at once
        reg_cov_xx = cov_xx + 1e-6 * torch.eye(cov_xx.shape[-1], device=self.device).unsqueeze(0)
        inv_cov_xx = torch.linalg.inv(reg_cov_xx)  # (n_components, feature_dim, feature_dim)
        log_det_xx = torch.linalg.slogdet(reg_cov_xx)[1]  # (n_components,)
        
        # Vectorized Mahalanobis distance
        temp = torch.einsum('nkf,kfg->nkg', diff_x, inv_cov_xx)
        mahal = torch.einsum('nkf,nkf->nk', temp, diff_x)
        
        # Compute log responsibilities
        log_resp = -0.5 * mahal - 0.5 * log_det_xx.unsqueeze(0)
        log_resp += self.weights_.log().unsqueeze(0)
        log_resp -= torch.logsumexp(log_resp, dim=1, keepdim=True)
        resps = log_resp.exp()  # (n_samples, n_components)
        
        # Compute transformation matrices for all components at once
        # A = Cov_yx @ Cov_xx^(-1)
        A = torch.einsum('kij,kjl->kil', cov_yx, inv_cov_xx)  # (n_components, feature_dim, feature_dim)
        
        # Compute conditional means for all components at once
        # Y = mu_y + (X - mu_x) @ A^T
        transformed = torch.einsum('nkf,kgf->nkg', diff_x, A)  # (n_samples, n_components, feature_dim)
        Y_components = mu_y.unsqueeze(0) + transformed  # (n_samples, n_components, feature_dim)
        
        # Weighted sum of component predictions
        Y_hat = torch.einsum('nk,nkf->nf', resps, Y_components)  # (n_samples, feature_dim)
        
        return Y_hat
    
    def predict_responsibilities(self, X):
        """
        Compute frame-wise responsibilities (posterior probabilities) using source features only.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Source speech features

        Returns
        -------
        responsibilities : torch.Tensor of shape (n_samples, n_components)
            Posterior probability p(k|x) for each frame and component
        """
        X = X.to(self.device)

        # Split means and covariances into source part only
        mu_x = self.means_[:, :self.feature_dim]  # (n_components, feature_dim)
        cov_xx = self.covariances_[:, :self.feature_dim, :self.feature_dim]  # (n_components, feature_dim, feature_dim)

        # Compute differences
        diff_x = X.unsqueeze(1) - mu_x.unsqueeze(0)  # (n_samples, n_components, feature_dim)

        # Add regularization and compute inverses
        reg_cov_xx = cov_xx + 1e-6 * torch.eye(cov_xx.shape[-1], device=self.device).unsqueeze(0)
        inv_cov_xx = torch.linalg.inv(reg_cov_xx)  # (n_components, feature_dim, feature_dim)
        log_det_xx = torch.linalg.slogdet(reg_cov_xx)[1]  # (n_components,)

        # Vectorized Mahalanobis distance
        temp = torch.einsum('nkf,kfg->nkg', diff_x, inv_cov_xx)
        mahal = torch.einsum('nkf,nkf->nk', temp, diff_x)  # (n_samples, n_components)

        # Compute log responsibilities
        log_resp = -0.5 * mahal - 0.5 * log_det_xx.unsqueeze(0)
        log_resp -= 0.5 * self.feature_dim * np.log(2 * np.pi)
        log_resp += self.weights_.log().unsqueeze(0)

        # Normalize using log-sum-exp
        log_resp -= torch.logsumexp(log_resp, dim=1, keepdim=True)
        responsibilities = log_resp.exp()  # (n_samples, n_components)

        return responsibilities

    def save_model(self, filepath):
        """Save model parameters to file"""
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
            if self.verbose > 0:
                print(f"Directory '{dir_path}' ensured.")

        # Move tensors to CPU for saving
        save_dict = {
            'n_components': self.n_components,
            'feature_dim': self.feature_dim,
            'means': self.means_.cpu().numpy(),
            'covariances': self.covariances_.cpu().numpy(),
            'weights': self.weights_.cpu().numpy(),
        }
        
        np.savez_compressed(filepath, **save_dict)
        if self.verbose > 0:
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model parameters from file"""
        data = np.load(filepath)
        
        self.n_components = int(data["n_components"])
        self.feature_dim = int(data["feature_dim"])
        self.means_ = torch.from_numpy(data["means"]).to(self.device).float()
        self.covariances_ = torch.from_numpy(data["covariances"]).to(self.device).float()
        self.weights_ = torch.from_numpy(data["weights"]).to(self.device).float()
        
        if self.verbose > 0:
            print(f"Model loaded from {filepath}")

class FullJointGMMCPU(GaussianMixture):
    def __init__(self, n_components=1, covariance_type='full', tol=1e-6,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='k-means++',
                 weights_init=None, means_init=None, precisions_init=None,
                 random_state=None, warm_start=False, verbose=1,
                 verbose_interval=1, feature_dim=1024):
        
        super().__init__(
            n_components=n_components,
            covariance_type=covariance_type,
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init,
            random_state=random_state,
            warm_start=warm_start,
            verbose=verbose,
            verbose_interval=verbose_interval
        )
        
        self.feature_dim = feature_dim
    
    def convert(self, X):
        """
        Convert source features to target features.

        The conversion formula for each component k is:
            E[y|x, k] = A_k @ x + b_k

        where:
            A_k = Σ_yx @ Σ_xx^{-1}  (projection matrix)
            b_k = μ_y - A_k @ μ_x   (bias vector)

        Parameters
        ----------
        X : np.array of shape (n_samples, feature_dim)
            Source speech features

        Returns
        -------
        Y_hat : np.array of shape (n_samples, feature_dim)
            Converted target speech features
        """
        mu_x = self.means_[:, :self.feature_dim]  # (n_components, feature_dim)
        mu_y = self.means_[:, self.feature_dim:]  # (n_components, feature_dim)

        cov_yx = self.covariances_[:, self.feature_dim:, :self.feature_dim]  # (n_components, feature_dim, feature_dim)
        var_x = self.covariances_[:, :self.feature_dim, :self.feature_dim]  # (n_components, feature_dim, feature_dim)
        inv_var_x = np.linalg.inv(var_x)  # (n_components, feature_dim, feature_dim)

        # Projection matrix: A_k = Σ_yx @ Σ_xx^{-1}, shape: (n_components, feature_dim, feature_dim)
        A = cov_yx @ inv_var_x

        # Bias vector: b_k = μ_y - A_k @ μ_x, shape: (n_components, feature_dim)
        b = mu_y - np.einsum('kij,kj->ki', A, mu_x)

        # Compute responsibilities
        resps = self.predict_responsibilities(X).T  # (n_components, n_samples)

        # Apply linear transformation: y_k = A_k @ x + b_k
        linear = np.einsum('kij,nj->kni', A, X) + b[:, None, :]  # (n_components, n_samples, feature_dim)

        # Weighted sum across components
        Y_hat = np.sum(linear * resps[:, :, None], axis=0)  # (n_samples, feature_dim)

        return Y_hat
    
    def save_model(self, filepath):
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
            print(f"Directory '{dir_path}' ensured.")
            
        np.savez_compressed(
            filepath,
            n_components = self.n_components,
            means = self.means_,
            covariances = self.covariances_,
            weights=self.weights_,
        )
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        data = np.load(filepath)
        self.n_components = data["n_components"]
        self.means_ = data["means"]
        self.covariances_ = data["covariances"]
        self.weights_ = data["weights"]

        print(f"Model loaded from {filepath}")

    def predict_responsibilities(self, X):
        """
        Compute frame-wise responsibilities (posterior probabilities) using source features only.

        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            Source speech features

        Returns
        -------
        responsibilities : np.array of shape (n_samples, n_components)
            Posterior probability p(k|x) for each frame and component
        """
        mu_x = self.means_[:, :self.feature_dim]  # (n_components, feature_dim)
        var_x = self.covariances_[:, :self.feature_dim, :self.feature_dim]  # (n_components, feature_dim, feature_dim)

        # Compute log probabilities using multivariate normal
        log_probs = np.array([
            multivariate_normal.logpdf(X, mean=mu_x[k], cov=var_x[k]) + np.log(self.weights_[k] + 1e-12)
            for k in range(self.n_components)
        ])  # (n_components, n_samples)

        # Normalize using log-sum-exp
        log_probs -= np.max(log_probs, axis=0, keepdims=True)
        probs = np.exp(log_probs)
        responsibilities = probs / np.sum(probs, axis=0, keepdims=True)  # (n_components, n_samples)

        return responsibilities.T  # (n_samples, n_components)


class CrossDiagJointGMM:
    def __init__(self, n_components=1):
        self.n_components = n_components
    
    def _initialize_parameters(self, X, Y):
        """
        Parameters
        X: np.array of shape (n_samples, n_features)
            Source speech samples
        Y: np.array of shape (n_samples, n_features) 
            Target speech samples
            
        Returns
        None
        """
        
        print("Initializing parameters")
        
        n_samples_source, n_features_source = X.shape
        n_samples_target, n_features_target = Y.shape
        
        if n_samples_source != n_samples_target:
            raise ValueError(f"Source and target samples must match in number, got {n_samples_source} and {n_samples_target}") 
        
        if n_features_source != n_features_target:
            raise ValueError(f"Source and target feature dimention must match.")
        
        self.n_features = n_features_source
        
        # Concatenate source and target features
        self.joint_features = np.hstack((X, Y)) # (n_samples, 2 * n_features)
        n_samples, self.n_features_joint = self.joint_features.shape
        
        # kmeans clustering for initializing clusters
        kmeans = KMeans(n_clusters=self.n_components, n_init=1, random_state=0)
        labels = kmeans.fit_predict(self.joint_features)
        
        # Calculate cluster counts (Nk)
        Nk = np.bincount(labels, minlength=self.n_components)
        Nk_safe = np.maximum(Nk, EPSILON)
        
        # Initialize weights
        self.weights_ = Nk / n_samples # (n_components, 1)
        
        # Initialize mean vector
        sums = np.zeros((self.n_components, self.n_features_joint))
        np.add.at(sums, labels, self.joint_features)
        self.means_ = sums / Nk_safe[:, np.newaxis] # (n_components, 2 * n_features)
        
        # Initialize diagonal covariances
        sum_sq = np.zeros((self.n_components, self.n_features_joint))
        np.add.at(sum_sq, labels, self.joint_features ** 2)
        diag_cov = (sum_sq / Nk_safe[:, np.newaxis]) - (self.means_ ** 2)
        self.diagonal_covariances_ = np.maximum(diag_cov, EPSILON) # (n_components, 2 * n_features)
        
        
        # Initialize cross covariance
        X_features, Y_features = self.joint_features[:, :self.n_features], self.joint_features[:, self.n_features:]
        XY_product = X_features * Y_features
        sum_xy = np.zeros((self.n_components, self.n_features))
        np.add.at(sum_xy, labels, XY_product)
        mu_x = self.means_[:, :self.n_features]
        mu_y = self.means_[:, self.n_features:]
        self.cross_covariances_ = (sum_xy / Nk_safe[:, np.newaxis]) - (mu_x * mu_y) # (n_components, n_features)
        
        # Handle potentially empty clusters by re-initializing their params to the global mean and covariance
        empty_clusters = np.where(Nk == 0)[0]
        if len(empty_clusters) > 0:
            print(f"Warning: Found {len(empty_clusters)} empty clusters. Re-initializing them.")
            global_mean = np.mean(self.joint_features, axis=0)
            global_diag_cov = np.var(self.joint_features, axis=0)
            global_cross_cov = np.mean((X_features - global_mean[:self.n_features]) * (Y_features - global_mean[self.n_features:]), axis=0)
            
            for k in empty_clusters:
                self.means_[k] = global_mean
                self.diagonal_covariances_[k] = global_diag_cov
                self.cross_covariances_[k] = global_cross_cov
        
        print("Parameter initialization completed.")
        
    def _e_step(self, joint_features):
        n_samples, d = joint_features.shape
        k = self.n_components
        d_half = d // 2
        
        x_slice = slice(0, d_half)
        y_slice = slice(d_half, d)
        
        d_sigma = self.diagonal_covariances_
        c_sigma = self.cross_covariances_
        
        # Σ_xx - Σ_xy Σ_yy^(-1) Σ_yx
        schur1 = np.maximum(d_sigma[:, x_slice] - (c_sigma ** 2) / d_sigma[:, y_slice], EPSILON) # (n_components, n_features)
        # Σ_yy - Σ_yx Σ_xx^(-1) Σ_xy
        schur2 = np.maximum(d_sigma[:, y_slice] - (c_sigma ** 2) / d_sigma[:, x_slice], EPSILON) # (n_components, n_features)
        
        # T^(-1) and S^(-1)
        d_prec = np.hstack([1. / schur1, 1. / schur2]) # (n_components, 2 * n_features)
        # - Σ_xx^(-1) Σ_xy S^(-1) 
        c_prec = -c_sigma / (d_sigma[:, x_slice] * schur2) # (n_components, n_features)
        
        mu = self.means_
        mu_x, mu_y = mu[:, x_slice], mu[:, y_slice] # (n_components, n_features)
        X = joint_features
        X_x, X_y = joint_features[:, x_slice], joint_features[:, y_slice] # (n_samples, n_features)

        # x^T A x + y^T D y
        X_sq = ((X**2)[None, :, :] * d_prec[:, None, :]).sum(axis=2, keepdims=True) # (n_components, n_samples, 1)
        
        # 2 x^T B y
        cross_term = 2 * ((X_x * X_y)[None, :, :] * c_prec[:, None, :]).sum(axis=2, keepdims=True) # (n_components, n_samples, 1)
        
        # -2 [x^T A μ_x + y^T D μ_y]
        mu_term_diag = -2 * (X[None, :, :] * (d_prec * mu)[:, None, :]).sum(axis=2, keepdims=True) # (n_components, n_samples, 1)
        
        # -2 [x^T B μ_y + y^T B μ_x]
        mu_term_cross = -2 * (X[None, :, :] * (np.hstack([mu_y, mu_x]) * np.hstack([c_prec, c_prec]))[:, None, :]).sum(axis=2, keepdims=True) # (n_components, n_samples, 1)
        
        # μ_x^T A μ_x + μ_y^T D μ_y
        mu_sq = (mu * mu * d_prec).sum(axis=1) # (n_components, 1)
        
        # 2 μ_x^T B μ_y
        mu_cross_term = 2 * (mu_x * mu_y * c_prec).sum(axis=1) # (n_components, 1)
       
        mahal_dists = X_sq + cross_term + mu_term_diag + mu_term_cross + (mu_sq + mu_cross_term)[:, None, None] # (n_components, n_samples, 1)
        mahal_dists = mahal_dists.squeeze(-1)  # (n_components, n_samples)



        # log det(Σ) using Schur complement formula:
        # log|Σ| = log|Σ_xx| + log|S_D| where S_D = Σ_yy - Σ_yx Σ_xx^{-1} Σ_xy = schur2
        # This is mathematically equivalent to log|Σ_yy| + log|S_A| but must use consistent pairs
        log_det_xx = np.log(d_sigma[:, x_slice]).sum(axis=1)
        log_det_schur2 = np.log(schur2).sum(axis=1)
        log_det = log_det_xx + log_det_schur2

        # log probability
        log_prob = -0.5 * (mahal_dists + d * np.log(2 * np.pi) + log_det[:, None])  # (n_components, n_samples)

        # 混合係数（重み）を加味
        weighted_log_prob = log_prob + np.log(self.weights_[:, None])  # (n_components, n_samples)

        # 総和での正規化：log-sum-exp trick
        log_likelihood = np.logaddexp.reduce(weighted_log_prob, axis=0)  # (n_samples, 1)
        responsibilities = np.exp(weighted_log_prob - log_likelihood[None, :])  # (n_components, n_samples)

        # 合計対数尤度（全データのスカラー）
        total_log_likelihood = log_likelihood.sum()

        return responsibilities, total_log_likelihood
        
    
    def _m_step(self, joint_features, responsibilities):
        n_samples, d = joint_features.shape
        k = self.n_components
        d_half = d // 2
        x_slice = slice(0, d_half)
        y_slice = slice(d_half, d)
        
        X = joint_features
        resp = responsibilities  # (n_components, n_samples)

        # Each component's total responsibility (Nk)
        Nk = resp.sum(axis=1)  # (n_components,)
        Nk_safe = np.maximum(Nk, EPSILON) # Use a safe version for division

        # Update means (mu)
        weighted_sum = resp @ X  # (n_components, n_features_joint)
        self.means_ = weighted_sum / Nk_safe[:, None]

        # Split into x and y parts
        X_x, X_y = X[:, x_slice], X[:, y_slice]
        mu_x, mu_y = self.means_[:, x_slice], self.means_[:, y_slice]

        # Update diagonal covariances (var[x] and var[y])
        diff_x = X_x[None, :, :] - mu_x[:, None, :]
        diff_y = X_y[None, :, :] - mu_y[:, None, :]

        weighted_diff_x2 = (resp[:, :, None] * diff_x ** 2).sum(axis=1)
        weighted_diff_y2 = (resp[:, :, None] * diff_y ** 2).sum(axis=1)

        diag_cov = np.hstack([
            weighted_diff_x2 / Nk_safe[:, None],
            weighted_diff_y2 / Nk_safe[:, None]
        ])

        # --- THIS IS THE CRITICAL FIX ---
        # Enforce a minimum variance to prevent collapse
        self.diagonal_covariances_ = np.maximum(diag_cov, EPSILON)

        # Update cross-covariance (cov[x, y])
        cross_cov = (resp[:, :, None] * (diff_x * diff_y)).sum(axis=1) / Nk_safe[:, None]
        self.cross_covariances_ = cross_cov

        # Update weights
        self.weights_ = Nk / n_samples


        print("M-step completed.")
    
    def fit(self, X, Y, n_iter=100, tol=1e-6, verbose=True, log_likelihood_file=None):
        self._initialize_parameters(X, Y)
        
        prev_ll = -np.inf
        log_likelihoods_per_sample = []
        
        for iteration in range(n_iter):
            # Eステップ：log-likelihoodとresponsibilitiesを更新
            responsibilities, log_likelihood = self._e_step(self.joint_features)
            
            # Normalize the log-likelihood
            n_samples = self.joint_features.shape[0]
            log_likelihood_per_sample = log_likelihood / n_samples
            
            log_likelihoods_per_sample.append(log_likelihood_per_sample)
            
            # Mステップ：パラメータを更新
            self._m_step(self.joint_features, responsibilities)

            if verbose:
                print(f"Iter {iteration + 1}, Log-Likelihood per sample: {log_likelihood_per_sample:.4f}")
            
            # 収束判定
            if prev_ll > -np.inf:
                rise_rate = (log_likelihood - prev_ll) / abs(prev_ll)
                if verbose:
                    print(f"Log-Likelihood rise rate: {rise_rate}")
                if rise_rate < tol:
                    if verbose:
                        print("Convergence reached.")
                    break
                
            prev_ll = log_likelihood

        if verbose:
            print("Training finished.")
            
        # Save log-likelihood history to CSV if a path is provided
        if log_likelihood_file:
            try:
                log_likelihoods_arr = np.array(log_likelihoods_per_sample)
                # Ensure the directory exists
                output_dir = os.path.dirname(log_likelihood_file)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                # Save the file with a header
                np.savetxt(log_likelihood_file, log_likelihoods_arr, delimiter=",", header="log_likelihood", comments="")
                if verbose:
                    print(f"Log-likelihood history saved to {log_likelihood_file}")
            except IOError as e:
                print(f"Could not save log-likelihood file: {e}")
            
    def convert(self, X):
        """
        Convert source speech features X to target speech features Y using trained joint GMM.

        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            Source speech features

        Returns
        -------
        Y_hat : np.array of shape (n_samples, n_features)
            Converted target speech features
        """
        n_samples, d = X.shape
        k = self.n_components
        
        x_slice = slice(0, d)
        y_slice = slice(d, 2 * d)

        mu = self.means_  # (k, 2d)
        mu_x, mu_y = mu[:, x_slice], mu[:, y_slice]
        
        var_x = self.diagonal_covariances_[:, x_slice]  # (k, d)
        cov_yx = self.cross_covariances_  # (k, d)

        # Responsibility: compute p(k | x)
        # Using only source dimension
        diff = X[None, :, :] - mu_x[:, None, :]  # (k, n_samples, d)
        inv_var_x = 1.0 / np.maximum(var_x, EPSILON)  # (k, d)
        
        log_prob = -0.5 * np.sum((diff ** 2) * inv_var_x[:, None, :], axis=2)  # (k, n_samples)
        log_prob -= 0.5 * np.sum(np.log(2 * np.pi * var_x), axis=1)[:, None]  # (k, n_samples)
        log_prob += np.log(self.weights_[:, None] + EPSILON)  # (k, n_samples)
        
        # Normalize using log-sum-exp
        log_sum = np.logaddexp.reduce(log_prob, axis=0)  # (n_samples,)
        responsibilities = np.exp(log_prob - log_sum[None, :])  # (k, n_samples)

        # Conditional mean computation for each component
        cov_yx_div_varx = cov_yx / np.maximum(var_x, EPSILON)  # (k, d)
        diff = X[None, :, :] - mu_x[:, None, :]  # (k, n_samples, d)
        linear = mu_y[:, None, :] + cov_yx_div_varx[:, None, :] * diff  # (k, n_samples, d)

        # Weighted sum across components
        Y_hat = np.sum(responsibilities[:, :, None] * linear, axis=0)  # (n_samples, d)
        return Y_hat
    
    def save_model(self, filepath):
        
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
            print(f"Directory '{dir_path}' ensured.")
            
        np.savez_compressed(
            filepath,
            means=self.means_,
            diag_cov=self.diagonal_covariances_,
            cross_cov=self.cross_covariances_,
            weights=self.weights_,
            n_features=self.n_features,
            n_components=self.n_components
        )
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        data = np.load(filepath)
        self.means_ = data['means']
        self.diagonal_covariances_ = data['diag_cov']
        self.cross_covariances_ = data['cross_cov']
        self.weights_ = data['weights']
        self.n_features = int(data['n_features'])
        self.n_components = int(data['n_components'])
        print(f"Model loaded from {filepath}")

    def predict_responsibilities(self, X):
        """
        Compute frame-wise responsibilities (posterior probabilities) using source features only.

        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            Source speech features

        Returns
        -------
        responsibilities : np.array of shape (n_samples, n_components)
            Posterior probability p(k|x) for each frame and component
        """
        mu_x = self.means_[:, :self.n_features]  # (n_components, n_features)
        var_x = self.diagonal_covariances_[:, :self.n_features]  # (n_components, n_features)

        diff = X[None, :, :] - mu_x[:, None, :]  # (n_components, n_samples, n_features)
        inv_var_x = 1.0 / np.maximum(var_x, EPSILON)  # (n_components, n_features)

        # Compute log probabilities (diagonal covariance)
        log_prob = -0.5 * np.sum((diff ** 2) * inv_var_x[:, None, :], axis=2)  # (n_components, n_samples)
        log_prob -= 0.5 * np.sum(np.log(2 * np.pi * var_x), axis=1)[:, None]  # (n_components, n_samples)
        log_prob += np.log(self.weights_[:, None] + EPSILON)  # (n_components, n_samples)

        # Normalize using log-sum-exp
        log_sum = np.logaddexp.reduce(log_prob, axis=0)  # (n_samples,)
        responsibilities = np.exp(log_prob - log_sum[None, :])  # (n_components, n_samples)

        return responsibilities.T  # (n_samples, n_components)


class CrossDiagJointGMMGPU:
    """
    GPU-accelerated Cross-Diagonal Joint GMM for voice conversion.

    This class uses a cross-diagonal covariance structure where the covariance
    matrix has the form:
        Σ = [[Σ_xx,  Σ_xy],
             [Σ_yx,  Σ_yy]]
    where Σ_xx, Σ_yy are diagonal matrices and Σ_xy = Σ_yx^T is also diagonal
    (element-wise covariance between corresponding dimensions).

    This structure is memory-efficient compared to full covariance while still
    capturing cross-covariance between source and target features.
    """

    def __init__(self, n_components=1, device='cuda', verbose=1):
        self.n_components = n_components
        self.device = device
        self.verbose = verbose  # 0: silent, 1: progress, 2: detailed

    def _kmeans_plusplus(self, joint_features, n_clusters):
        """Initialize means using k-means++ algorithm (GPU-accelerated)"""
        n_samples, feature_dims = joint_features.shape
        centers = torch.empty(n_clusters, feature_dims, device=self.device, dtype=joint_features.dtype)

        # Pick first center randomly
        first_idx = torch.randint(0, n_samples, (1,), device=self.device).item()
        centers[0] = joint_features[first_idx]

        # Initialize closest distances to infinity
        closest_dist_sq = torch.full((n_samples,), float('inf'), device=self.device)

        if self.verbose > 1:
            print("Initializing with k-means++...")

        # Choose remaining centers
        for i in range(1, n_clusters):
            # Update closest distances with the last added center using cdist
            dist_sq = torch.cdist(joint_features, centers[i-1:i], p=2.0).squeeze() ** 2
            closest_dist_sq = torch.minimum(closest_dist_sq, dist_sq)

            # Choose next center with probability proportional to squared distance
            probabilities = closest_dist_sq / torch.sum(closest_dist_sq)
            next_idx = torch.multinomial(probabilities, 1).item()
            centers[i] = joint_features[next_idx]

            if self.verbose > 1 and (i + 1) % 10 == 0:
                print(f"  Selected center {i+1}/{n_clusters}")

        return centers

    def _initialize_parameters(self, X, Y):
        """
        Initialize GMM parameters using k-means++ clustering.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Source speech samples
        Y : torch.Tensor of shape (n_samples, n_features)
            Target speech samples
        """
        if self.verbose > 0:
            print("Initializing parameters...")

        n_samples_source, n_features_source = X.shape
        n_samples_target, n_features_target = Y.shape

        if n_samples_source != n_samples_target:
            raise ValueError(f"Source and target samples must match in number, got {n_samples_source} and {n_samples_target}")

        if n_features_source != n_features_target:
            raise ValueError("Source and target feature dimensions must match.")

        self.n_features = n_features_source

        # Concatenate source and target features
        joint_features = torch.cat([X, Y], dim=1)  # (n_samples, 2 * n_features)
        n_samples, self.n_features_joint = joint_features.shape

        # K-means++ initialization for cluster centers
        centers = self._kmeans_plusplus(joint_features, self.n_components)

        # Assign samples to nearest centers
        distances = torch.cdist(joint_features, centers)  # (n_samples, n_components)
        labels = torch.argmin(distances, dim=1)  # (n_samples,)

        # Calculate cluster counts (Nk)
        Nk = torch.bincount(labels, minlength=self.n_components).float()
        Nk_safe = torch.clamp(Nk, min=EPSILON)

        # Initialize weights
        self.weights_ = Nk / n_samples  # (n_components,)

        # Initialize means using one-hot encoding for efficiency
        one_hot = F.one_hot(labels, num_classes=self.n_components).float()  # (n_samples, n_components)
        sums = one_hot.T @ joint_features  # (n_components, n_features_joint)
        self.means_ = sums / Nk_safe.unsqueeze(1)  # (n_components, 2 * n_features)

        # Initialize diagonal covariances
        # Compute E[X^2] - E[X]^2 per cluster
        sum_sq = one_hot.T @ (joint_features ** 2)  # (n_components, n_features_joint)
        diag_cov = (sum_sq / Nk_safe.unsqueeze(1)) - (self.means_ ** 2)
        self.diagonal_covariances_ = torch.clamp(diag_cov, min=EPSILON)  # (n_components, 2 * n_features)

        # Initialize cross covariance
        X_features = joint_features[:, :self.n_features]
        Y_features = joint_features[:, self.n_features:]
        XY_product = X_features * Y_features  # (n_samples, n_features)
        sum_xy = one_hot.T @ XY_product  # (n_components, n_features)
        mu_x = self.means_[:, :self.n_features]
        mu_y = self.means_[:, self.n_features:]
        self.cross_covariances_ = (sum_xy / Nk_safe.unsqueeze(1)) - (mu_x * mu_y)  # (n_components, n_features)

        # Handle empty clusters
        empty_clusters = torch.where(Nk == 0)[0]
        if len(empty_clusters) > 0:
            if self.verbose > 0:
                print(f"Warning: Found {len(empty_clusters)} empty clusters. Re-initializing them.")
            global_mean = joint_features.mean(dim=0)
            global_diag_cov = joint_features.var(dim=0)
            global_cross_cov = ((X_features - global_mean[:self.n_features]) *
                               (Y_features - global_mean[self.n_features:])).mean(dim=0)

            for k in empty_clusters:
                self.means_[k] = global_mean
                self.diagonal_covariances_[k] = global_diag_cov
                self.cross_covariances_[k] = global_cross_cov

        if self.verbose > 0:
            print("Parameter initialization completed.")

        return joint_features

    def _e_step(self, joint_features):
        """
        E-step: Compute responsibilities using cross-diagonal covariance structure.

        Uses the block matrix inverse formula for efficient computation:
        For Σ = [[A, B], [B^T, D]] where A, D are diagonal and B is diagonal:

        The precision matrix Σ^{-1} can be computed using Schur complements:
        - Schur complement of D: S_A = A - B D^{-1} B^T
        - Schur complement of A: S_D = D - B^T A^{-1} B

        For diagonal matrices, these simplify to element-wise operations.

        The log determinant is computed as:
        log|Σ| = log|A| + log|S_D| = log|D| + log|S_A|

        (Both formulas give the same result for valid covariance matrices)
        """
        n_samples, d = joint_features.shape
        k = self.n_components
        d_half = d // 2

        # Extract x and y parts
        X_x = joint_features[:, :d_half]  # (n_samples, d_half)
        X_y = joint_features[:, d_half:]  # (n_samples, d_half)

        # Get covariance parameters
        var_x = self.diagonal_covariances_[:, :d_half]  # (k, d_half) - Σ_xx diagonal
        var_y = self.diagonal_covariances_[:, d_half:]  # (k, d_half) - Σ_yy diagonal
        cov_xy = self.cross_covariances_  # (k, d_half) - Σ_xy diagonal

        # Compute Schur complements (element-wise for diagonal matrices)
        # S_A = Σ_xx - Σ_xy Σ_yy^{-1} Σ_yx = var_x - cov_xy^2 / var_y
        schur_A = torch.clamp(var_x - (cov_xy ** 2) / var_y, min=EPSILON)  # (k, d_half)
        # S_D = Σ_yy - Σ_yx Σ_xx^{-1} Σ_xy = var_y - cov_xy^2 / var_x
        schur_D = torch.clamp(var_y - (cov_xy ** 2) / var_x, min=EPSILON)  # (k, d_half)

        # Compute precision matrix elements (for diagonal cross-covariance structure)
        # Using block inverse formula:
        # Σ^{-1} = [[S_A^{-1}, -S_A^{-1} B D^{-1}], [-D^{-1} B^T S_A^{-1}, S_D^{-1} + D^{-1} B^T S_A^{-1} B D^{-1}]]
        #
        # For our simplified structure (diagonal blocks):
        # prec_xx = 1/S_A (diagonal)
        # prec_yy = 1/S_D (diagonal)
        # prec_xy = -cov_xy / (var_x * S_D) = -cov_xy / (var_y * S_A) (diagonal)

        prec_x = 1.0 / schur_A  # (k, d_half)
        prec_y = 1.0 / schur_D  # (k, d_half)
        prec_xy = -cov_xy / (var_x * schur_D)  # (k, d_half)

        # Get means
        mu_x = self.means_[:, :d_half]  # (k, d_half)
        mu_y = self.means_[:, d_half:]  # (k, d_half)

        # Compute differences from means
        diff_x = X_x.unsqueeze(1) - mu_x.unsqueeze(0)  # (n_samples, k, d_half)
        diff_y = X_y.unsqueeze(1) - mu_y.unsqueeze(0)  # (n_samples, k, d_half)

        # Compute Mahalanobis distance: (x-μ)^T Σ^{-1} (x-μ)
        # = diff_x^T prec_xx diff_x + diff_y^T prec_yy diff_y + 2 * diff_x^T prec_xy diff_y
        mahal_xx = (diff_x ** 2 * prec_x.unsqueeze(0)).sum(dim=2)  # (n_samples, k)
        mahal_yy = (diff_y ** 2 * prec_y.unsqueeze(0)).sum(dim=2)  # (n_samples, k)
        mahal_xy = 2 * (diff_x * diff_y * prec_xy.unsqueeze(0)).sum(dim=2)  # (n_samples, k)

        mahal_dists = mahal_xx + mahal_yy + mahal_xy  # (n_samples, k)

        # Log determinant: log|Σ| = log|Σ_xx| + log|S_D| (using Schur complement formula)
        # For diagonal Σ_xx: log|Σ_xx| = sum(log(var_x))
        # FIX: Use consistent formula - log|Σ| = log|Σ_xx| + log|S_D|
        log_det_xx = torch.log(var_x).sum(dim=1)  # (k,)
        log_det_schur_D = torch.log(schur_D).sum(dim=1)  # (k,)
        log_det = log_det_xx + log_det_schur_D  # (k,)

        # Log probability: log N(x|μ,Σ) = -0.5 * (mahal + d*log(2π) + log|Σ|)
        log_prob = -0.5 * (mahal_dists + d * np.log(2 * np.pi) + log_det.unsqueeze(0))  # (n_samples, k)

        # Add log mixture weights
        weighted_log_prob = log_prob + torch.log(self.weights_ + EPSILON).unsqueeze(0)  # (n_samples, k)

        # Normalize using log-sum-exp for numerical stability
        log_likelihood = torch.logsumexp(weighted_log_prob, dim=1)  # (n_samples,)
        responsibilities = torch.exp(weighted_log_prob - log_likelihood.unsqueeze(1))  # (n_samples, k)

        # Total log-likelihood
        total_log_likelihood = log_likelihood.sum()

        return responsibilities.T, total_log_likelihood  # (k, n_samples), scalar

    def _m_step(self, joint_features, responsibilities):
        """
        M-step: Update GMM parameters given responsibilities.

        Parameters
        ----------
        joint_features : torch.Tensor of shape (n_samples, 2*n_features)
        responsibilities : torch.Tensor of shape (n_components, n_samples)
        """
        n_samples, d = joint_features.shape
        d_half = d // 2

        resp = responsibilities  # (k, n_samples)

        # Extract x and y parts
        X_x = joint_features[:, :d_half]  # (n_samples, d_half)
        X_y = joint_features[:, d_half:]  # (n_samples, d_half)

        # Each component's total responsibility (Nk)
        Nk = resp.sum(dim=1)  # (k,)
        Nk_safe = torch.clamp(Nk, min=EPSILON)

        # Update weights
        self.weights_ = Nk / n_samples

        # Update means: μ_k = Σ_n r_{nk} x_n / N_k
        weighted_sum = resp @ joint_features  # (k, d)
        self.means_ = weighted_sum / Nk_safe.unsqueeze(1)

        # Get updated means
        mu_x = self.means_[:, :d_half]  # (k, d_half)
        mu_y = self.means_[:, d_half:]  # (k, d_half)

        # Compute differences (broadcasting for efficiency)
        diff_x = X_x.unsqueeze(1) - mu_x.unsqueeze(0)  # (n_samples, k, d_half)
        diff_y = X_y.unsqueeze(1) - mu_y.unsqueeze(0)  # (n_samples, k, d_half)

        # Update diagonal covariances
        # Var_k[x] = Σ_n r_{nk} (x_n - μ_k)^2 / N_k
        weighted_diff_x2 = (resp.T.unsqueeze(2) * diff_x ** 2).sum(dim=0)  # (k, d_half)
        weighted_diff_y2 = (resp.T.unsqueeze(2) * diff_y ** 2).sum(dim=0)  # (k, d_half)

        diag_cov = torch.cat([
            weighted_diff_x2 / Nk_safe.unsqueeze(1),
            weighted_diff_y2 / Nk_safe.unsqueeze(1)
        ], dim=1)  # (k, d)

        # Enforce minimum variance
        self.diagonal_covariances_ = torch.clamp(diag_cov, min=EPSILON)

        # Update cross-covariances
        # Cov_k[x,y] = Σ_n r_{nk} (x_n - μ_k)(y_n - μ_k) / N_k
        cross_cov = (resp.T.unsqueeze(2) * diff_x * diff_y).sum(dim=0) / Nk_safe.unsqueeze(1)  # (k, d_half)
        self.cross_covariances_ = cross_cov

        if self.verbose > 1:
            print("M-step completed.")

    def _check_invalid_values(self, tensor, name="tensor"):
        """Check for NaN or Inf values in a tensor"""
        if torch.isnan(tensor).any():
            if self.verbose > 0:
                print(f"\n⚠️  NaN values detected in {name}")
            return True
        if torch.isinf(tensor).any():
            if self.verbose > 0:
                print(f"\n⚠️  Inf values detected in {name}")
            return True
        return False

    def _check_invalid_parameters(self):
        """Check if model parameters are valid"""
        if self._check_invalid_values(self.means_, "means"):
            return True
        if self._check_invalid_values(self.weights_, "weights"):
            return True
        if self._check_invalid_values(self.diagonal_covariances_, "diagonal_covariances"):
            return True
        if self._check_invalid_values(self.cross_covariances_, "cross_covariances"):
            return True

        # Check if weights are valid probabilities
        if (self.weights_ < 0).any() or (self.weights_ > 1).any():
            if self.verbose > 0:
                print(f"\n⚠️  Invalid weight values detected (outside [0,1])")
            return True

        return False

    def fit(self, X, max_iter=100, tol=1e-6, log_likelihood_file=None):
        """
        Fit the GMM model to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, 2*n_features)
            Concatenated source and target speech samples (can be numpy array or torch tensor)
        max_iter : int, default=100
            Maximum number of EM iterations
        tol : float, default=1e-6
            Convergence tolerance (relative change in log-likelihood)
        log_likelihood_file : str, optional
            Path to save log-likelihood history

        Returns
        -------
        self
        """
        # Convert to torch tensors if needed
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()

        # Move to device
        X = X.to(self.device)

        n_samples, n_features = X.shape
        self.feature_dim = n_features // 2  # Store feature dimension for convert function

        if self.verbose > 0:
            print(f"Starting Cross-Diagonal Joint GMM fitting with {self.n_components} components")
            print(f"Data shape: {n_samples} samples, {n_features} features")
            print(f"Device: {self.device}")
            print("-" * 50)

        # Initialize parameters (split X into source and target parts)
        X_source = X[:, :self.feature_dim]
        Y_target = X[:, self.feature_dim:]
        joint_features = self._initialize_parameters(X_source, Y_target)

        prev_ll = float('-inf')
        log_likelihoods_per_sample = []
        converged = False
        invalid_learning = False
        rise_rate = None

        for iteration in range(max_iter):
            try:
                # E-step
                responsibilities, log_likelihood = self._e_step(joint_features)

                # Check for invalid values
                if self._check_invalid_values(responsibilities, "responsibilities"):
                    invalid_learning = True
                    break

                # M-step
                self._m_step(joint_features, responsibilities)

                # Check parameters
                if self._check_invalid_parameters():
                    invalid_learning = True
                    break

                # Check for invalid log-likelihood
                if torch.isnan(log_likelihood) or torch.isinf(log_likelihood):
                    if self.verbose > 0:
                        print(f"\n⚠️  Invalid log-likelihood detected at iteration {iteration + 1}")
                    invalid_learning = True
                    break

                # Normalize log-likelihood
                n_samples = joint_features.shape[0]
                log_likelihood_per_sample = log_likelihood.item() / n_samples
                log_likelihoods_per_sample.append(log_likelihood_per_sample)

                # Calculate rise rate and check convergence
                if prev_ll > float('-inf'):
                    rise_rate = abs((log_likelihood.item() - prev_ll) / abs(prev_ll)) if prev_ll != 0 else 0

                    if log_likelihood.item() < prev_ll - 1e-10:
                        if self.verbose > 0:
                            warnings.warn(f"Log-likelihood decreased at iteration {iteration + 1}!")

                    if rise_rate < tol:
                        converged = True
                        if self.verbose > 0:
                            print(f"\n✓ Converged at iteration {iteration + 1} (rise rate {rise_rate:.2e} < {tol:.2e})")
                        break

                # Verbose output
                if self.verbose > 0:
                    if iteration == 0 or (iteration + 1) % 10 == 0 or self.verbose > 1:
                        if rise_rate is not None:
                            print(f"Iteration {iteration + 1:3d} | Log-likelihood/sample: {log_likelihood_per_sample:12.6f} | Rise rate: {rise_rate:12.8f}")
                        else:
                            print(f"Iteration {iteration + 1:3d} | Log-likelihood/sample: {log_likelihood_per_sample:12.6f} | Rise rate: N/A (first iteration)")

                prev_ll = log_likelihood.item()

            except Exception as e:
                if self.verbose > 0:
                    print(f"\n❌ Error at iteration {iteration + 1}: {str(e)}")
                invalid_learning = True
                break

        # Final status report
        if self.verbose > 0:
            print("-" * 50)
            if invalid_learning:
                print("❌ Training failed due to invalid learning!")
            elif not converged:
                print(f"⚠️  Did not converge within {max_iter} iterations")
                if rise_rate is not None:
                    print(f"   Final rise rate: {rise_rate:.2e}")
            else:
                print(f"✓ Training completed successfully!")
                print(f"   Final log-likelihood/sample: {log_likelihood_per_sample:.6f}")

            if self.verbose > 1 and not invalid_learning:
                print(f"\nComponent weights: {self.weights_.cpu().numpy()}")

        if invalid_learning:
            raise RuntimeError("GMM training failed due to invalid learning")

        # Save log-likelihood history
        if log_likelihood_file:
            try:
                log_likelihoods_arr = np.array(log_likelihoods_per_sample)
                output_dir = os.path.dirname(log_likelihood_file)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                np.savetxt(log_likelihood_file, log_likelihoods_arr, delimiter=",",
                          header="log_likelihood", comments="")
                if self.verbose > 0:
                    print(f"Log-likelihood history saved to {log_likelihood_file}")
            except IOError as e:
                print(f"Could not save log-likelihood file: {e}")

        return self

    def convert(self, X):
        """
        Convert source speech features X to target speech features Y.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Source speech features

        Returns
        -------
        Y_hat : torch.Tensor of shape (n_samples, n_features)
            Converted target speech features
        """
        # Convert to torch tensor if needed
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        X = X.to(self.device)

        n_samples = X.shape[0]

        # Get parameters using stored feature_dim
        mu_x = self.means_[:, :self.feature_dim]  # (k, feature_dim)
        mu_y = self.means_[:, self.feature_dim:]  # (k, feature_dim)
        var_x = self.diagonal_covariances_[:, :self.feature_dim]  # (k, feature_dim)
        cov_xy = self.cross_covariances_  # (k, feature_dim)

        # Compute responsibilities based on source features only
        # p(k|x) ∝ π_k N(x|μ_x^k, Σ_xx^k)
        diff = X.unsqueeze(1) - mu_x.unsqueeze(0)  # (n_samples, k, feature_dim)
        inv_var_x = 1.0 / torch.clamp(var_x, min=EPSILON)  # (k, feature_dim)

        # Log probability under each component's marginal distribution
        log_prob = -0.5 * (diff ** 2 * inv_var_x.unsqueeze(0)).sum(dim=2)  # (n_samples, k)
        log_prob -= 0.5 * torch.log(2 * np.pi * var_x).sum(dim=1).unsqueeze(0)  # (n_samples, k)
        log_prob += torch.log(self.weights_ + EPSILON).unsqueeze(0)  # (n_samples, k)

        # Normalize using log-sum-exp
        log_sum = torch.logsumexp(log_prob, dim=1)  # (n_samples,)
        responsibilities = torch.exp(log_prob - log_sum.unsqueeze(1))  # (n_samples, k)

        # Compute conditional mean for each component
        # E[y|x,k] = μ_y^k + Σ_yx^k (Σ_xx^k)^{-1} (x - μ_x^k)
        # For diagonal: E[y|x,k] = μ_y + (cov_xy / var_x) * (x - μ_x)
        cov_xy_div_varx = cov_xy / torch.clamp(var_x, min=EPSILON)  # (k, feature_dim)
        diff = X.unsqueeze(1) - mu_x.unsqueeze(0)  # (n_samples, k, feature_dim)
        conditional_means = mu_y.unsqueeze(0) + cov_xy_div_varx.unsqueeze(0) * diff  # (n_samples, k, feature_dim)

        # Weighted sum across components
        Y_hat = (responsibilities.unsqueeze(2) * conditional_means).sum(dim=1)  # (n_samples, feature_dim)

        return Y_hat

    def save_model(self, filepath):
        """Save model parameters to file"""
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
            if self.verbose > 0:
                print(f"Directory '{dir_path}' ensured.")

        np.savez_compressed(
            filepath,
            means=self.means_.cpu().numpy(),
            diag_cov=self.diagonal_covariances_.cpu().numpy(),
            cross_cov=self.cross_covariances_.cpu().numpy(),
            weights=self.weights_.cpu().numpy(),
            n_features=self.n_features,
            n_components=self.n_components,
            feature_dim=self.feature_dim
        )
        if self.verbose > 0:
            print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load model parameters from file"""
        data = np.load(filepath)
        self.means_ = torch.from_numpy(data['means']).to(self.device).float()
        self.diagonal_covariances_ = torch.from_numpy(data['diag_cov']).to(self.device).float()
        self.cross_covariances_ = torch.from_numpy(data['cross_cov']).to(self.device).float()
        self.weights_ = torch.from_numpy(data['weights']).to(self.device).float()
        self.n_features = int(data['n_features'])
        self.n_components = int(data['n_components'])
        # Load feature_dim, with fallback to n_features for backward compatibility
        self.feature_dim = int(data['feature_dim']) if 'feature_dim' in data else self.n_features

        if self.verbose > 0:
            print(f"Model loaded from {filepath}")

    def predict_responsibilities(self, X):
        """
        Compute frame-wise responsibilities (posterior probabilities) using source features only.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Source speech features

        Returns
        -------
        responsibilities : torch.Tensor of shape (n_samples, n_components)
            Posterior probability p(k|x) for each frame and component
        """
        # Convert to torch tensor if needed
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        X = X.to(self.device)

        # Get parameters
        mu_x = self.means_[:, :self.feature_dim]  # (k, feature_dim)
        var_x = self.diagonal_covariances_[:, :self.feature_dim]  # (k, feature_dim)

        # Compute differences
        diff = X.unsqueeze(1) - mu_x.unsqueeze(0)  # (n_samples, k, feature_dim)
        inv_var_x = 1.0 / torch.clamp(var_x, min=EPSILON)  # (k, feature_dim)

        # Compute log probabilities (diagonal covariance)
        log_prob = -0.5 * (diff ** 2 * inv_var_x.unsqueeze(0)).sum(dim=2)  # (n_samples, k)
        log_prob -= 0.5 * torch.log(2 * np.pi * var_x).sum(dim=1).unsqueeze(0)  # (n_samples, k)
        log_prob += torch.log(self.weights_ + EPSILON).unsqueeze(0)  # (n_samples, k)

        # Normalize using log-sum-exp
        log_sum = torch.logsumexp(log_prob, dim=1)  # (n_samples,)
        responsibilities = torch.exp(log_prob - log_sum.unsqueeze(1))  # (n_samples, k)

        return responsibilities
    

class SharedGMMGPU:
    """
    GPU-accelerated Shared (Tied) Covariance GMM for voice conversion.

    This class uses a tied covariance structure where all components share
    the same covariance matrix:
        Σ = [[Σ_xx,  Σ_xy],
             [Σ_yx,  Σ_yy]]

    All components share this single full covariance matrix, but have
    different means. This reduces the number of parameters compared to
    full covariance GMM while still capturing cross-covariance between
    source and target features.
    """

    def __init__(self, n_components, device='cuda', verbose=1):
        self.n_components = n_components
        self.device = device
        self.verbose = verbose  # 0: silent, 1: progress, 2: detailed

    def fit(self, X, max_iter=100, tol=1e-6):
        """
        Fit the GMM model to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, 2*n_features)
            Concatenated source and target speech samples
        max_iter : int, default=100
            Maximum number of EM iterations
        tol : float, default=1e-6
            Convergence tolerance (relative change in log-likelihood)

        Returns
        -------
        self
        """
        # Convert to torch tensor if needed
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()

        X = X.to(self.device)
        n_samples, n_features = X.shape
        self.feature_dim = n_features // 2

        if self.verbose > 0:
            print(f"Starting Shared (Tied) Covariance GMM fitting with {self.n_components} components")
            print(f"Data shape: {n_samples} samples, {n_features} features")
            print(f"Device: {self.device}")
            print("-" * 50)

        # Initialize with k-means++
        self.means_ = self._kmeans_plusplus(X, self.n_components)
        # Initialize with identity matrix (shared across all components)
        self.covariance_ = torch.eye(n_features, device=self.device)
        self.weights_ = torch.ones(self.n_components, device=self.device) / self.n_components

        log_likelihood_old = None
        converged = False
        invalid_learning = False
        rise_rate = None

        for iteration in range(max_iter):
            try:
                # E-step
                resp = self._e_step(X)

                # Check for invalid responsibilities
                if self._check_invalid_values(resp, "responsibilities"):
                    invalid_learning = True
                    break

                # M-step
                self._m_step(X, resp)

                # Check for invalid parameters after M-step
                if self._check_invalid_parameters():
                    invalid_learning = True
                    break

                # Check convergence
                log_likelihood = self._compute_log_likelihood(X)

                # Check for invalid log-likelihood
                if torch.isnan(log_likelihood) or torch.isinf(log_likelihood):
                    if self.verbose > 0:
                        print(f"\n⚠️  Invalid log-likelihood detected at iteration {iteration + 1}")
                    invalid_learning = True
                    break

                # Calculate rise rate
                if log_likelihood_old is not None:
                    rise_rate = abs((log_likelihood - log_likelihood_old) / abs(log_likelihood_old) if log_likelihood_old != 0 else 0)

                    if log_likelihood < log_likelihood_old - 1e-10:
                        if self.verbose > 0:
                            warnings.warn(f"Log-likelihood decreased at iteration {iteration + 1}! "
                                        f"({log_likelihood:.6f} < {log_likelihood_old:.6f})")

                    if rise_rate < tol:
                        converged = True
                        if self.verbose > 0:
                            print(f"\n✓ Converged at iteration {iteration + 1} (rise rate {rise_rate:.2e} < {tol:.2e})")
                        break
                else:
                    rise_rate = None

                # Verbose output
                if self.verbose > 0:
                    if iteration == 0 or (iteration + 1) % 10 == 0 or self.verbose > 1:
                        if rise_rate is not None:
                            print(f"Iteration {iteration + 1:3d} | Log-likelihood: {log_likelihood:12.6f} | Rise rate: {rise_rate:12.8f}")
                        else:
                            print(f"Iteration {iteration + 1:3d} | Log-likelihood: {log_likelihood:12.6f} | Rise rate: N/A (first iteration)")

                log_likelihood_old = log_likelihood

            except Exception as e:
                if self.verbose > 0:
                    print(f"\n❌ Error at iteration {iteration + 1}: {str(e)}")
                invalid_learning = True
                break

        # Final status report
        if self.verbose > 0:
            print("-" * 50)
            if invalid_learning:
                print("❌ Training failed due to invalid learning!")
                print("   Possible causes:")
                print("   - Singular covariance matrices")
                print("   - Numerical instability")
                print("   - Data contains NaN or Inf values")
                print("   - Too many components for the data")
            elif not converged:
                print(f"⚠️  Did not converge within {max_iter} iterations")
                if rise_rate is not None:
                    print(f"   Final rise rate: {rise_rate:.2e}")
            else:
                print(f"✓ Training completed successfully!")
                print(f"   Final log-likelihood: {log_likelihood:.6f}")

            if self.verbose > 1 and not invalid_learning:
                print(f"\nComponent weights: {self.weights_.cpu().numpy()}")

        if invalid_learning:
            raise RuntimeError("GMM training failed due to invalid learning")

        return self

    def _check_invalid_values(self, tensor, name="tensor"):
        """Check for NaN or Inf values in a tensor"""
        if torch.isnan(tensor).any():
            if self.verbose > 0:
                print(f"\n⚠️  NaN values detected in {name}")
            return True
        if torch.isinf(tensor).any():
            if self.verbose > 0:
                print(f"\n⚠️  Inf values detected in {name}")
            return True
        return False

    def _check_invalid_parameters(self):
        """Check if model parameters are valid"""
        if self._check_invalid_values(self.means_, "means"):
            return True
        if self._check_invalid_values(self.weights_, "weights"):
            return True
        if (self.weights_ < 0).any() or (self.weights_ > 1).any():
            if self.verbose > 0:
                print(f"\n⚠️  Invalid weight values detected (outside [0,1])")
            return True
        if self._check_invalid_values(self.covariance_, "covariance"):
            return True

        # Check for singular covariance
        try:
            eigenvalues = torch.linalg.eigvalsh(self.covariance_)
            min_abs_eigenvalue = abs(eigenvalues).min()
            if min_abs_eigenvalue < 1e-10:
                if self.verbose > 1:
                    print(f"\n⚠️  Near-singular covariance (min eigenvalue: {min_abs_eigenvalue:.2e})")
        except Exception as e:
            if self.verbose > 0:
                print(f"\n⚠️  Error checking covariance matrix: {str(e)}")
            return True

        return False

    def _kmeans_plusplus(self, X, k):
        """Initialize means using k-means++ (optimized)"""
        n_samples = X.shape[0]
        centers = [X[torch.randint(n_samples, (1,))].squeeze()]

        if self.verbose > 1:
            print("Initializing with k-means++...")

        for i in range(1, k):
            centers_tensor = torch.stack(centers)
            dists = torch.cdist(X.unsqueeze(0), centers_tensor.unsqueeze(0)).squeeze(0)
            min_dists = dists.min(dim=1)[0]

            probs = min_dists / min_dists.sum()
            cumprobs = probs.cumsum(0)
            r = torch.rand(1, device=self.device)
            idx = (cumprobs >= r).nonzero()[0]
            centers.append(X[idx].squeeze())

            if self.verbose > 1:
                print(f"  Selected center {i+1}/{k}")

        return torch.stack(centers)

    def _e_step(self, X):
        """Compute responsibilities using shared (tied) covariance"""
        n_samples = X.shape[0]

        # Compute differences for all components
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)  # (n_samples, n_components, n_features)

        # Add regularization for numerical stability
        reg_covariance = self.covariance_ + 1e-6 * torch.eye(self.covariance_.shape[0], device=self.device)

        try:
            # Cholesky decomposition (more stable)
            L = torch.linalg.cholesky(reg_covariance)
            log_det = 2 * torch.log(torch.diagonal(L)).sum()
            inv_cov = torch.cholesky_inverse(L)
        except:
            # Fall back to standard inverse
            inv_cov = torch.linalg.inv(reg_covariance)
            log_det = torch.linalg.slogdet(reg_covariance)[1]
            if self.verbose > 0:
                print(f"\n⚠️  Cholesky decomposition failed, using standard inverse")

        # Vectorized Mahalanobis distance computation
        # diff: (n_samples, n_components, n_features)
        # inv_cov: (n_features, n_features) - shared across components
        temp = torch.einsum('nkf,fg->nkg', diff, inv_cov)  # (n_samples, n_components, n_features)
        mahal = torch.einsum('nkf,nkf->nk', temp, diff)  # (n_samples, n_components)

        # Compute log responsibilities
        log_resp = -0.5 * mahal - 0.5 * log_det
        log_resp += self.weights_.log().unsqueeze(0)

        # Normalize
        log_resp -= torch.logsumexp(log_resp, dim=1, keepdim=True)
        return log_resp.exp()

    def _m_step(self, X, resp):
        """Update parameters with shared (tied) covariance"""
        n_samples, n_features = X.shape
        Nk = resp.sum(0) + 1e-6  # (n_components,)

        if self.verbose > 1:
            min_Nk = Nk.min().item()
            if min_Nk < 1.0:
                print(f"  ⚠️  Component with very few points: min Nk = {min_Nk:.2e}")

        # Update weights
        self.weights_ = Nk / n_samples

        # Update means
        self.means_ = torch.einsum('nk,nf->kf', resp, X) / Nk.unsqueeze(1)

        # Update shared covariance (weighted average across all components)
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)  # (n_samples, n_components, n_features)

        # Compute weighted outer products and sum across components
        # For tied covariance: Σ = Σ_k Σ_n r_nk (x_n - μ_k)(x_n - μ_k)^T / N
        weighted_diff = diff * resp.unsqueeze(-1).sqrt()  # (n_samples, n_components, n_features)

        # Sum over samples and components
        cov = torch.einsum('nkf,nkg->fg', weighted_diff, weighted_diff) / n_samples

        # Add regularization
        reg_term = 1e-6 * torch.eye(n_features, device=self.device)
        self.covariance_ = cov + reg_term

    def _compute_log_likelihood(self, X):
        """Compute log likelihood for convergence check"""
        n_samples, n_features = X.shape

        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)

        reg_covariance = self.covariance_ + 1e-6 * torch.eye(n_features, device=self.device)

        try:
            L = torch.linalg.cholesky(reg_covariance)
            log_det = 2 * torch.log(torch.diagonal(L)).sum()
            inv_cov = torch.cholesky_inverse(L)
        except:
            inv_cov = torch.linalg.inv(reg_covariance)
            log_det = torch.linalg.slogdet(reg_covariance)[1]

        temp = torch.einsum('nkf,fg->nkg', diff, inv_cov)
        mahal = torch.einsum('nkf,nkf->nk', temp, diff)

        log_prob = -0.5 * (mahal + log_det + n_features * np.log(2 * np.pi))
        log_prob += self.weights_.log().unsqueeze(0)

        log_likelihood = torch.logsumexp(log_prob, dim=1).sum()

        return log_likelihood

    def convert(self, X):
        """
        Convert source features to target features using the learned GMM mapping.

        Parameters
        ----------
        X : array-like of shape (n_samples, feature_dim)
            Source speech features

        Returns
        -------
        Y_hat : torch.Tensor of shape (n_samples, feature_dim)
            Converted target speech features
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        X = X.to(self.device)

        # Split means into source and target parts
        mu_x = self.means_[:, :self.feature_dim]  # (n_components, feature_dim)
        mu_y = self.means_[:, self.feature_dim:]  # (n_components, feature_dim)

        # Split shared covariance into blocks
        cov_xx = self.covariance_[:self.feature_dim, :self.feature_dim]  # (feature_dim, feature_dim)
        cov_yx = self.covariance_[self.feature_dim:, :self.feature_dim]  # (feature_dim, feature_dim)

        # Add regularization and compute inverse of source covariance
        reg_cov_xx = cov_xx + 1e-6 * torch.eye(self.feature_dim, device=self.device)
        inv_cov_xx = torch.linalg.inv(reg_cov_xx)

        # Compute responsibilities based on source features
        resps = self.predict_responsibilities(X)  # (n_samples, n_components)

        # Projection matrix: A = Σ_yx @ Σ_xx^{-1} (shared across components)
        A = cov_yx @ inv_cov_xx  # (feature_dim, feature_dim)

        # Bias vector: b_k = μ_y - A @ μ_x (per component)
        b = mu_y - (A @ mu_x.T).T  # (n_components, feature_dim)

        # Apply linear transformation: y = A @ x + b_k
        linear_base = (A @ X.T).T  # (n_samples, feature_dim)
        linear = linear_base.unsqueeze(1) + b.unsqueeze(0)  # (n_samples, n_components, feature_dim)

        # Weighted sum across components
        Y_hat = torch.einsum('nk,nkf->nf', resps, linear)  # (n_samples, feature_dim)

        return Y_hat

    def predict_responsibilities(self, X):
        """
        Compute frame-wise responsibilities (posterior probabilities) using source features only.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Source speech features

        Returns
        -------
        responsibilities : torch.Tensor of shape (n_samples, n_components)
            Posterior probability p(k|x) for each frame and component
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        X = X.to(self.device)

        # Extract source means and covariance
        mu_x = self.means_[:, :self.feature_dim]  # (n_components, feature_dim)
        var_x = self.covariance_[:self.feature_dim, :self.feature_dim]  # (feature_dim, feature_dim)

        # Compute differences
        diff_x = X.unsqueeze(1) - mu_x.unsqueeze(0)  # (n_samples, n_components, feature_dim)

        # Add regularization and compute inverse
        reg_var_x = var_x + 1e-6 * torch.eye(self.feature_dim, device=self.device)

        try:
            L = torch.linalg.cholesky(reg_var_x)
            log_det_x = 2 * torch.log(torch.diagonal(L)).sum()
            inv_var_x = torch.cholesky_inverse(L)
        except:
            inv_var_x = torch.linalg.inv(reg_var_x)
            log_det_x = torch.linalg.slogdet(reg_var_x)[1]

        # Vectorized Mahalanobis distance (shared covariance)
        temp = torch.einsum('nkf,fg->nkg', diff_x, inv_var_x)
        mahal = torch.einsum('nkf,nkf->nk', temp, diff_x)  # (n_samples, n_components)

        # Compute log responsibilities
        log_resp = -0.5 * mahal - 0.5 * log_det_x
        log_resp -= 0.5 * self.feature_dim * np.log(2 * np.pi)
        log_resp += self.weights_.log().unsqueeze(0)

        # Normalize using log-sum-exp
        log_resp -= torch.logsumexp(log_resp, dim=1, keepdim=True)
        responsibilities = log_resp.exp()  # (n_samples, n_components)

        return responsibilities

    def save_model(self, filepath):
        """Save model parameters to file"""
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
            if self.verbose > 0:
                print(f"Directory '{dir_path}' ensured.")

        save_dict = {
            'n_components': self.n_components,
            'feature_dim': self.feature_dim,
            'means': self.means_.cpu().numpy(),
            'covariance': self.covariance_.cpu().numpy(),
            'weights': self.weights_.cpu().numpy(),
        }

        np.savez_compressed(filepath, **save_dict)
        if self.verbose > 0:
            print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load model parameters from file"""
        data = np.load(filepath)

        self.n_components = int(data["n_components"])
        self.feature_dim = int(data["feature_dim"])
        self.means_ = torch.from_numpy(data["means"]).to(self.device).float()
        self.covariance_ = torch.from_numpy(data["covariance"]).to(self.device).float()
        self.weights_ = torch.from_numpy(data["weights"]).to(self.device).float()

        if self.verbose > 0:
            print(f"Model loaded from {filepath}")


class SharedGMMCPU(GaussianMixture):
    def __init__(self, n_components=1, covariance_type='tied', tol=1e-6,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='k-means++',
                 weights_init=None, means_init=None, precisions_init=None,
                 random_state=None, warm_start=False, verbose=1,
                 verbose_interval=1, feature_dim=1024):

        super().__init__(
            n_components=n_components,
            covariance_type=covariance_type,
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init,
            random_state=random_state,
            warm_start=warm_start,
            verbose=verbose,
            verbose_interval=verbose_interval
        )

        self.feature_dim = feature_dim

    def convert(self, X):
        mu_x = self.means_[:, :self.feature_dim]  # (n_components, feature_dim)
        mu_y = self.means_[:, self.feature_dim:]  # (n_components, feature_dim)

        # With tied covariance, self.covariances_ is (2*feature_dim, 2*feature_dim)
        cov_yx = self.covariances_[self.feature_dim:, :self.feature_dim]  # (feature_dim, feature_dim)
        var_x = self.covariances_[:self.feature_dim, :self.feature_dim]  # (feature_dim, feature_dim)
        inv_var_x = np.linalg.inv(var_x)  # (feature_dim, feature_dim)

        # Projection matrix: A = Σ_yx @ Σ_xx^{-1}, shape: (feature_dim, feature_dim)
        # Shared across all components
        A = cov_yx @ inv_var_x

        # Bias vector: b_k = μ_y - A @ μ_x, shape: (n_components, feature_dim)
        b = mu_y - (A @ mu_x.T).T  # A @ mu_x.T gives (feature_dim, n_components), transpose to (n_components, feature_dim)

        # Compute responsibilities
        resps = self.predict_responsibilities(X).T  # (n_components, n_samples)

        # Apply linear transformation: y_k = A @ x + b_k
        # A @ X.T gives (feature_dim, n_samples), transpose to (n_samples, feature_dim)
        linear_base = (A @ X.T).T  # (n_samples, feature_dim)
        # Add component-specific biases: (n_components, n_samples, feature_dim)
        linear = linear_base[None, :, :] + b[:, None, :]

        # Weighted sum across components
        Y_hat = np.sum(linear * resps[:, :, None], axis=0)  # (n_samples, feature_dim)

        return Y_hat

    def save_model(self, filepath):
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
            print(f"Directory '{dir_path}' ensured.")

        np.savez_compressed(
            filepath,
            n_components=self.n_components,
            feature_dim=self.feature_dim,
            means=self.means_,
            covariances=self.covariances_,
            weights=self.weights_,
        )
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        data = np.load(filepath)
        self.n_components = int(data["n_components"])
        self.feature_dim = int(data["feature_dim"])
        self.means_ = data["means"]
        self.covariances_ = data["covariances"]
        self.weights_ = data["weights"]

        print(f"Model loaded from {filepath}")

    def predict_responsibilities(self, X):
        """
        Compute frame-wise responsibilities (posterior probabilities) using source features only.

        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            Source speech features

        Returns
        -------
        responsibilities : np.array of shape (n_samples, n_components)
            Posterior probability p(k|x) for each frame and component
        """
        mu_x = self.means_[:, :self.feature_dim]  # (n_components, feature_dim)
        # With tied covariance, var_x is shared: (feature_dim, feature_dim)
        var_x = self.covariances_[:self.feature_dim, :self.feature_dim]

        # Compute log probabilities using multivariate normal (same covariance for all components)
        log_probs = np.array([
            multivariate_normal.logpdf(X, mean=mu_x[k], cov=var_x) + np.log(self.weights_[k] + 1e-12)
            for k in range(self.n_components)
        ])  # (n_components, n_samples)

        # Normalize using log-sum-exp
        log_probs -= np.max(log_probs, axis=0, keepdims=True)
        probs = np.exp(log_probs)
        responsibilities = probs / np.sum(probs, axis=0, keepdims=True)  # (n_components, n_samples)

        return responsibilities.T  # (n_samples, n_components)