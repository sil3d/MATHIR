"""
Immunological Memory — Anomaly detection.
Detects novel inputs via distance threshold.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import NearestNeighbors


class ImmunologicalMemory(nn.Module):
    """
    Immunological memory: anomaly detection.
    
    Learns what "normal" looks like. Flags novel inputs.
    
    Capacity: configurable (default 100 samples)
    Update: on event
    Detection: distance threshold from memory bank
    """
    
    def __init__(self, capacity: int = 100, feature_dim: int = 272, threshold: float = 2.0):
        super().__init__()
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.threshold = threshold
        
        # Memory bank of "normal" patterns
        self.register_buffer("bank", torch.zeros(capacity, feature_dim))
        self.register_buffer("ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))
    
    def store(self, features: torch.Tensor) -> None:
        """
        Store features as "normal" patterns.
        """
        with torch.no_grad():
            idx = self.ptr % self.capacity
            self.bank[idx] = features.detach().mean(0)
            
            self.ptr = (self.ptr + 1) % self.capacity
            self.count = torch.minimum(self.count + 1, torch.tensor(self.capacity, dtype=torch.long))
    
    def recognize(self, features: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Detect anomalies.
        
        Args:
            features: [B, D] tensor to check
            
        Returns:
            [B, D] anomaly signal (features if anomaly, zeros if normal)
            or None if not enough data
        """
        count = self.count.item()
        if count < 10:
            return None
        
        dists = torch.cdist(features, self.bank[:count])
        min_dist = dists.min(dim=1)[0]
        anomaly = (min_dist > self.threshold).float().unsqueeze(-1)
        return anomaly * features
    
    def get_anomaly_score(self, features: torch.Tensor) -> torch.Tensor:
        """
        Get anomaly score (distance to nearest "normal" pattern).
        """
        count = self.count.item()
        if count < 10:
            return torch.zeros(features.size(0), device=features.device)
        dists = torch.cdist(features, self.bank[:count])
        return dists.min(dim=1)[0]
    
    def reset(self) -> None:
        """Reset immune memory."""
        self.bank.zero_()
        self.ptr = torch.tensor(0, dtype=torch.long)
        self.count = torch.tensor(0, dtype=torch.long)
    
    def get_usage(self) -> int:
        """Get number of patterns stored."""
        return self.count.item()


class MahalanobisImmunologicalMemory(nn.Module):
    """
    Immunological memory with Mahalanobis distance.

    Replaces Euclidean distance with covariance-weighted Mahalanobis.
    NP-optimal for Gaussian-distributed normal patterns (Theorem 4).

    Adaptive covariance:
        Sigma_t = (1 - gamma) * Sigma_{t-1} + gamma * (x - mu)(x - mu)^T
    """

    def __init__(self, capacity: int = 100, feature_dim: int = 272,
                 threshold: float = 2.0, ema_decay: float = 0.01,
                 regularization: float = 1e-4):
        super().__init__()
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.threshold = threshold
        self.ema_decay = ema_decay
        self.regularization = regularization

        # Memory bank
        self.register_buffer("bank", torch.zeros(capacity, feature_dim))
        self.register_buffer("ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

        # Running statistics for Mahalanobis
        self.register_buffer("running_mean", torch.zeros(feature_dim))
        self.register_buffer("running_cov", torch.eye(feature_dim))
        self.register_buffer("n_updates", torch.tensor(0, dtype=torch.long))

    def store(self, features: torch.Tensor) -> None:
        """Store features and update running statistics."""
        with torch.no_grad():
            x = features.detach().mean(0)

            # Store
            idx = self.ptr % self.capacity
            self.bank[idx] = x
            self.ptr = (self.ptr + 1) % self.capacity
            self.count = torch.minimum(
                self.count + 1,
                torch.tensor(self.capacity, dtype=torch.long, device=self.count.device),
            )

            # Update running statistics
            self.running_mean = (1 - self.ema_decay) * self.running_mean + self.ema_decay * x

            diff = (x - self.running_mean).unsqueeze(0)
            new_cov = diff.T @ diff
            self.running_cov = (1 - self.ema_decay) * self.running_cov + self.ema_decay * new_cov

            # Regularize
            self.running_cov = self.running_cov + self.regularization * torch.eye(
                self.feature_dim, device=self.running_cov.device
            )

            self.n_updates = self.n_updates + 1

    def mahalanobis_distance(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Mahalanobis distance from running mean."""
        diff = x - self.running_mean.unsqueeze(0)  # [B, D]
        cov_inv = torch.linalg.inv(self.running_cov)
        # D_M^2 = diff^T @ cov_inv @ diff
        left = diff @ cov_inv  # [B, D]
        dist_sq = (left * diff).sum(dim=-1)  # [B]
        return torch.sqrt(dist_sq.clamp(min=0))

    def recognize(self, features: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Detect anomalies using Mahalanobis distance.

        Returns features if anomaly, None if normal.
        """
        count = self.count.item() if torch.is_tensor(self.count) else self.count
        n_updates = self.n_updates.item() if torch.is_tensor(self.n_updates) else self.n_updates
        if count < 10 or n_updates < 10:
            return None

        with torch.no_grad():
            # Use minimum distance to bank OR Mahalanobis to mean
            dists_to_bank = torch.cdist(features, self.bank[:count])
            min_bank_dist = dists_to_bank.min(dim=1)[0]

            mahalanobis = self.mahalanobis_distance(features)

            # Combined: use the more sensitive
            dist = torch.minimum(min_bank_dist, mahalanobis)

            # Adaptive threshold based on chi-squared distribution
            # chi^2_{D, 0.95} ~ D + 2*sqrt(D) for large D
            chi2_threshold = self.feature_dim + 2 * (self.feature_dim ** 0.5)
            chi2_threshold = chi2_threshold / 10  # Scale to similar magnitude

            anomaly = (dist > max(self.threshold, chi2_threshold ** 0.5)).float().unsqueeze(-1)

        return anomaly * features

    def get_anomaly_score(self, features: torch.Tensor) -> torch.Tensor:
        """Get continuous anomaly score (not just binary)."""
        n_updates = self.n_updates.item() if torch.is_tensor(self.n_updates) else self.n_updates
        if n_updates < 10:
            return torch.zeros(features.size(0), device=features.device)
        return self.mahalanobis_distance(features)

    def get_stats(self) -> dict:
        """Get statistics."""
        count = self.count.item() if torch.is_tensor(self.count) else self.count
        n_updates = self.n_updates.item() if torch.is_tensor(self.n_updates) else self.n_updates
        return {
            "count": count,
            "n_updates": n_updates,
            "cov_trace": torch.diagonal(self.running_cov).sum().item(),
            "cov_condition": torch.linalg.cond(self.running_cov.unsqueeze(0)).item(),
        }


class EnsembleColdStartImmunologicalMemory(nn.Module):
    """
    Immunological memory with ensemble cold-start detector.

    Combines multiple lightweight detectors for the first ~10 samples
    when covariance matrix is not yet stable, then transitions to
    Mahalanobis-based detection.

    Ensemble detectors (cold-start):
        1. Z-score: mean + std based threshold
        2. Isolation Forest: sklearn IsolationForest
        3. One-Class SVM: sklearn OneClassSVM
        4. K-NN Density: distance to k nearest neighbors

    Voting: if 2+ detectors flag anomaly → anomaly
    """

    def __init__(
        self,
        capacity: int = 100,
        feature_dim: int = 272,
        threshold: float = 2.0,
        ema_decay: float = 0.01,
        regularization: float = 1e-4,
        cold_start_k: int = 5,
        voting_threshold: int = 2,
    ):
        super().__init__()
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.threshold = threshold
        self.ema_decay = ema_decay
        self.regularization = regularization
        self.cold_start_k = cold_start_k
        self.voting_threshold = voting_threshold

        # Core Mahalanobis memory (used after warmup)
        self.mahalanobis = MahalanobisImmunologicalMemory(
            capacity=capacity,
            feature_dim=feature_dim,
            threshold=threshold,
            ema_decay=ema_decay,
            regularization=regularization,
        )

        # Cold-start ensemble detectors
        self._zscore_mean: Optional[np.ndarray] = None
        self._zscore_std: float = 1.0
        self._iforest: Optional[IsolationForest] = None
        self._ocsvm: Optional[OneClassSVM] = None
        self._knn: Optional[NearestNeighbors] = None
        self._knn_data: Optional[np.ndarray] = None

        # Warmup threshold
        self._warmup_threshold = 10

    def store(self, features: torch.Tensor) -> None:
        """Store features and update all detectors."""
        with torch.no_grad():
            x = features.detach().mean(0)
            x_np = x.cpu().numpy()

            # Update Mahalanobis memory
            self.mahalanobis.store(features)

            # Update cold-start ensemble when we have enough samples
            count = self.mahalanobis.count.item()
            n_updates = self.mahalanobis.n_updates.item()

            if n_updates <= self._warmup_threshold:
                # Accumulate data for cold-start detectors
                if self._knn_data is None:
                    self._knn_data = x_np.reshape(1, -1)
                else:
                    self._knn_data = np.vstack([self._knn_data, x_np])

                # Re-fit detectors when we have enough samples
                if count >= 3:
                    self._fit_cold_start_detectors()

    def _fit_cold_start_detectors(self) -> None:
        """Fit/update cold-start ensemble detectors."""
        if self._knn_data is None or len(self._knn_data) < 3:
            return

        data = self._knn_data

        # 1. Z-score statistics
        self._zscore_mean = data.mean(axis=0)
        self._zscore_std = data.std(axis=0) + 1e-8

        # 2. Isolation Forest (fast, contamination=0.1 for sensitivity)
        self._iforest = IsolationForest(
            n_estimators=50,
            contamination=0.1,
            random_state=42,
            n_jobs=1,
        )
        self._iforest.fit(data)

        # 3. One-Class SVM (fast, nu=0.1)
        self._ocsvm = OneClassSVM(
            kernel="rbf",
            gamma="scale",
            nu=0.1,
        )
        self._ocsvm.fit(data)

        # 4. K-NN Density
        k = min(self.cold_start_k, len(data) - 1)
        if k >= 1:
            self._knn = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree")
            self._knn.fit(data)

    def _zscore_score(self, x: np.ndarray) -> np.ndarray:
        """Z-score based anomaly score (higher = more anomalous)."""
        if self._zscore_mean is None:
            return np.zeros(len(x))
        z = np.abs(x - self._zscore_mean) / self._zscore_std
        return z.mean(axis=1)

    def _zscore_anomaly(self, x: np.ndarray, threshold: float = 2.5) -> np.ndarray:
        """Z-score based anomaly flag."""
        scores = self._zscore_score(x)
        return (scores > threshold).astype(int)

    def _iforest_score(self, x: np.ndarray) -> np.ndarray:
        """Isolation Forest anomaly score (higher = more anomalous)."""
        if self._iforest is None:
            return np.zeros(len(x))
        # score_samples returns negative scores (more negative = more anomalous)
        scores = -self._iforest.score_samples(x)
        return scores

    def _iforest_anomaly(self, x: np.ndarray) -> np.ndarray:
        """Isolation Forest anomaly flag."""
        if self._iforest is None:
            return np.zeros(len(x))
        return self._iforest.predict(x) # 1=normal, -1=anomaly

    def _ocsvm_score(self, x: np.ndarray) -> np.ndarray:
        """One-Class SVM distance (higher = more anomalous)."""
        if self._ocsvm is None:
            return np.zeros(len(x))
        # decision_function returns signed distance to hyperplane
        scores = -self._ocsvm.decision_function(x)
        return scores

    def _ocsvm_anomaly(self, x: np.ndarray) -> np.ndarray:
        """One-Class SVM anomaly flag."""
        if self._ocsvm is None:
            return np.zeros(len(x))
        preds = self._ocsvm.predict(x)  # 1=normal, -1=anomaly
        return (preds == -1).astype(int)

    def _knn_density_score(self, x: np.ndarray) -> np.ndarray:
        """K-NN density based anomaly score (higher = more anomalous)."""
        if self._knn is None:
            return np.zeros(len(x))
        distances, _ = self._knn.kneighbors(x)
        # Mean distance to k nearest neighbors (excluding self)
        mean_dists = distances[:, 1:].mean(axis=1) if distances.shape[1] > 1 else distances[:, 0]
        return mean_dists

    def _knn_anomaly(self, x: np.ndarray, threshold: float = None) -> np.ndarray:
        """K-NN density based anomaly flag."""
        scores = self._knn_density_score(x)
        if threshold is None:
            # Adaptive threshold: mean + 2*std
            threshold = scores.mean() + 2 * scores.std() + 1e-8
        return (scores > threshold).astype(int)

    def _ensemble_score(self, x: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Compute ensemble anomaly scores using all cold-start detectors.

        Returns:
            scores: [N] ensemble scores (weighted average, higher = more anomalous)
            votes: dict with individual detector votes
        """
        n = len(x)
        scores = np.zeros(n)
        votes = {}

        # Z-score
        z_scores = self._zscore_score(x)
        z_anomaly = self._zscore_anomaly(x)
        scores += z_scores
        votes["zscore"] = z_anomaly

        # Isolation Forest
        if_scores = self._iforest_score(x)
        if_anomaly = self._iforest_anomaly(x)
        scores += if_scores
        votes["iforest"] = if_anomaly

        # One-Class SVM
        svm_scores = self._ocsvm_score(x)
        svm_anomaly = self._ocsvm_anomaly(x)
        scores += svm_scores
        votes["ocsvm"] = svm_anomaly

        # K-NN Density
        knn_scores = self._knn_density_score(x)
        knn_anomaly = self._knn_anomaly(x)
        scores += knn_scores
        votes["knn"] = knn_anomaly

        # Normalize by number of active detectors
        n_active = sum([
            self._zscore_mean is not None,
            self._iforest is not None,
            self._ocsvm is not None,
            self._knn is not None,
        ])
        if n_active > 0:
            scores /= n_active

        return scores, votes

    def _ensemble_anomaly(self, x: np.ndarray) -> np.ndarray:
        """
        Ensemble voting: if 2+ detectors flag anomaly → anomaly.

        Returns:
            [N] binary anomaly flags
        """
        n = len(x)
        votes = np.zeros(n, dtype=int)

        # Z-score vote
        if self._zscore_mean is not None:
            z_anomaly = self._zscore_anomaly(x)
            votes += z_anomaly

        # Isolation Forest vote
        if self._iforest is not None:
            if_anomaly = self._iforest_anomaly(x)
            votes += if_anomaly

        # One-Class SVM vote
        if self._ocsvm is not None:
            svm_anomaly = self._ocsvm_anomaly(x)
            votes += svm_anomaly

        # K-NN Density vote
        if self._knn is not None:
            knn_anomaly = self._knn_anomaly(x)
            votes += knn_anomaly

        # Majority voting
        return (votes >= self.voting_threshold).astype(int)

    def recognize(self, features: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Detect anomalies using ensemble for cold-start, Mahalanobis after warmup.

        Returns features if anomaly, None if normal.
        """
        n_updates = self.mahalanobis.n_updates.item()
        count = self.mahalanobis.count.item()

        if count < 3 or n_updates < 3:
            # Not enough data for any detector
            return None

        if n_updates < self._warmup_threshold:
            # Cold-start phase: use ensemble
            x_np = features.detach().cpu().numpy()
            anomaly_flags = self._ensemble_anomaly(x_np)
            anomaly = torch.from_numpy(anomaly_flags).float().unsqueeze(-1).to(features.device)
            return anomaly * features

        # Warmup complete: use Mahalanobis
        return self.mahalanobis.recognize(features)

    def get_anomaly_score(self, features: torch.Tensor) -> torch.Tensor:
        """
        Get continuous anomaly score.

        Cold-start: ensemble weighted average
        After warmup: Mahalanobis distance
        """
        n_updates = self.mahalanobis.n_updates.item()
        count = self.mahalanobis.count.item()

        if count < 3 or n_updates < 3:
            return torch.zeros(features.size(0), device=features.device)

        if n_updates < self._warmup_threshold:
            # Cold-start phase: use ensemble scores
            x_np = features.detach().cpu().numpy()
            scores, _ = self._ensemble_score(x_np)
            return torch.from_numpy(scores).float().to(features.device)

        # Warmup complete: use Mahalanobis
        return self.mahalanobis.get_anomaly_score(features)

    def get_cold_start_stats(self) -> dict:
        """Get cold-start detector statistics."""
        return {
            "zscore_mean": self._zscore_mean.tolist() if self._zscore_mean is not None else None,
            "zscore_std": float(self._zscore_std),
            "has_iforest": self._iforest is not None,
            "has_ocsvm": self._ocsvm is not None,
            "has_knn": self._knn is not None,
            "knn_data_size": len(self._knn_data) if self._knn_data is not None else 0,
        }

    def reset(self) -> None:
        """Reset all memory."""
        self.mahalanobis.reset()
        self._zscore_mean = None
        self._zscore_std = 1.0
        self._iforest = None
        self._ocsvm = None
        self._knn = None
        self._knn_data = None

    def get_usage(self) -> int:
        """Get number of patterns stored."""
        return self.mahalanobis.get_usage()

    def get_stats(self) -> dict:
        """Get combined statistics."""
        stats = self.mahalanobis.get_stats()
        stats["cold_start"] = self.get_cold_start_stats()
        return stats
