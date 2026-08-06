"""Sliced inverse regression utilities."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh, fractional_matrix_power
from scipy.sparse.linalg import eigs


class SIR:
    """Sliced Inverse Regression for supervised dimension reduction."""

    def __init__(self, H: int, K: int | None = None, estdim: bool = False, threshold: float = 0.9, Cn: float = 1):
        self.H = H
        self.K = K
        self.estdim = estdim
        self.threshold = threshold
        self.Cn = Cn

    def fit(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y)
        self.mean_x = np.mean(X, axis=0)
        self.sigma_x = np.cov(X, rowvar=False)

        Z = (X - self.mean_x) @ fractional_matrix_power(self.sigma_x, -0.5)
        n, p = Z.shape

        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if Y.shape[0] != n or Y.shape[1] != 1:
            raise ValueError("X and Y must have matching samples, and Y must be 1D.")

        weights = np.ones((n, 1)) / n
        base_count = n // self.H
        extra = n % self.H
        counts = np.array([base_count + (1 if i < extra else 0) for i in range(self.H)])
        cum_counts = np.cumsum(counts)

        data = np.hstack((Z, Y, weights))
        data = data[np.argsort(data[:, p], axis=0)]
        z, w = data[:, :p], data[:, p + 1:p + 2]

        slice_means = np.zeros((self.H, p))
        slice_weights = np.zeros((self.H, 1))
        slice_idx = 0
        for i in range(n):
            if i >= cum_counts[slice_idx]:
                slice_idx += 1
            slice_means[slice_idx] += z[i]
            slice_weights[slice_idx] += w[i]
        slice_means = slice_means / (slice_weights * n)

        mat = np.zeros((p, p))
        for h in range(self.H):
            mat += slice_weights[h] * np.outer(slice_means[h], slice_means[h])
        self.M = mat

        if self.estdim:
            eigvals, eigvecs = eigh(mat)
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
            cumvar = np.cumsum(eigvals) / np.sum(eigvals)
            self.K = int(np.searchsorted(cumvar, self.threshold) + 1)
            self.D = eigvals[:self.K]
            self.V = eigvecs[:, :self.K]
        else:
            if self.K is None:
                raise ValueError("K must be specified when estdim=False.")
            vals, vecs = eigs(A=mat, k=self.K, which="LM")
            self.D = np.real(vals)
            self.V = np.real(vecs)

        return self.V, self.K, self.M, self.D

    def transform(self):
        return fractional_matrix_power(self.sigma_x, -0.5) @ self.V
