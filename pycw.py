import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics.pairwise import euclidean_distances


class ChineseWhispers(BaseEstimator, ClusterMixin):
    """Implementation of chinese whispers clustering algorithmself.

    Source paper: https://pdfs.semanticscholar.org/c64b/9ed6a42b93a24316c7d1d6b\
                  3fddbd96dbaf5.pdf?_ga=2.16343989.248353595.1538147473-1437352660.1538147473
    """

    def __init__(self, n_iterations=10):
        self.n_iterations_ = n_iterations

    def fit_predict(self, X):
        """Fits the estimator on X, and returns the labels of X as predictions.

        """

        n_samples = X.shape[0]
        adjacency_mat = (1 / (euclidean_distances(X, X) + np.identity(n_samples, dtype=X.dtype))) *\
                        (np.ones((n_samples, n_samples), dtype=X.dtype) -
                         np.identity(n_samples, dtype=X.dtype))
        labels_mat = np.identity(n_samples, dtype=np.int8)
        for _ in range(self.n_iterations_):
            for i in range(n_samples):
                labels_mat[i, :] = self.maxrow(np.dot(labels_mat[i, :], adjacency_mat))
        return np.where(labels_mat == 1)[1]

    @staticmethod
    def maxrow(self, row):
        argmax = np.argmax(row)
        out = np.zeros(row.shape, dtype=np.int8)
        out[argmax] = 1
        return out
