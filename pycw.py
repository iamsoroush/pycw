import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics.pairwise import euclidean_distances


class ChineseWhispers(BaseEstimator, ClusterMixin):
    """Implementation of chinese whispers clustering algorithm.

    Source paper: https://pdfs.semanticscholar.org/c64b/9ed6a42b93a24316c7d1d6b\
                  3fddbd96dbaf5.pdf?_ga=2.16343989.248353595.1538147473-1437352660.1538147473

    """

    def __init__(self, n_iterations=10):
        """Init an estimator object.

        Parameters
        ----------
        n_iterations: int
            Number of iterations.

        Attributes
        ----------
        labels_: Input data clusters will save in this attribute,
            after calling fit or fit_predict methods. This attribute
            will be a vector of shape X.shape[0] .

        adjacency_mat_: Contains pair-wise similarity measure of input data.
            Similarity measure is: 1/euclidean_distance.

        """
        self.n_iterations = n_iterations
        self.labels_ = None
        self.adjacency_mat_ = None

    def fit(self, X, y=None):
        """Fits the estimator on X, and returns the object (self).

        Parameters
        ----------
        X: :obj: np.ndarray with ndim=2
            This is the input array, rows represent data points while cols are data features.

        returns: self

        """

        assert isinstance(self.n_iterations, int), "parameter n_iterations must be of type int"
        assert isinstance(X, np.ndarray), "X must be an instance of np.ndarray"
        assert X.ndim == 2, "X must be of ndim=2"
        n_samples = X.shape[0]
        adjacency_mat = (1 / (euclidean_distances(X, X) + np.identity(n_samples, dtype=X.dtype))) *\
                        (np.ones((n_samples, n_samples), dtype=X.dtype) -
                         np.identity(n_samples, dtype=X.dtype))
        labels_mat = np.identity(n_samples, dtype=np.int8)
        for _ in range(self.n_iterations):
            for i in range(n_samples):
                labels_mat[i, :] = self.maxrow(np.dot(labels_mat[i, :], adjacency_mat))
        self.adjacency_mat_ = adjacency_mat
        self.labels_ = np.where(labels_mat == 1)[1]
        return self

    def fit_predict(self, X, y=None):
        """Fits the estimator on X, and returns the labels of X as predictions.

        Parameters
        ----------
        X: :obj: np.ndarray with ndim=2
            This is the input array, rows represent data points while cols are data features.

        returns: self.labels_

        """

        assert isinstance(self.n_iterations, int), "parameter n_iterations must be of type int"
        assert isinstance(X, np.ndarray), "X must be an instance of np.ndarray"
        assert X.ndim == 2, "X must be of ndim=2"
        n_samples = X.shape[0]
        adjacency_mat = (1 / (euclidean_distances(X, X) + np.identity(n_samples, dtype=X.dtype))) *\
                        (np.ones((n_samples, n_samples), dtype=X.dtype) -
                         np.identity(n_samples, dtype=X.dtype))
        labels_mat = np.identity(n_samples, dtype=np.int8)
        for _ in range(self.n_iterations):
            for i in range(n_samples):
                labels_mat[i, :] = self.maxrow(np.dot(labels_mat[i, :], adjacency_mat))
        self.adjacency_mat_ = adjacency_mat
        self.labels_ = np.where(labels_mat == 1)[1]
        return self.labels_

    @staticmethod
    def maxrow(self, row):
        """Returns a sparse vector of same size as input vector,
        containing zeros except for maximum element, which is 1.

        """

        argmax = np.argmax(row)
        out = np.zeros(row.shape, dtype=np.int8)
        out[argmax] = 1
        return out

    def score(self, X, y):
        pass
