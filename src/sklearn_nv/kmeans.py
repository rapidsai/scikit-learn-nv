import time
import numpy as np
import scipy.sparse as sp

import cupy as cp

from sklearn.utils import check_random_state, check_array
from sklearn.utils.validation import _check_sample_weight, _is_arraylike_not_scalar

from pylibraft.distance import fused_l2_nn_argmin
from pylibraft.cluster.kmeans import compute_new_centroids, init_plus_plus


def _get_namespace(X):
    """Get namespace of array.

    This will get thge "Array API namespace of the array, if it has one. CuPY and
    Numpy arrays are special cases where the respective namespace is returned,
    even if the arrays are not defined via the Array API.
    """
    if isinstance(X, cp.ndarray):
        return cp
    elif isinstance(X, np.ndarray):
        return np

    if hasattr(X, "__array_namespace__"):
        return X.__array_namespace__()


def pairwise_distances_argmin(X, Y, handle=None):
    labels = cp.zeros((X.shape[0], 1), dtype=np.int32)
    fused_l2_nn_argmin(X, Y, labels, handle=handle)
    return labels


def compute_centroids_(X, sample_weight, labels, centroids, new_centroids, handle=None):
    compute_new_centroids(
        X,
        centroids,
        labels,
        new_centroids,
        sample_weights=sample_weight,
        handle=handle,
    )


def _tolerance(X, tol):
    """Scale `tol` according to the dataset."""
    xp = _get_namespace(X)
    variances = xp.var(X, axis=0)
    return xp.mean(variances) * tol


def _is_same_clustering(labels1, labels2, n_clusters):
    """Check if two arrays of labels are the same up to a permutation of the labels"""
    xp = _get_namespace(labels1)
    mapping = xp.full(fill_value=-1, shape=(n_clusters,), dtype=xp.int32)

    for i in range(labels1.shape[0]):
        if mapping[labels1[i]] == -1:
            mapping[labels1[i]] = labels2[i]
        elif mapping[labels1[i]] != labels2[i]:
            return False
    return True


class KMeansEngine:  # (_kmeans.KMeansCythonEngine):
    def __init__(self, estimator):
        self.estimator = estimator

    def accepts(self, X, y=None, sample_weight=None):
        """Determine if input data and hyper-parameters are supported by
        this engine.

        Determine if this engine can handle the hyper-parameters of the
        estimator as well as the input data. If not, return `False`. This
        method is called during engine selection, where each enabled engine
        is tried in the user defined order.

        Should fail as quickly as possible.
        """
        if self.estimator.init != "k-means++":
            return False

        if self.estimator.algorithm != "lloyd":
            return False

        if sp.issparse(X):
            return False

        return True

    def prepare_fit(self, X, y=None, sample_weight=None):
        estimator = self.estimator
        X = estimator._validate_data(
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            copy=estimator.copy_x,
            accept_large_sparse=False,
        )
        # this sets estimator _algorithm implicitly
        # XXX: shall we explose this logic as part of then engine API?
        # or is the current API flexible enough?
        estimator._check_params_vs_input(X)

        # TODO: delegate rng and sample weight checks to engine
        random_state = check_random_state(estimator.random_state)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        # Validate init array
        init = estimator.init
        init_is_array_like = _is_arraylike_not_scalar(init)
        if init_is_array_like:
            init = check_array(init, dtype=X.dtype, copy=True, order="C")
            estimator._validate_center_shape(X, init)

        # subtract of mean of x for more accurate distance computations
        if not sp.issparse(X):
            xp = _get_namespace(X)
            X_mean = xp.mean(X, axis=0)
            # The copy was already done above
            X -= X_mean

            if init_is_array_like:
                init -= X_mean

            self.X_mean = X_mean

        self.random_state = random_state
        self.tol = _tolerance(X, estimator.tol)
        self.init = init

        # Used later to unshift X again, need to explicity convert to
        # cupy array
        self.X_mean = cp.asarray(self.X_mean)

        X_ = cp.asarray(X)
        sample_weight_ = cp.asarray(sample_weight)

        return X_, y, sample_weight_

    def unshift_centers(self, X, best_centers):
        if not sp.issparse(X):
            if not self.estimator.copy_x:
                X += self.X_mean
            best_centers += self.X_mean

    def init_centroids(self, X):
        # XXX What is a better way to pass the random state to raft?
        seed = self.random_state.randint(
            np.iinfo(np.int64).min, np.iinfo(np.int64).max
        ) + abs(np.iinfo(np.int64).min)
        initial_centroids = init_plus_plus(
            X,
            self.estimator.n_clusters,
            seed=seed,
        )
        return cp.asarray(initial_centroids)

    def count_distinct_clusters(self, cluster_labels):
        # XXX need to check that the clusters are actually unique
        # XXX cp.unique() doesn't support the axis arg yet :(
        return cluster_labels.shape[0]

    def prepare_prediction(self, X, sample_weight=None):
        X = cp.asarray(X)
        if sample_weight is not None:
            sample_weight = cp.asarray(sample_weight)
        return X, sample_weight

    def get_labels(self, X, sample_weight):
        X = cp.asarray(X)
        labels = pairwise_distances_argmin(
            X,
            self.estimator.cluster_centers_,
        )
        return labels.ravel()

    def is_same_clustering(self, labels, best_labels, n_clusters):
        return _is_same_clustering(labels, best_labels, n_clusters)

    def kmeans_single(self, X, sample_weight, current_centroids):
        labels = None
        new_centroids = cp.zeros_like(current_centroids)
        for n_iter in range(self.estimator.max_iter):
            # E(xpectation)
            new_labels = pairwise_distances_argmin(
                X,
                current_centroids,
            )

            # M(aximization)
            # Compute the new centroids using the weighted sum of points in each cluster
            compute_centroids_(
                X,
                sample_weight,
                new_labels,
                current_centroids,
                new_centroids,
            )

            if n_iter > 0:
                if cp.array_equal(labels, new_labels):
                    break
                else:
                    center_shift = cp.power(current_centroids - new_centroids, 2).sum()
                    if center_shift <= self.tol:
                        break

            current_centroids = new_centroids
            labels = new_labels

        labels = labels.ravel()
        inertia = 0.0
        for n, center in enumerate(current_centroids):
            inertia += (
                cp.power(X[labels == n] - center, 2).sum(1) * sample_weight[labels == n]
            ).sum()
        return labels, inertia, current_centroids, n_iter + 1
