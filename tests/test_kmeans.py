import time

import numpy as np
import cupy as cp
import cupy.array_api

import sklearn
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from sklearn_nv.kmeans import _is_same_clustering


def same_rows(a, b):
    """Determine if the rows are the same, up to a permutation.

    Check that all rows that appear in `b` are also in `a`, allowing
    for the order of the rows to be different.
    """
    if a.shape != b.shape:
        return False

    for row_a in a:
        for row_b in b:
            if np.allclose(row_a, row_b):
                break
        else:
            return False

    return True


def test_sklearn_equivalence():
    X, y_true = make_blobs(
        n_samples=30, centers=4, n_features=3, cluster_std=0.60, random_state=10
    )

    n_clusters = 4
    kmeans_args = dict(n_clusters=n_clusters, random_state=42, n_init=1)

    # Default implementation
    km = KMeans(**kmeans_args)
    km.fit(X)

    y_pred = km.predict(X)

    # Using the accelerated version
    with sklearn.config_context(engine_provider="sklearn_nv"):
        km2 = KMeans(**kmeans_args)
        km2.fit(X)

        y_pred2 = km2.predict(X)

    assert same_rows(km.cluster_centers_, km2.cluster_centers_)
    assert np.allclose(km.inertia_, km2.inertia_)
    assert _is_same_clustering(y_pred, y_pred2, n_clusters)
    assert km.n_iter_ == km2.n_iter_


def test_cupy_array_api_input():
    """Check that using cupy Array API arrays as input works."""
    with sklearn.config_context(array_api_dispatch=True):
        X, y_true = make_blobs(
            n_samples=30, centers=4, n_features=3, cluster_std=0.60, random_state=10
        )
        X = cp.array_api.asarray(X)
        y_true = cp.array_api.asarray(y_true)

        n_clusters = 4
        kmeans_args = dict(n_clusters=n_clusters, random_state=42, n_init=1)

        with sklearn.config_context(engine_provider="sklearn_nv"):
            km2 = KMeans(**kmeans_args)

            km2.fit(X)

            y_pred2 = km2.predict(X)


def test_cupy_input():
    """Check that using cupy arrays as input works."""
    with sklearn.config_context(array_api_dispatch=True):
        X, y_true = make_blobs(
            n_samples=30, centers=4, n_features=3, cluster_std=0.60, random_state=10
        )

        n_clusters = 4
        kmeans_args = dict(n_clusters=n_clusters, random_state=42, n_init=1)

        # Default implementation
        km = KMeans(**kmeans_args)
        km.fit(X)
        y_pred = km.predict(X)

        X = cp.asarray(X)
        y_true = cp.asarray(y_true)

        with sklearn.config_context(engine_provider="sklearn_nv"):
            km2 = KMeans(**kmeans_args)

            km2.fit(X)

            y_pred2 = km2.predict(X)

        assert same_rows(km.cluster_centers_, km2.cluster_centers_)
        assert np.allclose(km.inertia_, km2.inertia_)
        assert _is_same_clustering(y_pred, y_pred2, n_clusters)
        assert km.n_iter_ == km2.n_iter_
