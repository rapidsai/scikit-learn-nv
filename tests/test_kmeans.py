import time

import numpy as np

import sklearn
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


def test_sklearn_equivalence():
    X, y_true = make_blobs(
        n_samples=30, centers=4, n_features=3, cluster_std=0.60, random_state=10
    )

    kmeans_args = dict(n_clusters=4, random_state=42, n_init=1)

    # Default implementation
    km = KMeans(**kmeans_args)
    km.fit(X)

    y_pred = km.predict(X)

    # Using the accelerated version
    with sklearn.config_context(engine_provider="sklearn_gpu"):
        km2 = KMeans(**kmeans_args)
        km2.fit(X)

        y_pred2 = km2.predict(X)

    assert np.allclose(km.cluster_centers_, km2.cluster_centers_)
    assert np.allclose(km.inertia_, km2.inertia_)
    assert np.allclose(y_pred, y_pred2)
    assert km.n_iter_ == km2.n_iter_