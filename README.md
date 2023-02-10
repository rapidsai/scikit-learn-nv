# A GPU acceleration plugin for scikit-learn

> This is a proof-of-concept. Everything might change.


## Install

To install this plugin you need to install a in development version of scikit-learn as well
as this plugin, and the its dependencies:

1.  Create a new conda environment with `conda create -n sklearn-nv -c conda-forge python=3.9 numpy scipy matplotlib cython compilers joblib threadpoolctl`
2. Install the nightly build of pylibraft: `conda install -n sklearn-nv -c conda-forge -c rapidsai-nightly -c nvidia pylibraft=23.04 cupy rmm`
3. Activate the environment: `conda activate sklearn-nv`
4. Checkout the in development version of scikit-learn from [pull request #25535](https://github.com/scikit-learn/scikit-learn/pull/25535) ([get gh](https://cli.github.com/))
```
gh repo clone scikit-learn/scikit-learn
(cd scikit-learn; gh pr checkout 25535)
```
5. Install the scikit-learn development version with `pip install --no-build-isolation -e scikit-learn`
6. Install this plugin by checking out this repository
```
gh repo clone rapidsai/scikit-learn-nv
pip install -e scikit-learn-nv
```


## Running

To try it out, enable the plugin using:

```python
import sklearn
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

import cupy as cp

X, y_true = make_blobs(
    n_samples=300, centers=4, n_features=3, cluster_std=0.60, random_state=10
)
with sklearn.config_context(engine_provider="sklearn_nv"):
    km = KMeans(random_state=42)
    km.fit(X)
    y_pred = km.predict(X)
```
