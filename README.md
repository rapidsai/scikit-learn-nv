# A GPU acceleration plugin for scikit-learn

> This is a proof-of-concept. Everything might change.


## Install

To install this plugin you need to install a in development version of scikit-learn as well
as this plugin, and its dependencies:

1. Create a new conda environment:

   ```commandline
   conda create -n sklearn-nv -c conda-forge python=3.9 numpy scipy matplotlib cython compilers joblib threadpoolctl
   ```

2. Activate the environment:

   ```commandline
   conda activate sklearn-nv
   ```

3. Install the nightly build of `pylibraft`:

   ```commandline
   conda install -c conda-forge -c rapidsai-nightly -c nvidia --no-channel-priority pylibraft=23.04 cupy rmm
   ```

4. Checkout the development version of scikit-learn from the `feature/engine-api` branch
   (see [`scikit-learn#25535`](https://github.com/scikit-learn/scikit-learn/pull/25535). Using [`gh`](https://cli.github.com/):

   ```commandline
   gh repo clone scikit-learn/scikit-learn
   (cd scikit-learn; gh pr checkout 25535)
   ```

5. Install the scikit-learn development version by running the following in the folder you checked the code out to:

   ```commandline
   pip install --no-build-isolation -e .
   ```

6. Install this plugin by checking out this repository. Using [`gh`](https://cli.github.com/):

   ```commandline
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
