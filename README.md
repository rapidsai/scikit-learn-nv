# A GPU acceleration plugin for scikit-learn

> This is a proof-of-concept. Everything might change.

## Install

To install this you need to install a custom verison of scikit-learn as well
as this plugin.


## Running

To try it out enable the plugin using:

```python
with sklearn.config_context(engine_provider="sklearn_nv"):
    km = KMeans()
    km.fit(X)
    y_pred = km.predict(X)
```
