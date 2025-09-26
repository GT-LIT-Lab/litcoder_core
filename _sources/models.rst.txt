Models
======

Core models
-----------

- ``encoding.models.ridge_regression.RidgeRegressionModel``: efficient voxel-wise ridge
- ``encoding.models.nested_cv``: nested CV utilities and pipelines
- ``encoding.models.sklearn_model.SklearnModel``: wrapper for scikit-learn estimators
- ``encoding.models.linear`` and ``encoding.models.base``: common interfaces

Training modes
--------------

- Concatenated (LPP/Narratives): stack all stories, fit cross-validated ridge
- Train/Test split (Lebel): fit on early segments, test on held-out segments

Example
-------

.. code-block:: python

   from encoding.models.ridge_regression import RidgeRegressionModel
   model = RidgeRegressionModel(n_alphas=20) 