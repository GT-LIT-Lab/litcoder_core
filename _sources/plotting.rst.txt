Plotting & Logging
===================

Logging backends
----------------

- ``encoding.plotting.plotting_utils.WandBLogger``: logs scalars and figures to Weights & Biases
- ``encoding.plotting.plotting_utils.TensorBoardLogger``: logs to TensorBoard event files

Brain plots
-----------

- ``encoding.plotting.plotting_utils.BrainPlotter``: formats and logs correlation maps
- Use ``encoding.utils.unmask_correlations_for_plotting`` to expand masked results 