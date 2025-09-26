Assemblies
==========

Assemblies load and expose dataset-aligned structures used by the trainer:

- ``stories``: ordered identifiers
- ``get_brain_data()``: list of arrays per story
- ``get_data_times()``, ``get_tr_times()``: timestamp arrays
- ``get_split_indices()``: train/test folds or concatenation boundaries

Modules
-------

- ``encoding.assembly.assemblies``: user-facing assembly types
- ``encoding.assembly.base_processor``: common I/O, validation, contracts
- ``encoding.assembly.narratives_processor`` | ``lpp_processor`` | ``lebel_processor``: dataset-specific logic
- ``encoding.assembly.assembly_loader``: loading helpers
- ``encoding.assembly.assembly_generator``: composite assembly creation

Typical usage
-------------

.. code-block:: python

   from encoding.assembly.assemblies import NarrativesAssembly
   assembly = NarrativesAssembly(assembly_path=<path_to_narratives.h5>)
   print(len(assembly.stories))
   print(assembly.get_tr_times()[0].shape) 