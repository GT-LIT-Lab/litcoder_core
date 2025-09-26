Installation
============

Requirements
------------

- Python 3.9+
- NumPy, SciPy, h5py
- Sphinx (for docs) and ``sphinx_rtd_theme`` (for this site)

Install
-------

From source:

.. code-block:: bash

   git clone https://github.com/your-org/litcoder_final_core.git
   cd litcoder_final_core/litcoder_core
   pip install -e .

Docs dependencies (if you build locally):

.. code-block:: bash

   pip install sphinx sphinx-rtd-theme sphinx-autobuild

Build docs locally
------------------

.. code-block:: bash

   cd litcoder_final_core/litcoder_core/docs
   make html
   open _build/html/index.html

If you see ``ThemeError: no theme named 'sphinx_rtd_theme'``, install it:

.. code-block:: bash

   pip install sphinx-rtd-theme 