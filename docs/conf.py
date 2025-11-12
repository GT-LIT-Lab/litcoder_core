# docs/conf.py
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Project information
project = "litcoder"
copyright = "2024, Taha Binhuraib, Ruimin Gao, Anya Ivanova"
author = "Taha Binhuraib, Ruimin Gao, Anya Ivanova"
release = "0.0.1"

# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# Mock heavy or optional dependencies during autodoc to avoid installing the full stack
autodoc_mock_imports = [
    # Core scientific stack / plotting
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "seaborn",
    "plotly",
    "umap",
    "xarray",
    "netCDF4",
    "h5py",
    "tables",
    "pyarrow",
    # ML / DL frameworks and ecosystems
    "torch",
    "torchaudio",
    "transformers",
    "accelerate",
    "tokenizers",
    "sentencepiece",
    "huggingface_hub",
    "datasets",
    "nltk",
    "scikit-learn",
    "sklearn",
    "statsmodels",
    "shap",
    "numba",
    "llvmlite",
    # IO / cloud / tracking
    "boto3",
    "botocore",
    "wandb",
    "PIL",
    # Neuroimaging and transformer dependencies
    "nibabel",
    "transformer_lens",
    "nilearn",
    "tqdm",
    "gensim",
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Enable autodoc to generate API documentation
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Enable autosummary to automatically generate module documentation
autosummary_generate = True

# HTML output options
try:
    import sphinx_rtd_theme  # noqa: F401

    html_theme = "sphinx_rtd_theme"
except Exception:
    html_theme = "alabaster"

# Exclude patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
