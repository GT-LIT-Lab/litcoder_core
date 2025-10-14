from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .assemblies import SimpleNeuroidAssembly
    from .assembly_loader import AssemblyLoader

__all__ = [
    "SimpleNeuroidAssembly",
    "load_assembly",
    "save_assembly",
    "AssemblyLoader",
]

def __getattr__(name: str):
    """Lazy load assembly classes and functions."""
    if name == "SimpleNeuroidAssembly":
        from .assemblies import SimpleNeuroidAssembly
        return SimpleNeuroidAssembly
    elif name in ("load_assembly", "save_assembly", "AssemblyLoader"):
        from . import assembly_loader
        return getattr(assembly_loader, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")