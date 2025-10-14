from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .project import (
        SurfaceProcessor,
        VolumeProcessor,
        BaseBrainDataProcessor,
        SurfaceData,
        VolumeData,
    )

__all__ = [
    "SurfaceProcessor",
    "VolumeProcessor", 
    "BaseBrainDataProcessor",
    "SurfaceData",
    "VolumeData",
]

def __getattr__(name: str):
    """Lazy load classes on first access."""
    if name in __all__:
        from . import project
        return getattr(project, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")