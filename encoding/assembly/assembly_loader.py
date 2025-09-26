# encoding/assembly/assembly_loader.py
"""
Simple assembly loader with pickle support.(TODO: add support for other formats like online?)
"""

import pickle
import logging
from typing import Optional
from pathlib import Path

from .assemblies import SimpleNeuroidAssembly

logger = logging.getLogger(__name__)


class AssemblyLoaderError(Exception):
    """Exception for assembly loading errors."""
    pass


def validate_assembly(func):
    """Decorator to validate assembly after loading."""
    def wrapper(self, *args, **kwargs):
        assembly = func(self, *args, **kwargs)
        if not self._validate_assembly(assembly):
            raise AssemblyLoaderError("Assembly validation failed")
        return assembly
    return wrapper


class AssemblyLoader:
    """Simple loader for pickle-based assemblies."""
    
    def _validate_assembly(self, assembly: SimpleNeuroidAssembly) -> bool:
        """Basic validation of assembly structure."""
        if not hasattr(assembly, 'stories') or not assembly.stories:
            logger.error("Assembly missing stories")
            return False
        
        if not hasattr(assembly, 'story_data') or not assembly.story_data:
            logger.error("Assembly missing story_data")
            return False
        
        return True
    
    @validate_assembly
    def load(self, filepath: str) -> SimpleNeuroidAssembly:
        """Load assembly from pickle file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Assembly file not found: {filepath}")
        
        try:
            with open(filepath, 'rb') as f:
                assembly = pickle.load(f)
            
            logger.info(f"Assembly loaded from {filepath}")
            return assembly
            
        except Exception as e:
            raise AssemblyLoaderError(f"Failed to load assembly from {filepath}: {e}")
    
    def save(self, assembly: SimpleNeuroidAssembly, filepath: str) -> None:
        """Save assembly to pickle file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(assembly, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f"Assembly saved to {filepath}")
            
        except Exception as e:
            raise AssemblyLoaderError(f"Failed to save assembly to {filepath}: {e}")


def load_assembly(filepath: str) -> SimpleNeuroidAssembly:
    """Load assembly from pickle file."""
    loader = AssemblyLoader()
    return loader.load(filepath)


def save_assembly(assembly: SimpleNeuroidAssembly, filepath: str) -> None:
    """Save assembly to pickle file."""
    loader = AssemblyLoader()
    loader.save(assembly, filepath)