"""Simple cache for surface data processing."""

import hashlib
import pickle
from pathlib import Path
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


class SimpleSurfaceCache:
    """Simple cache for surface data processing."""

    def __init__(self, cache_dir: str = ".surface_cache"):
        """Initialize the cache.

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        logger.info(f"Surface cache initialized at: {self.cache_dir}")

    def _get_cache_key(self, subject: str, volume_path: str) -> str:
        """Generate cache key from subject and volume path.

        Args:
            subject: Subject identifier
            volume_path: Path to volume file

        Returns:
            Cache key string
        """
        # Use file modification time to detect changes
        volume_file = Path(volume_path)
        if volume_file.exists():
            stat = volume_file.stat()
            hash_input = f"{subject}:{volume_path}:{stat.st_mtime}"
        else:
            hash_input = f"{subject}:{volume_path}"

        return hashlib.md5(hash_input.encode()).hexdigest()

    def get(self, subject: str, volume_path: str) -> Optional[Any]:
        """Get cached surface data.

        Args:
            subject: Subject identifier
            volume_path: Path to volume file

        Returns:
            Cached surface data if found, None otherwise
        """
        cache_key = self._get_cache_key(subject, volume_path)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    data = pickle.load(f)
                logger.info(f"Cache hit for {subject}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

        logger.info(f"Cache miss for {subject}")
        return None

    def set(self, subject: str, volume_path: str, data: Any) -> None:
        """Cache surface data.

        Args:
            subject: Subject identifier
            volume_path: Path to volume file
            data: Surface data to cache
        """
        cache_key = self._get_cache_key(subject, volume_path)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(data, f)
            logger.info(f"Cached data for {subject}")
        except Exception as e:
            logger.error(f"Failed to cache data: {e}")


# Global cache instance, this works better when we have multiple processes for some reason.
# Still trying to figure out why

_surface_cache = SimpleSurfaceCache()


def get_surface_cache() -> SimpleSurfaceCache:
    """Get the global surface cache instance."""
    return _surface_cache


def set_cache_directory(cache_dir: str) -> None:
    """Set the cache directory."""
    global _surface_cache
    _surface_cache = SimpleSurfaceCache(cache_dir)
