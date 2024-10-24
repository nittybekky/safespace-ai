import json
import pylru
from pathlib import Path
from typing import Optional, Dict, Any
import logging
from .config import Config

class CacheManager:
    def __init__(self):
        self.cache = pylru.lrucache(Config.CACHE_SIZE)
        self.cache_file = Config.CACHE_FILE
        self._load_cache()
    
    def _load_cache(self):
        """Load cache from disk if it exists."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    for key, value in data.items():
                        self.cache[key] = value
        except Exception as e:
            logging.error(f"Error loading cache: {e}")
    
    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(dict(self.cache.items()), f)
        except Exception as e:
            logging.error(f"Error saving cache: {e}")
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get item from cache."""
        return self.cache.get(key)
    
    def set(self, key: str, value: Dict[str, Any]):
        """Set item in cache and save to disk."""
        self.cache[key] = value
        self._save_cache()