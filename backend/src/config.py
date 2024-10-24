from pathlib import Path
from typing import Dict, Any
import json

class Config:
    BASE_DIR = Path(__file__).parent.parent
    MODELS_DIR = BASE_DIR / "models"
    CACHE_DIR = BASE_DIR / "cache"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Model settings
    MODEL_NAME = "google/gemma-2b"
    GGUF_MODEL_PATH = MODELS_DIR / "gemma-2b-gguf"
    QUANTIZED_MODEL_PATH = MODELS_DIR / "gemma-2b-quantized"
    
    # Cache settings
    CACHE_SIZE = 1000
    CACHE_FILE = CACHE_DIR / "analysis_cache.json"
    
    # Worker thread settings
    MAX_WORKERS = 4
    
    # Logging settings
    LOG_FILE = LOGS_DIR / "safespace.log"
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist."""
        for directory in [cls.MODELS_DIR, cls.CACHE_DIR, cls.LOGS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)