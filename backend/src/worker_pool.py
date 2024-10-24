from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Any
import logging
from .config import Config

class WorkerPool:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=Config.MAX_WORKERS)
    
    async def submit(self, func: Callable, *args, **kwargs) -> Any:
        """Submit a task to the worker pool."""
        try:
            future = self.executor.submit(func, *args, **kwargs)
            return await future
        except Exception as e:
            logging.error(f"Error in worker thread: {e}")
            raise
    
    def shutdown(self):
        """Shutdown the worker pool."""
        self.executor.shutdown(wait=True)