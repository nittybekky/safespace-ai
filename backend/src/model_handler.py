import torch
import ctranslate2
from transformers import AutoTokenizer
import logging
from pathlib import Path
import hashlib
from typing import Dict, Any

from .config import Config
from .cache_manager import CacheManager
from .worker_pool import WorkerPool

class GemmaHandler:
    def __init__(self):
        self.cache_manager = CacheManager()
        self.worker_pool = WorkerPool()
        self.tokenizer = None
        self.model = None
        self.gguf_model = None
        self._load_models()
    
    def _load_models(self):
        """Load both GGUF and quantized models."""
        try:
            logging.info("Loading models...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(str(Config.GGUF_MODEL_PATH))
            
            # Load GGUF model
            self.gguf_model = ctranslate2.Generator(
                str(Config.GGUF_MODEL_PATH),
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            # Load quantized model as backup
            self.model = torch.jit.load(str(Config.QUANTIZED_MODEL_PATH))
            
            logging.info("Models loaded successfully")
            
        except Exception as e:
            logging.error(f"Error loading models: {e}")
            raise
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for input text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    async def generate(self, text: str) -> Dict[str, Any]:
        """Generate analysis using worker pool and caching."""
        cache_key = self._get_cache_key(text)
        
        # Check cache first
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            logging.info("Cache hit")
            return cached_result
        
        try:
            # Submit to worker pool
            result = await self.worker_pool.submit(self._generate_internal, text)
            
            # Cache the result
            self.cache_manager.set(cache_key, result)
            
            return result
            
        except Exception as e:
            logging.error(f"Error during generation: {e}")
            # Fallback to quantized model if GGUF fails
            return await self.worker_pool.submit(self._generate_fallback, text)
    
    def _generate_internal(self, text: str) -> Dict[str, Any]:
        """Internal generation using GGUF model."""
        tokens = self.tokenizer.encode(text)
        results = self.gguf_model.generate_batch(
            [tokens],
            max_length=512,
            sampling_temperature=0.7,
            sampling_topk=50
        )
        
        generated_text = self.tokenizer.decode(results[0].sequences_ids[0])
        return {"generated_text": generated_text}
    
    def _generate_fallback(self, text: str) -> Dict[str, Any]:
        """Fallback generation using quantized model."""
        logging.warning("Using fallback model")
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=512)
        return {"generated_text": self.tokenizer.decode(outputs[0])}

# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from pathlib import Path

# class GemmaHandler:
#     def __init__(self):
#         self.model_path = Path("../models/gemma-local")
#         self.tokenizer = None
#         self.model = None
#         self.load_model()
    
#     def load_model(self):
#         try:
#             print("Loading Gemma model...")
#             self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 str(self.model_path),
#                 device_map="auto",
#                 torch_dtype=torch.float16
#             )
#             self.model.eval()  # Set to evaluation mode
#             print("Model loaded successfully")
#         except Exception as e:
#             print(f"Error loading model: {e}")
#             raise

#     def generate(self, prompt: str) -> str:
#         try:
#             inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
#             outputs = self.model.generate(
#                 inputs.input_ids,
#                 max_length=512,
#                 temperature=0.7,
#                 top_p=0.9,
#                 do_sample=True
#             )
#             return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         except Exception as e:
#             print(f"Error during generation: {e}")
#             raise

# # Test the model handler
# if __name__ == "__main__":
#     handler = GemmaHandler()
#     test_prompt = "Analyze this text: 'Hello, how are you?'"
#     result = handler.generate(test_prompt)
#     print(f"Test result: {result}")