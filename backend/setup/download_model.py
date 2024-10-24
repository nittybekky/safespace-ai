import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import ctranslate2
import logging
from sys import path
path.append(str(Path(__file__).parent.parent))
from src.config import Config

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Config.LOG_FILE),
            logging.StreamHandler()
        ]
    )

def convert_to_gguf():
    """Convert the model to GGUF format for optimized inference."""
    logging.info("Converting model to GGUF format...")
    
    converter = ctranslate2.converters.TransformersConverter(
        model=str(Config.MODELS_DIR / "original"),
        output_dir=str(Config.GGUF_MODEL_PATH),
        quantization="int8"
    )
    converter.convert()
    logging.info("GGUF conversion complete")

def quantize_model():
    """Quantize the model for better performance."""
    logging.info("Quantizing model...")
    model = AutoModelForCausalLM.from_pretrained(
        str(Config.MODELS_DIR / "original"),
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Quantize to 8-bit
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    model.save_pretrained(str(Config.QUANTIZED_MODEL_PATH))
    logging.info("Model quantization complete")

def main():
    Config.create_directories()
    setup_logging()
    
    try:
        logging.info("Starting model download and optimization process...")
        
        # Download original model
        logging.info("Downloading original model...")
        tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(Config.MODEL_NAME)
        
        # Save original model
        original_path = Config.MODELS_DIR / "original"
        tokenizer.save_pretrained(str(original_path))
        model.save_pretrained(str(original_path))
        
        # Convert to GGUF
        convert_to_gguf()
        
        # Quantize model
        quantize_model()
        
        logging.info("Model setup completed successfully")
        
    except Exception as e:
        logging.error(f"Error during model setup: {e}")
        raise

if __name__ == "__main__":
    main()