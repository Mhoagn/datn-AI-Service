"""
Model management and caching
"""
import torch
import whisperx
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import WHISPER_MODEL, QWEN_MODEL, DEVICE, COMPUTE_TYPE
import os
import warnings

# Tắt các warning không cần thiết
warnings.filterwarnings("ignore", category=UserWarning)

# Bật TF32 để tăng tốc độ trên GPU Ampere (RTX 30xx, A100, etc.)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

class ModelManager:
    """Singleton class to manage and cache models"""
    
    _instance = None
    _models_loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.whisper_model = None
            cls._instance.qwen_model = None
            cls._instance.tokenizer = None
        return cls._instance
    
    def load_models(self):
        """Load all models once"""
        if self._models_loaded:
            print("Models already loaded, skipping...")
            return
        
        print(f"Loading WhisperX model on {DEVICE}...")
        os.environ["WHISPERX_NO_VAD"] = "1"
        self.whisper_model = whisperx.load_model(
            WHISPER_MODEL, 
            DEVICE, 
            compute_type=COMPUTE_TYPE
        )
        print(f"WhisperX model '{WHISPER_MODEL}' loaded!")
        
        print(f"Loading Qwen 2.5 model on {DEVICE}...")
        self.tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto"
        )
        print("Qwen model loaded!")
        
        self._models_loaded = True
    
    def get_whisper_model(self):
        """Get WhisperX model (lazy load if needed)"""
        if self.whisper_model is None:
            print(f"Loading WhisperX model '{WHISPER_MODEL}' on CPU...")
            print("⚠️ Using CPU for WhisperX to avoid GPU memory issues")
            print("   (Slower but more stable)")
            
            # Load on CPU để tránh out of memory với VAD
            self.whisper_model = whisperx.load_model(
                WHISPER_MODEL, 
                "cpu",  # Force CPU
                compute_type="int8"  # CPU mode
            )
            print(f"✅ WhisperX model '{WHISPER_MODEL}' loaded on CPU!")
        return self.whisper_model
    
    def unload_whisper_model(self):
        """Unload WhisperX to free VRAM"""
        if self.whisper_model is not None:
            print("Unloading WhisperX model to free VRAM...")
            self.whisper_model = None
            torch.cuda.empty_cache()
            print("WhisperX unloaded!")
    
    def get_qwen_model(self):
        """Get Qwen model (lazy load if needed)"""
        if self.qwen_model is None:
            print(f"Loading Qwen 2.5 model on {DEVICE}...")
            self.tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
            self.qwen_model = AutoModelForCausalLM.from_pretrained(
                QWEN_MODEL,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                device_map="auto"
            )
            print("Qwen model loaded!")
        return self.qwen_model
    
    def get_tokenizer(self):
        """Get tokenizer (lazy load if needed)"""
        if self.tokenizer is None:
            self.get_qwen_model()  # Load both together
        return self.tokenizer
    
    def cleanup(self):
        """Clean up models from memory"""
        print("Cleaning up models...")
        self.whisper_model = None
        self.qwen_model = None
        self.tokenizer = None
        self._models_loaded = False
        torch.cuda.empty_cache()
        print("Cleanup complete!")

# Global model manager instance
model_manager = ModelManager()
