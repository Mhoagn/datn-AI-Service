"""
Configuration file for AI Service
"""
import os

# Model Configuration
# WhisperX models: tiny, base, small, medium, large-v2
# Recommendation for production: medium (best balance of speed vs accuracy)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")  # ⭐ Balance: 96-98% accuracy, 5-10x faster than large-v2
WHISPER_DEVICE = "cpu"  # Force CPU để tránh out of memory
QWEN_MODEL = os.getenv("QWEN_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

# Device Configuration
DEVICE = "cuda" if os.getenv("DEVICE", "cuda") == "cuda" else "cpu"  # For Qwen
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

# Summary Configuration
MAX_INPUT_LENGTH = 2000
MAX_SUMMARY_LENGTH = 300

# Server Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Timeout Configuration (seconds)
# Medium model: ~5-10 minutes for 10-minute video
REQUEST_TIMEOUT = 900  # 15 minutes for medium model
