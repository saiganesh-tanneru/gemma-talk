import os
from pathlib import Path
from dotenv import load_dotenv

# Load root .env
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT_DIR / ".env")

# Server Network Settings
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))

# Ollama Settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4")
try:
    OLLAMA_KEEP_ALIVE = int(os.getenv("OLLAMA_KEEP_ALIVE", "-1"))
except ValueError:
    OLLAMA_KEEP_ALIVE = -1
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "1024"))

# Audio Pipeline Settings
AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
AUDIO_CHUNK_SAMPLES = int(os.getenv("AUDIO_CHUNK_SAMPLES", "512"))  # 32ms at 16kHz
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.35"))
VAD_SILENCE_DURATION_MS = int(os.getenv("VAD_SILENCE_DURATION_MS", "400"))

# Speech-to-Text Engine Selection
# Options: "ai4bharat" (AI4Bharat IndicConformer ONNX) or "whisper" (faster-whisper)
STT_ENGINE = os.getenv("STT_ENGINE", "ai4bharat")
AI4BHARAT_LANG = os.getenv("AI4BHARAT_LANG", "te")  # Default: Telugu (te)

# Speech-to-Text (faster-whisper fallback)
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", "4"))
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "") or None

# Text-to-Speech (Kokoro-82M ONNX)
TTS_VOICE = os.getenv("TTS_VOICE", "af_heart")
TTS_SPEED = float(os.getenv("TTS_SPEED", "1.0"))
TTS_SAMPLE_RATE = 24000  # Kokoro-82M native output

# Model Paths
MODELS_DIR = ROOT_DIR / "server" / "models"
SILERO_MODEL_PATH = MODELS_DIR / "silero_vad.onnx"
KOKORO_MODEL_PATH = MODELS_DIR / "kokoro-v1.0.onnx"
KOKORO_VOICES_PATH = MODELS_DIR / "voices-v1.0.bin"

# Voice-optimized System Prompt (Natural conversational speech in English)
VOICE_SYSTEM_PROMPT = (
    "You are a friendly, expressive, and natural conversational voice assistant. "
    "Respond in clear, natural, and concise English. "
    "Keep responses short and conversational for voice dialogue (1-3 sentences), "
    "avoid markdown, bullet points, asterisks, emojis, and code blocks unless specifically requested."
)
