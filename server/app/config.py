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
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
try:
    OLLAMA_KEEP_ALIVE = int(os.getenv("OLLAMA_KEEP_ALIVE", "-1"))
except ValueError:
    OLLAMA_KEEP_ALIVE = -1

# Audio Pipeline Settings
AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
AUDIO_CHUNK_SAMPLES = int(os.getenv("AUDIO_CHUNK_SAMPLES", "512"))  # 32ms at 16kHz
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.35"))
VAD_SILENCE_DURATION_MS = int(os.getenv("VAD_SILENCE_DURATION_MS", "400"))

# Speech-to-Text Engine Selection
# Options: "ai4bharat" (AI4Bharat IndicConformer ONNX) or "whisper" (faster-whisper)
STT_ENGINE = os.getenv("STT_ENGINE", "ai4bharat")
AI4BHARAT_LANG = os.getenv("AI4BHARAT_LANG", "hi")  # hi, te, ta, kn, mr, bn, gu, pa, ml, ur

# Speech-to-Text (faster-whisper fallback)
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", "4"))
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "") or None  # None = auto-detect Indic & world languages

# Text-to-Speech (Kokoro-82M ONNX)
TTS_VOICE = os.getenv("TTS_VOICE", "af_heart")
TTS_SPEED = float(os.getenv("TTS_SPEED", "1.0"))
TTS_SAMPLE_RATE = 24000  # Kokoro-82M native output

# Model Paths
MODELS_DIR = ROOT_DIR / "server" / "models"
SILERO_MODEL_PATH = MODELS_DIR / "silero_vad.onnx"
KOKORO_MODEL_PATH = MODELS_DIR / "kokoro-v1.0.onnx"
KOKORO_VOICES_PATH = MODELS_DIR / "voices-v1.0.bin"

# Voice-optimized System Prompt (Gemini Live-style conversational speech with Romanized Indic support)
VOICE_SYSTEM_PROMPT = (
    "You are a helpful, friendly, and expressive real-time voice assistant conversing directly over audio. "
    "Listen carefully to the user's language and respond naturally in that exact same language or dialect "
    "(such as English, Hindi, Telugu, Tamil, Kannada, Bengali, or colloquial Hinglish). "
    "CRITICAL RULE FOR VOICE OUTPUT: ALWAYS write your response using the standard English / Latin alphabet (Romanized Indic). "
    "For example: "
    "- If the user speaks Hindi, reply in Romanized Hindi (Hinglish), e.g.: 'Namaste! Main badhiya hoon, aap bataiye aaj kya chal raha hai?' "
    "- If the user speaks Telugu, reply in Romanized Telugu, e.g.: 'Namaskaram! Nenu baagunnanu, meeru ela unnaru?' "
    "- If the user speaks Tamil, reply in Romanized Tamil, e.g.: 'Vanakkam! Naan nallaa irukken, ungalukku enna venum?' "
    "- If the user speaks English, reply in English. "
    "NEVER output native scripts like Devanagari (हिन्दी), Telugu (తెలుగు), Tamil (தமிழ்), or Bengali, "
    "because the local speech synthesizer can only pronounce Latin alphabet characters out loud. "
    "Keep replies to 1 to 2 short, crisp, conversational spoken sentences. "
    "NEVER use markdown formatting, asterisks, bullet points, numbered lists, emojis, or code blocks. "
    "NEVER include internal thought blocks or <think> tags. "
    "Speak directly and warmly as if on a real-time telephone call."
)
