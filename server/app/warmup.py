import time
import logging
import numpy as np
import httpx
from server.app.pipeline.vad import SileroVAD
from server.app.pipeline.stt import SpeechToText
from server.app.pipeline.tts import TextToSpeech
from server.app import config

logger = logging.getLogger(__name__)

async def run_warmup(vad: SileroVAD, stt: SpeechToText, tts: TextToSpeech):
    """Pre-warm all neural models to avoid cold-start delays on the first conversational turn."""
    logger.info("Starting system warmup routine...")
    t0 = time.perf_counter()

    # 1. Warm VAD
    try:
        zero_chunk = np.zeros(config.AUDIO_CHUNK_SAMPLES, dtype=np.float32)
        vad.process_chunk(zero_chunk)
        vad.reset()
        logger.info("✓ Silero VAD warmed")
    except Exception as e:
        logger.warning(f"VAD warmup warning: {e}")

    # 2. Warm STT (faster-whisper)
    try:
        dummy_audio = np.zeros(16000, dtype=np.float32)
        await stt.transcribe(dummy_audio)
        logger.info("✓ faster-whisper STT warmed")
    except Exception as e:
        logger.warning(f"STT warmup warning: {e}")

    # 3. Warm TTS (Kokoro-82M)
    try:
        await tts.synthesize_sentence("Ready.")
        logger.info("✓ Kokoro-82M TTS warmed")
    except Exception as e:
        logger.warning(f"TTS warmup warning: {e}")

    # 4. Warm Ollama LLM into GPU memory with keep_alive=-1
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{config.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": config.OLLAMA_MODEL,
                    "prompt": "Hello",
                    "keep_alive": config.OLLAMA_KEEP_ALIVE,
                    "options": {"num_predict": 1}
                }
            )
            if resp.status_code == 200:
                logger.info(f"✓ Ollama model '{config.OLLAMA_MODEL}' loaded into GPU memory (keep_alive={config.OLLAMA_KEEP_ALIVE})")
            else:
                logger.warning(f"Ollama warmup returned status {resp.status_code}")
    except Exception as e:
        logger.warning(f"Ollama warmup warning: {e}")

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(f"Warmup completed in {elapsed:.1f}ms. System is 100% hot and ready for real-time speech.")
