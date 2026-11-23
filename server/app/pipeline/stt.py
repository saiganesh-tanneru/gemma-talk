import asyncio
import logging
import numpy as np
from faster_whisper import WhisperModel
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class SpeechToText:
    """In-memory faster-whisper STT transcriber with multilingual auto-detection."""

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 4
    ):
        logger.info(f"Loading faster-whisper multilingual model '{model_size}' on {device} ({compute_type})...")
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads
        )
        logger.info(f"faster-whisper '{model_size}' loaded successfully.")

    def _transcribe_sync(
        self,
        audio_float32: np.ndarray,
        language: Optional[str] = None
    ) -> Tuple[str, str, float]:
        """Run Whisper inference synchronously on CPU with beam_size=1 for lowest latency."""
        if audio_float32.dtype != np.float32:
            audio_float32 = audio_float32.astype(np.float32)

        # beam_size=1 (greedy) delivers sub-250ms transcription
        segments, info = self.model.transcribe(
            audio_float32,
            language=language,  # None enables automatic Indic/multilingual language detection
            beam_size=1,
            condition_on_previous_text=False,
            temperature=0.0
        )
        transcript = " ".join(seg.text.strip() for seg in segments).strip()
        detected_lang = info.language or "unknown"
        confidence = info.language_probability or 0.0
        logger.info(f"STT: detected language '{detected_lang}' (prob: {confidence:.2f})")
        return transcript, detected_lang, confidence

    async def transcribe(
        self,
        audio_float32: np.ndarray,
        language: Optional[str] = None
    ) -> Tuple[str, str, float]:
        """Asynchronously transcribe in-memory float32 PCM without blocking the event loop."""
        return await asyncio.to_thread(self._transcribe_sync, audio_float32, language)
