import asyncio
import logging
import numpy as np
from faster_whisper import WhisperModel
from typing import Optional

logger = logging.getLogger(__name__)

class SpeechToText:
    """In-memory faster-whisper STT transcriber."""

    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 4
    ):
        logger.info(f"Loading faster-whisper model '{model_size}' on {device} ({compute_type})...")
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads
        )
        logger.info("faster-whisper loaded successfully.")

    def _transcribe_sync(self, audio_float32: np.ndarray) -> str:
        """Run Whisper inference synchronously on CPU with beam_size=1 for lowest latency."""
        if audio_float32.dtype != np.float32:
            audio_float32 = audio_float32.astype(np.float32)

        # beam_size=1 (greedy) delivers sub-150ms transcription for short conversational turns
        segments, _ = self.model.transcribe(
            audio_float32,
            language="en",
            beam_size=1,
            condition_on_previous_text=False,
            temperature=0.0
        )
        transcript = " ".join(seg.text.strip() for seg in segments).strip()
        return transcript

    async def transcribe(self, audio_float32: np.ndarray) -> str:
        """Asynchronously transcribe in-memory float32 PCM without blocking the event loop."""
        return await asyncio.to_thread(self._transcribe_sync, audio_float32)
