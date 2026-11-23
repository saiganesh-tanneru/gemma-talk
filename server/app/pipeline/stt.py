import asyncio
import logging
import numpy as np
from typing import Optional, Tuple
from server.app import config

logger = logging.getLogger(__name__)

class SpeechToText:
    """Pluggable STT Transcriber supporting AI4Bharat IndicConformer ONNX and faster-whisper."""

    def __init__(
        self,
        engine: str = "ai4bharat",
        indic_lang: str = "hi",
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 4
    ):
        self.engine = engine
        self.indic_lang = indic_lang

        if self.engine == "ai4bharat":
            model_id = f"OpenVoiceOS/ai4bharat-indicconformer-{indic_lang}-onnx"
            logger.info(f"Loading AI4Bharat IndicConformer model '{model_id}'...")
            import onnx_asr
            self.model = onnx_asr.load_model(model_id)
            logger.info("✓ AI4Bharat IndicConformer loaded successfully with CoreML/CPU acceleration.")
        else:
            logger.info(f"Loading faster-whisper model '{model_size}' on {device} ({compute_type})...")
            from faster_whisper import WhisperModel
            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                cpu_threads=cpu_threads
            )
            logger.info(f"✓ faster-whisper '{model_size}' loaded successfully.")

    def _transcribe_sync(
        self,
        audio_float32: np.ndarray,
        language: Optional[str] = None
    ) -> Tuple[str, str, float]:
        """Run speech recognition inference synchronously in worker thread."""
        if audio_float32.dtype != np.float32:
            audio_float32 = audio_float32.astype(np.float32)

        if self.engine == "ai4bharat":
            # AI4Bharat IndicConformer direct in-memory inference
            result = self.model.recognize(audio_float32, sample_rate=16000)
            transcript = str(result).strip() if result else ""
            logger.info(f"AI4Bharat IndicConformer ({self.indic_lang}): '{transcript}'")
            return transcript, self.indic_lang, 1.0
        else:
            # faster-whisper inference
            segments, info = self.model.transcribe(
                audio_float32,
                language=language,
                beam_size=1,
                condition_on_previous_text=False,
                temperature=0.0
            )
            transcript = " ".join(seg.text.strip() for seg in segments).strip()
            detected_lang = info.language or "unknown"
            confidence = info.language_probability or 0.0
            logger.info(f"Whisper STT: detected language '{detected_lang}' (prob: {confidence:.2f}): '{transcript}'")
            return transcript, detected_lang, confidence

    async def transcribe(
        self,
        audio_float32: np.ndarray,
        language: Optional[str] = None
    ) -> Tuple[str, str, float]:
        """Asynchronously transcribe in-memory float32 PCM without blocking the event loop."""
        return await asyncio.to_thread(self._transcribe_sync, audio_float32, language)
