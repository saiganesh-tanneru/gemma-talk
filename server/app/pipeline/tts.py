import asyncio
import logging
from pathlib import Path
from typing import AsyncGenerator, Optional
import numpy as np
from kokoro_onnx import Kokoro

logger = logging.getLogger(__name__)

class TextToSpeech:
    """Kokoro-82M ONNX neural speech synthesizer."""

    def __init__(
        self,
        model_path: Path,
        voices_path: Path,
        default_voice: str = "af_heart",
        speed: float = 1.0
    ):
        logger.info(f"Initializing Kokoro-82M TTS from {model_path}...")
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.kokoro = Kokoro(str(model_path), str(voices_path))
        self.kokoro.sess = session
        self.default_voice = default_voice
        self.speed = speed
        self.sample_rate = 24000
        logger.info("Kokoro-82M TTS initialized successfully with optimized multithreading.")

    def _synthesize_sentence_sync(self, text: str, voice: Optional[str] = None) -> bytes:
        """Synthesize a single sentence to 24kHz Int16 linear PCM bytes."""
        voice_name = voice or self.default_voice
        samples, sr = self.kokoro.create(
            text,
            voice=voice_name,
            speed=self.speed
        )
        # Convert float32 [-1.0, 1.0] to int16 PCM bytes
        int16_samples = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
        return int16_samples.tobytes()

    async def synthesize_sentence(self, text: str, voice: Optional[str] = None) -> bytes:
        """Synthesize a sentence asynchronously in a worker thread."""
        return await asyncio.to_thread(self._synthesize_sentence_sync, text, voice)
