import asyncio
import logging
import os
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import AsyncGenerator, Optional
import numpy as np
from kokoro_onnx import Kokoro

logger = logging.getLogger(__name__)

# Mapping of Indic script Unicode code point ranges to macOS system voices
INDIC_VOICE_MAP = [
    (0x0C00, 0x0C7F, "Geeta"),   # Telugu
    (0x0900, 0x097F, "Lekha"),   # Hindi / Devanagari
    (0x0B80, 0x0BFF, "Vani"),    # Tamil
    (0x0C80, 0x0CFF, "Soumya"),  # Kannada
    (0x0980, 0x09FF, "Piya"),    # Bengali
]

class TextToSpeech:
    """Hybrid neural speech synthesizer supporting native macOS Indic voices (Geeta, Lekha)
    and Kokoro-82M ONNX for English speech at 24kHz Int16 linear PCM.
    """

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
        logger.info("Hybrid TTS initialized successfully (Kokoro English + macOS Indic).")

    @staticmethod
    def detect_indic_voice(text: str) -> Optional[str]:
        """Detect if text contains Indic characters and return the matching macOS voice."""
        for char in text:
            cp = ord(char)
            for start, end, voice in INDIC_VOICE_MAP:
                if start <= cp <= end:
                    return voice
        return None

    def _synthesize_mac_say(self, text: str, voice: str) -> bytes:
        """Synthesize Indic speech using native macOS say at 24kHz Int16 PCM."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name

        try:
            cmd = [
                "say",
                "--file-format=WAVE",
                "--data-format=LEI16@24000",
                "-v", voice,
                "-o", tmp_path,
                text
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            with wave.open(tmp_path, "rb") as wf:
                return wf.readframes(wf.getnframes())
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _synthesize_sentence_sync(self, text: str, voice: Optional[str] = None) -> bytes:
        """Synthesize a single sentence to 24kHz Int16 linear PCM bytes."""
        indic_voice = self.detect_indic_voice(text)
        if indic_voice:
            return self._synthesize_mac_say(text, indic_voice)

        # Fallback to Kokoro-82M ONNX for English / Latin script
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
