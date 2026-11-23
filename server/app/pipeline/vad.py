import logging
import collections
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class SileroVAD:
    """Official Silero VAD v5 ONNX runner.
    
    Silero VAD v5 requires an input tensor shape [batch, 64 + 512] at 16kHz,
    where 64 is the trailing context from the previous chunk.
    """

    def __init__(self, model_path: Path):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        self.sr = np.array(16000, dtype=np.int64)
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, 64), dtype=np.float32)

    def reset(self):
        """Reset internal recurrent state and context."""
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, 64), dtype=np.float32)

    def process_chunk(self, chunk: np.ndarray) -> float:
        """Process 512 samples of float32 audio at 16kHz (32ms).
        
        Args:
            chunk: 1D numpy array of float32 with length 512.
        Returns:
            probability of speech in [0.0, 1.0].
        """
        if chunk.ndim == 1:
            chunk = chunk[np.newaxis, :]
        
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)

        # Concatenate 64-sample previous context with current 512-sample chunk (576 samples total)
        model_input = np.concatenate([self.context, chunk], axis=1)

        outputs = self.session.run(None, {
            "input": model_input,
            "state": self.state,
            "sr": self.sr
        })

        # Update recurrent state and rolling 64-sample context
        self.state = outputs[1]
        self.context = chunk[:, -64:]

        return float(outputs[0][0][0])


class VoiceActivityDetector:
    """Detects start-of-speech, end-of-speech (silence), and interruption (barge-in).
    
    Includes a ring-buffer pre-roll to preserve consonant onsets (e.g., 'h' in 'hi').
    """

    def __init__(
        self,
        vad_model: SileroVAD,
        sample_rate: int = 16000,
        chunk_samples: int = 512,
        speech_threshold: float = 0.40,
        barge_in_threshold: float = 0.60,
        silence_duration_ms: int = 400,
        min_speech_duration_ms: int = 120,
        pre_roll_chunks: int = 4
    ):
        self.vad = vad_model
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples
        self.speech_threshold = speech_threshold
        self.barge_in_threshold = barge_in_threshold
        self.silence_chunks_threshold = max(2, int((silence_duration_ms / 1000.0) * (sample_rate / chunk_samples)))
        self.min_speech_chunks = max(2, int((min_speech_duration_ms / 1000.0) * (sample_rate / chunk_samples)))
        self.pre_roll_chunks = pre_roll_chunks

        # Ring buffer for pre-speech chunks
        self.pre_roll_buffer = collections.deque(maxlen=pre_roll_chunks)
        self.audio_buffer: list[np.ndarray] = []
        self.is_speech_active = False
        self.silence_chunk_count = 0
        self.speech_chunk_count = 0
        self.consecutive_barge_in_count = 0

    def reset(self):
        """Reset internal buffers and VAD state."""
        self.vad.reset()
        self.pre_roll_buffer.clear()
        self.audio_buffer.clear()
        self.is_speech_active = False
        self.silence_chunk_count = 0
        self.speech_chunk_count = 0
        self.consecutive_barge_in_count = 0

    def process(
        self,
        chunk_pcm16: bytes,
        is_assistant_speaking: bool = False
    ) -> Tuple[Optional[np.ndarray], bool]:
        """Process incoming 512-sample Int16 PCM chunk."""
        int16_arr = np.frombuffer(chunk_pcm16, dtype=np.int16)
        if len(int16_arr) != self.chunk_samples:
            if len(int16_arr) < self.chunk_samples:
                int16_arr = np.pad(int16_arr, (0, self.chunk_samples - len(int16_arr)))
            else:
                int16_arr = int16_arr[:self.chunk_samples]
        
        float_chunk = int16_arr.astype(np.float32) / 32768.0
        prob = self.vad.process_chunk(float_chunk)

        barge_in_triggered = False

        # Barge-in check during assistant speech
        if is_assistant_speaking:
            if prob >= self.barge_in_threshold:
                self.consecutive_barge_in_count += 1
                if self.consecutive_barge_in_count >= 2:
                    barge_in_triggered = True
                    self.consecutive_barge_in_count = 0
            else:
                self.consecutive_barge_in_count = max(0, self.consecutive_barge_in_count - 1)
        else:
            self.consecutive_barge_in_count = 0

        completed_utterance = None

        if prob >= self.speech_threshold:
            self.speech_chunk_count += 1
            self.silence_chunk_count = 0

            if not self.is_speech_active:
                self.is_speech_active = True
                self.audio_buffer = list(self.pre_roll_buffer)
                logger.info(f"🎤 Speech started (VAD prob: {prob:.2f})")

            self.audio_buffer.append(float_chunk)
        else:
            if self.is_speech_active:
                self.audio_buffer.append(float_chunk)
                self.silence_chunk_count += 1

                if self.silence_chunk_count >= self.silence_chunks_threshold:
                    if self.speech_chunk_count >= self.min_speech_chunks:
                        completed_utterance = np.concatenate(self.audio_buffer, axis=0)
                        logger.info(
                            f"⏹️ Utterance ended ({len(completed_utterance)/self.sample_rate:.2f}s, "
                            f"{self.speech_chunk_count} speech chunks). Sending to Whisper STT..."
                        )

                    self.audio_buffer.clear()
                    self.pre_roll_buffer.clear()
                    self.is_speech_active = False
                    self.silence_chunk_count = 0
                    self.speech_chunk_count = 0
            else:
                self.pre_roll_buffer.append(float_chunk)

        return completed_utterance, barge_in_triggered
