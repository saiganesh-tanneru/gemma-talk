import time
import asyncio
import logging
from enum import Enum
from typing import Optional, List, Dict
from fastapi import WebSocket, WebSocketDisconnect

from server.app.pipeline.vad import SileroVAD, VoiceActivityDetector
from server.app.pipeline.stt import SpeechToText
from server.app.pipeline.llm import OllamaStreamingClient
from server.app.pipeline.chunker import SentenceChunker
from server.app.pipeline.tts import TextToSpeech
from server.app import config

logger = logging.getLogger(__name__)

class SessionState(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    TRANSCRIBING = "TRANSCRIBING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    INTERRUPTED = "INTERRUPTED"


class VoiceSession:
    """Manages the full duplex audio conversation session over a single WebSocket."""

    def __init__(
        self,
        websocket: WebSocket,
        vad_model: SileroVAD,
        stt: SpeechToText,
        tts: TextToSpeech,
        llm: OllamaStreamingClient
    ):
        self.ws = websocket
        self.vad_model = vad_model
        self.stt = stt
        self.tts = tts
        self.llm = llm

        self.state: SessionState = SessionState.IDLE
        self.detector = VoiceActivityDetector(
            vad_model=self.vad_model,
            sample_rate=config.AUDIO_SAMPLE_RATE,
            chunk_samples=config.AUDIO_CHUNK_SAMPLES,
            speech_threshold=config.VAD_THRESHOLD,
            silence_duration_ms=config.VAD_SILENCE_DURATION_MS
        )
        self.chunker = SentenceChunker()
        self.conversation_history: List[Dict[str, str]] = []
        self.chunks_received_count = 0

        # In-flight task references for cancellation on barge-in
        self.active_generation_task: Optional[asyncio.Task] = None
        self.is_active = False

        # Latency metrics trackers
        self.utterance_end_time: float = 0.0
        self.stt_latency_ms: float = 0.0
        self.llm_ttft_ms: float = 0.0
        self.tts_first_chunk_ms: float = 0.0
        self.total_e2e_ms: float = 0.0

    async def set_state(self, new_state: SessionState):
        """Update and broadcast session state to frontend client."""
        if self.state != new_state:
            self.state = new_state
            logger.debug(f"Session State -> {new_state.value}")
            await self.send_json({
                "type": "state_change",
                "state": new_state.value
            })

    async def send_json(self, payload: dict):
        """Safely send JSON message to client."""
        try:
            await self.ws.send_json(payload)
        except Exception as e:
            logger.debug(f"Failed to send JSON (client disconnected?): {e}")

    async def send_audio_bytes(self, data: bytes):
        """Send raw binary audio PCM chunk to client."""
        try:
            await self.ws.send_bytes(data)
        except Exception as e:
            logger.debug(f"Failed to send audio bytes: {e}")

    async def handle_barge_in(self):
        """Interrupt any in-flight assistant speech, flush client audio, and revert to LISTENING."""
        logger.info("⚡ BARGE-IN DETECTED: Aborting generation and flushing client audio queue!")
        
        # 1. Cancel in-flight generation task (LLM stream + TTS)
        if self.active_generation_task and not self.active_generation_task.done():
            self.active_generation_task.cancel()
            try:
                await self.active_generation_task
            except asyncio.CancelledError:
                pass
            self.active_generation_task = None

        # 2. Reset chunker
        self.chunker.reset()

        # 3. Notify frontend to purge its audio playback buffer immediately
        await self.send_json({"type": "flush"})

        # 4. Update state to INTERRUPTED then back to LISTENING
        await self.set_state(SessionState.INTERRUPTED)
        await asyncio.sleep(0.05)
        await self.set_state(SessionState.LISTENING)

    async def process_utterance(self, audio_data):
        """Handle completed user speech: Transcribe -> LLM -> Chunk -> TTS -> Stream."""
        self.utterance_end_time = time.perf_counter()
        
        # 1. TRANSCRIBING
        await self.set_state(SessionState.TRANSCRIBING)
        t_stt_start = time.perf_counter()
        transcript, detected_lang, _ = await self.stt.transcribe(audio_data, language=config.WHISPER_LANGUAGE)
        self.stt_latency_ms = (time.perf_counter() - t_stt_start) * 1000.0

        if not transcript or len(transcript.strip()) == 0:
            logger.info("Empty transcript from audio segment, resuming LISTENING.")
            await self.set_state(SessionState.LISTENING)
            return

        logger.info(f"User ({detected_lang}): '{transcript}' [STT: {self.stt_latency_ms:.1f}ms]")
        await self.send_json({
            "type": "transcript",
            "role": "user",
            "text": transcript,
            "language": detected_lang
        })

        # Append user turn to conversation history (keep last 6 turns for context)
        self.conversation_history.append({"role": "user", "content": transcript})
        if len(self.conversation_history) > 6:
            self.conversation_history = self.conversation_history[-6:]

        # 2. Launch in-flight LLM & TTS pipeline as a cancellable task
        self.active_generation_task = asyncio.create_task(self._generate_and_stream_response())

    async def _generate_and_stream_response(self):
        """Stream LLM tokens, split into sentences, and synthesize TTS without blocking."""
        try:
            await self.set_state(SessionState.THINKING)
            self.chunker.reset()

            t_llm_start = time.perf_counter()
            first_token_received = False
            first_audio_sent = False
            full_assistant_reply = []

            # Stream LLM tokens from Ollama
            token_generator = self.llm.stream_response(
                messages=self.conversation_history,
                system_prompt=config.VOICE_SYSTEM_PROMPT
            )

            async for token in token_generator:
                if not first_token_received:
                    first_token_received = True
                    self.llm_ttft_ms = (time.perf_counter() - t_llm_start) * 1000.0

                full_assistant_reply.append(token)
                sentences = self.chunker.add_token(token)

                for sentence in sentences:
                    clean_sentence = OllamaStreamingClient.clean_voice_text(sentence)
                    if not clean_sentence:
                        continue

                    # Synthesize and send this sentence
                    t_tts_start = time.perf_counter()
                    pcm_bytes = await self.tts.synthesize_sentence(clean_sentence)
                    tts_duration_ms = (time.perf_counter() - t_tts_start) * 1000.0

                    if not first_audio_sent:
                        first_audio_sent = True
                        self.tts_first_chunk_ms = tts_duration_ms
                        self.total_e2e_ms = (time.perf_counter() - self.utterance_end_time) * 1000.0
                        await self.set_state(SessionState.SPEAKING)

                        # Emit telemetry
                        logger.info(
                            f"⚡ E2E Latency: {self.total_e2e_ms:.1f}ms "
                            f"(STT: {self.stt_latency_ms:.1f}ms, "
                            f"TTFT: {self.llm_ttft_ms:.1f}ms, "
                            f"TTS: {self.tts_first_chunk_ms:.1f}ms)"
                        )
                        await self.send_json({
                            "type": "telemetry",
                            "stt_ms": round(self.stt_latency_ms, 1),
                            "ttft_ms": round(self.llm_ttft_ms, 1),
                            "tts_first_chunk_ms": round(self.tts_first_chunk_ms, 1),
                            "total_e2e_ms": round(self.total_e2e_ms, 1)
                        })

                    # Send audio PCM to client
                    await self.send_audio_bytes(pcm_bytes)

            # Flush any remaining text in chunker
            for remaining_sentence in self.chunker.flush():
                clean_sentence = OllamaStreamingClient.clean_voice_text(remaining_sentence)
                if clean_sentence:
                    pcm_bytes = await self.tts.synthesize_sentence(clean_sentence)
                    if not first_audio_sent:
                        first_audio_sent = True
                        await self.set_state(SessionState.SPEAKING)
                    await self.send_audio_bytes(pcm_bytes)

            # Record assistant turn
            complete_reply = "".join(full_assistant_reply).strip()
            if complete_reply:
                clean_reply = OllamaStreamingClient.clean_voice_text(complete_reply)
                self.conversation_history.append({"role": "assistant", "content": clean_reply})
                logger.info(f"Assistant: '{clean_reply}'")
                await self.send_json({
                    "type": "transcript",
                    "role": "assistant",
                    "text": clean_reply
                })

            # Sent all audio chunks for this turn
            await self.send_json({"type": "turn_complete"})

        except asyncio.CancelledError:
            logger.info("Active generation task was cancelled during synthesis.")
            raise
        except Exception as e:
            logger.error(f"Error in response generation: {e}", exc_info=True)
            await self.set_state(SessionState.LISTENING)
        finally:
            self.active_generation_task = None

    async def run(self):
        """Main WebSocket loop handling incoming binary audio and JSON control frames."""
        self.is_active = True
        await self.set_state(SessionState.LISTENING)

        try:
            while self.is_active:
                message = await self.ws.receive()

                if message.get("type") == "websocket.disconnect":
                    break

                # Case A: Binary message (16kHz Int16 PCM chunk from browser microphone)
                if "bytes" in message and message["bytes"]:
                    pcm_chunk = message["bytes"]
                    self.chunks_received_count += 1
                    
                    # Inspect amplitude of incoming audio
                    import numpy as np
                    raw_arr = np.frombuffer(pcm_chunk, dtype=np.int16)
                    max_amp = int(np.max(np.abs(raw_arr))) if len(raw_arr) > 0 else 0
                    
                    if self.chunks_received_count == 1:
                        logger.info(f"🎙️ Receiving live microphone audio stream (initial amp: {max_amp})")
                    elif self.chunks_received_count % 30 == 0:  # Log every ~1 second
                        logger.info(f"Audio chunk #{self.chunks_received_count}: peak amp = {max_amp}")

                    is_speaking = (self.state == SessionState.SPEAKING or self.state == SessionState.THINKING)
                    
                    utterance, barge_in = self.detector.process(
                        chunk_pcm16=pcm_chunk,
                        is_assistant_speaking=is_speaking
                    )

                    if barge_in:
                        await self.handle_barge_in()

                    if utterance is not None:
                        # Process user speech turn
                        await self.process_utterance(utterance)

                # Case B: Text JSON control messages
                elif "text" in message and message["text"]:
                    data = self.ws.json() if hasattr(self.ws, "json") else None
                    # Client can signal playback queue drained
                    if message["text"].startswith("{"):
                        import json
                        try:
                            msg_obj = json.loads(message["text"])
                            if msg_obj.get("type") == "playback_ended":
                                if self.state == SessionState.SPEAKING:
                                    await self.set_state(SessionState.LISTENING)
                            elif msg_obj.get("type") == "stop":
                                self.is_active = False
                                break
                        except Exception:
                            pass

        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected normally.")
        except Exception as e:
            logger.error(f"WebSocket session error: {e}", exc_info=True)
        finally:
            self.is_active = False
            if self.active_generation_task and not self.active_generation_task.done():
                self.active_generation_task.cancel()
            self.detector.reset()
            logger.info("Session closed cleanly.")
