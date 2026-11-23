# Gemma-Talk

A **100% local, zero-API-key, hands-free conversational voice assistant** (Gemini Live-style) powered by Ollama, Silero VAD, faster-whisper, and Kokoro-82M TTS.

---

## ⚡ Highlights

- **Hands-Free Conversational Voice (No Push-to-Talk):** Continuous duplex streaming audio over WebSockets at 16kHz linear PCM.
- **True Barge-In Interruption (<10ms cutoff):** Interrupt the assistant at any time while it is speaking. In-flight LLM token generation is immediately aborted via cooperative cancellation tokens, and the client's audio buffer is flushed instantly.
- **Sub-1.5s Perceived Latency via Sentence Streaming:** Tokens stream from Ollama into an abbreviation-aware sentence boundary detector and directly into Kokoro-82M ONNX TTS without waiting for the full LLM response.
- **Hardware & VRAM Discipline:** Ollama runs on Apple Silicon GPU/Metal with `keep_alive=-1` to prevent cold-start reloads. VAD, Whisper, and Kokoro run on CPU using ONNX Runtime with multithreading.
- **Acoustic Hygiene:** Browser-level hardware acoustic echo cancellation (AEC), noise suppression, and auto-gain control to eliminate self-interruption loops.
- **Indic Language Ready:** Support for multilingual Whisper and specialized models (AI4Bharat IndicWhisper, IndicConformer).

---

## 🏗️ Architecture

```
[ Browser Microphone ]
        │  (getUserMedia: echoCancellation=true, noiseSuppression=true, autoGainControl=true)
        ▼
[ Web Audio Pipeline ] ──(Continuous 16kHz Int16 PCM, 512-sample chunks)──► [ WebSocket Duplex ]
                                                                                   │
 ┌──────────────────────── Python Backend Pipeline ────────────────────────────────┴─────────┐
 │                                                                                           │
 │  1. Silero VAD v5 (ONNX CPU, ~1ms/chunk):                                                 │
 │     • Stateful 576-sample tensor (64-sample context + 512 audio chunk)                    │
 │     • 4-chunk (~128ms) pre-roll buffer to preserve leading consonant onsets               │
 │     • Barge-in trigger when speech prob >= 0.6 during playback                            │
 │                                                                                           │
 │  2. faster-whisper (base.en / small, int8 CPU, greedy beam_size=1):                       │
 │     • Direct in-memory Float32 NumPy PCM array transcription                              │
 │                                                                                           │
 │  3. Ollama (gemma4 on Metal GPU, keep_alive=-1):                                          │
 │     • Cooperative async token streaming with instant cancellation                         │
 │     • Voice-optimized system prompt (short conversational sentences, no markdown)         │
 │                                                                                           │
 │  4. Sentence Chunker:                                                                     │
 │     • Token-to-sentence splitter with abbreviation protection (Dr., Mr., e.g., $3.50)     │
 │                                                                                           │
 │  5. Kokoro-82M v1.0 ONNX (CPU 4-thread multithreaded, 24kHz):                             │
 │     • Sub-800ms synthesis per sentence into 24kHz Int16 PCM chunks                        │
 └────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                          │ (Binary 24kHz PCM chunks)
                                          ▼
                           [ Web Audio API Queue Player ]
                                          │
             (Seamless gapless playback ──┴── Instantaneous .stop() flush on barge-in)
```

---

## 🚀 Quick Start

### 1. Prerequisites
- **macOS (Apple Silicon)** with 16GB+ Unified Memory (or Linux/Windows with NVIDIA GPU).
- **Ollama** installed and running (`ollama serve`).
- **Node.js** (v18+) & **`uv`** package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

### 2. Automated Setup
Run the single bootstrap script to pull the model, download ONNX weights, and install all dependencies:
```bash
git clone <repo-url> gemma-talk
cd gemma-talk
./setup.sh
```

### 3. Launch the Assistant
```bash
./start.sh
```
This automatically starts:
- **Backend Server:** `http://localhost:8000` (pre-warms all neural weights in ~2.4s)
- **Frontend Client:** `http://localhost:5173`

Open your browser at **[http://localhost:5173](http://localhost:5173)**, click **Start Call**, and begin speaking naturally.

---

## 🌐 Multilingual & Indic Language Support

To enable transcription for Indian languages (Hindi, Telugu, Tamil, Kannada, Marathi, Bengali, etc.):

1. Open `.env` and set:
   ```ini
   WHISPER_MODEL_SIZE=small
   ```
2. In `server/app/pipeline/stt.py`, set:
   ```python
   language=None  # Auto-detect language, or specify "hi", "te", "ta", etc.
   ```
3. For native code-mixed speech (Hinglish/Tanglish), you can drop in **AI4Bharat IndicWhisper** (`ai4bharat/indic-whisper-*`).

---

## ⚙️ Configuration Reference (`.env`)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `SERVER_HOST` | `0.0.0.0` | Backend bind host |
| `SERVER_PORT` | `8000` | Backend port |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `gemma4` | Target LLM model |
| `OLLAMA_KEEP_ALIVE` | `-1` | Keeps model resident in GPU VRAM indefinitely |
| `AUDIO_SAMPLE_RATE` | `16000` | STT and VAD sample rate |
| `AUDIO_CHUNK_SAMPLES` | `512` | 32ms audio frame chunk size |
| `VAD_THRESHOLD` | `0.35` | Silero VAD speech detection threshold |
| `VAD_SILENCE_DURATION_MS` | `400` | Silence duration to mark end of utterance |
| `WHISPER_MODEL_SIZE` | `base.en` | Whisper model (`base.en`, `small`, etc.) |
| `WHISPER_COMPUTE_TYPE` | `int8` | Quantization for CPU inference |
| `TTS_VOICE` | `af_heart` | Kokoro voice embedding |
| `TTS_SPEED` | `1.0` | Speech playback rate |

---

## 🧪 Verification Benchmarks

Run the automated test suite to verify pipeline latency on your local hardware:
```bash
PYTHONPATH=. server/.venv/bin/python3 server/test_pipeline.py
```

- **Cold-Start Warmup:** ~2.4s (models pre-loaded before turn 1)
- **Barge-in Cancellation Latency:** 0.43 ms
- **LLM Time-To-First-Token (TTFT):** ~590 ms
- **TTS Synthesis:** ~800 ms per conversational sentence
- **End-to-End Latency:** ~1.35s – 1.45s (Sub-1.5s requirement met)

---

## 📄 License
MIT
