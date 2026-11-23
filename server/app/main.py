import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from server.app import config
from server.app.pipeline.vad import SileroVAD
from server.app.pipeline.stt import SpeechToText
from server.app.pipeline.tts import TextToSpeech
from server.app.pipeline.llm import OllamaStreamingClient
from server.app.session import VoiceSession
from server.app.warmup import run_warmup

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("gemma-talk")

# Global singleton pipeline models
vad_model: SileroVAD = None
stt_model: SpeechToText = None
tts_model: TextToSpeech = None
llm_client: OllamaStreamingClient = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context: Load models and pre-warm on server startup."""
    global vad_model, stt_model, tts_model, llm_client

    logger.info("Initializing neural models for Gemma-Talk server...")
    vad_model = SileroVAD(config.SILERO_MODEL_PATH)
    
    stt_model = SpeechToText(
        engine=config.STT_ENGINE,
        indic_lang=config.AI4BHARAT_LANG,
        model_size=config.WHISPER_MODEL_SIZE,
        device=config.WHISPER_DEVICE,
        compute_type=config.WHISPER_COMPUTE_TYPE,
        cpu_threads=config.WHISPER_CPU_THREADS
    )

    tts_model = TextToSpeech(
        model_path=config.KOKORO_MODEL_PATH,
        voices_path=config.KOKORO_VOICES_PATH,
        default_voice=config.TTS_VOICE,
        speed=config.TTS_SPEED
    )

    llm_client = OllamaStreamingClient(
        base_url=config.OLLAMA_BASE_URL,
        model=config.OLLAMA_MODEL,
        keep_alive=config.OLLAMA_KEEP_ALIVE,
        num_predict=config.OLLAMA_NUM_PREDICT
    )

    # Run warmup routine
    await run_warmup(vad_model, stt_model, tts_model)

    yield

    # Teardown
    logger.info("Shutting down Gemma-Talk server...")
    if llm_client:
        await llm_client.close()

app = FastAPI(title="Gemma-Talk Real-Time Voice Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "stt_engine": config.STT_ENGINE,
        "indic_lang": config.AI4BHARAT_LANG,
        "ollama_model": config.OLLAMA_MODEL,
        "whisper_model": config.WHISPER_MODEL_SIZE,
        "tts_voice": config.TTS_VOICE,
        "sample_rate_in": config.AUDIO_SAMPLE_RATE,
        "sample_rate_out": config.TTS_SAMPLE_RATE
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = VoiceSession(
        websocket=websocket,
        vad_model=vad_model,
        stt=stt_model,
        tts=tts_model,
        llm=llm_client
    )
    await session.run()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server.app.main:app",
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        reload=False
    )
