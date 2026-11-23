import json
import logging
import re
import httpx
from typing import AsyncGenerator, List, Dict, Optional

logger = logging.getLogger(__name__)

class OllamaStreamingClient:
    """Async streaming client for Ollama LLM with cooperative cancellation."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "gemma4",
        keep_alive: int = -1,
        num_predict: int = 1024
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        try:
            self.keep_alive = int(keep_alive)
        except (ValueError, TypeError):
            self.keep_alive = -1
        self.num_predict = num_predict
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))

    async def close(self):
        await self.client.aclose()

    @staticmethod
    def clean_voice_text(text: str) -> str:
        """Strip markdown asterisks, hashes, backticks, emojis, and thought tags for natural TTS."""
        # Strip <think>...</think>
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Strip markdown bold/italic
        text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
        # Strip markdown headers (#, ##)
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        # Strip bullet points
        text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
        # Strip code blocks
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        return text.strip()

    async def stream_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str
    ) -> AsyncGenerator[str, None]:
        """Stream response tokens from Ollama /v1/chat/completions or /api/chat.
        
        Yields raw text tokens as they arrive. If the surrounding asyncio.Task is cancelled,
        the HTTP stream is immediately aborted to release GPU resources.
        """
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}] + messages,
            "stream": True,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": self.num_predict,
            },
            "keep_alive": self.keep_alive
        }

        url = f"{self.base_url}/api/chat"

        try:
            async with self.client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        chunk = data.get("message", {}).get("content", "")
                        if chunk:
                            yield chunk
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama HTTP error {e.response.status_code}: {e}")
            yield "I encountered a communication error with the language model."
        except Exception as e:
            logger.error(f"Ollama stream exception: {e}")
            raise
