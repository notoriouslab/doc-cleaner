"""
NVIDIA NIM backend.

Uses NVIDIA's OpenAI-compatible chat completions API (integrate.api.nvidia.com).
Supports vision models like Qwen 3.5 VL for document understanding.
"""
import base64
import io
import json
import logging
import ssl
from typing import Optional
from urllib import error, request

from .base import AIBackend

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "doc-cleaner/0.1"


class NvidiaBackend(AIBackend):
    """NVIDIA NIM backend via the OpenAI-compatible chat completions API."""

    MAX_IMAGES = 1  # NVIDIA NIM limits to 1 image per request

    def __init__(
        self,
        api_key: str,
        model: str = "meta/llama-3.2-90b-vision-instruct",
        base_url: str = "https://integrate.api.nvidia.com/v1",
        timeout: int = 180,
    ):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def call(self, prompt: str, images: Optional[list] = None, text: Optional[str] = None) -> str:
        """Send prompt + optional images/text to NVIDIA NIM."""
        user_content = []

        if text:
            user_content.append({
                "type": "text",
                "text": f"--- TEXT CONTENT ---\n{text}",
            })

        if images:
            if len(images) > self.MAX_IMAGES:
                logger.warning(
                    f"Truncating {len(images)} images to {self.MAX_IMAGES} (NVIDIA limit)."
                )
                images = images[:self.MAX_IMAGES]

            for img in images:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded}",
                    },
                })

        if not user_content:
            user_content.append({
                "type": "text",
                "text": "No extracted text or images were available.",
            })

        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            "max_tokens": 16384,
            "temperature": 0.2,
            "stream": False,
        }

        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self._base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": DEFAULT_USER_AGENT,
            },
            method="POST",
        )

        ssl_context = None
        try:
            import certifi
            ssl_context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ssl_context = ssl.create_default_context()

        try:
            with request.urlopen(req, timeout=self._timeout, context=ssl_context) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            logger.error(f"NVIDIA NIM API call failed: {exc.code} {detail}")
            raise RuntimeError(f"NVIDIA NIM API returned HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            logger.error(f"NVIDIA NIM API call failed: {exc}")
            raise

        choices = data.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content")
            if isinstance(content, str) and content:
                return content

        logger.error(f"Unexpected NVIDIA response shape: {data}")
        raise RuntimeError("NVIDIA NIM API returned no message content")
