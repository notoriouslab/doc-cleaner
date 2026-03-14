"""
xAI Grok backend.

Uses xAI's official REST API directly so doc-cleaner can support Grok without
adding a heavyweight SDK dependency.
"""
import base64
import io
import json
import logging
from typing import Optional
from urllib import error, request

from .base import AIBackend

logger = logging.getLogger(__name__)


class GrokBackend(AIBackend):
    """xAI Grok backend via the REST API."""

    def __init__(
        self,
        api_key: str,
        model: str = "grok-4",
        base_url: str = "https://api.x.ai/v1",
        timeout: int = 120,
        store: bool = False,
    ):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._store = store

    def call(self, prompt: str, images: Optional[list] = None, text: Optional[str] = None) -> str:
        """Send prompt + optional images/text to xAI Grok."""
        user_parts = []

        if images:
            for img in images:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
                user_parts.append({
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{encoded}",
                    "detail": "high",
                })

        if text:
            user_parts.append({
                "type": "input_text",
                "text": f"--- TEXT CONTENT ---\n{text}",
            })

        if not user_parts:
            user_parts.append({
                "type": "input_text",
                "text": "No extracted text or images were available.",
            })

        payload = {
            "model": self._model,
            "store": self._store,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": user_parts,
                },
            ],
            "text": {
                "format": {
                    "type": "text",
                }
            },
        }

        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self._base_url}/responses",
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            logger.error(f"Grok API call failed: {exc.code} {detail}")
            raise RuntimeError(f"xAI API returned HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            logger.error(f"Grok API call failed: {exc}")
            raise

        output = data.get("output", [])
        chunks = []
        for item in output:
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    text_chunk = content.get("text")
                    if text_chunk:
                        chunks.append(text_chunk)

        if chunks:
            return "\n".join(chunks)

        logger.error(f"Unexpected Grok response shape: {data}")
        raise RuntimeError("xAI API returned no output_text content")