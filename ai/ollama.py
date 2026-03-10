"""
Local Ollama AI backend.

Runs entirely on your machine — no API key, no cloud, full privacy.
Requires Ollama to be running locally (default: http://localhost:11434).
"""
import logging
from typing import Optional
from urllib.parse import urlparse

from .base import AIBackend

logger = logging.getLogger(__name__)


class OllamaBackend(AIBackend):
    """Local Ollama backend for privacy-first document processing."""

    # Built-in vision model prefixes. Override via config "ollama.vision_models".
    DEFAULT_VISION_MODELS = ("qwen3.5", "llava", "bakllava", "moondream", "minicpm-v", "gemma3")

    def __init__(self, model: str = "qwen3.5:9b", host: str = "http://localhost:11434",
                 vision_models: Optional[list] = None):
        self._model = model

        # P2 security: only allow localhost to prevent SSRF via malicious config
        parsed = urlparse(host)
        if parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError(
                f"Ollama host must be localhost, got {parsed.hostname!r}. "
                f"Remote Ollama endpoints are blocked to prevent accidental data leakage."
            )
        self._host = host.rstrip("/")
        self._vision_models = tuple(vision_models) if vision_models else self.DEFAULT_VISION_MODELS

    def call(self, prompt: str, images: Optional[list] = None, text: Optional[str] = None) -> str:
        """
        Send prompt + text to local Ollama instance.

        Note: not all Ollama models support vision (images).
        Images are sent but may be ignored unless using a vision-capable model
        (e.g. qwen3.5, gemma3, llava).

        Args:
            prompt: instruction prompt
            images: ignored for non-vision models
            text: optional extracted document text

        Returns:
            raw model response text
        """
        try:
            import ollama as ollama_lib
        except ImportError:
            raise ImportError(
                "ollama package is required for Ollama backend. "
                "Install it with: pip install ollama"
            )

        # qwen3/qwen3.5 specific: /no_think disables the model's built-in
        # chain-of-thought "thinking" mode, which otherwise generates a lengthy
        # <think>...</think> block before the actual answer. This saves 30-60%
        # inference time. See: https://huggingface.co/Qwen/Qwen3-8B#thinking-mode
        if "qwen3" in self._model.lower():
            full_prompt = "/no_think\n" + prompt
        else:
            full_prompt = prompt
        if text:
            full_prompt += f"\n\n--- TEXT CONTENT ---\n{text}"

        # Vision support: encode images as base64 if the model supports it
        kwargs = {}
        if images:
            if not any(v in self._model.lower() for v in self._vision_models):
                logger.warning(
                    f"Model {self._model} may not support vision. "
                    f"Images will be sent but might be ignored. "
                    f"Consider a vision model (qwen3.5, gemma3, llava, etc.) for scanned PDFs."
                )
            import base64
            import io

            encoded_images = []
            for img in images:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                encoded_images.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
            kwargs["images"] = encoded_images

        try:
            client = ollama_lib.Client(host=self._host)
            response = client.generate(
                model=self._model,
                prompt=full_prompt,
                **kwargs,
            )
            # Support both dict (old SDK) and object (new SDK) response types
            if hasattr(response, "response"):
                return response.response or ""
            return response.get("response", "")
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            raise
