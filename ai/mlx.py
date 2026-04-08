"""
Apple MLX backend for doc-cleaner.

Uses mlx-lm for fast on-device inference on Apple Silicon Macs.
No API key, no cloud, full privacy. Significantly faster than Ollama on M-series chips.

Requires: pip install mlx-lm
"""
import logging
from typing import Optional

from .base import AIBackend

logger = logging.getLogger(__name__)


class MLXBackend(AIBackend):
    """Apple MLX backend using mlx-lm for Apple Silicon acceleration."""

    def __init__(self, model: str = "mlx-community/Qwen3-4B-4bit",
                 max_tokens: int = 4096):
        self._model_name = model
        self._max_tokens = max_tokens
        self._model = None
        self._tokenizer = None
        self._generate = None

    def _load_model(self):
        """Lazy-load model and generate function on first call."""
        if self._model is not None:
            return
        try:
            from mlx_lm import load, generate
        except ImportError:
            raise ImportError(
                "mlx-lm is required for the MLX backend. "
                "Install with: pip install mlx-lm"
            )
        logger.info(f"Loading MLX model: {self._model_name} (first call, may take a moment)")
        self._model, self._tokenizer = load(self._model_name)
        self._generate = generate
        logger.info("MLX model loaded.")

    def call(self, prompt: str, images: Optional[list] = None, text: Optional[str] = None) -> str:
        """
        Generate text using mlx-lm.

        Args:
            prompt: instruction prompt
            images: not supported (MLX text models only; logged as warning)
            text: optional extracted document text

        Returns:
            raw model response text
        """
        self._load_model()

        if images:
            logger.warning(
                "MLX text backend does not support images. "
                "Images will be ignored. Consider mlx-vlm for vision tasks."
            )

        # Qwen3 specific: /no_think disables chain-of-thought thinking mode,
        # saving 30-60% inference time. Same approach as the Ollama backend.
        if "qwen3" in self._model_name.lower():
            full_prompt = "/no_think\n" + prompt
        else:
            full_prompt = prompt
        if text:
            full_prompt += f"\n\n--- TEXT CONTENT ---\n{text}"

        # Use chat template if tokenizer supports it
        if hasattr(self._tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": full_prompt}]
            formatted = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = full_prompt

        try:
            response = self._generate(
                self._model,
                self._tokenizer,
                prompt=formatted,
                max_tokens=self._max_tokens,
                verbose=False,
            )
            return response or ""
        except Exception as e:
            logger.error(f"MLX generation failed: {e}")
            raise
