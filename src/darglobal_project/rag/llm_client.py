"""Direct OpenRouter LLM client with Gemini fallback using httpx."""

import asyncio
import logging
from typing import Optional
import httpx
from darglobal_project.config import settings

logger = logging.getLogger(__name__)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_GENERATE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class OpenRouterClient:
    """Async client for generating completions via OpenRouter with automatic Gemini fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        gemini_chat_model: Optional[str] = None,
        timeout: float = 45.0,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.openrouter_api_key
        self.model = model or settings.openrouter_model
        self.gemini_api_key = gemini_api_key if gemini_api_key is not None else settings.gemini_api_key
        self.gemini_chat_model = gemini_chat_model or settings.gemini_chat_model
        self.timeout = timeout
        self.max_retries = max_retries

    async def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        client: Optional[httpx.AsyncClient] = None,
    ) -> str:
        """Call OpenRouter completions endpoint with provided prompts.

        When OpenRouter returns HTTP 429 due to quota exhaustion or rate limits,
        automatically falls back to Google Gemini using GEMINI_API_KEY.
        """
        should_close = False
        if client is None:
            client = httpx.AsyncClient(timeout=self.timeout)
            should_close = True

        try:
            # 1. Primary: Attempt OpenRouter if API key configured
            if self.api_key:
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": temperature,
                }
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://darglobal-wasalt-assistant.local",
                    "X-Title": "DarGlobal Wasalt Assistant",
                }

                for attempt in range(1, self.max_retries + 1):
                    try:
                        resp = await client.post(
                            OPENROUTER_API_URL,
                            headers=headers,
                            json=payload,
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            choices = data.get("choices", [])
                            if not choices:
                                raise ValueError("No choices returned from OpenRouter response")
                            message = choices[0].get("message", {}).get("content", "")
                            return message.strip()

                        if resp.status_code == 429:
                            logger.warning(
                                f"[OpenRouter] Quota/Rate limited (429) on attempt {attempt}/{self.max_retries}. "
                                "Falling back to Gemini LLM provider."
                            )
                            # Fallback immediately to Gemini on 429 quota exhaustion
                            return await self._generate_gemini_fallback(
                                system_prompt=system_prompt,
                                user_prompt=user_prompt,
                                temperature=temperature,
                                client=client,
                            )

                        resp.raise_for_status()

                    except (httpx.TimeoutException, httpx.NetworkError) as net_err:
                        if attempt == self.max_retries:
                            logger.warning(
                                f"[OpenRouter] Network failure after {self.max_retries} attempts: {type(net_err).__name__}. "
                                "Falling back to Gemini LLM provider."
                            )
                            return await self._generate_gemini_fallback(
                                system_prompt=system_prompt,
                                user_prompt=user_prompt,
                                temperature=temperature,
                                client=client,
                            )
                        await asyncio.sleep(1.5 * attempt)
            else:
                logger.info("[LLM] No OpenRouter API key found. Using Gemini provider.")
                return await self._generate_gemini_fallback(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    client=client,
                )

            raise RuntimeError(f"Failed to obtain response from OpenRouter after {self.max_retries} attempts.")

        finally:
            if should_close:
                await client.aclose()

    async def _generate_gemini_fallback(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        client: Optional[httpx.AsyncClient] = None,
    ) -> str:
        """Fallback LLM generation using official Google Gemini generateContent API via httpx."""
        if not self.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured in .env or environment variables. "
                "Unable to execute Gemini chat fallback."
            )

        headers = {
            "x-goog-api-key": self.gemini_api_key,
            "Content-Type": "application/json",
        }

        payload = {
            "system_instruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
            },
        }

        should_close = False
        if client is None:
            client = httpx.AsyncClient(timeout=self.timeout)
            should_close = True

        # Models to try: primary configured model (e.g. gemini-2.5-flash), with resilient backup (gemini-3-flash-preview)
        candidate_models = [self.gemini_chat_model]
        if self.gemini_chat_model != "gemini-3-flash-preview":
            candidate_models.append("gemini-3-flash-preview")

        try:
            last_err = None
            for model_name in candidate_models:
                url = f"{GEMINI_GENERATE_BASE_URL}/{model_name}:generateContent"
                try:
                    resp = await client.post(url, headers=headers, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        candidates = data.get("candidates", [])
                        if candidates and "content" in candidates[0]:
                            parts = candidates[0]["content"].get("parts", [])
                            if parts and "text" in parts[0]:
                                logger.info(f"[Gemini Fallback] Successfully generated response using {model_name}.")
                                return parts[0]["text"].strip()
                        raise ValueError("Empty or malformed candidate in Gemini response")

                    if resp.status_code == 404:
                        logger.warning(
                            f"[Gemini Fallback] Model '{model_name}' returned 404. Attempting alternative Gemini model."
                        )
                        last_err = RuntimeError(f"Gemini model {model_name} not available (404)")
                        continue

                    resp.raise_for_status()

                except Exception as exc:
                    last_err = exc
                    logger.warning(f"[Gemini Fallback] Error with {model_name}: {type(exc).__name__}")

            raise RuntimeError(f"Gemini fallback failed: {last_err}") from last_err

        finally:
            if should_close:
                await client.aclose()
