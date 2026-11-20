"""Direct Gemini embedding client using httpx and gemini-embedding-001."""

import asyncio
import logging
from typing import Optional
import httpx
from darglobal_project.config import settings

logger = logging.getLogger(__name__)

GEMINI_EMBED_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"
)
EXPECTED_DIMENSION = 768
TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_QUERY = "RETRIEVAL_QUERY"


class GeminiEmbedder:
    """Async client for generating text embeddings using Google's gemini-embedding-001."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_concurrency: int = 5,
        timeout: float = 30.0,
        max_retries: int = 6,
        dimension: int = EXPECTED_DIMENSION,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.gemini_api_key
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.timeout = timeout
        self.max_retries = max_retries
        self.dimension = dimension

    async def embed_single(
        self,
        text: str,
        task_type: str = TASK_DOCUMENT,
        client: Optional[httpx.AsyncClient] = None,
    ) -> list[float]:
        """Generate embedding for a single text using embedContent with 768 dimensions."""
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured in .env or environment variables. "
                "Please configure a valid key before generating embeddings."
            )

        sanitized_text = (text or "").strip()
        if not sanitized_text:
            sanitized_text = "property"

        payload = {
            "content": {
                "parts": [{"text": sanitized_text}]
            },
            "outputDimensionality": self.dimension,
            "taskType": task_type,
        }
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        should_close = False
        if client is None:
            client = httpx.AsyncClient(timeout=self.timeout)
            should_close = True

        try:
            async with self.semaphore:
                for attempt in range(1, self.max_retries + 1):
                    try:
                        resp = await client.post(
                            GEMINI_EMBED_ENDPOINT,
                            headers=headers,
                            json=payload,
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            embedding = data.get("embedding", {}).get("values", [])
                            if not embedding:
                                raise ValueError("Empty embedding vector returned by Gemini API")
                            if len(embedding) != self.dimension:
                                raise ValueError(
                                    f"Expected {self.dimension}-dimensional embedding, got {len(embedding)}"
                                )
                            return embedding

                        if resp.status_code == 429:
                            # Rate limit encountered - wait for quota sliding window (Gemini free tier has 15 RPM)
                            wait_time = max(15.0, 10.0 * attempt)
                            logger.warning(
                                f"[Gemini Embeddings] Rate limited (429). Retrying in {wait_time:.1f}s (attempt {attempt}/{self.max_retries})"
                            )
                            await asyncio.sleep(wait_time)
                            continue

                        # Other HTTP error
                        resp.raise_for_status()

                    except (httpx.TimeoutException, httpx.NetworkError) as net_err:
                        if attempt == self.max_retries:
                            raise RuntimeError(f"Network error calling Gemini API: {net_err}") from net_err
                        await asyncio.sleep(1.5 * attempt)

                raise RuntimeError(
                    f"Failed to generate embedding after {self.max_retries} attempts."
                )
        finally:
            if should_close:
                await client.aclose()

    async def embed_document(self, text: str, client: Optional[httpx.AsyncClient] = None) -> list[float]:
        """Convenience method to embed a document with taskType=RETRIEVAL_DOCUMENT."""
        return await self.embed_single(text, task_type=TASK_DOCUMENT, client=client)

    async def embed_query(self, text: str, client: Optional[httpx.AsyncClient] = None) -> list[float]:
        """Convenience method to embed a user query with taskType=RETRIEVAL_QUERY."""
        return await self.embed_single(text, task_type=TASK_QUERY, client=client)

    async def embed_batch(
        self,
        texts: list[str],
        task_type: str = TASK_DOCUMENT,
        progress_callback: Optional[callable] = None,
    ) -> list[list[float]]:
        """Embed a list of texts with controlled concurrency."""
        if not texts:
            return []

        completed_count = 0

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async def _embed_with_progress(text: str) -> list[float]:
                nonlocal completed_count
                res = await self.embed_single(text, task_type=task_type, client=client)
                completed_count += 1
                if progress_callback:
                    progress_callback(completed_count, len(texts))
                return res

            tasks = [_embed_with_progress(text) for text in texts]
            embeddings = await asyncio.gather(*tasks)

        return list(embeddings)
