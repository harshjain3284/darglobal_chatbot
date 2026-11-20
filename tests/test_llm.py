"""Unit tests for OpenRouter LLM client and Gemini fallback."""

from unittest.mock import AsyncMock, patch
import httpx
import pytest
from darglobal_project.rag.llm_client import OpenRouterClient


def test_openrouter_missing_key() -> None:
    """Verify error raised when both OpenRouter and Gemini keys are missing."""
    client = OpenRouterClient(api_key="", gemini_api_key="")
    with pytest.raises(ValueError, match="GEMINI_API_KEY is not configured"):
        import asyncio
        asyncio.run(client.generate_response("System prompt", "User prompt"))


@pytest.mark.asyncio
async def test_openrouter_generate_response_success() -> None:
    """Verify OpenRouter response is returned when OpenRouter returns 200 OK."""
    client = OpenRouterClient(api_key="mock-openrouter-key", model="openrouter/free")
    mock_resp = httpx.Response(
        status_code=200,
        json={
            "choices": [
                {
                    "message": {
                        "content": "Sedra in Riyadh has modern villas starting from 3,500,000 SAR."
                    }
                }
            ]
        },
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        answer = await client.generate_response("You are an assistant", "Tell me about Sedra")

        assert "Sedra in Riyadh" in answer
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        headers = kwargs["headers"]
        payload = kwargs["json"]

        assert headers["Authorization"] == "Bearer mock-openrouter-key"
        assert payload["model"] == "openrouter/free"
        assert payload["messages"][0]["content"] == "You are an assistant"
        assert payload["messages"][1]["content"] == "Tell me about Sedra"


@pytest.mark.asyncio
async def test_openrouter_429_triggers_gemini_fallback() -> None:
    """Verify that OpenRouter 429 rate limit triggers seamless Gemini generateContent fallback."""
    client = OpenRouterClient(
        api_key="mock-openrouter-key",
        gemini_api_key="mock-gemini-key",
        gemini_chat_model="gemini-2.5-flash",
        max_retries=1,
    )
    resp_429 = httpx.Response(
        status_code=429,
        json={"error": {"message": "Rate limit exceeded: free-models-per-day"}},
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
    )
    resp_gemini_200 = httpx.Response(
        status_code=200,
        json={
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Gemini fallback: Riyadh villas are available in Sedra."}]
                    }
                }
            ]
        },
        request=httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = [resp_429, resp_gemini_200]
        answer = await client.generate_response("System", "Find villas")

        assert "Gemini fallback" in answer
        assert mock_post.call_count == 2

        # Verify second call was to Gemini endpoint with x-goog-api-key
        gemini_call = mock_post.call_args_list[1]
        url = gemini_call[0][0]
        kwargs = gemini_call[1]
        assert "generativelanguage.googleapis.com" in url
        assert kwargs["headers"]["x-goog-api-key"] == "mock-gemini-key"


@pytest.mark.asyncio
async def test_gemini_fallback_model_404_resilience() -> None:
    """Verify Gemini fallback gracefully handles model deprecation 404 by trying alternate candidate model."""
    client = OpenRouterClient(
        api_key="mock-openrouter-key",
        gemini_api_key="mock-gemini-key",
        gemini_chat_model="gemini-2.5-flash",
        max_retries=1,
    )
    resp_openrouter_429 = httpx.Response(
        status_code=429,
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
    )
    resp_gemini_404 = httpx.Response(
        status_code=404,
        json={"error": {"message": "models/gemini-2.5-flash is no longer available"}},
        request=httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"),
    )
    resp_gemini_backup_200 = httpx.Response(
        status_code=200,
        json={
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Recovered via Gemini candidate model."}]
                    }
                }
            ]
        },
        request=httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite-preview:generateContent"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = [resp_openrouter_429, resp_gemini_404, resp_gemini_backup_200]
        answer = await client.generate_response("System", "Question")

        assert answer == "Recovered via Gemini candidate model."
        assert mock_post.call_count == 3


@pytest.mark.asyncio
async def test_gemini_fallback_model_429_cascading() -> None:
    """Verify Gemini fallback cascades to alternate model when first model returns 429 rate limit."""
    client = OpenRouterClient(
        api_key="",
        gemini_api_key="mock-gemini-key",
        gemini_chat_model="gemini-3-flash-preview",
    )
    resp_gemini_429 = httpx.Response(
        status_code=429,
        json={"error": {"message": "Resource has been exhausted (e.g. check quota)."}},
        request=httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent"),
    )
    resp_gemini_lite_200 = httpx.Response(
        status_code=200,
        json={
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Response from gemini-3.1-flash-lite-preview."}]
                    }
                }
            ]
        },
        request=httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite-preview:generateContent"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = [resp_gemini_429, resp_gemini_lite_200]
        answer = await client.generate_response("System", "Find properties")

        assert answer == "Response from gemini-3.1-flash-lite-preview."
        assert mock_post.call_count == 2
