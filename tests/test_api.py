"""Integration tests for FastAPI endpoints, RAG pipeline, and frontend serving."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from darglobal_project.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    """Verify /health returns 200 with service status and collection count."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "records_indexed" in data
    assert data["records_indexed"] >= 90
    assert "model" in data


def test_static_frontend_served() -> None:
    """Verify root / serves the static HTML interface."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "DarGlobal &amp; Wasalt Real Estate" in response.text or "DarGlobal & Wasalt Real Estate" in response.text


def test_chat_endpoint_validation_empty_query() -> None:
    """Verify empty question returns HTTP 422 Unprocessable Entity."""
    response = client.post("/api/chat", json={"question": ""})
    assert response.status_code == 422


@patch("darglobal_project.rag.embeddings.GeminiEmbedder.embed_query", new_callable=AsyncMock)
@patch("darglobal_project.rag.retriever.VectorStore.query")
@patch("darglobal_project.rag.llm_client.OpenRouterClient.generate_response", new_callable=AsyncMock)
def test_chat_endpoint_success(mock_llm, mock_vector_query, mock_embed) -> None:
    """Verify /api/chat generates grounded answer and returns source citations."""
    mock_embed.return_value = [0.05] * 768
    mock_vector_query.return_value = [
        {
            "id": "wasalt-001",
            "document": "Title: Modern Villa in Sedra\nPrice: 3,500,000 SAR\nLocation: Riyadh\nSource URL: https://wasalt.sa/item1",
            "metadata": {
                "title": "Modern Villa in Sedra",
                "source_url": "https://wasalt.sa/item1",
                "source_platform": "Wasalt",
                "location": "Riyadh",
                "price_formatted": "3,500,000 SAR",
                "property_type": "Villa",
            },
            "similarity": 0.85,
        }
    ]
    # First call: classifier JSON, Second call: synthesized answer
    mock_llm.side_effect = [
        '{"intent": "PROPERTY_SEARCH", "relevant_platforms": ["Wasalt"], "needs_retrieval": true, "rewritten_query": "Modern Villa in Sedra Riyadh", "confidence": 0.95}',
        "Based on Wasalt records, a Modern Villa in Sedra, Riyadh is available for 3,500,000 SAR (https://wasalt.sa/item1).",
    ]

    response = client.post("/api/chat", json={"question": "Find villas in Sedra"})
    assert response.status_code == 200
    data = response.json()

    assert "Sedra" in data["answer"]
    assert len(data["sources"]) == 1
    src = data["sources"][0]
    assert src["title"] == "Modern Villa in Sedra"
    assert src["source_url"] == "https://wasalt.sa/item1"
    assert src["source_platform"] == "Wasalt"
    assert src["location"] == "Riyadh"
    assert src["price"] == "3,500,000 SAR"
    assert data["intent"] == "PROPERTY_SEARCH"


@patch("darglobal_project.rag.embeddings.GeminiEmbedder.embed_query", new_callable=AsyncMock)
@patch("darglobal_project.rag.retriever.VectorStore.query")
@patch("darglobal_project.rag.llm_client.OpenRouterClient.generate_response", new_callable=AsyncMock)
def test_chat_endpoint_insufficient_evidence(mock_llm, mock_vector_query, mock_embed) -> None:
    """Verify that irrelevant questions return clear fallback message and zero citations without calling embedding."""
    # Classifier detects OUT_OF_SCOPE
    mock_llm.return_value = '{"intent": "OUT_OF_SCOPE", "relevant_platforms": [], "needs_retrieval": false, "rewritten_query": "", "confidence": 0.98}'

    response = client.post("/api/chat", json={"question": "What is the best recipe for cheesecake?"})
    assert response.status_code == 200
    data = response.json()

    assert "I do not have sufficient information" in data["answer"]
    assert data["sources"] == []
    # Embedding and vector store should not be called
    mock_embed.assert_not_called()
    mock_vector_query.assert_not_called()
