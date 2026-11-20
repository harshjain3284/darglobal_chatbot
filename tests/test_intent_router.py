"""Comprehensive tests for dynamic LLM-based intent classification and query routing."""

import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from darglobal_project.rag.service import ChatIntent, IntentAnalysis, RAGChatbot, parse_classifier_json


def test_parse_classifier_json_clean() -> None:
    """Verify parsing clean JSON string."""
    raw = '{"intent": "GREETING", "needs_retrieval": false, "relevant_platforms": [], "rewritten_query": "", "confidence": 0.95}'
    parsed = parse_classifier_json(raw)
    assert parsed["intent"] == "GREETING"
    assert parsed["needs_retrieval"] is False


def test_parse_classifier_json_markdown_wrapped() -> None:
    """Verify parsing JSON wrapped in markdown code fence."""
    raw = """```json
    {
      "intent": "PROPERTY_SEARCH",
      "relevant_platforms": ["Wasalt"],
      "needs_retrieval": true,
      "rewritten_query": "villas in Riyadh",
      "confidence": 0.92
    }
    ```"""
    parsed = parse_classifier_json(raw)
    assert parsed["intent"] == "PROPERTY_SEARCH"
    assert parsed["relevant_platforms"] == ["Wasalt"]


@pytest.mark.asyncio
async def test_dynamic_classification_greetings() -> None:
    """Verify natural language greetings are classified as GREETING without retrieval."""
    mock_llm = MagicMock()
    mock_llm.generate_response = AsyncMock(
        return_value='{"intent": "GREETING", "relevant_platforms": [], "needs_retrieval": false, "rewritten_query": "", "confidence": 0.98}'
    )
    bot = RAGChatbot(llm_client=mock_llm)

    greetings = ["Hello there!", "Good morning", "Salam, how are you?", "Hey friend"]
    for g in greetings:
        analysis = await bot.classify_intent_with_llm(g)
        assert analysis.intent == "GREETING"
        assert analysis.needs_retrieval is False
        assert analysis.confidence > 0.9


@pytest.mark.asyncio
async def test_dynamic_classification_company_comparison() -> None:
    """Verify different ways of asking about both companies map to COMPANY_COMPARISON with both platforms."""
    mock_llm = MagicMock()
    mock_llm.generate_response = AsyncMock(
        return_value=json.dumps({
            "intent": "COMPANY_COMPARISON",
            "relevant_platforms": ["DarGlobal", "Wasalt"],
            "needs_retrieval": True,
            "rewritten_query": "DarGlobal luxury developments Wasalt Saudi real estate properties",
            "confidence": 0.96,
        })
    )
    bot = RAGChatbot(llm_client=mock_llm)

    comparison_queries = [
        "Tell me about both companies.",
        "Compare DarGlobal and Wasalt.",
        "What is the difference between these two platforms?",
        "Show me properties from both sources.",
    ]
    for q in comparison_queries:
        analysis = await bot.classify_intent_with_llm(q)
        assert analysis.intent == "COMPANY_COMPARISON"
        assert "DarGlobal" in analysis.relevant_platforms
        assert "Wasalt" in analysis.relevant_platforms
        assert analysis.needs_retrieval is True
        assert len(analysis.rewritten_query) > 0


@pytest.mark.asyncio
async def test_dynamic_classification_property_search() -> None:
    """Verify property searches extract intent and rewrite search query."""
    mock_llm = MagicMock()
    mock_llm.generate_response = AsyncMock(
        return_value=json.dumps({
            "intent": "PROPERTY_SEARCH",
            "relevant_platforms": ["Wasalt"],
            "needs_retrieval": True,
            "rewritten_query": "4 bedroom modern villa Sedra Riyadh",
            "confidence": 0.95,
        })
    )
    bot = RAGChatbot(llm_client=mock_llm)

    analysis = await bot.classify_intent_with_llm("Could you show me modern villas in Sedra Riyadh with 4 bedrooms?")
    assert analysis.intent == "PROPERTY_SEARCH"
    assert analysis.needs_retrieval is True
    assert "Wasalt" in analysis.relevant_platforms
    assert analysis.rewritten_query == "4 bedroom modern villa Sedra Riyadh"


@pytest.mark.asyncio
async def test_dynamic_classification_project_search() -> None:
    """Verify DarGlobal luxury project searches route with DarGlobal platform."""
    mock_llm = MagicMock()
    mock_llm.generate_response = AsyncMock(
        return_value=json.dumps({
            "intent": "PROJECT_SEARCH",
            "relevant_platforms": ["DarGlobal"],
            "needs_retrieval": True,
            "rewritten_query": "Trump International Hotel Tower Dubai",
            "confidence": 0.97,
        })
    )
    bot = RAGChatbot(llm_client=mock_llm)

    analysis = await bot.classify_intent_with_llm("Tell me about the Trump tower project in Dubai")
    assert analysis.intent == "PROJECT_SEARCH"
    assert analysis.needs_retrieval is True
    assert "DarGlobal" in analysis.relevant_platforms
    assert "Trump" in analysis.rewritten_query


@pytest.mark.asyncio
async def test_dynamic_classification_out_of_scope() -> None:
    """Verify unrelated questions are classified as OUT_OF_SCOPE with needs_retrieval=False."""
    mock_llm = MagicMock()
    mock_llm.generate_response = AsyncMock(
        return_value=json.dumps({
            "intent": "OUT_OF_SCOPE",
            "relevant_platforms": [],
            "needs_retrieval": False,
            "rewritten_query": "",
            "confidence": 0.99,
        })
    )
    bot = RAGChatbot(llm_client=mock_llm)

    unrelated = ["What is Python?", "What is the capital of India?", "How do I bake a chocolate cake?"]
    for q in unrelated:
        analysis = await bot.classify_intent_with_llm(q)
        assert analysis.intent == "OUT_OF_SCOPE"
        assert analysis.needs_retrieval is False


@pytest.mark.asyncio
async def test_classifier_malformed_json_failure() -> None:
    """Verify malformed JSON returns None and answer() does NOT trigger retrieval."""
    mock_vector = MagicMock()
    mock_embed = MagicMock()
    mock_llm = MagicMock()
    mock_llm.generate_response = AsyncMock(return_value="I am an LLM and I think this is about villas {bad_json")

    bot = RAGChatbot(vector_store=mock_vector, embedder=mock_embed, llm_client=mock_llm)

    # 1. Direct classification returns None
    analysis = await bot.classify_intent_with_llm("luxury villas in Sedra")
    assert analysis is None

    # 2. End-to-end answer() handles failure safely without triggering retrieval
    result = await bot.answer("luxury villas in Sedra")
    assert result["intent"] is None
    assert "unable to process your request" in result["answer"].lower()
    assert result["sources"] == []

    # Assert Gemini embeddings and ChromaDB were NOT triggered
    mock_embed.embed_query.assert_not_called()
    mock_vector.query.assert_not_called()


@pytest.mark.asyncio
async def test_classifier_llm_network_error_failure() -> None:
    """Verify network error during classification returns safe message and does NOT trigger retrieval."""
    mock_vector = MagicMock()
    mock_embed = MagicMock()
    mock_llm = MagicMock()
    mock_llm.generate_response = AsyncMock(side_effect=RuntimeError("OpenRouter 503 Service Unavailable"))

    bot = RAGChatbot(vector_store=mock_vector, embedder=mock_embed, llm_client=mock_llm)

    # 1. Direct classification returns None
    analysis = await bot.classify_intent_with_llm("apartments in Riyadh")
    assert analysis is None

    # 2. End-to-end answer() handles failure safely without triggering retrieval
    result = await bot.answer("apartments in Riyadh")
    assert result["intent"] is None
    assert "unable to process your request" in result["answer"].lower()
    assert result["sources"] == []

    mock_embed.embed_query.assert_not_called()
    mock_vector.query.assert_not_called()


@pytest.mark.asyncio
async def test_classifier_invalid_intent_response_failure() -> None:
    """Verify unrecognized intent returned by LLM is treated as failure and does NOT trigger retrieval."""
    mock_vector = MagicMock()
    mock_embed = MagicMock()
    mock_llm = MagicMock()
    mock_llm.generate_response = AsyncMock(
        return_value='{"intent": "UNRECOGNIZED_ACTION_INTENT", "relevant_platforms": ["FakePlatform"], "needs_retrieval": true}'
    )

    bot = RAGChatbot(vector_store=mock_vector, embedder=mock_embed, llm_client=mock_llm)

    # 1. Direct classification returns None (never invent fake intent)
    analysis = await bot.classify_intent_with_llm("tell me something")
    assert analysis is None

    # 2. End-to-end answer() handles failure safely without triggering retrieval
    result = await bot.answer("tell me something")
    assert result["intent"] is None
    assert "unable to process your request" in result["answer"].lower()
    assert result["sources"] == []

    mock_embed.embed_query.assert_not_called()
    mock_vector.query.assert_not_called()


@pytest.mark.asyncio
async def test_chatbot_answer_greeting_flow() -> None:
    """Verify GREETING returns fast response without calling embedding or vector store."""
    mock_vector = MagicMock()
    mock_embed = MagicMock()
    mock_llm = MagicMock()
    mock_llm.generate_response = AsyncMock(
        return_value='{"intent": "GREETING", "relevant_platforms": [], "needs_retrieval": false, "rewritten_query": "", "confidence": 0.98}'
    )

    bot = RAGChatbot(vector_store=mock_vector, embedder=mock_embed, llm_client=mock_llm)
    result = await bot.answer("Hello")

    assert result["intent"] == "GREETING"
    assert "Hello" in result["answer"]
    assert result["sources"] == []

    # Embedding and vector store should NOT be called
    mock_embed.embed_query.assert_not_called()
    mock_vector.query.assert_not_called()
    # Only 1 LLM call was made (for classification)
    assert mock_llm.generate_response.call_count == 1


@pytest.mark.asyncio
async def test_chatbot_answer_property_search_flow() -> None:
    """Verify PROPERTY_SEARCH uses rewritten query for embedding and generates grounded answer with citations."""
    mock_vector = MagicMock()
    mock_vector.query.return_value = [
        {
            "id": "wasalt-001",
            "document": "Villa in Sedra Riyadh 3,500,000 SAR",
            "metadata": {
                "title": "Modern Villa in Sedra",
                "source_url": "https://wasalt.sa/1",
                "source_platform": "Wasalt",
                "location": "Riyadh",
                "price_formatted": "3,500,000 SAR",
            },
            "similarity": 0.88,
        }
    ]
    mock_embed = MagicMock()
    mock_embed.embed_query = AsyncMock(return_value=[0.05] * 768)

    mock_llm = MagicMock()
    # Call 1: classification, Call 2: grounded answer
    mock_llm.generate_response = AsyncMock(
        side_effect=[
            json.dumps({
                "intent": "PROPERTY_SEARCH",
                "relevant_platforms": ["Wasalt"],
                "needs_retrieval": True,
                "rewritten_query": "Sedra Riyadh villa 3.5m SAR",
                "confidence": 0.95,
            }),
            "Based on Wasalt listings, a Modern Villa in Sedra is available for 3,500,000 SAR.",
        ]
    )

    bot = RAGChatbot(vector_store=mock_vector, embedder=mock_embed, llm_client=mock_llm)
    result = await bot.answer("Show me villas in Sedra")

    assert result["intent"] == "PROPERTY_SEARCH"
    assert result["rewritten_query"] == "Sedra Riyadh villa 3.5m SAR"
    assert "Modern Villa" in result["answer"]
    assert len(result["sources"]) == 1

    # Verify query embedding used the rewritten query
    mock_embed.embed_query.assert_called_once_with("Sedra Riyadh villa 3.5m SAR")
    mock_vector.query.assert_called_once()
    assert mock_llm.generate_response.call_count == 2
