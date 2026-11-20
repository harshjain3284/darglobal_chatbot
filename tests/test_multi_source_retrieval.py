"""Unit tests for platform-aware multi-source retrieval and deduplication."""

import json
from unittest.mock import AsyncMock, MagicMock
import pytest
from darglobal_project.rag.service import RAGChatbot


def test_retrieve_evidence_darglobal_only() -> None:
    """Verify DarGlobal-only query triggers query with filter_platform='DarGlobal'."""
    mock_vector = MagicMock()
    mock_vector.query.return_value = [
        {"id": "dg-1", "document": "Trump Tower", "metadata": {"source_platform": "DarGlobal", "source_url": "https://dg.com/1"}, "similarity": 0.8}
    ]
    bot = RAGChatbot(vector_store=mock_vector)

    matches = bot.retrieve_evidence(
        query_embedding=[0.01] * 768,
        relevant_platforms=["DarGlobal"],
        top_k=4,
    )
    assert len(matches) == 1
    assert matches[0]["metadata"]["source_platform"] == "DarGlobal"
    mock_vector.query.assert_called_once_with([0.01] * 768, top_k=4, filter_platform="DarGlobal")


def test_retrieve_evidence_wasalt_only() -> None:
    """Verify Wasalt-only query triggers query with filter_platform='Wasalt'."""
    mock_vector = MagicMock()
    mock_vector.query.return_value = [
        {"id": "w-1", "document": "Villa in Sedra", "metadata": {"source_platform": "Wasalt", "source_url": "https://wasalt.sa/1"}, "similarity": 0.85}
    ]
    bot = RAGChatbot(vector_store=mock_vector)

    matches = bot.retrieve_evidence(
        query_embedding=[0.01] * 768,
        relevant_platforms=["Wasalt"],
        top_k=4,
    )
    assert len(matches) == 1
    assert matches[0]["metadata"]["source_platform"] == "Wasalt"
    mock_vector.query.assert_called_once_with([0.01] * 768, top_k=4, filter_platform="Wasalt")


def test_retrieve_evidence_both_platforms_balanced() -> None:
    """Verify multi-source query retrieves from both platforms separately and combines results."""
    mock_vector = MagicMock()

    dg_record = {"id": "dg-1", "document": "Trump Tower", "metadata": {"source_platform": "DarGlobal", "source_url": "https://dg.com/1"}, "similarity": 0.82}
    wasalt_record = {"id": "w-1", "document": "Sedra Villa", "metadata": {"source_platform": "Wasalt", "source_url": "https://wasalt.sa/1"}, "similarity": 0.85}

    def mock_query(embedding, top_k, filter_platform):
        if filter_platform == "DarGlobal":
            return [dg_record]
        elif filter_platform == "Wasalt":
            return [wasalt_record]
        return []

    mock_vector.query.side_effect = mock_query
    bot = RAGChatbot(vector_store=mock_vector)

    matches = bot.retrieve_evidence(
        query_embedding=[0.01] * 768,
        relevant_platforms=["DarGlobal", "Wasalt"],
        top_k=4,
    )
    assert len(matches) == 2
    platforms = [m["metadata"]["source_platform"] for m in matches]
    assert "DarGlobal" in platforms
    assert "Wasalt" in platforms
    assert mock_vector.query.call_count == 2


def test_retrieve_evidence_duplicate_removal() -> None:
    """Verify duplicate records sharing the same ID or source URL are deduplicated."""
    mock_vector = MagicMock()
    duplicate_records = [
        {"id": "w-1", "document": "Sedra Villa", "metadata": {"source_platform": "Wasalt", "source_url": "https://wasalt.sa/1"}, "similarity": 0.85},
        {"id": "w-1", "document": "Sedra Villa Duplicate", "metadata": {"source_platform": "Wasalt", "source_url": "https://wasalt.sa/1"}, "similarity": 0.85},
        {"id": "w-2", "document": "Sedra Villa Duplicate URL", "metadata": {"source_platform": "Wasalt", "source_url": "https://wasalt.sa/1"}, "similarity": 0.84},
    ]
    mock_vector.query.return_value = duplicate_records
    bot = RAGChatbot(vector_store=mock_vector)

    matches = bot.retrieve_evidence(
        query_embedding=[0.01] * 768,
        relevant_platforms=["Wasalt"],
        top_k=4,
    )
    # Only 1 unique record should remain
    assert len(matches) == 1
    assert matches[0]["id"] == "w-1"


@pytest.mark.asyncio
async def test_both_platforms_missing_evidence_honestly_noted() -> None:
    """Verify that when one platform has no evidence above threshold, it is noted in prompt context."""
    mock_vector = MagicMock()
    # DarGlobal has relevant match, Wasalt has match below similarity threshold 0.35
    dg_match = {"id": "dg-1", "document": "Trump Tower Dubai", "metadata": {"title": "Trump Tower", "source_platform": "DarGlobal", "source_url": "https://dg.com/1"}, "similarity": 0.75}
    wasalt_low = {"id": "w-1", "document": "Unrelated", "metadata": {"title": "Low Sim", "source_platform": "Wasalt", "source_url": "https://w.sa/1"}, "similarity": 0.20}

    def mock_query(embedding, top_k, filter_platform):
        if filter_platform == "DarGlobal":
            return [dg_match]
        elif filter_platform == "Wasalt":
            return [wasalt_low]
        return []

    mock_vector.query.side_effect = mock_query

    mock_embed = MagicMock()
    mock_embed.embed_query = AsyncMock(return_value=[0.01] * 768)

    mock_llm = MagicMock()
    # Call 1: classifier, Call 2: answer
    mock_llm.generate_response = AsyncMock(
        side_effect=[
            json.dumps({
                "intent": "COMPANY_COMPARISON",
                "relevant_platforms": ["DarGlobal", "Wasalt"],
                "needs_retrieval": True,
                "rewritten_query": "DarGlobal Wasalt comparison",
                "confidence": 0.95,
            }),
            "DarGlobal is an international luxury developer. No verified Wasalt records met the threshold.",
        ]
    )

    bot = RAGChatbot(vector_store=mock_vector, embedder=mock_embed, llm_client=mock_llm)
    result = await bot.answer("Compare DarGlobal and Wasalt")

    assert result["intent"] == "COMPANY_COMPARISON"
    assert len(result["sources"]) == 1
    assert result["sources"][0]["source_platform"] == "DarGlobal"

    # Verify the LLM prompt received the missing platform observation note
    call_args_list = mock_llm.generate_response.call_args_list
    prompt_to_llm = call_args_list[1].kwargs["user_prompt"]
    assert "No verified evidence meeting the relevance threshold was found for Wasalt" in prompt_to_llm


@pytest.mark.asyncio
async def test_similarity_threshold_filtering_all_below_threshold() -> None:
    """Verify that if all matches fall below 0.35, the chatbot refuses without calling LLM synthesis."""
    mock_vector = MagicMock()
    mock_vector.query.return_value = [
        {"id": "dg-1", "document": "Low similarity", "metadata": {"source_platform": "DarGlobal", "source_url": "https://dg.com/1"}, "similarity": 0.25}
    ]
    mock_embed = MagicMock()
    mock_embed.embed_query = AsyncMock(return_value=[0.01] * 768)

    mock_llm = MagicMock()
    mock_llm.generate_response = AsyncMock(
        return_value=json.dumps({
            "intent": "PROPERTY_SEARCH",
            "relevant_platforms": ["DarGlobal"],
            "needs_retrieval": True,
            "rewritten_query": "properties in Antarctica",
            "confidence": 0.9,
        })
    )

    bot = RAGChatbot(vector_store=mock_vector, embedder=mock_embed, llm_client=mock_llm)
    result = await bot.answer("Show me properties in Antarctica")

    assert "I do not have sufficient information" in result["answer"]
    assert result["sources"] == []
    # Only 1 LLM call was made (for classification), answer generation was skipped
    assert mock_llm.generate_response.call_count == 1


def test_cosine_distance_to_similarity_conversion() -> None:
    """Verify Chroma cosine distance is converted correctly to similarity and filters at 0.35 threshold."""
    from darglobal_project.rag.retriever import VectorStore

    mock_client = MagicMock()
    mock_coll = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_coll

    store = VectorStore.__new__(VectorStore)
    store.collection = mock_coll

    # Simulate Chroma query returning 4 items with distinct cosine distances:
    # item 0: distance 0.10 -> similarity 0.90 (pass)
    # item 1: distance 0.65 -> similarity 0.35 (pass boundary)
    # item 2: distance 0.66 -> similarity 0.34 (fail boundary)
    # item 3: distance 1.20 -> similarity -0.20 (fail)
    mock_coll.query.return_value = {
        "ids": [["id-0", "id-1", "id-2", "id-3"]],
        "documents": [["Doc 0", "Doc 1", "Doc 2", "Doc 3"]],
        "metadatas": [[{"k": "v0"}, {"k": "v1"}, {"k": "v2"}, {"k": "v3"}]],
        "distances": [[0.10, 0.65, 0.66, 1.20]],
    }

    matches = store.query([0.01] * 768, top_k=4)

    assert len(matches) == 4
    assert matches[0]["similarity"] == 0.90
    assert matches[1]["similarity"] == 0.35
    assert matches[2]["similarity"] == 0.34
    assert matches[3]["similarity"] == -0.20

    # Filter with 0.35 threshold
    relevant = [m for m in matches if m.get("similarity", 0.0) >= 0.35]
    assert len(relevant) == 2
    assert relevant[0]["id"] == "id-0"
    assert relevant[1]["id"] == "id-1"


def test_system_prompt_corporate_evidence_grounding() -> None:
    """Verify SYSTEM_PROMPT forbids unverified corporate descriptions or business models."""
    from darglobal_project.rag.service import SYSTEM_PROMPT

    prompt_lower = SYSTEM_PROMPT.lower()

    # Rule: do not present company descriptions as verified unless supported by evidence
    assert "company descriptions" in prompt_lower
    assert "business models" in prompt_lower
    assert "unless explicitly supported by the retrieved evidence" in prompt_lower

    # Rule: do not describe Wasalt as developer or owner
    assert "do not describe wasalt as a property owner, landlord, or developer" in prompt_lower

    # Rule: corporate details unavailable notice
    assert "corporate details are unavailable" in prompt_lower
