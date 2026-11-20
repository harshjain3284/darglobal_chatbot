"""Unit and integration tests for RAG embedding and ChromaDB retriever modules."""

from unittest.mock import AsyncMock, patch
import httpx
import pytest
from darglobal_project.rag.embeddings import GeminiEmbedder
from darglobal_project.rag.retriever import VectorStore, build_document_text, build_metadata
from darglobal_project.scrapers.cleaner import PropertyItem


def test_build_document_text_wasalt() -> None:
    prop = PropertyItem(
        id="wasalt-001",
        title="Modern Villa in Sedra",
        location="Sedra, Riyadh",
        price=3500000.0,
        price_formatted="3,500,000 SAR",
        currency="SAR",
        property_type="Villa",
        bedrooms=4,
        area_sqm=300.0,
        status="For Sale",
        description="Luxury family villa with modern finishes",
        source_url="https://wasalt.sa/en/property/sale/villa-1",
        source_platform="Wasalt",
    )
    doc_text = build_document_text(prop)
    assert "Title: Modern Villa in Sedra" in doc_text
    assert "Price: 3,500,000 SAR" in doc_text
    assert "Bedrooms: 4" in doc_text
    assert "https://wasalt.sa/en/property/sale/villa-1" in doc_text


def test_build_document_text_darglobal() -> None:
    prop = PropertyItem(
        id="darglobal-001",
        title="Trump Tower Dubai",
        source_url="https://darglobal.co.uk/trump-tower",
        source_platform="DarGlobal",
    )
    doc_text = build_document_text(prop)
    assert "Title: Trump Tower Dubai" in doc_text
    assert "DarGlobal" in doc_text
    assert "unavailable during public automated collection" in doc_text


def test_gemini_embedder_missing_key() -> None:
    embedder = GeminiEmbedder(api_key="")
    with pytest.raises(ValueError, match="GEMINI_API_KEY is not configured"):
        import asyncio
        asyncio.run(embedder.embed_single("Test property"))


def test_chroma_vector_store_in_memory(tmp_path) -> None:
    # Test ChromaDB in temporary directory
    store = VectorStore(persist_dir=tmp_path / "test_chroma", collection_name="test_collection")
    assert store.count() == 0

    props = [
        PropertyItem(
            id="p-1",
            title="Luxury Villa Riyadh",
            location="Riyadh",
            price=2000000.0,
            price_formatted="2,000,000 SAR",
            source_url="https://wasalt.sa/p1",
            source_platform="Wasalt",
        ),
        PropertyItem(
            id="p-2",
            title="Sea View Apartment Jeddah",
            location="Jeddah",
            price=1200000.0,
            price_formatted="1,200,000 SAR",
            source_url="https://wasalt.sa/p2",
            source_platform="Wasalt",
        ),
    ]

    # Dummy 4-dimensional normalized vectors for testing
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]

    # First upsert
    upserted = store.upsert_records(props, embeddings)
    assert upserted == 2
    assert store.count() == 2

    # Repeatable indexing check (upsert should not duplicate)
    store.upsert_records(props, embeddings)
    assert store.count() == 2

    # Query test: querying vector close to p-1
    query_vec = [0.95, 0.05, 0.0, 0.0]
    matches = store.query(query_vec, top_k=1)
    assert len(matches) == 1
    assert matches[0]["id"] == "p-1"
    assert matches[0]["metadata"]["title"] == "Luxury Villa Riyadh"
    assert matches[0]["similarity"] > 0.9


@pytest.mark.asyncio
async def test_gemini_embedder_document_request_payload() -> None:
    """Verify document embedding requests explicitly specify 768 dimensions and RETRIEVAL_DOCUMENT."""
    embedder = GeminiEmbedder(api_key="mock-gemini-key")
    dummy_vec = [0.01] * 768
    mock_resp = httpx.Response(
        status_code=200,
        json={"embedding": {"values": dummy_vec}},
        request=httpx.Request("POST", "https://generativelanguage.googleapis.com"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        vec = await embedder.embed_document("Modern Villa in Sedra")

        assert len(vec) == 768
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        payload = kwargs["json"]
        assert payload["outputDimensionality"] == 768
        assert payload["taskType"] == "RETRIEVAL_DOCUMENT"
        assert payload["content"]["parts"][0]["text"] == "Modern Villa in Sedra"
        assert kwargs["headers"]["x-goog-api-key"] == "mock-gemini-key"


@pytest.mark.asyncio
async def test_gemini_embedder_query_request_payload() -> None:
    """Verify query embedding requests explicitly specify 768 dimensions and RETRIEVAL_QUERY."""
    embedder = GeminiEmbedder(api_key="mock-gemini-key")
    dummy_vec = [0.02] * 768
    mock_resp = httpx.Response(
        status_code=200,
        json={"embedding": {"values": dummy_vec}},
        request=httpx.Request("POST", "https://generativelanguage.googleapis.com"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        vec = await embedder.embed_query("luxury apartments in Jeddah")

        assert len(vec) == 768
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        payload = kwargs["json"]
        assert payload["outputDimensionality"] == 768
        assert payload["taskType"] == "RETRIEVAL_QUERY"
        assert payload["content"]["parts"][0]["text"] == "luxury apartments in Jeddah"
        assert kwargs["headers"]["x-goog-api-key"] == "mock-gemini-key"


@pytest.mark.asyncio
async def test_gemini_embedder_dimension_mismatch_error() -> None:
    """Verify ValueError is raised if the returned embedding does not match 768 dimensions."""
    embedder = GeminiEmbedder(api_key="mock-gemini-key")
    mock_resp = httpx.Response(
        status_code=200,
        json={"embedding": {"values": [0.1, 0.2, 0.3]}},  # only 3 dimensions instead of 768
        request=httpx.Request("POST", "https://generativelanguage.googleapis.com"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        with pytest.raises(ValueError, match="Expected 768-dimensional embedding, got 3"):
            await embedder.embed_single("Test text")


def test_chroma_vector_store_768_dim(tmp_path) -> None:
    """Verify ChromaDB collection accepts 768-dimensional embeddings and calculates cosine similarity."""
    store = VectorStore(persist_dir=tmp_path / "test_chroma_768", collection_name="real_estate")
    prop = PropertyItem(
        id="test-768",
        title="Sedra Villa",
        location="Riyadh",
        price=1500000.0,
        price_formatted="1,500,000 SAR",
        source_url="https://wasalt.sa/item1",
        source_platform="Wasalt",
    )
    vec = [0.05] * 768
    store.upsert_records([prop], [vec])
    assert store.count() == 1

    res = store.query(vec, top_k=1)
    assert len(res) == 1
    assert res[0]["id"] == "test-768"
    assert res[0]["similarity"] > 0.99
