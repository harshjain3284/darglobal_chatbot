"""ChromaDB persistent vector store and query retriever."""

import json
import logging
from pathlib import Path
from typing import Any, Optional
import chromadb
from darglobal_project.config import settings
from darglobal_project.scrapers.cleaner import PropertyItem

logger = logging.getLogger(__name__)


def build_document_text(prop: PropertyItem) -> str:
    """Format property attributes into rich document text for vector indexing."""
    if prop.source_platform == "DarGlobal":
        # Honest representation of DarGlobal sitemap entry without invented attributes
        return (
            f"Title: {prop.title}\n"
            f"Platform: {prop.source_platform}\n"
            f"Source URL: {prop.source_url}\n"
            "Status: Project listing from official DarGlobal sitemap. "
            "Property specifications, prices, and floor plans are currently protected by website anti-bot security "
            "and were unavailable during public automated collection."
        )

    # Wasalt verified company representation
    if prop.property_type == "Company Information":
        return (
            f"Title: {prop.title}\n"
            f"Platform: {prop.source_platform}\n"
            f"Location: {prop.location or 'Saudi Arabia'}\n"
            f"Type: Company Information\n"
            f"Description: {prop.description}\n"
            f"Source URL: {prop.source_url}"
        )

    # Wasalt verified property representation
    lines = [
        f"Title: {prop.title}",
        f"Platform: {prop.source_platform}",
        f"Location: {prop.location or 'Saudi Arabia'}",
        f"Property Type: {prop.property_type or 'Residential'}",
        f"Price: {prop.price_formatted or 'Price on request'}",
    ]
    if prop.bedrooms is not None:
        lines.append(f"Bedrooms: {prop.bedrooms}")
    if prop.area_sqm is not None:
        lines.append(f"Area: {prop.area_sqm} SQM")
    if prop.status:
        lines.append(f"Status: {prop.status}")
    if prop.description:
        lines.append(f"Description: {prop.description}")
    lines.append(f"Source URL: {prop.source_url}")

    return "\n".join(lines)


def build_metadata(prop: PropertyItem) -> dict[str, Any]:
    """Convert PropertyItem into flat primitive dictionary accepted by ChromaDB."""
    return {
        "property_id": prop.id,
        "title": prop.title or "",
        "location": prop.location or "",
        "price": float(prop.price) if prop.price is not None else 0.0,
        "price_formatted": prop.price_formatted or "Price on request",
        "currency": prop.currency or "",
        "property_type": prop.property_type or "",
        "bedrooms": int(prop.bedrooms) if prop.bedrooms is not None else 0,
        "area_sqm": float(prop.area_sqm) if prop.area_sqm is not None else 0.0,
        "status": prop.status or "",
        "source_url": prop.source_url,
        "source_platform": prop.source_platform,
        "scraped_at": prop.scraped_at,
        "image_url": prop.image_urls[0] if prop.image_urls else "",
    }


class VectorStore:
    """Persistent ChromaDB client for indexing and similarity search."""

    def __init__(self, persist_dir: Optional[Path] = None, collection_name: str = "real_estate") -> None:
        self.persist_dir = persist_dir or settings.vector_store_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_records(
        self,
        properties: list[PropertyItem],
        embeddings: list[list[float]],
    ) -> int:
        """Upsert properties and their precomputed embeddings into ChromaDB."""
        if not properties or not embeddings:
            return 0
        if len(properties) != len(embeddings):
            raise ValueError(
                f"Count mismatch: {len(properties)} properties vs {len(embeddings)} embeddings"
            )

        ids: list[str] = [p.id for p in properties]
        documents: list[str] = [build_document_text(p) for p in properties]
        metadatas: list[dict[str, Any]] = [build_metadata(p) for p in properties]

        # ChromaDB handles batch upserts efficiently
        batch_size = 50
        for i in range(0, len(ids), batch_size):
            end = i + batch_size
            self.collection.upsert(
                ids=ids[i:end],
                documents=documents[i:end],
                embeddings=embeddings[i:end],
                metadatas=metadatas[i:end],
            )

        return len(ids)

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_platform: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve top_k nearest neighbors by cosine similarity."""
        where = {"source_platform": filter_platform} if filter_platform else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        matches: list[dict[str, Any]] = []
        if results and results.get("ids") and results["ids"][0]:
            for idx in range(len(results["ids"][0])):
                # Extract cosine distance from Chroma (hnsw:space: cosine)
                # In Chroma: distance d = 1.0 - cosine_similarity
                # Therefore: cosine_similarity = 1.0 - distance
                dist = 1.0
                if results.get("distances") and len(results["distances"]) > 0 and len(results["distances"][0]) > idx:
                    raw_dist = results["distances"][0][idx]
                    if raw_dist is not None:
                        dist = float(raw_dist)

                # Clamp similarity to [-1.0, 1.0] to prevent floating point anomalies
                similarity = round(max(-1.0, min(1.0, 1.0 - dist)), 4)

                matches.append({
                    "id": results["ids"][0][idx],
                    "document": results["documents"][0][idx] if results.get("documents") else "",
                    "metadata": results["metadatas"][0][idx] if results.get("metadatas") else {},
                    "distance": round(dist, 6),
                    "similarity": similarity,
                })
        return matches

    def count(self) -> int:
        """Return total number of records in the collection."""
        return self.collection.count()
