"""Standalone indexing script: embeds validated properties via Gemini and populates ChromaDB."""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add src to python path for standalone execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from darglobal_project.config import settings
from darglobal_project.rag.embeddings import GeminiEmbedder
from darglobal_project.rag.retriever import VectorStore, build_document_text
from darglobal_project.scrapers.cleaner import PropertyItem


async def main() -> None:
    """Run vector indexing pipeline."""
    print("=" * 70)
    print("STARTING VECTOR INDEXING PIPELINE (GEMINI + CHROMADB)")
    print("=" * 70)

    # 1. Load validated properties
    processed_file = settings.processed_data_dir / "properties.json"
    if not processed_file.exists():
        print(f"Error: {processed_file} does not exist. Run scripts/run_scrapers.py first.")
        sys.exit(1)

    with open(processed_file, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    properties = [PropertyItem(**item) for item in raw_items]
    wasalt_count = sum(1 for p in properties if p.source_platform == "Wasalt")
    darglobal_count = sum(1 for p in properties if p.source_platform == "DarGlobal")

    print(f"Loaded {len(properties)} validated records from: {processed_file}")
    print(f"  Wasalt records:    {wasalt_count} (fully verified property details)")
    print(f"  DarGlobal records: {darglobal_count} (sitemap project listings, unverified fields set to null)")

    # 2. Check Gemini API Key
    api_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY")
    use_test_mode = "--test" in sys.argv or "--mock" in sys.argv

    if not api_key and not use_test_mode:
        print("\n" + "!" * 70)
        print("MISSING CONFIGURATION: GEMINI_API_KEY is not set.")
        print("Please configure GEMINI_API_KEY in your .env file or environment.")
        print("Example: echo GEMINI_API_KEY=AIzaSy... >> .env")
        print("Tip: Run with --test to test ChromaDB indexing & retrieval without API calls.")
        print("!" * 70)
        sys.exit(1)

    # 3. Initialize ChromaDB store to check existing cached embeddings
    print(f"\nInitializing ChromaDB persistent collection at: {settings.vector_store_dir}")
    store = VectorStore(persist_dir=settings.vector_store_dir, collection_name="real_estate")
    initial_count = store.count()
    print(f"Existing records in ChromaDB collection before update: {initial_count}")

    existing_embeddings = {}
    if initial_count > 0:
        existing_data = store.collection.get(include=["embeddings"])
        if existing_data.get("ids") and existing_data.get("embeddings") is not None:
            for rec_id, emb in zip(existing_data["ids"], existing_data["embeddings"]):
                if emb is not None and len(emb) == 768:
                    existing_embeddings[rec_id] = emb
        print(f"Loaded {len(existing_embeddings)} pre-existing 768-dim embeddings from ChromaDB.")

    # 4. Generate Embeddings (Reusing existing embeddings, embedding only new records)
    needed_properties = [p for p in properties if p.id not in existing_embeddings]
    print(f"Properties requiring new embeddings: {len(needed_properties)} / {len(properties)}")

    new_embeddings = {}
    if api_key and not use_test_mode:
        embedder = GeminiEmbedder(api_key=api_key, max_concurrency=2)
        if needed_properties:
            print(f"\nGenerating embeddings for {len(needed_properties)} new properties via Gemini embedContent...")
            import httpx
            async with httpx.AsyncClient(timeout=30.0) as client:
                for idx, prop in enumerate(needed_properties, start=1):
                    doc_text = build_document_text(prop)
                    emb = await embedder.embed_document(doc_text, client=client)
                    new_embeddings[prop.id] = emb
                    if idx % 5 == 0 or idx == len(needed_properties):
                        print(f"  [Embeddings] Embedded {idx}/{len(needed_properties)} new records...")
                    # Respect Gemini free-tier 15 RPM rate limit (wait 4.1s between requests)
                    if idx < len(needed_properties):
                        await asyncio.sleep(4.1)
    else:
        print(f"\n[Test Mode] Generating deterministic 768-dim test vectors for unindexed properties...")
        import hashlib
        import math

        def test_vector(text: str, dim: int = 768) -> list[float]:
            h = hashlib.sha256(text.encode("utf-8")).digest()
            vec = [float((b % 50) - 25) for b in h]
            while len(vec) < dim:
                vec.extend(vec[: dim - len(vec)])
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            return [round(x / norm, 6) for x in vec[:dim]]

        for prop in needed_properties:
            doc_text = build_document_text(prop)
            new_embeddings[prop.id] = test_vector(doc_text)

    # Assemble final embeddings matching properties list order
    all_embeddings = []
    for prop in properties:
        if prop.id in existing_embeddings:
            all_embeddings.append(existing_embeddings[prop.id])
        else:
            all_embeddings.append(new_embeddings[prop.id])

    embedding_dim = len(all_embeddings[0]) if all_embeddings else 0
    print(f"\nTotal vectors prepared: {len(all_embeddings)} with consistent dimension: {embedding_dim}")

    # 5. Populate ChromaDB
    upserted_count = store.upsert_records(properties, all_embeddings)
    print(f"Upserted {upserted_count} records into ChromaDB collection 'real_estate'")

    final_count = store.count()
    print(f"Total records in ChromaDB collection: {final_count}")

    # 6. Verification Test Queries
    print("\n" + "=" * 70)
    print("RUNNING VERIFICATION SIMILARITY QUERIES ACROSS LOCATIONS")
    print("=" * 70)

    test_queries = [
        "luxury villas with swimming pool in Riyadh",
        "apartments for sale in Jeddah",
        "villas in Dammam or Al Khobar",
        "Trump International Hotel and Tower",
        "Wasalt company information",
    ]

    for q in test_queries:
        print(f"\nQuery: '{q}'")
        if api_key and not use_test_mode:
            q_emb = await embedder.embed_query(q)
        else:
            q_emb = test_vector(q)
        matches = store.query(q_emb, top_k=2)
        for i, match in enumerate(matches, start=1):
            meta = match["metadata"]
            print(f"  Match {i}: [{meta.get('source_platform')}] {meta.get('title')}")
            print(f"    Similarity Score: {match['similarity']}")
            print(f"    Location: {meta.get('location') or 'Protected/Unavailable'}")
            print(f"    Price: {meta.get('price_formatted') or 'Unavailable'}")
            print(f"    URL: {meta.get('source_url')}")

    print("\n" + "=" * 70)
    print("INDEXING PIPELINE COMPLETE & VERIFIED")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
