"""Master CLI runner for DarGlobal and Wasalt real estate scrapers."""

import asyncio
import json
import sys
from pathlib import Path

# Add src to python path for standalone execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from darglobal_project.config import settings
from darglobal_project.scrapers.cleaner import deduplicate_properties
from darglobal_project.scrapers.darglobal import scrape_darglobal
from darglobal_project.scrapers.wasalt import scrape_wasalt


async def main() -> None:
    """Execute both scrapers, save raw outputs, and produce clean unified dataset."""
    print("=" * 70)
    print("STARTING REAL ESTATE SCRAPING PIPELINE")
    print("=" * 70)

    # Ensure directories exist
    settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Scrape Wasalt
    print("\n--- PHASE 1: WASALT SCRAPER ---")
    wasalt_raw, wasalt_norm = await scrape_wasalt(max_detail_pages=5)
    wasalt_raw_file = settings.raw_data_dir / "wasalt.json"
    with open(wasalt_raw_file, "w", encoding="utf-8") as f:
        json.dump(wasalt_raw, f, ensure_ascii=False, indent=2)
    print(f"[Wasalt] Saved {len(wasalt_raw)} raw records to: {wasalt_raw_file}")

    # 2. Scrape DarGlobal
    print("\n--- PHASE 2: DARGLOBAL SCRAPER ---")
    darglobal_raw, darglobal_norm = await scrape_darglobal(max_projects=40)
    darglobal_raw_file = settings.raw_data_dir / "darglobal.json"
    with open(darglobal_raw_file, "w", encoding="utf-8") as f:
        json.dump(darglobal_raw, f, ensure_ascii=False, indent=2)
    print(f"[DarGlobal] Saved raw data to: {darglobal_raw_file}")

    # 3. Combine and Deduplicate
    print("\n--- PHASE 3: NORMALIZATION & DEDUPLICATION ---")
    combined_items = wasalt_norm + darglobal_norm
    deduped_items = deduplicate_properties(combined_items)

    processed_file = settings.processed_data_dir / "properties.json"
    serializable_data = [item.model_dump() for item in deduped_items]
    with open(processed_file, "w", encoding="utf-8") as f:
        json.dump(serializable_data, f, ensure_ascii=False, indent=2)

    # 4. Summary & Verification
    print("\n" + "=" * 70)
    print("SCRAPING PIPELINE SUMMARY")
    print("=" * 70)
    print(f"Wasalt normalized records:    {len(wasalt_norm)}")
    print(f"DarGlobal normalized records: {len(darglobal_norm)}")
    print(f"Total before deduplication:   {len(combined_items)}")
    print(f"Total after deduplication:    {len(deduped_items)}")
    print(f"Clean catalog output file:    {processed_file}")
    print("=" * 70)

    # Sample record preview
    if deduped_items:
        print("\n--- SAMPLE EXTRACTED PROPERTY RECORD ---")
        sample = deduped_items[0].model_dump()
        print(json.dumps(sample, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
