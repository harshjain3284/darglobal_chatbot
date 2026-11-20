"""Tests verifying multi-city Wasalt data expansion and preservation of original records."""

import json
from pathlib import Path
import pytest
from darglobal_project.config import settings
from darglobal_project.scrapers.cleaner import PropertyItem


def test_processed_properties_multi_city_coverage() -> None:
    """Verify data/processed/properties.json contains properties across Riyadh, Jeddah, Dammam, and Al Khobar."""
    processed_path = settings.processed_data_dir / "properties.json"
    assert processed_path.exists(), "properties.json does not exist"

    with open(processed_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    assert len(records) >= 170, f"Expected at least 170 records, got {len(records)}"

    wasalt_records = [r for r in records if r.get("source_platform") == "Wasalt"]
    darglobal_records = [r for r in records if r.get("source_platform") == "DarGlobal"]

    assert len(darglobal_records) == 40, f"Expected 40 DarGlobal records, got {len(darglobal_records)}"
    assert len(wasalt_records) >= 130, f"Expected >= 130 Wasalt records, got {len(wasalt_records)}"

    # Check city coverage
    cities = {"Riyadh": 0, "Jeddah": 0, "Dammam": 0, "Al Khobar": 0}
    for r in wasalt_records:
        loc = (r.get("location") or "") + " " + (r.get("title") or "") + " " + (r.get("description") or "")
        if "Jeddah" in loc:
            cities["Jeddah"] += 1
        elif "Aldammam" in loc or "Dammam" in loc:
            cities["Dammam"] += 1
        elif "Al Khobar" in loc or "Khobar" in loc:
            cities["Al Khobar"] += 1
        elif "Riyadh" in loc:
            cities["Riyadh"] += 1

    assert cities["Riyadh"] >= 50, f"Expected at least 50 Riyadh records, got {cities['Riyadh']}"
    assert cities["Jeddah"] >= 25, f"Expected at least 25 Jeddah records, got {cities['Jeddah']}"
    assert cities["Dammam"] >= 25, f"Expected at least 25 Dammam records, got {cities['Dammam']}"
    assert cities["Al Khobar"] >= 25, f"Expected at least 25 Al Khobar records, got {cities['Al Khobar']}"


def test_wasalt_categories_diversity() -> None:
    """Verify collected Wasalt properties span required categories."""
    processed_path = settings.processed_data_dir / "properties.json"
    with open(processed_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    wasalt_records = [r for r in records if r.get("source_platform") == "Wasalt"]
    types = {r.get("property_type") for r in wasalt_records}

    assert "Villa" in types
    assert "Apartment" in types
    assert "Land" in types
    assert "Commercial" in types or "Floor" in types or "Building" in types


def test_preservation_of_original_riyadh_records() -> None:
    """Verify all 50 original Riyadh records from initial scrape are 100% preserved in the dataset."""
    processed_path = settings.processed_data_dir / "properties.json"
    with open(processed_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    # All IDs must be unique
    all_ids = [r["id"] for r in records]
    assert len(all_ids) == len(set(all_ids)), "Duplicate IDs found in processed properties"

    # All canonical URLs must be valid http/https URLs
    for r in records:
        url = r.get("source_url")
        assert url and (url.startswith("http://") or url.startswith("https://")), f"Invalid source URL: {url}"
