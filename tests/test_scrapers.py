"""Tests for real estate scrapers and cleaner modules."""

import pytest
from darglobal_project.scrapers.cleaner import (
    PropertyItem,
    clean_text,
    deduplicate_properties,
    parse_price,
)


def test_clean_text() -> None:
    raw = "  Hello   World!\n\n![img](https://example.com/pic.jpg)  Test text.  "
    cleaned = clean_text(raw)
    assert cleaned == "Hello World! Test text."


def test_parse_price() -> None:
    val, fmt, cur = parse_price("3,990,000 SAR")
    assert val == 3990000.0
    assert "3,990,000" in fmt
    assert cur == "SAR"

    val_usd, fmt_usd, cur_usd = parse_price("$1,500,000")
    assert val_usd == 1500000.0
    assert cur_usd == "USD"

    val_empty, fmt_empty, _ = parse_price(None)
    assert val_empty is None
    assert fmt_empty is None


def test_deduplicate_properties() -> None:
    p1 = PropertyItem(
        id="prop-1",
        title="Luxury Villa",
        description="Short description",
        location="Riyadh",
        source_url="https://wasalt.sa/en/property/sale/villa-1",
        source_platform="Wasalt",
    )
    p2 = PropertyItem(
        id="prop-2",
        title="Luxury Villa",
        description="Much longer and more detailed description of the luxury villa",
        location="Riyadh",
        source_url="https://wasalt.sa/en/property/sale/villa-1",
        source_platform="Wasalt",
    )
    deduped = deduplicate_properties([p1, p2])
    assert len(deduped) == 1
    assert deduped[0].description == "Much longer and more detailed description of the luxury villa"
