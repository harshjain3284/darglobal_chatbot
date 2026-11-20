"""Data cleaning, normalization, and deduplication for scraped property records."""

import re
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field


class PropertyItem(BaseModel):
    """Normalized schema for real estate properties and projects."""

    id: str = Field(description="Unique property identifier")
    title: str = Field(description="Property or project title")
    description: Optional[str] = Field(default=None, description="Detailed description")
    location: Optional[str] = Field(default=None, description="City, district, or neighborhood")
    price: Optional[float] = Field(default=None, description="Numeric price if available")
    price_formatted: Optional[str] = Field(default=None, description="Formatted price string")
    currency: Optional[str] = Field(default=None, description="Price currency (e.g., SAR, USD, GBP)")
    property_type: Optional[str] = Field(default=None, description="Type: Villa, Apartment, Floor, Land, etc.")
    bedrooms: Optional[int] = Field(default=None, description="Number of bedrooms")
    bathrooms: Optional[int] = Field(default=None, description="Number of bathrooms")
    area_sqm: Optional[float] = Field(default=None, description="Built-up or plot area in square meters")
    status: Optional[str] = Field(default=None, description="Status: For Sale, For Rent, Under Construction, etc.")
    image_urls: list[str] = Field(default_factory=list, description="Publicly accessible image URLs")
    source_url: str = Field(description="Canonical source URL")
    source_platform: str = Field(description="Source platform: Wasalt or DarGlobal")
    scraped_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp of collection",
    )


def clean_text(text: Optional[str]) -> str:
    """Normalize text by collapsing whitespace and stripping control characters."""
    if not text:
        return ""
    # Remove markdown image syntax like ![...](...) from plain text
    cleaned = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    # Collapse multiple whitespaces and newlines
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def parse_price(price_str: Optional[str]) -> tuple[Optional[float], Optional[str], Optional[str]]:
    """Extract numeric price, formatted string, and currency from raw text."""
    if not price_str or not price_str.strip():
        return None, None, None

    currency = "SAR"
    for cur in ["USD", "$", "SAR", "AED", "GBP", "£", "EUR", "€"]:
        if cur in price_str.upper():
            currency = cur.replace("$", "USD").replace("£", "GBP").replace("€", "EUR")
            break

    # Extract digits and commas/dots
    digits_match = re.findall(r"[\d,]+(?:\.\d+)?", price_str.replace(" ", ""))
    if digits_match:
        raw_num = digits_match[0].replace(",", "")
        try:
            val = float(raw_num)
            return val, f"{val:,.0f} {currency}", currency
        except ValueError:
            pass

    return None, price_str.strip(), currency


def deduplicate_properties(properties: list[PropertyItem]) -> list[PropertyItem]:
    """Deduplicate property records by source_url preserving the most complete record."""
    seen: dict[str, PropertyItem] = {}
    for prop in properties:
        url = prop.source_url.strip().lower()
        if not url:
            continue
        if url not in seen:
            seen[url] = prop
        else:
            # Keep whichever has longer description / more fields populated
            existing = seen[url]
            if len(prop.description) > len(existing.description) or len(prop.image_urls) > len(existing.image_urls):
                seen[url] = prop
    return list(seen.values())
