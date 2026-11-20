"""Wasalt real estate property scraper using Crawl4AI."""

import asyncio
import logging
import re
from typing import Any
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from darglobal_project.scrapers.cleaner import PropertyItem, clean_text, parse_price

logger = logging.getLogger(__name__)

WASALT_CATEGORIES = [
    # Riyadh categories (preserve existing coverage)
    "https://wasalt.sa/en/properties-for-sale-in-riyadh",
    # Jeddah categories (priority #1)
    "https://wasalt.sa/en/properties-for-sale-in-jeddah",
    # Dammam categories (priority #2)
    "https://wasalt.sa/en/properties-for-sale-in-dammam",
    # Al Khobar categories (priority #3)
    "https://wasalt.sa/en/properties-for-sale-in-al-khobar",
]


def extract_property_type(title: str, url: str) -> str:
    """Infer property type from title or URL slug."""
    text = (title + " " + url).lower()
    if "villa" in text:
        return "Villa"
    elif "apartment" in text:
        return "Apartment"
    elif "floor" in text:
        return "Floor"
    elif "land" in text:
        return "Land"
    elif "building" in text or "commercial" in text:
        return "Commercial"
    return "Residential Property"


def extract_cards_from_markdown(markdown: str, category_url: str) -> list[dict[str, Any]]:
    """Parse property cards from Wasalt listing page markdown."""
    properties: list[dict[str, Any]] = []
    
    # Pattern matching image, title, and property detail URL
    # Example: [![Title](image_url)](https://wasalt.sa/en/property/sale/slug "Title")
    card_pattern = re.compile(
        r"\[\!\[(.*?)\]\((https://[^\)]+)\)\]\((https://wasalt\.sa/en/property/sale/[^\s\)\"]+)",
        re.MULTILINE,
    )
    
    # Also find price and location blocks that follow property links
    # Matches [1,450,000 Floor...](url "Title")
    price_pattern = re.compile(
        r"\[([0-9,]{4,})\s*(.*?)\s*\]\((https://wasalt\.sa/en/property/sale/[^\s\)\"]+)",
        re.MULTILINE,
    )
    
    prices_by_url: dict[str, str] = {}
    for match in price_pattern.finditer(markdown):
        price_val, _, p_url = match.groups()
        prices_by_url[p_url.strip()] = price_val.strip()

    seen_urls = set()
    for match in card_pattern.finditer(markdown):
        title, img_url, prop_url = match.groups()
        prop_url = prop_url.strip()
        if prop_url in seen_urls:
            continue
        seen_urls.add(prop_url)

        # Extract bedrooms or area if in title
        bedrooms = None
        bed_match = re.search(r"(\d+)\s*(?:bedrooms|beds|bed)", title, re.IGNORECASE)
        if bed_match:
            bedrooms = int(bed_match.group(1))

        area_sqm = None
        area_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:sqm|sq m|m²)", title, re.IGNORECASE)
        if area_match:
            area_sqm = float(area_match.group(1))

        # Extract location from title and category URL
        location = "Saudi Arabia"
        c_low = category_url.lower()
        t_low = title.lower()

        if "khobar" in t_low or "khobar" in c_low:
            location = "Al Khobar, Saudi Arabia"
        elif "dammam" in t_low or "dammam" in c_low or "aldammam" in t_low:
            location = "Dammam, Saudi Arabia"
        elif "jeddah" in t_low or "jeddah" in c_low:
            location = "Jeddah, Saudi Arabia"
        elif "riyadh" in t_low or "riyadh" in c_low:
            location = "Riyadh, Saudi Arabia"

        loc_match = re.search(
            r"(?:in|,)\s*([A-Za-z\s-]+(?:Riyadh|Jeddah|Dammam|Aldammam|Khobar|Al Khobar))",
            title,
            re.IGNORECASE,
        )
        if loc_match:
            location = clean_text(loc_match.group(1))

        price_text = prices_by_url.get(prop_url, "")
        p_type = extract_property_type(title, prop_url)

        properties.append({
            "title": clean_text(title),
            "url": prop_url,
            "image_url": img_url.strip(),
            "price_text": price_text,
            "location": location,
            "property_type": p_type,
            "bedrooms": bedrooms,
            "area_sqm": area_sqm,
        })

    return properties


def parse_detail_markdown(markdown: str, base_item: dict[str, Any]) -> dict[str, Any]:
    """Enrich property item with full description and attributes from detail page."""
    item = dict(base_item)

    # Extract description
    desc_match = re.search(r"##\s*Property Description\s*(.*?)(?=\n##|\nREGA|\nRef no|\Z)", markdown, re.DOTALL)
    if desc_match:
        item["description"] = clean_text(desc_match.group(1))
    else:
        # Fallback to general lines
        item["description"] = clean_text(markdown[:800])

    # Extract price if missing
    if not item.get("price_text"):
        p_match = re.search(r"\b([1-9][0-9]{2,3}(?:,[0-9]{3})+)\b", markdown)
        if p_match:
            item["price_text"] = p_match.group(1)

    # Extract bedrooms if missing
    if not item.get("bedrooms"):
        b_match = re.search(r"\*\s*(\d+)\s*\n\s*bedrooms", markdown, re.IGNORECASE)
        if b_match:
            item["bedrooms"] = int(b_match.group(1))

    # Extract area if missing
    if not item.get("area_sqm"):
        a_match = re.search(r"\*\s*(\d+(?:\.\d+)?)\s*SQM", markdown, re.IGNORECASE)
        if a_match:
            item["area_sqm"] = float(a_match.group(1))

    # Additional images
    extra_images = re.findall(r"\((https://imagedelivery\.net/[^\)]+)\)", markdown)
    if extra_images:
        current_images = [item["image_url"]] if item.get("image_url") else []
        for img in extra_images:
            if img not in current_images:
                current_images.append(img)
        item["all_images"] = current_images[:5]

    return item


async def scrape_wasalt(max_detail_pages: int = 10) -> tuple[list[dict[str, Any]], list[PropertyItem]]:
    """Scrape Wasalt property categories and detail pages politely."""
    import json
    from darglobal_project.config import settings

    raw_records: list[dict[str, Any]] = []
    normalized_items: list[PropertyItem] = []

    # 0. Load existing raw records to guarantee zero loss of previously collected Riyadh properties
    existing_raw_file = settings.raw_data_dir / "wasalt.json"
    existing_cards: dict[str, dict[str, Any]] = {}
    if existing_raw_file.exists():
        try:
            with open(existing_raw_file, "r", encoding="utf-8") as f:
                for item in json.load(f):
                    u = item.get("url")
                    if u:
                        existing_cards[u] = item
            print(f"[Wasalt] Loaded {len(existing_cards)} existing verified records from {existing_raw_file}")
        except Exception as e:
            print(f"[Wasalt] Note reading existing raw records: {e}")

    print("[Wasalt] Initializing Crawl4AI scraper...")
    # 1. Scrape listing categories across Riyadh, Jeddah, Dammam, and Al Khobar
    discovered_cards: list[dict[str, Any]] = []
    for cat_url in WASALT_CATEGORIES:
        print(f"[Wasalt] Fetching category: {cat_url}")
        try:
            async with AsyncWebCrawler(verbose=False) as crawler:
                res = await crawler.arun(url=cat_url)
                if res.success and res.markdown:
                    cards = extract_cards_from_markdown(res.markdown, cat_url)
                    print(f"[Wasalt] Discovered {len(cards)} property cards from {cat_url}")
                    discovered_cards.extend(cards)
                else:
                    print(f"[Wasalt] Warning: Could not crawl {cat_url} (status: {res.status_code})")
        except Exception as e:
            print(f"[Wasalt] Error fetching category {cat_url}: {e}")

        # Polite delay between category requests (at least 3.5s)
        await asyncio.sleep(3.5)

    # Merge with existing cards (existing cards take precedence to preserve previously enriched details)
    unique_cards = dict(existing_cards)
    new_card_count = 0
    for c in discovered_cards:
        if c["url"] not in unique_cards:
            unique_cards[c["url"]] = c
            new_card_count += 1

    all_cards = list(unique_cards.values())
    print(f"[Wasalt] Discovered {new_card_count} new properties. Total unique properties: {len(all_cards)}")

    # 2. Enrich top new detail pages politely if max_detail_pages > 0
    new_cards = [c for c in all_cards if c["url"] not in existing_cards]
    detail_targets = new_cards[:max_detail_pages]
    for i, card in enumerate(detail_targets, start=1):
        prop_url = card["url"]
        print(f"[Wasalt] Fetching detail [{i}/{len(detail_targets)}]: {prop_url}")
        try:
            async with AsyncWebCrawler(verbose=False) as crawler:
                res = await crawler.arun(url=prop_url)
                if res.success and res.markdown:
                    enriched = parse_detail_markdown(res.markdown, card)
                    unique_cards[prop_url] = enriched
        except Exception as e:
            print(f"[Wasalt] Detail crawl failed for {prop_url}: {e}")

        await asyncio.sleep(3.0)

    # 2b. Add official Wasalt company information page if accessible
    try:
        async with AsyncWebCrawler(verbose=False) as crawler:
            biz_res = await crawler.arun(url="https://wasalt.sa/en/s/wasalt-business")
            if biz_res.success and biz_res.markdown:
                clean_biz_desc = clean_text(biz_res.markdown[:1200])
                biz_card = {
                    "title": "Wasalt Real Estate Platform & Business Services",
                    "url": "https://wasalt.sa/en/s/wasalt-business",
                    "image_url": "https://cdn.wasalt.sa/images/wasalt-logo.svg",
                    "price_text": "",
                    "location": "Saudi Arabia",
                    "property_type": "Company Information",
                    "bedrooms": None,
                    "area_sqm": None,
                    "description": f"Official Wasalt Business Overview: Wasalt is a leading Saudi real estate marketplace providing verified property listings, auctions, interactive masterplans, and brokerage services across the Kingdom of Saudi Arabia. {clean_biz_desc}",
                }
                unique_cards[biz_card["url"]] = biz_card
                print("[Wasalt] Successfully scraped official company information (/s/wasalt-business)")
    except Exception as e:
        print(f"[Wasalt] Company page crawl error: {e}")

    raw_records = list(unique_cards.values())

    # 3. Convert all raw records to normalized PropertyItem schema
    for idx, raw in enumerate(raw_records, start=1):
        price_num, price_fmt, currency = parse_price(raw.get("price_text"))
        
        # Build description strictly from verified scraped fields
        desc = raw.get("description")
        if not desc:
            parts = [raw.get("title", "")]
            if raw.get("location"):
                parts.append(f"Location: {raw.get('location')}")
            if raw.get("property_type"):
                parts.append(f"Type: {raw.get('property_type')}")
            if raw.get("bedrooms") is not None:
                parts.append(f"Bedrooms: {raw.get('bedrooms')}")
            if raw.get("area_sqm") is not None:
                parts.append(f"Area: {raw.get('area_sqm')} SQM")
            if price_fmt:
                parts.append(f"Price: {price_fmt}")
            desc = " | ".join(p for p in parts if p)

        images = raw.get("all_images", [])
        if not images and raw.get("image_url"):
            images = [raw["image_url"]]

        if raw.get("property_type") == "Company Information":
            prop_id = "wasalt-company-business"
            prop_status = "Active"
        else:
            slug_match = re.search(r"(\d+)$", raw.get("url", ""))
            prop_id = f"wasalt-{slug_match.group(1)}" if slug_match else f"wasalt-{idx:04d}"
            prop_status = "For Sale"

        item = PropertyItem(
            id=prop_id,
            title=raw.get("title", f"Property {idx}"),
            description=clean_text(desc),
            location=raw.get("location", "Saudi Arabia"),
            price=price_num,
            price_formatted=price_fmt,
            currency=currency,
            property_type=raw.get("property_type", "Residential Property"),
            bedrooms=raw.get("bedrooms"),
            area_sqm=raw.get("area_sqm"),
            status=prop_status,
            image_urls=images,
            source_url=raw.get("url", ""),
            source_platform="Wasalt",
        )
        normalized_items.append(item)

    print(f"[Wasalt] Normalization complete. Processed {len(normalized_items)} records.")
    return raw_records, normalized_items
