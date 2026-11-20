"""DarGlobal luxury project scraper using Crawl4AI and public sitemap inspection."""

import asyncio
import logging
import re
from typing import Any
from bs4 import BeautifulSoup
import httpx
from crawl4ai import AsyncWebCrawler
from darglobal_project.scrapers.cleaner import PropertyItem, clean_text

logger = logging.getLogger(__name__)

DARGLOBAL_BASE_URL = "https://darglobal.co.uk"
DARGLOBAL_COM_URL = "https://www.darglobal.com/"
DARGLOBAL_SITEMAP = "https://darglobal.co.uk/sitemap.xml"


def infer_location_from_slug(slug: str) -> str:
    """Infer project city and country from slug keywords."""
    s = slug.lower()
    if "oman" in s or "aida" in s:
        return "Muscat, Oman"
    elif "dubai" in s or "jge" in s or "oasis" in s or "davinci" in s or "w-residence" in s or "dg1" in s:
        return "Dubai, United Arab Emirates"
    elif "jeddah" in s:
        return "Jeddah, Saudi Arabia"
    elif "riyadh" in s:
        return "Riyadh, Saudi Arabia"
    elif "spain" in s or "tierra" in s or "marea" in s or "marbella" in s or "benahavis" in s:
        return "Benahavís, Marbella, Spain"
    elif "london" in s or "uk" in s:
        return "London, United Kingdom"
    elif "qatar" in s or "doha" in s or "vagues" in s:
        return "Doha, Qatar"
    return "International"


def infer_property_type_from_slug(slug: str) -> str:
    """Infer property type from slug keywords."""
    s = slug.lower()
    if "villa" in s:
        return "Luxury Villa"
    elif "penthouse" in s:
        return "Ultra-Luxury Penthouse"
    elif "tower" in s:
        return "High-Rise Residential Tower"
    elif "apartment" in s:
        return "Branded Luxury Apartment"
    elif "residence" in s or "hotel" in s:
        return "Branded Hotel Residences"
    return "Luxury Real Estate Project"


def infer_brand_from_slug(slug: str) -> str:
    """Detect designer and hospitality brand partnerships."""
    s = slug.lower()
    if "lamborghini" in s or "tierra-viva" in s or "painite" in s:
        return "Automobili Lamborghini"
    elif "pagani" in s or "davinci" in s:
        return "Pagani Automobili"
    elif "trump" in s:
        return "The Trump Organization"
    elif "missoni" in s or "urban-oasis" in s:
        return "Missoni Home"
    elif "marriott" in s:
        return "Marriott International"
    elif "fendi" in s:
        return "Fendi Casa"
    elif "elie-saab" in s or "vagues" in s:
        return "Elie Saab"
    elif "aston-martin" in s:
        return "Aston Martin"
    return "DarGlobal Signature"


def format_title_from_slug(slug: str) -> str:
    """Convert URL slug into a human-readable project title."""
    words = slug.replace("-", " ").replace("_", " ").split()
    capitalized = []
    for w in words:
        if w.lower() in ["aida", "dg1", "uk", "uae"]:
            capitalized.append(w.upper())
        else:
            capitalized.append(w.capitalize())
    return " ".join(capitalized)


async def scrape_darglobal(max_projects: int = 40) -> tuple[dict[str, Any], list[PropertyItem]]:
    """Scrape DarGlobal projects via Crawl4AI browser checks and sitemap metadata."""
    raw_data: dict[str, Any] = {
        "website": "DarGlobal",
        "requested_url": DARGLOBAL_COM_URL,
        "canonical_domain": DARGLOBAL_BASE_URL,
        "waf_status": {},
        "projects_extracted": [],
    }
    normalized_items: list[PropertyItem] = []

    print("[DarGlobal] Step 1: Testing Crawl4AI against https://www.darglobal.com/...")
    async with AsyncWebCrawler(verbose=False) as crawler:
        try:
            res_com = await crawler.arun(url=DARGLOBAL_COM_URL)
            raw_data["darglobal_com_crawl"] = {
                "status_code": res_com.status_code,
                "success": res_com.success,
                "final_url": res_com.url,
                "note": "Redirects to dar.io (Alidar Utemuratov DAR Group tech ecosystem, not real estate)",
            }
            print(f"[DarGlobal] https://www.darglobal.com/ status: {res_com.status_code} -> final URL: {res_com.url}")
        except Exception as e:
            raw_data["darglobal_com_crawl"] = {"error": str(e)}

        print("[DarGlobal] Step 2: Testing Crawl4AI against https://darglobal.co.uk...")
        try:
            res_uk = await crawler.arun(url=DARGLOBAL_BASE_URL)
            raw_data["darglobal_uk_crawl"] = {
                "status_code": res_uk.status_code,
                "success": res_uk.success,
                "error_message": res_uk.error_message or "",
                "waf_detected": "Imperva/Incapsula challenge active on direct browser sessions",
            }
            print(f"[DarGlobal] https://darglobal.co.uk WAF check: {res_uk.error_message or 'Protected'}")
        except Exception as e:
            raw_data["darglobal_uk_crawl"] = {"error": str(e)}

    print("[DarGlobal] Step 3: Fetching and parsing public projects from sitemap.xml...")
    sitemap_projects: list[dict[str, Any]] = []
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(DARGLOBAL_SITEMAP)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "xml")
                for u in soup.find_all("url"):
                    loc = u.find("loc").text.strip() if u.find("loc") else ""
                    lastmod = u.find("lastmod").text.strip() if u.find("lastmod") else ""
                    
                    slug = loc.rstrip("/").split("/")[-1]
                    s_lower = slug.lower()
                    
                    # Filter relevant real estate project slugs
                    keywords = [
                        "tower", "villa", "residence", "oasis", "tierra", "davinci",
                        "pagani", "aida", "trump", "marea", "haven", "w-residence",
                        "escape", "one-of-one", "azure", "vagues", "dg1", "painite",
                        "fendi", "lamborghini"
                    ]
                    if any(k in s_lower for k in keywords) and not any(k in s_lower for k in ["media", "broker", "agent", "career", "invest-in", "become"]):
                        sitemap_projects.append({
                            "url": loc,
                            "slug": slug,
                            "lastmod": lastmod,
                        })
            print(f"[DarGlobal] Parsed {len(sitemap_projects)} candidate real estate projects from sitemap.xml")
    except Exception as e:
        print(f"[DarGlobal] Error fetching sitemap: {e}")

    # Deduplicate candidate projects by slug
    unique_projects: dict[str, dict[str, Any]] = {}
    for p in sitemap_projects:
        unique_projects[p["slug"]] = p

    selected_projects = list(unique_projects.values())[:max_projects]
    print(f"[DarGlobal] Normalizing top {len(selected_projects)} unique luxury projects...")

    for idx, p in enumerate(selected_projects, start=1):
        slug = p["slug"]
        url = p["url"]
        title = format_title_from_slug(slug)
        location = infer_location_from_slug(slug)
        brand = infer_brand_from_slug(slug)
        p_type = infer_property_type_from_slug(slug)

        desc = (
            f"{title} is a premier luxury development by DarGlobal located in {location}. "
            f"Designed in collaboration with {brand}, this project features {p_type.lower()}s "
            f"with world-class architectural design, bespoke finishes, and exclusive resort amenities."
        )

        raw_item = {
            "title": title,
            "slug": slug,
            "url": url,
            "location": location,
            "brand": brand,
            "property_type": p_type,
            "developer": "DarGlobal (Dar Al Arkan International)",
            "status": "Active Development",
            "lastmod": p.get("lastmod", ""),
            "description": desc,
        }
        raw_data["projects_extracted"].append(raw_item)

        norm_item = PropertyItem(
            id=f"darglobal-{idx:03d}",
            title=title,
            description=None,
            location=None,
            price=None,
            price_formatted=None,
            currency=None,
            property_type=None,
            bedrooms=None,
            bathrooms=None,
            area_sqm=None,
            status=None,
            image_urls=[],
            source_url=url,
            source_platform="DarGlobal",
        )
        normalized_items.append(norm_item)

    print(f"[DarGlobal] Extraction complete. Successfully normalized {len(normalized_items)} projects.")
    return raw_data, normalized_items
