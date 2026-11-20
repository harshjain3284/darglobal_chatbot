"""RAG Service orchestrating dynamic LLM intent classification, query rewriting, and grounded response synthesis."""

from enum import Enum
import json
import logging
import re
from typing import Any, Optional
from pydantic import BaseModel, Field
from darglobal_project.rag.embeddings import GeminiEmbedder
from darglobal_project.rag.llm_client import OpenRouterClient
from darglobal_project.rag.retriever import VectorStore

logger = logging.getLogger(__name__)


class ChatIntent(str, Enum):
    """Supported intent categories."""
    GREETING = "GREETING"
    THANKS = "THANKS"
    PROPERTY_SEARCH = "PROPERTY_SEARCH"
    PROJECT_SEARCH = "PROJECT_SEARCH"
    COMPANY_INFORMATION = "COMPANY_INFORMATION"
    COMPANY_COMPARISON = "COMPANY_COMPARISON"
    FOLLOW_UP = "FOLLOW_UP"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


class IntentAnalysis(BaseModel):
    """Structured result from dynamic LLM intent classification."""
    intent: str = Field(
        ...,
        description="One of: GREETING, THANKS, PROPERTY_SEARCH, PROJECT_SEARCH, COMPANY_INFORMATION, COMPANY_COMPARISON, FOLLOW_UP, OUT_OF_SCOPE",
    )
    relevant_platforms: list[str] = Field(
        default_factory=list,
        description="Platforms involved: 'DarGlobal', 'Wasalt', both, or empty",
    )
    needs_retrieval: bool = Field(
        default=True,
        description="Whether this query requires searching the vector database",
    )
    rewritten_query: str = Field(
        default="",
        description="Search-optimized query for vector retrieval, or empty string",
    )
    confidence: float = Field(
        default=1.0,
        description="Confidence score between 0.0 and 1.0",
    )


INTENT_CLASSIFIER_PROMPT = """You are an intent classification and query analysis engine for a Real Estate AI Assistant specializing in DarGlobal (international luxury developments) and Wasalt (Saudi Arabia property listings).

Analyze the user's message and return a single valid JSON object with EXACTLY this structure:
{
  "intent": "<GREETING | THANKS | PROPERTY_SEARCH | PROJECT_SEARCH | COMPANY_INFORMATION | COMPANY_COMPARISON | FOLLOW_UP | OUT_OF_SCOPE>",
  "relevant_platforms": ["<Wasalt | DarGlobal>"],
  "needs_retrieval": <true | false>,
  "rewritten_query": "<search-optimized keywords for vector retrieval, or empty string if needs_retrieval is false>",
  "confidence": <float between 0.0 and 1.0>
}

INTENT DEFINITIONS:
1. GREETING: Conversational greetings, hello, hi, how are you, good morning. (needs_retrieval: false, relevant_platforms: [])
2. THANKS: Expressions of gratitude, thank you, appreciation. (needs_retrieval: false, relevant_platforms: [])
3. PROPERTY_SEARCH: Inquiries about residential/commercial properties, villas, apartments, lands, floors, prices, bedrooms, areas, or districts in Saudi Arabia (Riyadh, Jeddah, Sedra, Qurtubah, etc.). (needs_retrieval: true, relevant_platforms: ["Wasalt"])
4. PROJECT_SEARCH: Inquiries about luxury developments, master projects, or brand collaborations by DarGlobal (e.g. Trump Tower, AIDA Oman, Pagani DaVinci, Lamborghini Tierra Viva, DG1, Fendi Casa, Missoni). (needs_retrieval: true, relevant_platforms: ["DarGlobal"])
5. COMPANY_INFORMATION: Inquiries asking about what DarGlobal or Wasalt is as a single company or platform. (needs_retrieval: true)
6. COMPANY_COMPARISON: Inquiries comparing both companies, asking about the difference between both platforms, or requesting properties/projects from both sources (e.g., "Compare DarGlobal and Wasalt", "What is the difference between these two platforms?", "Tell me about both companies", "Show me properties from both sources"). (needs_retrieval: true, relevant_platforms: ["DarGlobal", "Wasalt"])
7. FOLLOW_UP: Continuing an existing line of real estate inquiry or asking for elaboration. (needs_retrieval: true)
8. OUT_OF_SCOPE: Questions completely unrelated to real estate or DarGlobal/Wasalt (e.g., programming, general world knowledge, math, recipes, weather, politics). (needs_retrieval: false, relevant_platforms: [])

REWRITTEN QUERY GUIDELINES:
- If needs_retrieval is true: formulate clear, focused search keywords for vector retrieval.
  Example: "Can you please tell me if there are any luxury villas in Riyadh with 4 bedrooms?" -> "4 bedroom luxury villas in Riyadh"
  Example: "Compare DarGlobal and Wasalt" -> "DarGlobal luxury developments Wasalt Saudi real estate properties"
- If needs_retrieval is false: set rewritten_query to "".

OUTPUT INSTRUCTIONS:
- You must output ONLY valid, parseable JSON.
- Do NOT include markdown code fences (no ```json).
- Do NOT include any commentary or preamble.
"""


def parse_classifier_json(raw_text: str) -> dict[str, Any]:
    """Safely extract and parse JSON object from LLM response."""
    text = (raw_text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


SYSTEM_PROMPT = """You are the official Real Estate AI Assistant for DarGlobal and Wasalt property data.
Your task is to answer user questions strictly and accurately based ONLY on the retrieved property evidence provided below.

MANDATORY RULES:
1. Ground your answer completely and strictly in the provided evidence. Do NOT extrapolate, speculate, or invent details.
2. Never hallucinate or invent prices, locations, bedrooms, unit dimensions, amenities, or project availability.
3. Corporate and Company Information Rules:
   - Do NOT present company descriptions, business models, ownership structures, or corporate background as verified facts unless explicitly supported by the retrieved evidence.
   - If company-level information, business operations, or corporate histories are not present in the evidence, explicitly state that corporate details are unavailable in the collected property data.
   - Do NOT describe Wasalt as a property owner, landlord, or developer based only on its property listings.
   - Do NOT describe DarGlobal's corporate history, business model, or financials unless verified evidence is provided.
4. For DarGlobal listings: Disclose honestly that these are project entries from the official DarGlobal sitemap. Specific property attributes (prices, specifications, floor plans) were protected by website anti-bot security during automated collection and were unavailable.
5. For Wasalt listings: Present only the verified attributes appearing in the evidence (title, location, price in SAR, bedrooms, area in SQM, property type).
6. Property listings in the evidence are individual samples and must NOT be presented as a complete or exhaustive catalogue of either platform.
7. For queries involving both DarGlobal and Wasalt:
   - Provide balanced information covering both platforms whenever evidence exists for both.
   - If relevant evidence is missing or unavailable for one of the platforms, explicitly state that verified evidence for that platform was not found in the collected data.
8. If the provided evidence does not contain relevant information to answer the question, state clearly: "I do not have sufficient information in the collected property data to answer this question."
9. Always cite the property or project title, platform (Wasalt or DarGlobal), and direct source URL when discussing a property. Format direct source links as valid Markdown: [Property Title](source_url). Never invent, alter, or fabricate URLs.
10. Keep answers structured, polite, and concise.
"""


class RAGChatbot:
    """Orchestrator for the RAG pipeline with dynamic LLM intent routing."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedder: Optional[GeminiEmbedder] = None,
        llm_client: Optional[OpenRouterClient] = None,
    ) -> None:
        self.vector_store = vector_store or VectorStore()
        self.embedder = embedder or GeminiEmbedder()
        self.llm_client = llm_client or OpenRouterClient()

    async def classify_intent_with_llm(self, query: str) -> Optional[IntentAnalysis]:
        """Dynamically classify query intent and rewrite search query via OpenRouter LLM."""
        user_prompt = f"User Message: {query}"
        try:
            raw_response = await self.llm_client.generate_response(
                system_prompt=INTENT_CLASSIFIER_PROMPT,
                user_prompt=user_prompt,
                temperature=0.0,
            )
            parsed = parse_classifier_json(raw_response)
            if not isinstance(parsed, dict):
                logger.warning("Classifier response is not a valid JSON object.")
                return None

            intent_raw = parsed.get("intent")
            if not intent_raw or not isinstance(intent_raw, str):
                logger.warning("Classifier response missing or invalid 'intent' field.")
                return None

            intent_str = intent_raw.upper().strip()
            if intent_str not in ChatIntent.__members__:
                logger.warning(f"Classifier returned unrecognized intent: '{intent_str}'.")
                return None

            platforms = parsed.get("relevant_platforms", [])
            if not isinstance(platforms, list):
                platforms = [str(platforms)]

            needs_retrieval = bool(parsed.get("needs_retrieval", True))
            if intent_str in ("GREETING", "THANKS", "OUT_OF_SCOPE"):
                needs_retrieval = False

            rewritten = str(parsed.get("rewritten_query", "")).strip() or query
            conf = float(parsed.get("confidence", 1.0))
            conf = max(0.0, min(1.0, conf))

            return IntentAnalysis(
                intent=intent_str,
                relevant_platforms=platforms,
                needs_retrieval=needs_retrieval,
                rewritten_query=rewritten,
                confidence=conf,
            )
        except Exception as err:
            logger.warning(f"Dynamic intent classification failed safely: {type(err).__name__}")
            return None

    def retrieve_evidence(
        self,
        query_embedding: list[float],
        relevant_platforms: list[str],
        top_k: int = 4,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant records respecting platform constraints and balancing multi-source queries."""
        normalized_platforms: list[str] = []
        for p in (relevant_platforms or []):
            p_lower = str(p).strip().lower()
            if "darglobal" in p_lower or "dar global" in p_lower:
                if "DarGlobal" not in normalized_platforms:
                    normalized_platforms.append("DarGlobal")
            elif "wasalt" in p_lower:
                if "Wasalt" not in normalized_platforms:
                    normalized_platforms.append("Wasalt")

        # Multi-source query: retrieve from both platforms separately to ensure balance
        if "DarGlobal" in normalized_platforms and "Wasalt" in normalized_platforms:
            k_per_platform = max(2, top_k // 2)
            dg_matches = self.vector_store.query(
                query_embedding, top_k=k_per_platform, filter_platform="DarGlobal"
            )
            wasalt_matches = self.vector_store.query(
                query_embedding, top_k=k_per_platform, filter_platform="Wasalt"
            )
            combined = dg_matches + wasalt_matches

        # Single-platform query: retrieve only from the specified platform
        elif len(normalized_platforms) == 1:
            combined = self.vector_store.query(
                query_embedding, top_k=top_k, filter_platform=normalized_platforms[0]
            )

        # Unconstrained query: query without metadata filter
        else:
            combined = self.vector_store.query(
                query_embedding, top_k=top_k, filter_platform=None
            )

        # Deduplicate records by ID and canonical source URL
        seen_ids = set()
        seen_urls = set()
        deduped: list[dict[str, Any]] = []
        for match in combined:
            m_id = match.get("id")
            url = match.get("metadata", {}).get("source_url")
            if m_id and m_id in seen_ids:
                continue
            if url and url in seen_urls:
                continue
            if m_id:
                seen_ids.add(m_id)
            if url:
                seen_urls.add(url)
            deduped.append(match)

        return deduped

    async def answer(self, question: str, top_k: int = 4) -> dict[str, Any]:
        """Process user question through dynamic intent routing and RAG pipeline."""
        query_text = (question or "").strip()
        if not query_text:
            raise ValueError("User question cannot be empty.")

        # 1. Dynamic intent classification & query rewriting via LLM
        analysis = await self.classify_intent_with_llm(query_text)

        # Handle classification failure safely without triggering retrieval or fabricating data
        if analysis is None:
            return {
                "answer": (
                    "I am currently unable to process your request due to a temporary service issue. "
                    "Please rephrase your question or try again in a moment."
                ),
                "sources": [],
                "intent": None,
                "rewritten_query": "",
            }

        # 2. Fast-path responses for intents that do not require retrieval
        if not analysis.needs_retrieval:
            if analysis.intent == ChatIntent.GREETING.value:
                return {
                    "answer": (
                        "Hello! I am your AI assistant for DarGlobal and Wasalt real estate. "
                        "How can I help you explore verified properties, luxury development projects, "
                        "or locations today?"
                    ),
                    "sources": [],
                    "intent": analysis.intent,
                    "rewritten_query": "",
                }

            if analysis.intent == ChatIntent.THANKS.value:
                return {
                    "answer": (
                        "You're welcome! Please let me know if you need any more details "
                        "on DarGlobal luxury projects or Wasalt property listings."
                    ),
                    "sources": [],
                    "intent": analysis.intent,
                    "rewritten_query": "",
                }

            if analysis.intent == ChatIntent.OUT_OF_SCOPE.value:
                return {
                    "answer": (
                        "I do not have sufficient information in the collected property data "
                        "to answer this inquiry. I specialize exclusively in DarGlobal and Wasalt "
                        "real estate properties, listings, and development projects. "
                        "Please feel free to ask about properties in Riyadh, Jeddah, or DarGlobal luxury developments."
                    ),
                    "sources": [],
                    "intent": analysis.intent,
                    "rewritten_query": "",
                }

        # 3. Retrieval path (PROPERTY_SEARCH, PROJECT_SEARCH, COMPANY_INFORMATION, COMPANY_COMPARISON, FOLLOW_UP)
        search_query = analysis.rewritten_query if analysis.rewritten_query else query_text

        # Generate query embedding using rewritten query (768-dim, RETRIEVAL_QUERY)
        query_embedding = await self.embedder.embed_query(search_query)

        # Retrieve nearest neighbor properties with platform-aware filtering and multi-source balance
        matches = self.retrieve_evidence(
            query_embedding=query_embedding,
            relevant_platforms=analysis.relevant_platforms,
            top_k=top_k,
        )

        # Relevance threshold filter
        relevant_matches = [m for m in matches if m.get("similarity", 0.0) >= 0.35]

        if not relevant_matches:
            return {
                "answer": (
                    "I do not have sufficient information in the collected property data "
                    "to answer your question. Please try asking about available properties in Riyadh, "
                    "villas, apartments, or DarGlobal projects."
                ),
                "sources": [],
                "intent": analysis.intent,
                "rewritten_query": search_query,
            }

        # Check multi-platform balance and detect missing evidence
        normalized_requested: list[str] = []
        for p in (analysis.relevant_platforms or []):
            p_low = str(p).lower()
            if "darglobal" in p_low or "dar global" in p_low:
                if "DarGlobal" not in normalized_requested:
                    normalized_requested.append("DarGlobal")
            elif "wasalt" in p_low:
                if "Wasalt" not in normalized_requested:
                    normalized_requested.append("Wasalt")

        platforms_with_evidence = {m.get("metadata", {}).get("source_platform") for m in relevant_matches}
        platform_notes: list[str] = []
        if "DarGlobal" in normalized_requested and "DarGlobal" not in platforms_with_evidence:
            platform_notes.append("Note: No verified evidence meeting the relevance threshold was found for DarGlobal.")
        if "Wasalt" in normalized_requested and "Wasalt" not in platforms_with_evidence:
            platform_notes.append("Note: No verified evidence meeting the relevance threshold was found for Wasalt.")

        # 4. Format evidence context for LLM
        evidence_blocks = []
        sources = []
        seen_urls = set()

        for idx, match in enumerate(relevant_matches, start=1):
            meta = match.get("metadata", {})
            doc = match.get("document", "")
            similarity = match.get("similarity", 0.0)
            url = meta.get("source_url", "")

            evidence_blocks.append(
                f"[Evidence #{idx}] (Similarity: {similarity})\n"
                f"{doc}\n"
            )

            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append({
                    "title": meta.get("title", "Property Listing"),
                    "source_url": url,
                    "source_platform": meta.get("source_platform", "Unknown"),
                    "location": meta.get("location") or None,
                    "price": meta.get("price_formatted") or None,
                    "property_type": meta.get("property_type") or None,
                })

        notes_header = f"\nPlatform Observations: {' '.join(platform_notes)}\n" if platform_notes else ""

        user_prompt = (
            f"User Question: {query_text}\n"
            f"Detected Intent: {analysis.intent}\n"
            f"{notes_header}\n"
            f"=== RETRIEVED PROPERTY EVIDENCE ===\n"
            f"{''.join(evidence_blocks)}\n"
            f"=== END OF EVIDENCE ===\n\n"
            f"Please answer the user's question based strictly on the evidence above."
        )

        # 5. Generate final grounded completion via OpenRouter
        llm_answer = await self.llm_client.generate_response(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        return {
            "answer": llm_answer,
            "sources": sources,
            "intent": analysis.intent,
            "rewritten_query": search_query,
        }
