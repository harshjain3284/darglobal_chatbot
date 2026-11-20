# PROJECT_CONTEXT.md — DarGlobal & Wasalt AI Real Estate Chatbot

> **Notice to All Coding Agents:**  
> Before making any changes or starting any development task, you **MUST** read both `AGENTS.md` and this `PROJECT_CONTEXT.md` file.  
> This file maintains the permanent context, current progress, architectural decisions, and open items across development sessions.

---

## 1. Project Overview

- **Project Name:** DarGlobal & Wasalt AI Real Estate Chatbot
- **Objective:** Build a reliable, production-conscious, demo-ready AI chatbot that answers user questions regarding properties, projects, locations, amenities, and pricing using publicly scraped data from **DarGlobal** and **Wasalt**.
- **Expected Outcome:** A containerized web application exposing clean FastAPI REST endpoints and a responsive, user-friendly chat UI that delivers grounded responses with clickable source citations, publicly deployed with a working test URL.

---

## 2. Assignment Requirements

1. **Website Scraping:** Collect publicly accessible data from DarGlobal (`darglobal.co.uk`) and Wasalt (`wasalt.com`) respecting `robots.txt`, access limits, and terms of service.
2. **Data Cleaning & Processing:** Clean, normalize, structure, and deduplicate property listings and project descriptions using pure Python.
3. **RAG Pipeline:** Implement a direct Retrieval-Augmented Generation (RAG) architecture grounded strictly in retrieved website data (zero LangChain/LangGraph).
4. **LLM Integration:** Integrate OpenRouter using a suitable free model via direct `httpx` calls with configurable parameters and resilient error handling.
5. **API & Interface:** Provide FastAPI REST endpoints and an intuitive web chat interface displaying answers alongside source URLs.
6. **Containerization:** Package the application using Docker adhering to best practices (lightweight footprint, non-root, environment-based configuration, zero browser dependencies in production).
7. **Public Deployment:** Deploy the containerized application to a public cloud platform and provide a live URL.

---

## 3. Approved Final Technology Stack

| Component | Technology | Approval Status | Notes / Rationale |
| :--- | :--- | :--- | :--- |
| **Language & Runtime** | Python 3.11 | **Approved** | Pinned in `.python-version` and `pyproject.toml`. |
| **Project & Package Manager** | Astral `uv` | **Approved** | Packaged application structure (`src/` layout); lockfile-driven. |
| **Scraping Framework** | Crawl4AI | **Approved** | Primary offline scraping framework. Test static extraction first; browser rendering used only where dynamic JS is required. |
| **Chunking Engine** | Pure Python Chunker | **Approved** | Domain-tailored real estate text splitter in standard Python (zero LangChain). |
| **Embedding Engine** | Google Gemini `gemini-embedding-001` | **Approved** | Direct REST API calls via `httpx` (`embedContent` endpoint; no batch embedding on free tier; zero SDKs). |
| **Vector Database** | ChromaDB (`chromadb`) | **Approved** | Native Python client (`chromadb.PersistentClient`) for local vector indexing and metadata storage. |
| **LLM Gateway** | OpenRouter API (via `httpx`) | **Approved** | Direct async HTTP requests; supports free models with custom headers and fallback handling. |
| **Backend Framework** | FastAPI + Uvicorn | **Approved** | High performance ASGI, native Pydantic validation; directly serves both API and static UI. |
| **Frontend** | Vanilla HTML5 + CSS3 + JavaScript | **Approved** | Zero Node.js/npm build chain; served directly from `ui/static/` by FastAPI. |
| **Framework Exclusions** | No LangChain, No LangGraph | **Approved** | RAG implemented using pure Python, native ChromaDB, and `httpx`. |
| **Containerization** | Docker (`python:3.11-slim`) | **Approved** | Lightweight production image strictly excluding Playwright/browser binaries. |
| **Cloud Hosting Platform** | Render / Hugging Face Spaces / Northflank | **Pending Verification** | Choose based on no-credit-card requirements and resource constraints. |

---

## 4. Architecture: Subsystem Separation

The project enforces a **Modular Layered (Service-Based) Architecture** with an explicit boundary between two operational subsystems:

### Subsystem A: Offline Ingestion & Indexing Pipeline (Batch Execution)
- **Role:** Independent, offline workflow executed via CLI scripts (`scripts/run_scrapers.py`, `scripts/build_index.py`).
- **Operation:**
  1. Crawls target websites politely using Crawl4AI (`scrapers/darglobal.py`, `scrapers/wasalt.py`).
  2. Evaluates pages for static vs. browser rendering requirements to avoid unnecessary browser overhead.
  3. Cleans, normalizes, and deduplicates records into `data/processed/` using `scrapers/cleaner.py`.
  4. Chunks text via pure Python while preserving real estate metadata (prices, locations, amenities, canonical URLs).
  5. Computes vector embeddings via Gemini `gemini-embedding-001` (`embedContent` per chunk or polite sequential loop) and writes persistent index to `data/vector_store/`.
- **Isolation:** Never runs inside the live chatbot API process. Uses development dependencies only.

### Subsystem B: Online Chatbot Runtime Service (Live Production)
- **Role:** The containerized web service deployed publicly.
- **Operation:**
  1. Bootstraps FastAPI from `src/darglobal_project/main.py`.
  2. Loads the pre-built `data/vector_store/` via native ChromaDB in read-only / persistent mode at startup.
  3. Serves the REST API (`api/chat_routes.py`) for incoming chat queries.
  4. Directly serves static frontend assets from `src/darglobal_project/ui/static/` at `/`.
  5. Retrieves relevant property context (`rag/retriever.py`) and returns grounded OpenRouter completions (`rag/llm_client.py`) with source citations.
- **Isolation:** Contains zero scraping code execution, zero Playwright/browser dependencies, and zero cold-start scraping delays.

### Source Code Placement
All Python application source code must strictly remain inside `src/darglobal_project/`.

### Approved Directory Structure

```text
darglobal_project/
│
├── data/
│   ├── raw/                           # Raw crawl outputs (Markdown & metadata from Crawl4AI)
│   ├── processed/                     # Cleaned, normalized, and deduplicated property records
│   └── vector_store/                  # Persistent ChromaDB vector index and metadata store
│
├── src/
│   └── darglobal_project/
│       ├── __init__.py                # Package declaration
│       ├── config.py                  # Pydantic BaseSettings (keys, paths, models)
│       ├── main.py                    # FastAPI entrypoint: index loader, API routes & static UI mount
│       │
│       ├── scrapers/                  # Ingestion & Normalization Layer
│       │   ├── __init__.py
│       │   ├── darglobal.py           # DarGlobal scraper module (Crawl4AI)
│       │   ├── wasalt.py              # Wasalt scraper module (Crawl4AI)
│       │   └── cleaner.py             # Normalizer & pure Python chunker
│       │
│       ├── rag/                       # RAG & Intelligence Layer (Native integrations)
│       │   ├── __init__.py
│       │   ├── embeddings.py          # Gemini gemini-embedding-001 direct httpx client
│       │   ├── retriever.py           # Native ChromaDB search & context assembly
│       │   └── llm_client.py          # OpenRouter client via httpx (prompts & retries)
│       │
│       ├── api/                       # API Layer
│       │   ├── __init__.py
│       │   ├── schemas.py             # Pydantic models (ChatRequest, ChatResponse, CitationSource)
│       │   └── chat_routes.py         # Endpoints: /api/chat, /health
│       │
│       └── ui/                        # Frontend Assets (Served directly by main.py)
│           └── static/
│               ├── index.html         # Chatbot interface
│               ├── style.css          # Modern real-estate styling
│               └── app.js             # Client-side chat logic & citation badges
│
├── scripts/                           # Offline Automation Scripts
│   ├── run_scrapers.py                # Standalone crawler runner
│   └── build_index.py                 # Standalone vector index builder
│
├── tests/                             # Unit & integration test suite
├── AGENTS.md                          # Permanent project rules & guidelines
├── PROJECT_CONTEXT.md                 # Permanent project reference (this file)
├── pyproject.toml                     # Dependencies & package configuration
├── uv.lock                            # Dependency lockfile
├── .python-version                    # Python 3.11 marker
├── .env.example                       # Environment variable template
├── .gitignore                         # Git exclusion rules
├── .dockerignore                      # Docker build exclusion rules
├── Dockerfile                         # Production container specification
└── README.md                          # Documentation & usage guide
```

---

## 5. End-to-End Data Flow

```text
========================================================================
1. OFFLINE INGESTION & INDEXING (scripts/run_scrapers.py & build_index.py)
========================================================================
Target Websites (darglobal.co.uk, wasalt.com)
      │
      ▼ (Crawl4AI: evaluate static vs. browser rendering per page)
data/raw/*.json (Markdown text, page titles, canonical URLs, timestamps)
      │
      ▼ (scrapers/cleaner.py: normalize currencies/areas, deduplicate)
data/processed/properties.json (Structured property entities)
      │
      ▼ (Pure Python chunker: preserve project specs, price, location, URL)
Text Chunks + Metadata Dictionaries
      │
      ▼ (rag/embeddings.py: Gemini gemini-embedding-001 embedContent via httpx)
data/vector_store/ (Native ChromaDB persistent collection)

========================================================================
2. ONLINE CHATBOT RUNTIME (FastAPI: main.py + chat_routes.py)
========================================================================
User enters query via Browser UI (served from ui/static/)
      │
      ▼ (POST /api/chat)
rag/retriever.py:
  a. Embed query using Gemini gemini-embedding-001 via httpx.
  b. Query native ChromaDB collection for top-k nearest chunks.
  c. Extract snippets and metadata (canonical URLs, titles, prices).
  d. Assemble Grounded Context with Source IDs:
     "[Source 1] (DarGlobal - Project X) Details: ... URL: https://darglobal.co.uk/..."
      │
      ▼ (Grounded Context + Query + Strict Prompt)
rag/llm_client.py:
  a. Sends request to OpenRouter API (free model via httpx).
  b. Enforces strict grounding (no hallucinations; explicit "unavailable" fallbacks).
  c. Formats bracketed citations [1], [2].
      │
      ▼ (HTTP response)
FastAPI JSON Response:
{
  "answer": "DarGlobal offers...",
  "sources": [{"id": 1, "title": "Project X", "url": "https://...", "platform": "DarGlobal"}]
}
      │
      ▼
Browser UI: Renders formatted response with interactive, clickable citation links.
```

---

## 6. Coding Rules & Best Practices

All contributors and coding agents must strictly comply with `AGENTS.md`:
- **Read First:** Review `AGENTS.md` and `PROJECT_CONTEXT.md` before starting any task.
- **Minimum Code, Maximum Clarity:** Write short, idiomatic, and readable Python. Avoid boilerplate and over-abstraction.
- **DRY (Don't Repeat Yourself):** Reuse utilities, modules, and schemas. Never create duplicate implementations.
- **Inspect Before Creating:** Always inspect existing modules before writing new code.
- **No Unapproved Redesigns:** Do not alter the approved directory structure or introduce new frameworks/services without explicit user approval.
- **Modularity Discipline:** Maintain the approved modular structure (`config.py`, `main.py`, `scrapers/`, `rag/`, `api/`, `ui/static/`, `scripts/`). Do not collapse modules into mega-files, but do not create unnecessary abstractions.

---

## 7. RAG Quality and Data Reliability

- **Strict Grounding:** The chatbot must rely exclusively on retrieved context.
- **Zero Hallucination:** The LLM must never invent prices, locations, amenities, or property specifications.
- **Honest Fallbacks:** If the requested information is absent from the indexed data, the model must explicitly state that the information is unavailable.
- **Canonical Citations:** Every scraped property retains its canonical source URL and metadata; the API and UI must display direct links alongside every answer.
- **Deduplication:** Raw records must be deduplicated by canonical URL and attributes during processing to prevent redundant index chunks.

---

## 8. Docker and Production Deployment

- **Lightweight Production Image:** The `Dockerfile` is built on `python:3.11-slim`.
- **Exclusion of Browser Dependencies:** Because scraping and index building are performed offline, **Playwright, Chromium binaries, and browser automation drivers are strictly excluded from the production Docker image**.
- **Pre-Built Index Bundling:** The production container packages the application and the pre-computed `data/vector_store/`, enabling instantaneous startup.
- **Dynamic Port & Host:** The application reads `PORT` and `HOST` dynamically from environment variables (e.g., `${PORT:-8000}`).
- **Secret Hygiene:** Never commit `.env` or credentials. All API keys must be loaded via environment variables (`OPENROUTER_API_KEY`, `GEMINI_API_KEY`).
- **Public Deployment URL:** *TBD (To be added upon deployment verification).*

---

## 9. Development Progress

### Phase 1: Project Setup & Guidelines
- [x] Create and approve `AGENTS.md` with permanent rules and constraints.
- [x] Add explicit rules: no duplicate code, mandatory architecture approval, and no unnecessary code generation.
- [x] Initialize uv project using official packaged application structure (`uv init --app --python 3.11`).
- [x] Verify project structure, `.python-version`, and `pyproject.toml`.
- [x] Create and refine `PROJECT_CONTEXT.md` as the authoritative cross-session reference document.

### Phase 2: Architecture & Decision Finalization
- [x] Delineate Subsystem A (Offline Ingestion/Index) vs. Subsystem B (Online Chatbot).
- [x] Finalize directory structure approval.
- [x] Approve native ChromaDB as vector store.
- [x] Approve Gemini `gemini-embedding-001` via direct `httpx` (no LangChain, no batch embedding on free tier).
- [x] Approve pure Python chunking (zero LangChain).
- [x] Approve OpenRouter via direct `httpx` (no LangChain).
- [x] Approve production and dev dependency list.

### Phase 3: Dependencies & Scaffolding
- [x] Add approved production dependencies (`fastapi`, `uvicorn[standard]`, `pydantic-settings`, `chromadb`, `httpx`) via `uv add`.
- [x] Add approved dev dependencies (`crawl4ai`, `playwright`, `pytest`, `pytest-asyncio`) via `uv add --dev`.
- [x] Generate and verify `uv.lock`.
- [x] Scaffold folder structure (`src/darglobal_project/scrapers/`, `scripts/`, `data/raw/`, `data/processed/`, `tests/`).

### Phase 4: Data Scraping & Processing (Subsystem A)
- [x] Verify static vs. browser rendering requirements for DarGlobal and Wasalt pages.
- [x] Implement DarGlobal scraper (`scrapers/darglobal.py`).
- [x] Implement Wasalt scraper (`scrapers/wasalt.py`).
- [x] Implement data cleaner, normalizer, and pure Python chunker (`scrapers/cleaner.py`).
- [x] Implement standalone runner script (`scripts/run_scrapers.py`).
- [x] Execute scraping, verify collected data, and save raw/processed outputs.

### Phase 5: RAG & Index Building (Subsystem A & B)
- [x] Validate raw dataset and clean all unbacked attributes (Part A).
- [x] Implement Gemini embedding client (`rag/embeddings.py`).
- [x] Implement native ChromaDB index builder script (`scripts/build_index.py`).
- [x] Build and verify persistent vector index (`data/vector_store/`).
- [x] Implement retriever with citation formatting (`rag/retriever.py`).
- [x] Implement OpenRouter client with fallback handling (`rag/llm_client.py`).
- [x] Implement dynamic LLM-based intent routing with graceful classification fallback (`rag/service.py`).
- [x] Implement platform-aware multi-source balanced retrieval and deduplication (`rag/service.py`, `rag/retriever.py`).

### Phase 6: API & Frontend Development (Subsystem B)
- [x] Implement Pydantic request/response schemas (`api/chat_routes.py`).
- [x] Implement FastAPI endpoints (`api/chat_routes.py` and `main.py`).
- [x] Build responsive chat UI with source citation badges (`ui/static/`).
- [x] Implement safe frontend Markdown rendering with local vendored `marked.min.js` and `purify.min.js`.
- [x] Test end-to-end user query flow and automated test suite (46 passing tests).

### Phase 7: Containerization & Deployment
- [ ] Write and optimize lean production `Dockerfile` (excluding Playwright).
- [ ] Measure runtime memory with Docker (`docker stats`).
- [ ] Deploy to public cloud platform (Render / Hugging Face Spaces / Northflank).
- [ ] Verify live public URL, source links, and chatbot reliability.

---

## 10. Important Decisions and Verification Notes

### Approved Decisions
1. Use Python 3.11 with `uv` packaged-project layout (`src/` structure).
2. Modular Layered (Service-Based) Architecture strictly separating offline scraping/indexing from online chatbot execution.
3. Crawl4AI as the scraping framework, testing static extraction first and applying browser rendering only where necessary.
4. Native ChromaDB (`chromadb`) for vector store and metadata index.
5. Gemini `gemini-embedding-001` via direct HTTP `embedContent` requests (no LangChain, no batch embedding on free tier).
6. Pure Python real estate text chunker.
7. OpenRouter as the sole LLM inference gateway using free-tier models via direct `httpx` calls.
8. FastAPI as backend and Vanilla HTML/CSS/JS as frontend served directly from `ui/static/` by FastAPI.
9. Playwright and browser dependencies strictly excluded from the production Docker image.
10. Measure container memory empirically via Docker before cloud deployment.
