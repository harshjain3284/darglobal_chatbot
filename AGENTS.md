# AGENTS.md — Project Guidelines and Permanent Rules

> **Project:** DarGlobal & Wasalt Real Estate AI Chatbot  
> **Purpose:** Permanent rules, architectural standards, and workflow constraints governing all phases of development.

---

## 1. Project Overview

This project is a technical assignment focused on creating a functional, reliable, maintainable, and demo-ready AI Chatbot.

### Key Requirements
- **Data Collection:** Scrape publicly available data from DarGlobal and Wasalt websites.
- **Data Cleaning & Processing:** Clean, normalize, and organize the collected property and project data.
- **RAG Architecture:** Implement a Retrieval-Augmented Generation (RAG) pipeline to ground all responses in retrieved website data.
- **LLM Integration:** Integrate a suitable, free model available via OpenRouter with configurable settings.
- **User Interface & API:** Expose clean endpoints and a responsive, user-friendly chatbot interface with source citations.
- **Containerization:** Containerize the entire application using Docker following best practices.
- **Public Deployment:** Deploy the application publicly and provide a working, testable URL.

---

## 2. Technology & Project Management Rules

- **Python Version:** Python 3.11 strictly.
- **Package & Environment Manager:** Use `uv` exclusively for dependency and virtual environment management.
- **Project Initialization Standard:** Follow the official uv no-build-system initialization approach:
  ```bash
  uv init --no-package --python 3.11
  ```
- **Structure:** Follow the standard `no-package` structure. Do not introduce unnecessary build backends (e.g., `setuptools`, `hatchling`, `flit`) or packaging boilerplate unless explicitly required.
- **Lockfile Integrity:** Let `uv` manage the virtual environment and `uv.lock`.
- **Dependency Discipline:** Choose third-party libraries only when genuinely required by the architecture. Prefer lightweight, free, and open-source solutions.
- **LLM Gateway:** Use OpenRouter as the sole gateway for LLM inferences.

---

## 3. Architecture Rules

- **Simplicity First:** Follow a clear, modular, and easy-to-understand architecture.
- **Mandatory Architecture Approval:** The coding agent must propose the complete project directory structure and get user approval before creating application modules or adding major dependencies.
- **Separation of Concerns:** Maintain clean boundaries across modules:
  - Scraping (`scraper`)
  - Data processing & cleaning (`processing`)
  - Retrieval & vector search (`retrieval`)
  - LLM integration (`llm`)
  - API endpoints (`api`)
  - User interface (`frontend` / `ui`)
- **Single Responsibility:** Every module and file must have one distinct, well-defined responsibility.
- **Strict Adherence:** All code additions must adhere strictly to the approved architecture.
- **No Unapproved Redesigns:** Do not alter the approved architecture or introduce new layers, agents, services, or design patterns without providing clear justification and receiving explicit user approval.
- **Loose Coupling:** Keep components decoupled to ensure individual pieces can be tested, debugged, and replaced independently.

---

## 4. Code Quality Rules — Very Important

- **Concise & Readable:** Write short, clean, readable, and idiomatic Python.
- **Minimalism:** Write the minimum amount of code required to solve the task correctly.
- **No Unnecessary Code Generation:** Do not generate lengthy implementations, unnecessary boilerplate, repeated code blocks, or extra abstractions. Prefer the shortest maintainable implementation that satisfies the requirement.
- **DRY Principle:** Strictly avoid duplicating logic or data structures. Reuse existing utilities and functions.
- **No Duplicate Code:** Before writing new code, inspect existing modules and reuse their functionality. Never create duplicate functions, utilities, or logic when an existing implementation can be reused.
- **No Over-Abstraction:** Avoid unnecessary classes, inheritance hierarchies, abstract wrappers, and excessive helper functions. Write straightforward procedural or functional code when sufficient.
- **File Discipline:** Do not create files or folders without a clear, immediate purpose.
- **No Mega-Files:** Do not place the entire application into a single file to minimize file counts; maintain modularity.
- **Meaningful Documentation:** Avoid noisy, redundant inline comments that state the obvious. Include comments only where logic is non-trivial.
- **Naming Standards:** Use descriptive, consistent, and standard Python naming conventions (`snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants).
- **Error Handling:** Implement graceful, focused error handling without burying root causes or introducing bloated exception hierarchies.

> **Core Principle:** Minimum code, maximum clarity, zero duplication, and strict adherence to the approved architecture.

---

## 5. Website Scraping Rules

- **Public Data Only:** Scrape only publicly accessible information from DarGlobal and Wasalt websites.
- **Polite Crawling:** Respect website terms of service, `robots.txt`, and reasonable request rate limits (avoid aggressive concurrency or hammering target servers).
- **No Circumvention:** Never attempt to bypass access controls, authentication gates, anti-bot mechanisms, or CAPTCHA challenges.
- **Privacy First:** Avoid extracting personal or private contact details of individuals. Focus on property details, projects, amenities, locations, pricing, and specifications.
- **Source Preservation:** Every scraped item must store its canonical source URL and extraction timestamp.
- **Data Integrity:** Handle missing fields, partial data, and duplicate records gracefully during extraction.
- **Reusability & Efficiency:** Keep the scraping logic deterministic, modular, and runnable as standalone scripts/tasks without redundant repeated scraping.

---

## 6. Data Processing & RAG Rules

- **Normalization:** Clean, deduplicate, and normalize raw scraped data prior to chunking or indexing.
- **Chunking Strategy:** Employ an appropriate chunking strategy tailored to real estate data (preserving property specifications, project overviews, and location context).
- **Metadata Retention:** Retain source URLs, titles, and property attributes in chunk metadata so they can be surfaced to users.
- **Vector Storage:** Store embeddings in a lightweight, suitable vector database or local index.
- **Grounded Responses:** Chatbot responses must be strictly grounded in retrieved context.
- **No Hallucinations:** The LLM must never invent project names, prices, locations, unit sizes, or property amenities.
- **Honest Fallbacks:** If the requested information is absent from the indexed data, the model must explicitly state that the information is unavailable.
- **Citations:** Always provide direct source URLs/references with answers whenever relevant context is used.

---

## 7. LLM & OpenRouter Rules

- **Provider:** Exclusively use OpenRouter for LLM completions.
- **Model Selection:** Prefer suitable, active, free-tier models on OpenRouter (e.g., capable open models available at zero cost).
- **Configurability:** Ensure the model identifier, temperature, and related parameters are configurable via environment variables.
- **Credential Safety:** Store OpenRouter API keys strictly in environment variables (`.env`).
  - Never hardcode API keys or secrets in source code.
  - Never commit API keys to version control.
  - Never expose API keys to frontend clients, log messages, or Docker images.
- **Resilience:** Handle rate limits, temporary OpenRouter downtime, and invalid responses gracefully with actionable error logging.
- **Decoupled Logic:** Keep the LLM client call logic decoupled from RAG retrieval and API routes.

---

## 8. API & Frontend Rules

- **Framework:** Use FastAPI if it aligns with the approved project architecture.
- **Focused Endpoints:** Design simple, single-purpose REST endpoints (e.g., `/health`, `/chat`).
- **Input Validation:** Validate request payloads using Pydantic models.
- **User Interface:** Provide a clean, intuitive, and responsive web chat interface.
- **Transparent Sourcing:** Visibly render source links and references alongside answers in the UI.
- **Lightweight Dependencies:** Avoid heavy frontend frameworks or bloated toolchains unless strictly necessary.

---

## 9. Security Rules

- **Environment Separation:** Use `.env` for local secrets and runtime configuration.
- **Template Configuration:** Maintain an up-to-date `.env.example` file containing placeholder keys and instructions.
- **Git Hygiene:** Maintain a robust `.gitignore` excluding `.env`, virtual environments (`.venv`), cache files (`__pycache__`), build artifacts, and downloaded vector indexes/raw scrapes where appropriate.
- **Zero Secrets in Git:** Never commit credentials, private keys, or API tokens.
- **Sanitization:** Validate and sanitize all external user inputs before downstream retrieval or LLM inference.
- **Safe Error Responses:** Do not leak stack traces, internal paths, or environment variables to client responses.

---

## 10. Docker & Deployment Rules

- **Dockerfile Quality:** Write a clean, multi-stage or lean single-stage Dockerfile adhering to Docker best practices.
- **Lightweight Footprint:** Avoid installing extraneous compilers, debuggers, or unneeded dependencies in the final container image.
- **Dynamic Port & Host:** Ensure the application reads `PORT` and `HOST` from environment variables to satisfy cloud hosting environments.
- **Reproducibility:** Ensure the container build and startup can be tested and verified locally before cloud deployment.
- **Deployment Platform:** Choose an appropriate free/low-cost deployment platform (e.g., Render, Railway, Hugging Face Spaces, Fly.io) compatible with containerized FastAPI services.
- **Live Verification:** Thoroughly test and verify the publicly accessible URL before declaring the assignment complete.

---

## 11. Testing & Development Workflow

- **Incremental Progress:** Work step-by-step. Do not jump ahead to future tasks before completing and verifying the current task.
- **Inspect Before Acting:** Always inspect existing code and file structure before implementing new logic.
- **Test Before Moving Forward:** Test each feature (scraper, cleaner, indexer, retriever, API endpoint, UI) immediately after implementation.
- **Immediate Bug Fixing:** Resolve errors as soon as they arise; never leave known regressions behind.
- **No Silent Changes:** Never modify existing approved architecture, files, or configurations without documenting the change and getting user approval.
- **Transparent Communication:** Explain key commands, dependencies, file structures, and architecture choices in clear, beginner-friendly language.
- **Empirical Verification:** Never assume code works without running and verifying it.

---

## 12. Instructions for the Coding Agent

- **Primary Source of Truth:** Treat this `AGENTS.md` file as the primary governing specification for all development steps.
- **Review Requirement:** Consult these rules before starting any task or writing any code.
- **Scope Discipline:** Do not implement features or add dependencies outside the assignment's explicit scope without prior approval.
- **Prevent Overengineering:** Implement the simplest viable architecture that satisfies all assignment criteria.
- **Clarification Over Assumptions:** If any requirement or data source structure is ambiguous, request clarification before making sweeping implementation choices.

---

## 13. Core Development Philosophy

> **Build the simplest architecture that completely satisfies the assignment. Every file, function, dependency, and line of code must have a clear purpose.**
