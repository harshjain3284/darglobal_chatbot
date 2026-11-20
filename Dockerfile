# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Copy standalone uv binary from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Configure Python and uv runtime behavior
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONPATH=/app/src \
    HOST=0.0.0.0 \
    PORT=8000

# Copy dependency specifications first to leverage Docker layer caching
COPY pyproject.toml uv.lock ./

# Install production dependencies only (strictly excluding crawl4ai, playwright, pytest)
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code, pre-built ChromaDB index, and processed data
COPY src/ /app/src/
COPY data/processed/ /app/data/processed/
COPY data/vector_store/ /app/data/vector_store/
COPY README.md /app/README.md

# Install application package into the uv environment
RUN uv sync --frozen --no-dev

# Add virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Create non-root user and grant workspace permissions
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose default application port
EXPOSE 8000

# Run FastAPI via Uvicorn with dynamic HOST and PORT
CMD ["sh", "-c", "exec uvicorn darglobal_project.main:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000}"]
