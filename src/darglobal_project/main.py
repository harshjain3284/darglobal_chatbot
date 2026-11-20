"""Main FastAPI application entrypoint."""

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from darglobal_project.api.chat_routes import router as chat_router

app = FastAPI(
    title="DarGlobal & Wasalt Real Estate Chatbot",
    description="Grounded AI Assistant for DarGlobal and Wasalt real estate properties.",
    version="1.0.0",
)

# Enable CORS for local testing or cross-origin access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes first
app.include_router(chat_router)

# Mount UI static files
static_dir = Path(__file__).resolve().parent / "ui" / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


def run() -> None:
    """Run server with uvicorn CLI entrypoint."""
    import uvicorn
    from darglobal_project.config import settings

    uvicorn.run(
        "darglobal_project.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    run()
