"""FastAPI router for chatbot and health check endpoints."""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from darglobal_project.config import settings
from darglobal_project.rag.service import RAGChatbot

logger = logging.getLogger(__name__)

router = APIRouter()

# Global chatbot instance reused across requests (avoids reloading vector store)
_chatbot_instance: Optional[RAGChatbot] = None


def get_chatbot() -> RAGChatbot:
    """Provide singleton instance of RAGChatbot."""
    global _chatbot_instance
    if _chatbot_instance is None:
        _chatbot_instance = RAGChatbot()
    return _chatbot_instance


class ChatRequest(BaseModel):
    """User chat request payload."""
    question: str = Field(..., min_length=1, max_length=1000, description="User query about real estate")


class SourceCitation(BaseModel):
    """Grounding source citation for retrieved property."""
    title: str
    source_url: str
    source_platform: str
    location: Optional[str] = None
    price: Optional[str] = None
    property_type: Optional[str] = None


class ChatResponse(BaseModel):
    """Grounded chatbot response with citations."""
    answer: str
    sources: list[SourceCitation] = []
    intent: Optional[str] = None
    rewritten_query: Optional[str] = None


class HealthResponse(BaseModel):
    """Service health response."""
    status: str
    records_indexed: int
    model: str


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return application health and collection status."""
    try:
        chatbot = get_chatbot()
        count = chatbot.vector_store.count()
        return HealthResponse(
            status="healthy",
            records_indexed=count,
            model=settings.openrouter_model,
        )
    except Exception as err:
        logger.error(f"Health check failed: {err}")
        return HealthResponse(
            status="degraded",
            records_indexed=0,
            model=settings.openrouter_model,
        )


@router.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Process user real estate query and return grounded response with sources."""
    try:
        chatbot = get_chatbot()
        result = await chatbot.answer(request.question)
        return ChatResponse(
            answer=result["answer"],
            sources=[SourceCitation(**s) for s in result.get("sources", [])],
            intent=result.get("intent"),
            rewritten_query=result.get("rewritten_query"),
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as err:
        logger.error(f"Error handling chat request: {err}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your request. Please try again later.",
        )
