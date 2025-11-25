# src/routes/routes.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from src.services import CVService

router = APIRouter(prefix="/api")

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []

class ChatResponse(BaseModel):
    response: str

# Dependency provider - in production you'd use injection or use app.state
def get_service() -> CVService:
    # this function will be replaced by an instance set on app.state at startup.
    # FastAPI will call the dependency function each request; we will override
    # it with the real service using `app.dependency_overrides` in main.py or
    # provide via `app.state` and refer to it directly there.
    raise RuntimeError("CVService dependency was not configured.")


@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest, service: CVService = Depends(get_service)):
    try:
        result = service.answer_question(payload.message, payload.history)
        return ChatResponse(response=result)
    except Exception as e:
        # You can log the exception here
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
def health(service: CVService = Depends(get_service)):
    # lightweight health check - ensure index exists
    ok = service.index is not None
    return {"status": "ok" if ok else "not ready"}
