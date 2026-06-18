"""Available image models — only those whose provider has an API key configured."""

from fastapi import APIRouter
from pydantic import BaseModel

from .config import OPENAI_API_KEY, GEMINI_API_KEY, MODELS, Provider

router = APIRouter()


class Models:
    class Item(BaseModel):
        id: str
        label: str
        provider: Provider


@router.get("/api/models")
async def list_models() -> list[Models.Item]:
    available: list[Models.Item] = []
    for model_id, info in MODELS.items():
        provider = info["provider"]
        if provider == "openai" and OPENAI_API_KEY:
            available.append(
                Models.Item(id=model_id, label=info["label"], provider=provider)
            )
        elif provider == "gemini" and GEMINI_API_KEY:
            available.append(
                Models.Item(id=model_id, label=info["label"], provider=provider)
            )
    return available
