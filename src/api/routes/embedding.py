"""
Embedding configuration endpoints for OKT-RAG.

Provides API for managing multi-embedding slots.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from ...config import get_settings, EmbeddingSlotConfig


router = APIRouter(prefix="/embedding", tags=["Embedding"])


class SlotConfigResponse(BaseModel):
    """Response model for slot configuration."""

    name: str
    provider: str
    model: str
    dimension: int
    weight: float
    enabled: bool
    description: str


class EmbeddingConfigResponse(BaseModel):
    """Response model for full embedding configuration."""

    multi_embedding_enabled: bool
    default_slot: str
    slots: list[SlotConfigResponse]


class SlotStatsResponse(BaseModel):
    """Response model for slot statistics."""

    slot_name: str
    document_count: int
    dimension: int
    provider_type: str
    model: str
    weight: float
    enabled: bool
    has_provider: bool


class StoreStatsResponse(BaseModel):
    """Response model for store statistics."""

    working_dir: str
    total_slots: int
    enabled_slots: int
    slots: dict[str, SlotStatsResponse]


@router.get("/config", response_model=EmbeddingConfigResponse)
async def get_embedding_config():
    """
    Get current embedding configuration.

    Returns the multi-embedding slot configuration including:
    - Whether multi-embedding is enabled
    - Default embedding slot
    - List of configured slots with their settings
    """
    settings = get_settings()

    slots = [
        SlotConfigResponse(
            name=slot.name,
            provider=slot.provider,
            model=slot.model,
            dimension=slot.dimension,
            weight=slot.weight,
            enabled=slot.enabled,
            description=slot.description,
        )
        for slot in settings.embedding_slots
    ]

    return EmbeddingConfigResponse(
        multi_embedding_enabled=settings.multi_embedding_enabled,
        default_slot=settings.default_embedding_slot,
        slots=slots,
    )


@router.get("/slots")
async def list_embedding_slots():
    """
    List all configured embedding slots.

    Returns summary of each slot configuration.
    """
    settings = get_settings()

    return {
        "slots": [
            {
                "name": slot.name,
                "provider": slot.provider,
                "model": slot.model,
                "dimension": slot.dimension,
                "weight": slot.weight,
                "enabled": slot.enabled,
            }
            for slot in settings.embedding_slots
        ],
        "default_slot": settings.default_embedding_slot,
    }


@router.get("/slots/{slot_name}")
async def get_slot_details(slot_name: str):
    """
    Get details for a specific embedding slot.

    Args:
        slot_name: Name of the slot to retrieve.

    Returns:
        Detailed slot configuration.
    """
    settings = get_settings()

    for slot in settings.embedding_slots:
        if slot.name == slot_name:
            return {
                "name": slot.name,
                "provider": slot.provider,
                "model": slot.model,
                "dimension": slot.dimension,
                "weight": slot.weight,
                "enabled": slot.enabled,
                "description": slot.description,
            }

    raise HTTPException(
        status_code=404,
        detail=f"Slot '{slot_name}' not found",
    )


@router.get("/matryoshka")
async def get_matryoshka_info():
    """
    Get information about Matryoshka embedding support.

    Matryoshka embeddings allow dimension reduction while preserving
    semantic quality, enabling speed/quality tradeoffs.
    """
    return {
        "supported_models": {
            "text-embedding-3-small": {
                "default_dimension": 1536,
                "supported_dimensions": [256, 512, 1024, 1536],
                "recommended_for_speed": 512,
                "recommended_for_quality": 1536,
            },
            "text-embedding-3-large": {
                "default_dimension": 3072,
                "supported_dimensions": [256, 512, 1024, 2048, 3072],
                "recommended_for_speed": 1024,
                "recommended_for_quality": 3072,
            },
        },
        "benefits": [
            "Reduced storage costs with smaller dimensions",
            "Faster similarity search",
            "Same model, configurable quality/speed tradeoff",
            "No need to re-train or use different models",
        ],
    }
