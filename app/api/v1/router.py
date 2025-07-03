from fastapi import APIRouter
from app.api.v1.endpoints import ai_service

api_router = APIRouter()

# You can include new, non-database routers here in the future
# Example: from .endpoints import ai_service
api_router.include_router(ai_service.router, prefix="/ai", tags=["AI Service"]) 