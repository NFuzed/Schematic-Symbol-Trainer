from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import Services, services
from app.schemas import ModelInfo

router = APIRouter(tags=["models"])


@router.get("/models", response_model=list[ModelInfo])
def list_models(svc: Annotated[Services, Depends(services)]) -> list[ModelInfo]:
    return [record.info() for record in svc.store.list()]
