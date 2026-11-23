"""FastAPI app. Run with: uvicorn app.main:app --port 8000"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import detections, models, training
from app.api.deps import Services, services
from app.config import Settings
from app.inference.detector import Detector
from app.schemas import Health
from app.storage import ModelStore
from app.training.jobs import JobManager, Trainer


def create_app(settings: Settings | None = None, trainer: Trainer | None = None, device: str | None = None) -> FastAPI:
    """Builds the app. Tests pass a fake trainer and device so they don't need a GPU or Ultralytics."""
    settings = settings or Settings()
    if trainer is None or device is None:
        from app.training import trainer as yolo_trainer

        trainer = trainer or yolo_trainer.make_trainer(settings.weights_dir)
        device = device or yolo_trainer.device()

    store = ModelStore(settings.models_dir)
    app = FastAPI(title="Schematic Symbol Trainer", version="0.1.0")
    app.state.services = Services(
        settings=settings,
        store=store,
        jobs=JobManager(trainer, store, settings.jobs_dir, settings.keep_job_files),
        detector=Detector(store),
        device=device,
    )

    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"],
                           allow_headers=["*"])

    api = APIRouter(prefix="/api")

    @api.get("/health", response_model=Health, tags=["health"])
    def health(svc: Annotated[Services, Depends(services)]) -> Health:
        return Health(device=svc.device)

    api.include_router(training.router)
    api.include_router(models.router)
    api.include_router(detections.router)
    app.include_router(api)
    return app


def __getattr__(name: str) -> FastAPI:
    # Built on first access (by uvicorn) rather than on import, so tests can import create_app without loading
    # PyTorch, creating ./data or starting a training worker.
    if name == "app":
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
