"""Request and response models. These mirror the front end's src/lib/api.ts and src/types.ts."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

BaseModelName = Literal["yolo11n", "yolo11s", "yolo11m"]
JobStatus = Literal["queued", "running", "done", "failed"]
# [x, y, width, height] in pixels of the diagram's canonical raster, top-left origin.
BBox = tuple[float, float, float, float]


class CamelModel(BaseModel):
    """Python attributes in snake_case, JSON in camelCase to match the front end."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class TrainConfig(CamelModel):
    base_model: BaseModelName = "yolo11s"
    epochs: int = Field(100, ge=1, le=1000)
    image_size: int = Field(1280, ge=320, le=2048, multiple_of=32)
    val_split: float = Field(0.2, ge=0.05, le=0.5)


class JobCreated(CamelModel):
    job_id: str


class TrainingEvent(CamelModel):
    status: JobStatus
    epoch: int | None = None
    epochs: int | None = None
    metrics: dict[str, float] | None = None
    message: str | None = None
    model_id: str | None = None


class ModelInfo(CamelModel):
    id: str
    created_at: str
    entity_ids: list[str]
    metrics: dict[str, float] | None = None


class Detection(CamelModel):
    entity_id: str
    bbox: BBox
    confidence: float


class DetectionsResponse(CamelModel):
    detections: list[Detection]


class Health(CamelModel):
    status: Literal["ok"] = "ok"
    device: str
