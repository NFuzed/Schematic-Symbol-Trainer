"""Trained models on disk: data/models/{modelId}/weights.pt plus model.json."""

import json
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.schemas import ModelInfo, TrainConfig


@dataclass
class ModelRecord:
    id: str
    created_at: str
    base_model: str
    image_size: int
    epochs: int
    # Index i is YOLO class i.
    entity_ids: list[str]
    entity_names: list[str]
    metrics: dict[str, float]

    def info(self) -> ModelInfo:
        return ModelInfo(id=self.id, created_at=self.created_at, entity_ids=self.entity_ids, metrics=self.metrics)


class ModelStore:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def weights_path(self, model_id: str) -> Path:
        return self.root / model_id / "weights.pt"

    def save(
        self,
        weights: Path,
        config: TrainConfig,
        entity_ids: list[str],
        entity_names: list[str],
        metrics: dict[str, float],
    ) -> ModelRecord:
        record = ModelRecord(
            id=uuid.uuid4().hex,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            base_model=config.base_model,
            image_size=config.image_size,
            epochs=config.epochs,
            entity_ids=entity_ids,
            entity_names=entity_names,
            metrics=metrics,
        )
        # Write into a temporary folder and rename, so a half-written model is never listed.
        staging = self.root / f".{record.id}.tmp"
        staging.mkdir(parents=True)
        shutil.copyfile(weights, staging / "weights.pt")
        (staging / "model.json").write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
        staging.rename(self.root / record.id)
        return record

    def get(self, model_id: str) -> ModelRecord | None:
        # Ids are uuid hex strings; anything else can't name a model folder.
        if not model_id.isalnum():
            return None
        path = self.root / model_id / "model.json"
        if not path.is_file():
            return None
        return ModelRecord(**json.loads(path.read_text(encoding="utf-8")))

    def list(self) -> list[ModelRecord]:
        records = [
            ModelRecord(**json.loads(path.read_text(encoding="utf-8")))
            for path in self.root.glob("*/model.json")
            if not path.parent.name.startswith(".")
        ]
        return sorted(records, key=lambda r: r.created_at, reverse=True)
