"""Runs a stored model on a diagram and maps its classes back to the front end's entity ids."""

import threading
from collections import OrderedDict

from PIL import Image

from app.schemas import Detection
from app.storage import ModelRecord, ModelStore

# Loaded models kept in memory; each is tens of MB on the GPU.
CACHE_SIZE = 3


class Detector:
    def __init__(self, store: ModelStore):
        self._store = store
        self._models: OrderedDict[str, object] = OrderedDict()
        # Ultralytics predictors aren't thread-safe, and FastAPI runs sync endpoints on a thread pool.
        self._lock = threading.Lock()

    def detect(self, record: ModelRecord, image: Image.Image, min_confidence: float) -> list[Detection]:
        with self._lock:
            model = self._load(record)
            results = model.predict(image, imgsz=record.image_size, conf=min_confidence, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            for (x1, y1, x2, y2), cls, conf in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist()):
                detections.append(Detection(
                    entity_id=record.entity_ids[int(cls)],
                    bbox=(round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)),
                    confidence=round(conf, 4),
                ))
        return sorted(detections, key=lambda d: d.confidence, reverse=True)

    def _load(self, record: ModelRecord):
        from ultralytics import YOLO

        if record.id in self._models:
            self._models.move_to_end(record.id)
        else:
            self._models[record.id] = YOLO(self._store.weights_path(record.id))
            if len(self._models) > CACHE_SIZE:
                self._models.popitem(last=False)
        return self._models[record.id]
