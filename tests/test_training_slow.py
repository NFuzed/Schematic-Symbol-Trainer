"""Trains a real YOLO model for one epoch and runs detection with it. Run with: pytest -m slow"""

import io
import json

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

pytestmark = pytest.mark.slow


def test_real_training_and_detection(tmp_path, dataset_zip, diagrams):
    pytest.importorskip("ultralytics")
    app = create_app(Settings(data_dir=tmp_path))
    with TestClient(app) as client:
        config = json.dumps({"baseModel": "yolo11n", "epochs": 1, "imageSize": 640, "valSplit": 0.25})
        response = client.post("/api/training-jobs", files={"dataset": ("dataset.zip", dataset_zip)},
                               data={"config": config})
        assert response.status_code == 200, response.text

        with client.stream("GET", f"/api/training-jobs/{response.json()['jobId']}/events") as stream:
            events = [json.loads(line[6:]) for line in stream.iter_lines() if line.startswith("data: ")]
        assert events[-1]["status"] == "done", events[-1]
        epoch_events = [e for e in events if e.get("epoch") == 1 and e["status"] == "running"]
        assert {"box_loss", "mAP50"} <= set(epoch_events[0]["metrics"])

        buffer = io.BytesIO()
        diagrams[0][1].save(buffer, format="PNG")
        response = client.post("/api/detections", files={"image": ("d.png", buffer.getvalue(), "image/png")},
                               data={"model_id": events[-1]["modelId"], "min_confidence": "0.01"})
        assert response.status_code == 200, response.text
        entity_ids = set(client.get("/api/models").json()[0]["entityIds"])
        assert all(d["entityId"] in entity_ids for d in response.json()["detections"])
