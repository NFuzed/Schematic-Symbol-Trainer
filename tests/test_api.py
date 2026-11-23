"""API tests against the real routes, with the YOLO trainer and detector replaced by fakes."""

import io
import json
import threading

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.schemas import Detection
from app.training.jobs import TrainResult

CONFIG = json.dumps({"baseModel": "yolo11n", "epochs": 3, "imageSize": 640, "valSplit": 0.25})


class FakeTrainer:
    def __init__(self):
        self.fail = False
        self.release = threading.Event()
        self.release.set()

    def __call__(self, dataset, config, run_dir, on_epoch):
        self.release.wait(5)
        if self.fail:
            raise RuntimeError("CUDA out of memory")
        for epoch in range(1, config.epochs + 1):
            on_epoch(epoch, {"mAP50": epoch / 10})
        run_dir.mkdir(parents=True)
        weights = run_dir / "best.pt"
        weights.write_bytes(b"weights")
        return TrainResult(weights=weights, metrics={"mAP50": 0.3})


class FakeDetector:
    def detect(self, record, image, min_confidence):
        self.seen = (record.id, image.size, min_confidence)
        return [Detection(entity_id=record.entity_ids[0], bbox=(1, 2, 3, 4), confidence=0.9)]


@pytest.fixture
def trainer():
    return FakeTrainer()


@pytest.fixture
def client(tmp_path, trainer):
    app = create_app(Settings(data_dir=tmp_path), trainer=trainer, device="cpu")
    app.state.services.detector = FakeDetector()
    with TestClient(app) as test_client:
        yield test_client


def start_job(client, dataset_zip, config=CONFIG):
    return client.post("/api/training-jobs", files={"dataset": ("dataset.zip", dataset_zip, "application/zip")},
                       data={"config": config})


def read_events(client, job_id):
    with client.stream("GET", f"/api/training-jobs/{job_id}/events") as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        return [json.loads(line[len("data: "):]) for line in response.iter_lines() if line.startswith("data: ")]


def png(diagrams, index=0) -> bytes:
    buffer = io.BytesIO()
    diagrams[index][1].save(buffer, format="PNG")
    return buffer.getvalue()


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok", "device": "cpu"}


def test_no_models_at_first(client):
    assert client.get("/api/models").json() == []


def test_training_job_streams_progress_then_lists_the_model(client, dataset_zip):
    response = start_job(client, dataset_zip)
    assert response.status_code == 200
    job_id = response.json()["jobId"]

    events = read_events(client, job_id)
    assert [e["status"] for e in events] == ["queued", "running", "running", "running", "running", "done"]
    assert [e.get("epoch") for e in events[2:5]] == [1, 2, 3]
    assert all(e["epochs"] == 3 for e in events)
    done = events[-1]
    assert done["metrics"] == {"mAP50": 0.3}

    models = client.get("/api/models").json()
    assert [m["id"] for m in models] == [done["modelId"]]
    assert set(models[0]) == {"id", "createdAt", "entityIds", "metrics"}
    assert len(models[0]["entityIds"]) == 4


def test_late_subscriber_gets_the_whole_history(client, dataset_zip):
    job_id = start_job(client, dataset_zip).json()["jobId"]
    first = read_events(client, job_id)
    assert read_events(client, job_id) == first


def test_events_arrive_while_job_is_still_queued(client, dataset_zip, trainer):
    trainer.release.clear()
    job_id = start_job(client, dataset_zip).json()["jobId"]
    threading.Timer(0.5, trainer.release.set).start()
    events = read_events(client, job_id)
    assert events[0] == {"status": "queued", "epochs": 3}
    assert events[-1]["status"] == "done"


def test_failed_job_reports_the_error(client, dataset_zip, trainer):
    trainer.fail = True
    job_id = start_job(client, dataset_zip).json()["jobId"]
    events = read_events(client, job_id)
    assert events[-1] == {"status": "failed", "message": "Training failed: CUDA out of memory"}
    assert client.get("/api/models").json() == []


def test_unknown_job_is_404(client):
    assert client.get("/api/training-jobs/nope/events").status_code == 404


@pytest.mark.parametrize("config", ['{"baseModel": "yolo99"}', '{"epochs": 0}', '{"imageSize": 1000}', "not json"])
def test_bad_config_is_422(client, dataset_zip, config):
    response = start_job(client, dataset_zip, config)
    assert response.status_code == 422
    assert "Invalid training config" in response.json()["detail"]


def test_bad_dataset_is_400_and_leaves_nothing_behind(client, tmp_path):
    response = start_job(client, b"not a zip")
    assert response.status_code == 400
    assert response.json()["detail"] == "The dataset isn't a valid zip file."
    assert list((tmp_path / "jobs").iterdir()) == []


def test_detection_maps_to_entity_ids(client, dataset_zip, diagrams):
    events = read_events(client, start_job(client, dataset_zip).json()["jobId"])
    model_id = events[-1]["modelId"]
    entity_ids = client.get("/api/models").json()[0]["entityIds"]

    response = client.post("/api/detections", files={"image": ("diagram.png", png(diagrams), "image/png")},
                           data={"model_id": model_id, "min_confidence": "0.4"})
    assert response.status_code == 200
    assert response.json() == {"detections": [{"entityId": entity_ids[0], "bbox": [1, 2, 3, 4], "confidence": 0.9}]}
    assert client.app.state.services.detector.seen == (model_id, diagrams[0][1].size, 0.4)


def test_detection_with_unknown_model_is_404(client, diagrams):
    response = client.post("/api/detections", files={"image": ("d.png", png(diagrams), "image/png")},
                           data={"model_id": "abc123", "min_confidence": "0.5"})
    assert response.status_code == 404


def test_detection_rejects_path_like_model_id(client, diagrams):
    response = client.post("/api/detections", files={"image": ("d.png", png(diagrams), "image/png")},
                           data={"model_id": "../models", "min_confidence": "0.5"})
    assert response.status_code == 404


def test_detection_rejects_non_image(client, dataset_zip):
    model_id = read_events(client, start_job(client, dataset_zip).json()["jobId"])[-1]["modelId"]
    response = client.post("/api/detections", files={"image": ("d.png", b"nope", "image/png")},
                           data={"model_id": model_id, "min_confidence": "0.5"})
    assert response.status_code == 400


def test_detection_rejects_out_of_range_confidence(client, diagrams):
    response = client.post("/api/detections", files={"image": ("d.png", png(diagrams), "image/png")},
                           data={"model_id": "abc", "min_confidence": "1.5"})
    assert response.status_code == 422
