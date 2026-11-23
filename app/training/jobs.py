"""Background training jobs and their progress events.

Jobs run one at a time on a single worker thread, since they share one GPU. Each job keeps every event it has
published, so a client that connects late (or reconnects) gets the whole history before any new events.
"""

import logging
import queue
import shutil
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from app.dataset.yolo import YoloDataset
from app.schemas import TrainConfig, TrainingEvent
from app.storage import ModelStore

log = logging.getLogger(__name__)

OnEpoch = Callable[[int, dict[str, float]], None]


@dataclass(frozen=True)
class TrainResult:
    weights: Path
    metrics: dict[str, float]


# (dataset, config, run directory, per-epoch callback) -> best weights and final metrics
Trainer = Callable[[YoloDataset, TrainConfig, Path, OnEpoch], TrainResult]


@dataclass
class Job:
    id: str
    config: TrainConfig
    dataset: YoloDataset
    entity_names: list[str]
    directory: Path
    events: list[TrainingEvent] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def publish(self, event: TrainingEvent) -> None:
        with self._lock:
            self.events.append(event)

    def events_since(self, index: int) -> list[TrainingEvent]:
        with self._lock:
            return self.events[index:]

    @property
    def finished(self) -> bool:
        with self._lock:
            return bool(self.events) and self.events[-1].status in ("done", "failed")


class JobManager:
    def __init__(self, trainer: Trainer, store: ModelStore, jobs_dir: Path, keep_job_files: bool = False):
        self._trainer = trainer
        self._store = store
        self.jobs_dir = jobs_dir
        self._keep_job_files = keep_job_files
        self._jobs: dict[str, Job] = {}
        self._queue: queue.Queue[Job] = queue.Queue()
        threading.Thread(target=self._work, name="training-worker", daemon=True).start()

    def new_job_dir(self) -> tuple[str, Path]:
        job_id = uuid.uuid4().hex
        directory = self.jobs_dir / job_id
        directory.mkdir(parents=True)
        return job_id, directory

    def submit(self, job_id: str, directory: Path, dataset: YoloDataset, entity_names: list[str],
               config: TrainConfig) -> Job:
        job = Job(id=job_id, config=config, dataset=dataset, entity_names=entity_names, directory=directory)
        job.publish(TrainingEvent(status="queued", epochs=config.epochs))
        self._jobs[job.id] = job
        self._queue.put(job)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def _work(self) -> None:
        while True:
            self._run(self._queue.get())

    def _run(self, job: Job) -> None:
        epochs = job.config.epochs
        job.publish(TrainingEvent(
            status="running", epoch=0, epochs=epochs,
            message=f"Training on {job.dataset.train_count} diagrams, validating on {job.dataset.val_count}.",
        ))

        def on_epoch(epoch: int, metrics: dict[str, float]) -> None:
            job.publish(TrainingEvent(status="running", epoch=epoch, epochs=epochs, metrics=metrics))

        try:
            result = self._trainer(job.dataset, job.config, job.directory / "run", on_epoch)
            record = self._store.save(result.weights, job.config, job.dataset.entity_ids, job.entity_names,
                                      result.metrics)
        except Exception as err:  # noqa: BLE001 - any failure must reach the user as a "failed" event
            log.exception("Training job %s failed", job.id)
            job.publish(TrainingEvent(status="failed", message=f"Training failed: {err}"))
            return

        # The model is saved (and listed) before "done" goes out, so the front end's refresh finds it.
        job.publish(TrainingEvent(status="done", epoch=epochs, epochs=epochs, metrics=result.metrics,
                                  model_id=record.id))
        if not self._keep_job_files:
            shutil.rmtree(job.directory, ignore_errors=True)
