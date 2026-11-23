import asyncio
import shutil
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from app.api.deps import Services, services
from app.dataset.coco import DatasetError, read_dataset
from app.dataset.yolo import write_yolo_dataset
from app.schemas import JobCreated, TrainConfig

router = APIRouter(tags=["training"])

POLL_SECONDS = 0.25
KEEP_ALIVE_SECONDS = 15


@router.post("/training-jobs", response_model=JobCreated)
def start_training(
    dataset: UploadFile,
    config: Annotated[str, Form()],
    svc: Annotated[Services, Depends(services)],
) -> JobCreated:
    try:
        train_config = TrainConfig.model_validate_json(config)
    except ValidationError as err:
        raise HTTPException(422, f"Invalid training config: {err.errors(include_url=False)}") from err

    # Convert the dataset before answering, so a broken upload is reported straight away instead of as a failed job.
    job_id, directory = svc.jobs.new_job_dir()
    try:
        coco = read_dataset(dataset.file.read())
        yolo = write_yolo_dataset(coco, directory / "dataset", train_config.val_split)
    except DatasetError as err:
        shutil.rmtree(directory, ignore_errors=True)
        raise HTTPException(400, str(err)) from err

    job = svc.jobs.submit(job_id, directory, yolo, [c.name for c in coco.categories], train_config)
    return JobCreated(job_id=job.id)


@router.get("/training-jobs/{job_id}/events")
async def training_events(job_id: str, svc: Annotated[Services, Depends(services)]) -> StreamingResponse:
    job = svc.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "No training job with that id. Jobs don't survive a back-end restart.")

    async def stream() -> AsyncIterator[str]:
        sent = 0
        idle = 0.0
        while True:
            events = job.events_since(sent)
            for event in events:
                yield f"data: {event.model_dump_json(by_alias=True, exclude_none=True)}\n\n"
            sent += len(events)
            if events and events[-1].status in ("done", "failed"):
                return
            if events:
                idle = 0.0
            elif idle >= KEEP_ALIVE_SECONDS:
                # A comment line keeps proxies from closing a quiet connection during a long epoch.
                yield ": keep-alive\n\n"
                idle = 0.0
            await asyncio.sleep(POLL_SECONDS)
            idle += POLL_SECONDS

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
