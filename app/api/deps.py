from dataclasses import dataclass

from fastapi import Request

from app.config import Settings
from app.inference.detector import Detector
from app.storage import ModelStore
from app.training.jobs import JobManager


@dataclass
class Services:
    settings: Settings
    store: ModelStore
    jobs: JobManager
    detector: Detector
    device: str


def services(request: Request) -> Services:
    return request.app.state.services
