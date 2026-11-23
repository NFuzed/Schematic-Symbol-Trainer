"""Settings read from the environment, with defaults suited to local development."""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _origins() -> list[str]:
    raw = os.environ.get("SST_CORS_ORIGINS", "")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("SST_DATA_DIR", "data")).resolve())
    # Only needed when the front end isn't served through Vite's /api proxy.
    cors_origins: list[str] = field(default_factory=_origins)
    # Keep each job's unpacked dataset and training run after it succeeds (failed jobs are always kept).
    keep_job_files: bool = os.environ.get("SST_KEEP_JOB_FILES", "") == "1"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def weights_dir(self) -> Path:
        """Pretrained base weights (yolo11n.pt etc.), downloaded on first use."""
        return self.data_dir / "weights"
