"""Fine-tunes a pretrained Ultralytics YOLO11 model on a converted dataset."""

import logging
from pathlib import Path

from app.dataset.yolo import YoloDataset
from app.schemas import TrainConfig
from app.training.jobs import OnEpoch, TrainResult

log = logging.getLogger(__name__)

# Ultralytics metric keys -> the short names shown in the front end.
METRIC_NAMES = {
    "train/box_loss": "box_loss",
    "train/cls_loss": "cls_loss",
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50-95",
}


def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def base_weights(name: str, weights_dir: Path) -> Path:
    """Returns the pretrained weights for `name`, downloading them into weights_dir the first time."""
    from ultralytics.utils.downloads import attempt_download_asset

    weights_dir.mkdir(parents=True, exist_ok=True)
    return Path(attempt_download_asset(weights_dir / f"{name}.pt"))


# Ultralytics' default nominal batch: it accumulates gradients until this many images have been seen.
DEFAULT_NBS = 64
# A small dataset should still get several optimizer steps per epoch.
MIN_STEPS_PER_EPOCH = 4


def batch_plan(train_count: int, max_batch: int) -> tuple[int, int]:
    """Returns (batch, nbs) so that every epoch gets at least about MIN_STEPS_PER_EPOCH optimizer steps.

    Left alone, Ultralytics sizes the batch to GPU memory and then accumulates gradients over DEFAULT_NBS images
    before stepping. With 20 training diagrams that is one batch per epoch and one weight update every three
    epochs, so 100 epochs is only ~33 updates. Large datasets keep Ultralytics' defaults.
    """
    batch = max(1, min(max_batch, max(2, train_count // MIN_STEPS_PER_EPOCH)))
    nbs = min(DEFAULT_NBS, max(batch, train_count // MIN_STEPS_PER_EPOCH))
    return batch, nbs


def _metrics(trainer) -> dict[str, float]:
    raw = {**trainer.label_loss_items(trainer.tloss, prefix="train"), **(trainer.metrics or {})}
    return {short: round(float(raw[key]), 4) for key, short in METRIC_NAMES.items() if key in raw}


def make_trainer(weights_dir: Path):
    def train(dataset: YoloDataset, config: TrainConfig, run_dir: Path, on_epoch: OnEpoch) -> TrainResult:
        import ultralytics.utils
        from ultralytics import YOLO
        from ultralytics.models.yolo.detect import DetectionTrainer

        # The AMP check downloads its own small model into WEIGHTS_DIR, which defaults to ./weights in the working
        # directory. Keep it with the other base weights instead.
        ultralytics.utils.WEIGHTS_DIR = weights_dir

        class Trainer(DetectionTrainer):
            def auto_batch(self, *args, **kwargs):
                # Runs before the optimizer and gradient accumulation are set up, so nbs can still change here.
                batch, self.args.nbs = batch_plan(dataset.train_count, super().auto_batch(*args, **kwargs))
                log.info("Training with batch %d, nominal batch %d", batch, self.args.nbs)
                return batch

        model = YOLO(base_weights(config.base_model, weights_dir))
        model.add_callback("on_fit_epoch_end", lambda t: on_epoch(t.epoch + 1, _metrics(t)))

        on_gpu = device() == "cuda"
        # On GPU, Ultralytics measures how big a batch fits and Trainer.auto_batch caps it; on CPU, a small fixed one.
        batch, nbs = (-1, DEFAULT_NBS) if on_gpu else batch_plan(dataset.train_count, 4)
        model.train(
            trainer=Trainer,
            data=str(dataset.data_yaml),
            epochs=config.epochs,
            imgsz=config.image_size,
            batch=batch,
            nbs=nbs,
            device=0 if on_gpu else "cpu",
            # Dataloader worker processes need a __main__ guard on Windows, which a server thread can't provide.
            workers=0,
            project=str(run_dir.parent),
            name=run_dir.name,
            exist_ok=True,
            plots=False,
            verbose=False,
            # Schematics have a meaningful orientation for text, but symbols appear at any rotation and mirrored.
            fliplr=0.5,
            flipud=0.5,
            degrees=0.0,
        )

        trainer = model.trainer
        best = Path(trainer.best)
        if not best.is_file():
            best = Path(trainer.last)
        return TrainResult(weights=best, metrics=_metrics(trainer))

    return train
