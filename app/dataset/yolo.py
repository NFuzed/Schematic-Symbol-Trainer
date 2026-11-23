"""Converts a COCO dataset into the directory layout Ultralytics trains from."""

import random
from dataclasses import dataclass
from pathlib import Path

from app.dataset.coco import Box, CocoDataset, DatasetImage, extract_image


@dataclass(frozen=True)
class YoloDataset:
    data_yaml: Path
    # Index i is YOLO class i; the value is the front end's entity id.
    entity_ids: list[str]
    train_count: int
    val_count: int


def to_yolo_line(box: Box, class_index: int, width: int, height: int) -> str:
    cx = (box.x + box.w / 2) / width
    cy = (box.y + box.h / 2) / height
    return f"{class_index} {cx:.6f} {cy:.6f} {box.w / width:.6f} {box.h / height:.6f}"


def split(images: list[DatasetImage], val_split: float, seed: int = 0) -> tuple[list[DatasetImage], list[DatasetImage]]:
    """Splits diagrams into training and validation sets.

    Datasets here are often tiny, so there's always at least one training and one validation diagram. With a single
    diagram, it's used for both: the validation metrics are then optimistic, but training can still run.
    """
    if len(images) == 1:
        return images, images
    shuffled = images[:]
    random.Random(seed).shuffle(shuffled)
    val_count = min(len(images) - 1, max(1, round(len(images) * val_split)))
    return shuffled[val_count:], shuffled[:val_count]


def write_yolo_dataset(dataset: CocoDataset, root: Path, val_split: float) -> YoloDataset:
    class_index = {category.coco_id: i for i, category in enumerate(dataset.categories)}
    train, val = split(dataset.images, val_split)

    for subset, images in (("train", train), ("val", val)):
        image_dir = root / "images" / subset
        label_dir = root / "labels" / subset
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for image in images:
            extract_image(dataset, image, image_dir / image.file_name)
            lines = [to_yolo_line(box, class_index[box.category_id], image.width, image.height) for box in image.boxes]
            (label_dir / f"{Path(image.file_name).stem}.txt").write_text("\n".join(lines), encoding="utf-8")

    names = "\n".join(f"  {i}: {_yaml_string(c.name)}" for i, c in enumerate(dataset.categories))
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        f"path: {_yaml_string(root.resolve().as_posix())}\ntrain: images/train\nval: images/val\nnames:\n{names}\n",
        encoding="utf-8",
    )
    return YoloDataset(
        data_yaml=data_yaml,
        entity_ids=[c.entity_id for c in dataset.categories],
        train_count=len(train),
        val_count=len(val),
    )


def _yaml_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"
