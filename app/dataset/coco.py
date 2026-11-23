"""Reads and validates the dataset zip the front end uploads (see the README's "Dataset format")."""

import io
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

# Zip bomb guard: far more than any realistic set of diagrams.
MAX_UNCOMPRESSED_BYTES = 4 * 1024**3
# Image names come from the front end's diagram ids, so they are always plain file names.
SAFE_FILE_NAME = re.compile(r"^[\w.-]+\.png$")


class DatasetError(ValueError):
    """The upload isn't a usable dataset. The message is shown to the user."""


@dataclass(frozen=True)
class Category:
    coco_id: int
    name: str
    entity_id: str


@dataclass(frozen=True)
class Box:
    category_id: int
    x: float
    y: float
    w: float
    h: float


@dataclass
class DatasetImage:
    coco_id: int
    file_name: str
    width: int
    height: int
    boxes: list[Box]


@dataclass
class CocoDataset:
    categories: list[Category]
    images: list[DatasetImage]
    archive: zipfile.ZipFile

    def read_image(self, image: DatasetImage) -> bytes:
        return self.archive.read(f"images/{image.file_name}")

    @property
    def box_count(self) -> int:
        return sum(len(image.boxes) for image in self.images)


def read_dataset(data: bytes) -> CocoDataset:
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as err:
        raise DatasetError("The dataset isn't a valid zip file.") from err

    if sum(info.file_size for info in archive.infolist()) > MAX_UNCOMPRESSED_BYTES:
        raise DatasetError("The dataset is too large once unzipped.")

    try:
        coco = json.loads(archive.read("annotations.json"))
    except KeyError as err:
        raise DatasetError("annotations.json not found in the zip.") from err
    except (json.JSONDecodeError, UnicodeDecodeError) as err:
        raise DatasetError(f"annotations.json isn't valid JSON: {err}") from err

    try:
        dataset = _parse(coco, archive)
    except (KeyError, TypeError, ValueError) as err:
        if isinstance(err, DatasetError):
            raise
        raise DatasetError(f"annotations.json is missing a field or has the wrong type: {err}") from err

    if not dataset.categories:
        raise DatasetError("The dataset has no entities.")
    if dataset.box_count == 0:
        raise DatasetError("The dataset has no annotations.")
    return dataset


def _parse(coco: dict, archive: zipfile.ZipFile) -> CocoDataset:
    names = set(archive.namelist())

    categories = [
        Category(coco_id=int(c["id"]), name=str(c["name"]), entity_id=str(c.get("entity_id") or c["id"]))
        for c in coco["categories"]
    ]
    category_ids = {c.coco_id for c in categories}

    images: dict[int, DatasetImage] = {}
    for img in coco["images"]:
        file_name = str(img["file_name"])
        if not SAFE_FILE_NAME.match(file_name):
            raise DatasetError(f"Unexpected image name {file_name!r}; expected a .png file name.")
        if f"images/{file_name}" not in names:
            raise DatasetError(f"images/{file_name} is listed in annotations.json but missing from the zip.")
        images[int(img["id"])] = DatasetImage(
            coco_id=int(img["id"]),
            file_name=file_name,
            width=int(img["width"]),
            height=int(img["height"]),
            boxes=[],
        )

    for ann in coco["annotations"]:
        image = images.get(int(ann["image_id"]))
        category_id = int(ann["category_id"])
        if image is None or category_id not in category_ids:
            continue
        box = _clip(ann["bbox"], image.width, image.height)
        if box is not None:
            image.boxes.append(Box(category_id, *box))

    return CocoDataset(categories=categories, images=list(images.values()), archive=archive)


def _clip(bbox: list, width: int, height: int) -> tuple[float, float, float, float] | None:
    """Clips a [x, y, w, h] box to the image. Returns None if nothing is left."""
    x, y, w, h = (float(v) for v in bbox)
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(width), x + w), min(float(height), y + h)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return x1, y1, x2 - x1, y2 - y1


def check_image_size(data: bytes, image: DatasetImage) -> None:
    """Boxes are in pixels of the canonical raster, so the PNG must match the recorded size."""
    with Image.open(io.BytesIO(data)) as png:
        if png.size != (image.width, image.height):
            raise DatasetError(
                f"images/{image.file_name} is {png.size[0]}×{png.size[1]} px but annotations.json says "
                f"{image.width}×{image.height}."
            )


def extract_image(dataset: CocoDataset, image: DatasetImage, destination: Path) -> None:
    data = dataset.read_image(image)
    check_image_size(data, image)
    destination.write_bytes(data)
