import io
import json
import zipfile
import time

import pytest
import yaml

from app.dataset.coco import Box, DatasetError, read_dataset
from app.dataset.yolo import split, to_yolo_line, write_yolo_dataset


def make_zip(files: dict[str, bytes | str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def rewrite_annotations(dataset_zip: bytes, change) -> bytes:
    with zipfile.ZipFile(io.BytesIO(dataset_zip)) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    coco = json.loads(files["annotations.json"])
    change(coco)
    files["annotations.json"] = json.dumps(coco)
    return make_zip(files)


def test_reads_front_end_export(dataset_zip, diagrams):
    start_time = time.time()
    dataset = read_dataset(dataset_zip)
    assert [c.name for c in dataset.categories] == ["resistor", "capacitor", "ldr", "diode"]
    assert len(dataset.images) == len(diagrams)
    assert dataset.box_count == sum(len(symbols) for _, _, symbols in diagrams)
    print("--- %s seconds ---" % (time.time() - start_time))


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"not a zip", "valid zip"),
        (make_zip({"images/a.png": b""}), "annotations.json not found"),
        (make_zip({"annotations.json": "{"}), "isn't valid JSON"),
        (make_zip({"annotations.json": "{}"}), "missing a field"),
    ],
)
def test_rejects_broken_uploads(data, message):
    with pytest.raises(DatasetError, match=message):
        read_dataset(data)


def test_rejects_paths_outside_images(dataset_zip):
    def escape(coco):
        coco["images"][0]["file_name"] = "../../evil.png"

    with pytest.raises(DatasetError, match="Unexpected image name"):
        read_dataset(rewrite_annotations(dataset_zip, escape))


def test_rejects_missing_image(dataset_zip):
    def rename(coco):
        coco["images"][0]["file_name"] = "missing.png"

    with pytest.raises(DatasetError, match="missing from the zip"):
        read_dataset(rewrite_annotations(dataset_zip, rename))


def test_rejects_dataset_without_annotations(dataset_zip):
    with pytest.raises(DatasetError, match="no annotations"):
        read_dataset(rewrite_annotations(dataset_zip, lambda coco: coco.update(annotations=[])))


def test_clips_boxes_to_the_image_and_drops_empty_ones(dataset_zip):
    def move(coco):
        width = coco["images"][0]["width"]
        coco["annotations"] = [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [width - 10, -5, 50, 20]},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [width + 10, 0, 50, 20]},
        ]

    dataset = read_dataset(rewrite_annotations(dataset_zip, move))
    image = dataset.images[0]
    assert image.boxes == [Box(category_id=1, x=image.width - 10, y=0, w=10, h=15)]


def test_yolo_line_is_normalised_centre_and_size():
    assert to_yolo_line(Box(1, x=100, y=50, w=200, h=100), 3, width=1000, height=500) == (
        "3 0.200000 0.200000 0.200000 0.200000"
    )


@pytest.mark.parametrize(("count", "val_split", "expected"), [(1, 0.2, (1, 1)), (2, 0.2, (1, 1)), (10, 0.2, (8, 2)),
                                                               (10, 0.5, (5, 5))])
def test_split_keeps_at_least_one_diagram_on_each_side(count, val_split, expected):
    train, val = split(list(range(count)), val_split)
    assert (len(train), len(val)) == expected
    if count > 1:
        assert not set(train) & set(val)


def test_writes_ultralytics_layout(dataset_zip, tmp_path):
    dataset = read_dataset(dataset_zip)
    yolo = write_yolo_dataset(dataset, tmp_path, val_split=0.25)

    config = yaml.safe_load(yolo.data_yaml.read_text())
    assert config["names"] == {0: "resistor", 1: "capacitor", 2: "ldr", 3: "diode"}
    assert yolo.entity_ids == [c.entity_id for c in dataset.categories]
    assert (yolo.train_count, yolo.val_count) == (3, 1)

    labels = list(tmp_path.glob("labels/*/*.txt"))
    assert len(labels) == 4
    lines = [line for label in labels for line in label.read_text().splitlines()]
    assert len(lines) == dataset.box_count
    for line in lines:
        cls, *coords = line.split()
        assert 0 <= int(cls) < 4
        assert all(0 <= float(v) <= 1 for v in coords)


def test_rejects_image_whose_size_disagrees_with_annotations(dataset_zip, tmp_path):
    def resize(coco):
        coco["images"][0]["width"] += 1

    dataset = read_dataset(rewrite_annotations(dataset_zip, resize))
    with pytest.raises(DatasetError, match="px but annotations.json says"):
        write_yolo_dataset(dataset, tmp_path, val_split=0.2)
