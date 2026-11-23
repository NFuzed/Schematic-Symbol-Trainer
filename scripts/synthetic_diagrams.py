"""Draws synthetic circuit schematics with known symbol positions, for tests and end-to-end checks.

Each diagram is a grid of wires; some wire segments carry a resistor, capacitor, LDR or diode, drawn horizontally or
vertically with random size, line width and jitter, plus component labels as clutter.

    python scripts/synthetic_diagrams.py out/ --count 12 --seed 1

writes out/diagram-NN.png and out/dataset.zip, the latter in the front end's export format (COCO + PNGs).
"""

import argparse
import hashlib
import io
import json
import math
import random
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

ENTITIES = {
    "resistor": "#e4572e",
    "capacitor": "#378add",
    "ldr": "#29a35a",
    "diode": "#9b59b6",
}
LABEL_PREFIX = {"resistor": "R", "capacitor": "C", "ldr": "LDR", "diode": "D"}


@dataclass
class Symbol:
    entity: str
    bbox: tuple[float, float, float, float]  # x, y, w, h


class Pen:
    """Draws in symbol-local coordinates (x along the wire, y across it), centred on (cx, cy)."""

    def __init__(self, draw: ImageDraw.ImageDraw, cx: float, cy: float, vertical: bool, width: int):
        self.draw, self.cx, self.cy, self.vertical, self.width = draw, cx, cy, vertical, width
        self.points: list[tuple[float, float]] = []

    def _map(self, x: float, y: float) -> tuple[float, float]:
        point = (self.cx + y, self.cy + x) if self.vertical else (self.cx + x, self.cy + y)
        self.points.append(point)
        return point

    def line(self, *points: tuple[float, float]) -> None:
        self.draw.line([self._map(x, y) for x, y in points], fill="black", width=self.width, joint="curve")

    def polygon(self, *points: tuple[float, float], filled: bool = False) -> None:
        mapped = [self._map(x, y) for x, y in points]
        self.draw.polygon(mapped, outline="black", fill="black" if filled else None, width=self.width)

    def bbox(self, pad: float) -> tuple[float, float, float, float]:
        xs, ys = [p[0] for p in self.points], [p[1] for p in self.points]
        x0, y0 = min(xs) - pad, min(ys) - pad
        return x0, y0, max(xs) + pad - x0, max(ys) + pad - y0


def _zigzag(pen: Pen, half: float, height: float) -> None:
    peaks = 6
    step = 2 * half / peaks
    points = [(-half, 0.0)]
    for i in range(peaks):
        points.append((-half + step * (i + 0.5), height if i % 2 == 0 else -height))
    points.append((half, 0.0))
    pen.line(*points)


def _arrow(pen: Pen, start: tuple[float, float], end: tuple[float, float], head: float) -> None:
    pen.line(start, end)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    for side in (-1, 1):
        a = angle + math.pi - side * 0.45
        pen.line(end, (end[0] + head * math.cos(a), end[1] + head * math.sin(a)))


def draw_symbol(pen: Pen, entity: str, size: float) -> None:
    """Draws the symbol body between x = -size and x = +size."""
    if entity == "resistor":
        _zigzag(pen, size, size * 0.35)
    elif entity == "capacitor":
        gap, plate = size * 0.18, size * 0.6
        pen.line((-size, 0), (-gap, 0))
        pen.line((gap, 0), (size, 0))
        pen.line((-gap, -plate), (-gap, plate))
        pen.line((gap, -plate), (gap, plate))
    elif entity == "ldr":
        _zigzag(pen, size, size * 0.35)
        # Two arrows pointing in towards the body: what distinguishes an LDR from a plain resistor.
        for offset in (-0.35, 0.25):
            _arrow(pen, (size * (offset - 0.45), -size * 1.25), (size * offset, -size * 0.55), size * 0.22)
    elif entity == "diode":
        tri = size * 0.5
        pen.line((-size, 0), (-tri, 0))
        pen.line((tri, 0), (size, 0))
        pen.polygon((-tri, -tri), (-tri, tri), (tri, 0), filled=random.random() < 0.5)
        pen.line((tri, -tri), (tri, tri))
    else:
        raise ValueError(entity)


def draw_diagram(rng: random.Random, width: int = 1600, height: int = 1100) -> tuple[Image.Image, list[Symbol]]:
    image = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(image)
    cols, rows = rng.randint(4, 6), rng.randint(3, 5)
    margin = 110
    xs = [margin + i * (width - 2 * margin) / (cols - 1) for i in range(cols)]
    ys = [margin + j * (height - 2 * margin) / (rows - 1) for j in range(rows)]
    line_width = rng.choice([2, 2, 3, 4])
    symbols: list[Symbol] = []
    counters = dict.fromkeys(ENTITIES, 0)

    edges = [((i, j), (i + 1, j)) for i in range(cols - 1) for j in range(rows)]
    edges += [((i, j), (i, j + 1)) for i in range(cols) for j in range(rows - 1)]
    for (i0, j0), (i1, j1) in edges:
        if rng.random() < 0.3:
            continue
        x0, y0, x1, y1 = xs[i0], ys[j0], xs[i1], ys[j1]
        vertical = i0 == i1
        if rng.random() > 0.55:
            draw.line([(x0, y0), (x1, y1)], fill=0, width=line_width)
            continue

        entity = rng.choice(list(ENTITIES))
        length = (y1 - y0) if vertical else (x1 - x0)
        size = min(length * 0.3, rng.uniform(28, 55))
        cx = (x0 + x1) / 2 + (0 if vertical else rng.uniform(-0.1, 0.1) * length)
        cy = (y0 + y1) / 2 + (rng.uniform(-0.1, 0.1) * length if vertical else 0)
        pen = Pen(draw, cx, cy, vertical, line_width)
        draw_symbol(pen, entity, size)
        bbox = pen.bbox(pad=line_width + 2)
        # Leads from the wire ends to the symbol body.
        if vertical:
            draw.line([(x0, y0), (cx, cy - size)], fill=0, width=line_width)
            draw.line([(cx, cy + size), (x1, y1)], fill=0, width=line_width)
        else:
            draw.line([(x0, y0), (cx - size, cy)], fill=0, width=line_width)
            draw.line([(cx + size, cy), (x1, y1)], fill=0, width=line_width)
        symbols.append(Symbol(entity, bbox))

        counters[entity] += 1
        label = f"{LABEL_PREFIX[entity]}{counters[entity]}"
        bx, by, bw, bh = bbox
        draw.text((bx + bw + 6, by + bh / 2 - 6) if vertical else (bx + bw / 2 - 8, by + bh + 4), label, fill=0)

    for x in xs:
        for y in ys:
            if rng.random() < 0.3:
                r = line_width + 3
                draw.ellipse([x - r, y - r, x + r, y + r], fill=0)

    return image, symbols


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def dataset_zip(diagrams: list[tuple[str, Image.Image, list[Symbol]]]) -> bytes:
    """Builds a dataset zip in the front end's export format (see its src/lib/dataset.ts)."""
    entity_ids = {name: str(uuid.uuid4()) for name in ENTITIES}
    categories = [
        {"id": i + 1, "name": name, "entity_id": entity_ids[name], "colour": colour}
        for i, (name, colour) in enumerate(ENTITIES.items())
    ]
    category_for = {name: i + 1 for i, name in enumerate(ENTITIES)}
    images, annotations, files = [], [], {}
    for image_id, (name, image, symbols) in enumerate(diagrams, start=1):
        diagram_id = str(uuid.uuid4())
        data = png_bytes(image)
        files[f"images/{diagram_id}.png"] = data
        images.append({
            "id": image_id, "file_name": f"{diagram_id}.png", "width": image.width, "height": image.height,
            "diagram_id": diagram_id, "original_name": f"{name}.png", "sha256": hashlib.sha256(data).hexdigest(),
        })
        for symbol in symbols:
            annotations.append({
                "id": len(annotations) + 1, "image_id": image_id, "category_id": category_for[symbol.entity],
                "bbox": [round(v, 1) for v in symbol.bbox], "area": symbol.bbox[2] * symbol.bbox[3], "iscrowd": 0,
                "annotation_id": str(uuid.uuid4()), "source": "manual",
            })
    coco = {
        "info": {"description": "Synthetic schematics", "version": "1", "date_created": "2026-01-01T00:00:00Z"},
        "images": images, "categories": categories, "annotations": annotations,
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("annotations.json", json.dumps(coco, indent=2))
        for path, data in files.items():
            archive.writestr(path, data)
    return buffer.getvalue()


def generate(count: int, seed: int) -> list[tuple[str, Image.Image, list[Symbol]]]:
    rng = random.Random(seed)
    random.seed(seed)
    return [(f"diagram-{i + 1:02d}", *draw_diagram(rng)) for i in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    diagrams = generate(args.count, args.seed)
    for name, image, _ in diagrams:
        image.save(args.out / f"{name}.png")
    (args.out / "dataset.zip").write_bytes(dataset_zip(diagrams))
    total = sum(len(symbols) for _, _, symbols in diagrams)
    print(f"Wrote {len(diagrams)} diagrams with {total} symbols to {args.out}")


if __name__ == "__main__":
    main()
