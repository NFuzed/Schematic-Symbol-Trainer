# Schematic Symbol Trainer (Python back end)

The back end for [Schematic-Symbol-Trainer-TypeScript](https://github.com/NFuzed/Schematic-Symbol-Trainer-TypeScript),
a tool for training and running a symbol recogniser on old blueprints, scanned schematics and hand-drawn diagrams.

The front end handles everything a person touches: loading diagrams, defining entities (symbol types such as
`resistor` or `ldr`) and drawing boxes around examples. This repo handles everything that has to run on a server:
turning those labelled examples into an object-detection model, storing the trained models, and running them on new
diagrams to find the same symbols.

> **Status:** the API contract below is implemented and tested end to end with the front end: import a dataset,
> train, watch progress live, detect on unseen diagrams and review the results. Until the back end answers on
> `http://localhost:8000/api/health`, the front end shows **Local only** and its Train and Detect tabs stay disabled.

## What it does

```
            front end                                   back end (this repo)
 ┌───────────────────────────┐   dataset.zip + config   ┌──────────────────────────────────────┐
 │ diagrams, entities, boxes │ ───────────────────────▶ │ 1. unpack COCO dataset               │
 │ (IndexedDB)               │                          │ 2. convert to YOLO labels, split     │
 │                           │ ◀─────────────────────── │ 3. train in a background job         │
 │ training progress         │   progress events (SSE)  │ 4. save weights + class mapping      │
 │                           │                          │                                      │
 │ detect tab                │   image + model id       │ 5. run model on the image            │
 │                           │ ───────────────────────▶ │ 6. map class index back to entity id │
 │ accept / reject review    │ ◀─────────────────────── │                                      │
 └───────────────────────────┘   detections             └──────────────────────────────────────┘
```

1. **Receive a dataset.** The front end uploads a zip of whole diagrams plus a COCO `annotations.json`. It never crops
   symbols, so the model sees the wires, text and blank paper around each one and learns what *isn't* a symbol.
2. **Prepare it.** Convert each `[x, y, w, h]` pixel box to YOLO's normalised `class cx cy w h`, write the label files
   and split the diagrams into training and validation sets.
3. **Train.** Fine-tune a YOLO11 model in a background job and stream progress back to the front end.
4. **Store the model.** Save the weights along with the mapping from YOLO class index to the front end's `entity_id`.
5. **Detect.** Run a stored model on a new diagram and return boxes in the same pixel coordinates the front end uses.

## Getting started

Needs Python 3.11+. A CUDA GPU makes training much faster, but CPU works for small datasets.

```bash
python -m venv .venv
```

On Windows, activate it with `.venv\Scripts\activate`; elsewhere, `source .venv/bin/activate`. With an NVIDIA GPU,
install the CUDA build of PyTorch first (pick the CUDA version for your driver at https://pytorch.org):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

```bash
pip install -r requirements.txt
```

```bash
uvicorn app.main:app --port 8000
```

Run it from the repo root, since `data/` is created relative to the working directory. `python -m app --port 8000`
does the same and also works under PyCharm's debugger, whose patched `asyncio.run` rejects the arguments uvicorn's own
launcher passes.

The first training run downloads the pretrained base weights (a few MB to ~40 MB each) into `data/weights/`.

Then start the front end (`npm run dev` in the TypeScript repo) and open http://localhost:5173. Vite proxies `/api/*` to
`http://localhost:8000`, so the browser sees one origin and CORS isn't needed during development. The header should
switch to **Back end online** within 15 seconds.

The interactive API docs are at http://localhost:8000/docs, and the OpenAPI schema is at
http://localhost:8000/openapi.json. The front end's types should eventually be generated from that schema with
`openapi-typescript`, so a change to a Python model shows up as a TypeScript compile error.

### Settings

| Environment variable  | Default | Meaning                                                                                  |
|-----------------------|---------|------------------------------------------------------------------------------------------|
| `SST_DATA_DIR`        | `data`  | Where models, base weights and job working folders are stored                            |
| `SST_CORS_ORIGINS`    | none    | Comma-separated origins to allow, if the front end isn't served through Vite's proxy     |
| `SST_KEEP_JOB_FILES`  | unset   | Set to `1` to keep a job's unpacked dataset and training run after it succeeds           |

### Tests

```bash
pytest
```

runs the fast suite (dataset parsing, YOLO conversion and every endpoint, with the trainer and detector faked). The
slow test trains a real YOLO11n model for one epoch and runs detection with it:

```bash
pytest -m slow
```

### Test data

`scripts/synthetic_diagrams.py` draws circuit schematics with resistors, capacitors, LDRs and diodes at known positions,
and writes them as PNGs plus a `dataset.zip` in the front end's export format. Import the zip on the front end's
Dataset tab to get a labelled dataset without drawing boxes by hand, and load the PNGs from a second run (with a
different `--seed`) as unseen diagrams for the Detect tab.

```bash
python scripts/synthetic_diagrams.py out/train --count 16 --seed 1
```

As a reference point: 16 synthetic diagrams (193 boxes), trained with the front end's defaults (YOLO11s, 100 epochs,
1280 px) on an RTX 3060 Ti in about 3 minutes, found 34 of 36 symbols on 3 unseen diagrams with no false positives or
LDR/resistor mix-ups. Real scans and hand drawings are harder, so expect to need more examples.

## API contract

These are the endpoints the front end calls today, in its
[`src/lib/api.ts`](https://github.com/NFuzed/Schematic-Symbol-Trainer-TypeScript/blob/main/src/lib/api.ts). Every path
is under `/api`.

| Method & path                         | Request                                                                                          | Response                                                        |
|---------------------------------------|--------------------------------------------------------------------------------------------------|-----------------------------------------------------------------|
| `GET  /api/health`                    | none                                                                                             | `{ "status": "ok", "device": "cuda" \| "cpu" }`                 |
| `POST /api/training-jobs`             | multipart: `dataset` (zip, see below), `config` (JSON `{baseModel, epochs, imageSize, valSplit}`) | `{ "jobId": "…" }`, returned straight away                      |
| `GET  /api/training-jobs/{id}/events` | none                                                                                             | Server-Sent Events, see below                                   |
| `GET  /api/models`                    | none                                                                                             | `[{ id, createdAt, entityIds, metrics? }]`                      |
| `POST /api/detections`                | multipart: `image` (PNG), `model_id`, `min_confidence` (0–1)                                     | `{ "detections": [{ entityId, bbox: [x,y,w,h], confidence }] }` |

Errors should use a non-`2xx` status with a readable body (FastAPI's `{"detail": "…"}` is fine). The front end shows the
status and body text to the user.

### Training config

| Field       | Values the front end sends                    | Default   |
|-------------|-----------------------------------------------|-----------|
| `baseModel` | `yolo11n`, `yolo11s`, `yolo11m`               | `yolo11s` |
| `epochs`    | 1–1000                                        | `100`     |
| `imageSize` | `640`, `960`, `1280`, `1600`                  | `1280`    |
| `valSplit`  | `0.1`, `0.15`, `0.2`, `0.25`, `0.3`           | `0.2`     |

Large `imageSize` values matter here: schematics are big, and small symbols disappear if a whole sheet is shrunk to
640 px.

### Training progress events

Training takes minutes or longer, so `POST /api/training-jobs` only queues the job. The front end then opens an
`EventSource` on `/api/training-jobs/{id}/events`. Each `data:` line is one JSON object:

```json
{"status": "running", "epoch": 12, "epochs": 100, "metrics": {"box_loss": 1.23, "mAP50": 0.61, "mAP50-95": 0.38}}
```

| Field      | Type                                         | Notes                                              |
|------------|----------------------------------------------|----------------------------------------------------|
| `status`   | `queued` · `running` · `done` · `failed`     | required on every event                            |
| `epoch`    | number                                       | current epoch, 1-based                             |
| `epochs`   | number                                       | total epochs; the progress bar is `epoch / epochs` |
| `metrics`  | `{ [name]: number }`                         | any names; shown as-is in the UI                   |
| `message`  | string                                       | shown to the user, mainly for `failed`             |
| `modelId`  | string                                       | required on `done`                                 |

Notes for the implementation:

- Send the job's current state as soon as a client connects. The front end subscribes only after the `POST` returns,
  so early events would otherwise be missed.
- Close the stream after `done` or `failed`. The front end treats a dropped connection as a failure, so don't close it
  early.
- When a job finishes, the front end calls `GET /api/models` to refresh its model list, so the new model must be listed
  by the time `done` is sent.

### Detections

The `image` is the diagram's canonical PNG, the same raster the front end draws boxes on. Every `bbox` in the response
must be `[x, y, width, height]` in pixels of that image, with the origin at the top left. Only detections with
`confidence >= min_confidence` are returned.

`entityId` is the `entity_id` string from the training dataset's categories, not the YOLO class index. That's how the
front end maps a detection back to the entity's name and colour, so each model has to store its class-to-entity
mapping.

The front end lets the user accept or reject each detection. Accepted boxes are saved as new annotations with
`source: "detection-accepted"` and are included in the next training run.

## Dataset format

The training upload and the front end's export/import use the same zip:

```
dataset.zip
├── annotations.json      COCO: images, categories, annotations (bbox = [x, y, w, h] px)
└── images/
    └── {diagramId}.png   canonical rasters the bboxes refer to
```

Besides the standard COCO fields, each record carries extra fields so nothing is lost on a round trip:

| COCO section  | Extra fields                                      |
|---------------|---------------------------------------------------|
| `images`      | `diagram_id`, `original_name`, `page`, `sha256`   |
| `categories`  | `entity_id`, `colour`, `description`              |
| `annotations` | `annotation_id`, `source`                         |

The front end only sends diagrams that have at least one annotation and entities that have at least one example, so
every category in a training upload has data.

### Converting to YOLO

For each annotation, with the image's `width` and `height`:

```
class = index of category_id in the sorted category list
cx    = (x + w / 2) / width
cy    = (y + h / 2) / height
nw    = w / width
nh    = h / height
```

Each diagram gets a `labels/{diagramId}.txt` with one `class cx cy nw nh` line per box, and a `data.yaml` lists the
class names in index order. Save that index-to-`entity_id` mapping with the model.

Boxes are axis-aligned. A symbol rotated 90° just gets a taller box. Rotated boxes (YOLO-OBB) can be added later if
steep angles become a problem.

## Storage

Nothing is persisted between jobs except trained models.

```
data/
├── jobs/{jobId}/                 unpacked dataset, YOLO labels, training run output (can be deleted after)
└── models/{modelId}/
    ├── weights.pt                best weights from the run
    └── model.json                id, createdAt, baseModel, config, entityIds (in class order), final metrics
```

`GET /api/models` reads the `model.json` files, so models survive a restart.

**Later:** the back end becomes the source of truth for the dataset too. The front end's `DataStore` interface
(`src/lib/store.ts`) is the seam: these CRUD endpoints would replace the browser's IndexedDB store without changing the
UI.

| What                  | Now                         | Later                                              |
|-----------------------|-----------------------------|----------------------------------------------------|
| Entities, annotations | in the front end            | SQLite via SQLAlchemy; Postgres if multi-user      |
| Diagram images        | in the front end            | `data/diagrams/{sha256}.png`                       |
| Trained models        | `data/models/{modelId}/`    | same, plus a metadata row                          |
| New endpoints         | none                        | `/api/diagrams`, `/api/entities`, `/api/annotations` |

## Stack

| Concern           | Choice                           | Why                                                                        |
|-------------------|----------------------------------|----------------------------------------------------------------------------|
| Language          | Python                           | The computer-vision and deep-learning ecosystem lives here                 |
| Web framework     | FastAPI + Uvicorn                | Typed request models, streaming responses for SSE, OpenAPI schema         |
| Detection model   | Ultralytics YOLO11               | Strong small-object detection, simple fine-tuning, OBB variant for later  |
| Deep learning     | PyTorch                          | Required by Ultralytics; CUDA when available                               |
| Image handling    | Pillow, OpenCV                   | Reading PNGs and any preprocessing                                         |
| Background jobs   | In-process worker thread         | One GPU, so one job at a time; swap for a task queue if needed           |
| Tests             | pytest + FastAPI `TestClient`    |                                                                            |

## Project layout

```
app/
├── main.py                FastAPI app factory, /api router, health check
├── config.py              settings from environment variables
├── api/
│   ├── deps.py            shared services (model store, job manager, detector)
│   ├── training.py        POST /training-jobs, GET /training-jobs/{id}/events
│   ├── models.py          GET /models
│   └── detections.py      POST /detections
├── schemas.py             Pydantic models matching the front end's types
├── dataset/
│   ├── coco.py            read and validate the uploaded zip
│   └── yolo.py            COCO → YOLO labels, train/val split, data.yaml
├── training/
│   ├── jobs.py            job queue, status, event broadcasting
│   └── trainer.py         Ultralytics training run, per-epoch callbacks
├── inference/
│   └── detector.py        load a model, run it, map classes to entity ids
└── storage.py             data/ directory layout, model metadata
scripts/
└── synthetic_diagrams.py  synthetic schematics + dataset zip for testing
tests/
requirements.txt
```

## Roadmap

- [x] `GET /api/health`
- [x] Read and validate the dataset zip
- [x] COCO → YOLO conversion and train/validation split
- [x] Training jobs with SSE progress
- [x] Model store and `GET /api/models`
- [x] `POST /api/detections`
- [ ] Cancel a running training job
- [ ] Training jobs that survive a back-end restart
- [ ] Generate the front end's API types from `/openapi.json`
- [ ] Tiled inference for very large scans
- [ ] Server-side `DataStore` endpoints so several people can share one dataset
- [ ] Rotated boxes (YOLO-OBB)

## History

This repository used to hold an earlier Python desktop version of the project (PySide6 front end in
`Schematic-Symbol-Trainer-Graphical`, core library here). That code was removed when the project moved to the
TypeScript front end, and is still available in this repository's git history.

## License

TBD
