import io
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app.api.deps import Services, services
from app.schemas import DetectionsResponse

router = APIRouter(tags=["detections"])


@router.post("/detections", response_model=DetectionsResponse)
def detect(
    image: UploadFile,
    model_id: Annotated[str, Form()],
    svc: Annotated[Services, Depends(services)],
    min_confidence: Annotated[float, Form(ge=0, le=1)] = 0.25,
) -> DetectionsResponse:
    record = svc.store.get(model_id)
    if record is None:
        raise HTTPException(404, f"No model with id {model_id!r}.")

    try:
        with Image.open(io.BytesIO(image.file.read())) as uploaded:
            # Boxes come back in pixels of this image, which is the front end's canonical raster.
            rgb = uploaded.convert("RGB")
    except UnidentifiedImageError as err:
        raise HTTPException(400, "The uploaded file isn't an image.") from err

    return DetectionsResponse(detections=svc.detector.detect(record, rgb, min_confidence))
