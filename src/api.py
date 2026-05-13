from pathlib import Path
from io import BytesIO
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from .media_utils import transcribe_audio_bytes
from .search_engine import hybrid_search, search


DEFAULT_DB = "C:/Users/hp/.lancedb"
DEFAULT_TABLE = "myntra_5k"
DEFAULT_SCHEMA = "Myntra"
SUPPORTED_AUDIO_LANGUAGE_CODES = {"en-IN", "hi-IN", "mr-IN"}
DEFAULT_AUDIO_LANGUAGE_CODE = "en-IN"


class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    use_hybrid: bool = False
    image_weight: float = 0.6
    text_weight: float = 0.4
    filters: Optional[Dict[str, Any]] = None


def _format_results(items: List[Any]) -> List[Dict[str, Any]]:
    return [
        {
            "image_uri": item.image_uri,
            "name": getattr(item, "name", ""),
            "brand": getattr(item, "brand", ""),
            "price": getattr(item, "price", 0.0),
            "color": getattr(item, "color", ""),
            "description": getattr(item, "description", ""),
            "attributes": getattr(item, "attributes", ""),
        }
        for item in items
    ]


def _run_search(
    query: Any,
    limit: int,
    use_hybrid: bool,
    image_weight: float,
    text_weight: float,
    filters: Optional[Dict[str, Any]],
) -> List[Any]:
    if use_hybrid:
        return hybrid_search(
            DEFAULT_DB,
            DEFAULT_TABLE,
            DEFAULT_SCHEMA,
            query,
            limit=limit,
            image_weight=image_weight,
            text_weight=text_weight,
            filters=filters,
        )
    return search(
        DEFAULT_DB,
        DEFAULT_TABLE,
        DEFAULT_SCHEMA,
        query,
        limit=limit,
        filters=filters,
    )


app = FastAPI(title="Multimodal Fashion Search API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

dataset_dir = Path(__file__).parent.parent / "dataset"
if dataset_dir.exists():
    app.mount("/dataset", StaticFiles(directory=str(dataset_dir)), name="dataset")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/search/text")
def search_text(request: SearchRequest) -> Dict[str, Any]:
    try:
        items = _run_search(
            query=request.query,
            limit=request.limit,
            use_hybrid=request.use_hybrid,
            image_weight=request.image_weight,
            text_weight=request.text_weight,
            filters=request.filters,
        )
        return {"query_type": "text", "results": _format_results(items)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/search/image")
async def search_image(
    file: UploadFile = File(...),
    limit: int = Form(10),
    use_hybrid: bool = Form(False),
    image_weight: float = Form(0.6),
    text_weight: float = Form(0.4),
) -> Dict[str, Any]:
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        items = _run_search(
            query=image,
            limit=limit,
            use_hybrid=use_hybrid,
            image_weight=image_weight,
            text_weight=text_weight,
            filters=None,
        )
        return {"query_type": "image", "results": _format_results(items)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/search/audio")
async def search_audio(
    file: UploadFile = File(...),
    limit: int = Form(10),
    use_hybrid: bool = Form(False),
    image_weight: float = Form(0.6),
    text_weight: float = Form(0.4),
    language_code: str = Form(DEFAULT_AUDIO_LANGUAGE_CODE),
) -> Dict[str, Any]:
    try:
        language_code = (language_code or DEFAULT_AUDIO_LANGUAGE_CODE).strip()
        if language_code not in SUPPORTED_AUDIO_LANGUAGE_CODES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported language_code '{language_code}'. "
                    f"Use one of: {sorted(SUPPORTED_AUDIO_LANGUAGE_CODES)}"
                ),
            )

        audio_bytes = await file.read()
        transcript = transcribe_audio_bytes(audio_bytes, language_code=language_code)
        if not transcript:
            raise HTTPException(status_code=400, detail="Could not transcribe audio.")

        items = _run_search(
            query=transcript,
            limit=limit,
            use_hybrid=use_hybrid,
            image_weight=image_weight,
            text_weight=text_weight,
            filters=None,
        )
        return {
            "query_type": "audio",
            "language_code": language_code,
            "transcript": transcript,
            "results": _format_results(items),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
