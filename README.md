# Multimodal Search Engine (uv + FastAPI)

Local multimodal product retrieval backend using:

- OpenCLIP image/text embeddings
- SentenceTransformers text embeddings
- LanceDB local vector store
- Sarvam Speech-to-Text for audio queries
- FastAPI API server

## Project Structure

```text
multimodal-search-engine/
  dataset/
    data.csv
    data/*.jpg
  src/
    api.py
    search_engine.py
    make_table.py
    embeddings.py
    media_utils.py
    schema.py
  .env
  pyproject.toml / requirements.txt
```

## 1. Setup with uv

Install uv (if not installed):

```bash
pip install uv
```

Create virtual environment with Python 3.10 and install dependencies:

```bash
uv venv --python 3.10
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

Optional lock file:

```bash
uv pip freeze > requirements-lock.txt
```

## 2. Configure Environment

Create/update `.env` in project root:

```env
SARVAM_API_KEY=your_key_here
SARVAM_STT_ENDPOINT=https://api.sarvam.ai/speech-to-text
SARVAM_STT_MODEL=saaras:v3
SARVAM_STT_MODE=transcribe
SARVAM_STT_LANGUAGE_CODE=en-IN
```

Supported audio language codes in API:

- `en-IN` (primary/default)
- `hi-IN`
- `mr-IN`

## 3. Build Vector Table (first time)

```bash
uv run python src/make_table.py --database "C:/Users/hp/.lancedb" --table_name "myntra" --mode overwrite --num_samples 44441
```

## 4. Start the Project (Backend API)

```bash
uv run uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Server will be available at:

- `http://127.0.0.1:8000`
- Swagger docs: `http://127.0.0.1:8000/docs`

## 5. API Endpoints

- `GET /health`
- `POST /search/text`
- `POST /search/image`
- `POST /search/audio`

### Example: text search

```bash
curl -X POST "http://127.0.0.1:8000/search/text" ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"black sneakers\",\"limit\":10,\"use_hybrid\":true}"
```

### Example: audio search language selection

Send `language_code` as form-data in `/search/audio`:

- `en-IN`
- `hi-IN`
- `mr-IN`
