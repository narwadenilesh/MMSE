# Multimodal Fashion Search Engine — Full Project Information

## 1. Overview

This project is a **multimodal fashion search engine** that lets users find visually similar products using **text**, **images**, **videos**, or **voice**. It uses **LanceDB** as a vector database and **CLIP** (OpenCLIP) for embedding both text and images into a shared space, so you can search a catalog of product images by describing what you want, uploading a reference image, a video frame, or speaking your query.

The UI is built with **Streamlit** and styled to resemble a fashion e‑commerce experience (Myntra-like). All search logic runs in Python; there is no separate React/frontend—Streamlit is the only UI.

---

## 2. Purpose & Use Case

- **Fashion product discovery**: e.g. “red kurta”, “white sneakers”, or “similar to this image”.
- **Multimodal input**: same backend supports text, image upload, video (one frame), and audio (transcribed to text via Whisper).
- **Vector similarity**: CLIP embeddings + LanceDB for fast nearest-neighbor search over product images.

Typical workflow: index product images once with `make_table.py`, then search via the Streamlit app or the `vector_search.py` CLI.

---

## 3. Tech Stack Summary

| Layer | Technology |
|-------|------------|
| **Language** | Python 3.8 |
| **UI** | Streamlit |
| **Vector DB** | LanceDB |
| **Embeddings** | OpenCLIP (CLIP) via `open-clip-torch` |
| **Deep learning** | PyTorch, torchvision, timm |
| **Images** | Pillow (PIL), OpenCV (video frames) |
| **Audio → text** | Whisper (optional), FFmpeg (system) |
| **Data** | Pandas, Pydantic, PyArrow |
| **Environment** | Conda (`environment.yml`) |

Optional for full features:

- **Whisper**: `pip install openai-whisper` (audio search).
- **FFmpeg**: on PATH (for Whisper).
- **OpenCV**: `opencv-python-headless` (video search; listed in `environment.yml`).

---

## 4. Project Structure

```
multimodal-search-engine/
├── src/
│   ├── app.py              # Streamlit web app (UI + all search entry points)
│   ├── vector_search.py    # Core search: run_vector_search()
│   ├── make_table.py       # Build LanceDB table from image folder
│   ├── schema.py           # Myntra schema + CLIP registration
│   └── embedding_model.py  # LanceDB embedding registry (open-clip)
├── input/                  # Your image dataset (e.g. input/Images/*.jpg)
├── output/                 # Search result images written here
├── environment.yml         # Conda env + pip deps
├── README.md
├── .gitignore
└── PROJECT_OVERVIEW.md     # This file
```

- **No React**: The `frontend/` folder, if present, is leftover (e.g. `node_modules`); the active UI is only Streamlit in `src/app.py`.

---

## 5. Architecture & Data Flow

### 5.1 Indexing (one-time or when you add data)

1. **`make_table.py`**  
   - Connects to LanceDB (e.g. `~/.lancedb`).  
   - Scans a folder (e.g. `input/Images`) for `*.jpg`.  
   - Optionally samples up to `num_samples` images.  
   - Creates a table with schema `Myntra`: each row has `image_uri` and a **vector** from CLIP.  
   - The vector is produced by the **open-clip** embedding function registered in `schema.py` (via `embedding_model.py`).

### 5.2 Schema & embeddings

- **`schema.py`**  
  - Defines **Myntra** (LanceModel): `vector` (CLIP embedding), `image_uri` (path to image).  
  - Registers the **open-clip** model with LanceDB’s `EmbeddingFunctionRegistry` and sets `clip.VectorField()` and `clip.SourceField()` so that:  
    - On insert: each `image_uri` is passed to CLIP to compute the vector.  
    - On search: the query (text or image) is embedded with the same CLIP model.  
  - Exposes `get_schema_by_name("Myntra")` for the CLI.

- **`embedding_model.py`**  
  - Thin wrapper: `register_model("open-clip")` returns the CLIP model instance from LanceDB’s registry.

### 5.3 Search flow

1. **Query input** (in Streamlit or CLI):  
   - **Text**: e.g. “Blue Jeans” or transcribed audio text.  
   - **Image**: file path (CLI) or PIL Image (Streamlit upload or extracted video frame).  
   - **Video**: Streamlit uploads video → **OpenCV** extracts one frame (first/middle/last) → that frame is used as a PIL Image query.  
   - **Audio**: Streamlit record/upload → **Whisper** transcribes → text used as query.

2. **`vector_search.run_vector_search()`**  
   - Normalizes the query: if it’s an image path (string ending in `.jpg`/`.png`), opens it as PIL Image; otherwise keeps string or PIL Image.  
   - Connects to LanceDB, opens the table.  
   - Calls `table.search(search_query).limit(limit).to_pydantic(Myntra)`.  
   - LanceDB uses the same open-clip embedding function to embed the query and runs vector similarity search.  
   - Returns top‑k rows as Pydantic `Myntra` instances.

3. **Output**  
   - Saves each result’s image to `output_folder` (e.g. `image_0.jpg`, `image_1.jpg`, …).  
   - Streamlit then reads `output_folder` and displays results in a **grid** or **carousel** view.

---

## 6. File-by-File Role

| File | Role |
|------|------|
| **app.py** | Streamlit UI: page config, custom CSS (Myntra-like), sidebar (table name, text query, limit, output folder, image/video/audio uploads, record audio), Whisper + OpenCV helpers, call to `run_vector_search`, results grid/carousel. |
| **vector_search.py** | `run_vector_search(database, table_name, schema, search_query, limit, output_folder)`: normalizes query (path → PIL, keep text/PIL), runs LanceDB search, saves result images. CLI for text/image path search. |
| **make_table.py** | `create_table(database, table_name, data_path, schema, mode, num_samples)`: creates/opens table, globs `.jpg`, samples, adds rows; LanceDB + CLIP fill vectors. CLI to build the index. |
| **schema.py** | Defines `Myntra` (vector + image_uri), registers open-clip, `get_schema_by_name()`. |
| **embedding_model.py** | `register_model(model_name)` → returns embedding model from LanceDB registry (used for open-clip). |

---

## 7. Setup & Run

### 7.1 Environment

```bash
cd multimodal-search-engine
conda env create -f environment.yml
conda activate lance-env
```

Optional for audio search:

```bash
pip install openai-whisper
# Ensure ffmpeg is on PATH (e.g. install and add to PATH on Windows)
```

### 7.2 Data

- Put product images (e.g. `.jpg`) in a folder, e.g. `input/Images/`.  
- Optional: use [Myntra Fashion Product Dataset](https://www.kaggle.com/datasets/hiteshsuthar101/myntra-fashion-product-dataset) and place images under `input/Images/`.

### 7.3 Create the vector index

```bash
python src/make_table.py --database "~/.lancedb" --table_name "myntra" --data_path "input/Images" --num_samples 1000
```

### 7.4 Run the Streamlit UI

```bash
streamlit run src/app.py -- --table_name myntra
```

Or with defaults (table name “myntra” is in the sidebar):

```bash
streamlit run src/app.py
```

Then use the sidebar to: enter text, upload image/video/audio (or record audio), set result limit and output folder, and click **Run Vector Search**. Results appear in the main area in grid or carousel view.

### 7.5 CLI-only search (no UI)

**Text:**

```bash
python src/vector_search.py --database ~/.lancedb --table_name myntra --schema Myntra --search_query "Blue Jeans" --output_folder output
```

**Image (path):**

```bash
python src/vector_search.py --database ~/.lancedb --table_name myntra --schema Myntra --search_query "path/to/image.jpg" --output_folder output
```

---

## 8. Multimodal Search in the UI (Streamlit)

| Modality | How it works in the app |
|----------|--------------------------|
| **Text** | Sidebar “Search query” text box; optional hero chips (Kurta, Jeans, etc.) are display-only (you still type or use another modality). |
| **Image** | Sidebar “Upload an image” (jpg/png). Uploaded image is passed as PIL to `run_vector_search`. |
| **Video** | Sidebar “Upload a video” (mp4, webm, avi, mov, mkv). One frame is extracted (first/middle/last) via OpenCV; that frame is used as the search image. |
| **Audio** | Record in browser (Streamlit `audio_input`) or upload (wav, mp3, m4a, flac, ogg). Audio is transcribed with Whisper → text used as search query. |

Only one “active” query is used per search: image and video set the query to a PIL Image; audio sets it to the transcribed text; otherwise the text box value is used.

---

## 9. Configuration & Defaults

- **Database path**: `~/.lancedb` (overridable in sidebar or CLI).  
- **Table name**: default `myntra` (sidebar/CLI).  
- **Result limit**: slider in sidebar (e.g. 1–10, default 3); CLI `--limit`.  
- **Output folder**: default `output`; result images are written here and then shown in the app.  
- **FFmpeg (Windows)**: `app.py` can prepend a hardcoded FFmpeg path to `PATH` (see top of `app.py`); you can change or remove it.

---

## 10. Summary

- **What it is**: A Python-based multimodal fashion search engine with a Streamlit-only UI.  
- **What it does**: Indexes product images with CLIP in LanceDB; supports search by text, image, video frame, or voice (audio → Whisper → text).  
- **Tech**: Python, Streamlit, LanceDB, OpenCLIP, PyTorch, Pillow, OpenCV, optional Whisper.  
- **How to run**: Create conda env → run `make_table.py` on your image folder → run `streamlit run src/app.py` and use the sidebar for all search types and the main area for results (grid or carousel).

For a short feature list and quick start, see **README.md**. For this full description, use **PROJECT_OVERVIEW.md** (this file).
