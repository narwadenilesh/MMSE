import argparse
import random
from pathlib import Path

import lancedb
import numpy as np
import pandas as pd
import torch
from PIL import Image
from open_clip import create_model_and_transforms

try:
    from .embeddings import embed_text
except ImportError:
    # Allow direct script execution: `python src/make_table.py`
    from embeddings import embed_text


def _normalize_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def _parse_price(value):
    try:
        return float(value)
    except Exception:
        return 0.0


def _get_sorted_image_paths(data_path: str):
    p = Path(data_path).expanduser()
    paths = [f for f in p.glob("*.jpg") if f.is_file()]

    def sort_key(path: Path):
        stem = path.stem
        return int(stem) if stem.isdigit() else stem

    return sorted(paths, key=sort_key)


def create_table(
    database,
    table_name,
    data_path,
    metadata_csv,
    mode="overwrite",
    num_samples=1000,
    seed=42,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, preprocess = create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.to(device)
    model.eval()

    metadata_df = pd.read_csv(metadata_csv).fillna("")
    # Build fast lookup by image filename from the new dataset schema
    # (`image`, `display name`, `description`, `category`).
    if "image" in metadata_df.columns:
        metadata_by_image = metadata_df.set_index("image", drop=False)
    else:
        metadata_by_image = None

    db = lancedb.connect(database)

    print(f"Creating table '{table_name}'...")

    image_paths = _get_sorted_image_paths(data_path)
    print(f"Found {len(image_paths)} images")

    if len(image_paths) > num_samples:
        image_paths = random.Random(seed).sample(image_paths, num_samples)
        image_paths = sorted(image_paths, key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem)

    data = []
    print("Generating embeddings and attaching metadata...")

    for i, img_path in enumerate(image_paths):
        try:
            img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                vec = model.encode_image(img).cpu().numpy()[0]
            # L2-normalise for cosine similarity
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec = vec / norm
            vec = np.asarray(vec, dtype=np.float32)

            metadata = {
                "name": "",
                "price": 0.0,
                "color": "",
                "brand": "",
                "description": "",
                "attributes": "",
            }

            row = None
            img_filename = img_path.name
            if metadata_by_image is not None and img_filename in metadata_by_image.index:
                row = metadata_by_image.loc[img_filename]
            elif i < len(metadata_df):
                # Fallback for older CSV layouts
                row = metadata_df.iloc[i]

            if row is not None:
                metadata = {
                    # New dataset names
                    "name": _normalize_text(row.get("display name", row.get("name", ""))),
                    "price": _parse_price(row.get("price", 0.0)),
                    "color": _normalize_text(row.get("colour", row.get("color", ""))),
                    "brand": _normalize_text(row.get("brand", "")),
                    "description": _normalize_text(row.get("description", "")),
                    # Use category as a lightweight attribute channel
                    "attributes": _normalize_text(row.get("category", row.get("p_attributes", ""))),
                }

            # Compute text embedding for description
            text_embedding = np.asarray(
                embed_text(metadata["description"]), dtype=np.float32
            )

            data.append({
                "id": i,
                "vector": vec.tolist(),
                "image_uri": str(img_path),
                "text_embedding": text_embedding.tolist(),
                **metadata,
            })

            if i % 100 == 0:
                print(f"Processed {i}/{len(image_paths)}")

        except Exception as e:
            print(f"Skipping {img_path}: {e}")

    df = pd.DataFrame(data)
    table = db.create_table(table_name, data=df, mode=mode)

    print("✅ Table created with embeddings and metadata!")
    print("Columns:", table.to_pandas().columns)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--database", default="C:/Users/hp/.lancedb")
    parser.add_argument("--table_name", default="myntra")
    parser.add_argument("--data_path", default="dataset/data")
    parser.add_argument("--metadata_csv", default="dataset/data.csv")
    parser.add_argument("--mode", default="overwrite")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    create_table(
        args.database,
        args.table_name,
        args.data_path,
        args.metadata_csv,
        args.mode,
        args.num_samples,
        args.seed,
    )
