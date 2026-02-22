import argparse
import os
from typing import Any
from PIL import Image

import lancedb

from schema import Myntra, get_schema_by_name


def run_vector_search(
    database: str,
    table_name: str,
    schema: Any,
    search_query: Any,
    limit: int = 6,
    output_folder: str = "output",
) -> None:
    """
    Perform vector search using text or image queries.
    Supports:
    - Text queries
    - PIL Images (Streamlit uploads / video frames)
    - Image file paths (CLI usage)
    """

    # Recreate output folder cleanly
    if os.path.exists(output_folder):
        for file in os.listdir(output_folder):
            os.remove(os.path.join(output_folder, file))
    else:
        os.makedirs(output_folder)

    # Connect to LanceDB
    db = lancedb.connect(database)
    table = db.open_table(table_name)

    # ✅ SAFE multimodal query handling
    if isinstance(search_query, Image.Image):
        # Already a PIL Image → do nothing
        pass

    elif isinstance(search_query, str) and (
        search_query.endswith(".jpg") or search_query.endswith(".png")
    ):
        # Image path from CLI
        search_query = Image.open(search_query)

    elif isinstance(search_query, str):
        # Normal text query → do nothing
        pass

    else:
        raise ValueError(f"Unsupported search query type: {type(search_query)}")

    # Perform search
    rs = table.search(search_query).limit(limit).to_pydantic(schema)

    if len(rs) == 0:
        print("No results found.")
        return

    # ✅ Save results safely
    for i, item in enumerate(rs):
        image_path = os.path.join(output_folder, f"image_{i}.jpg")
        item.image.save(image_path, "JPEG")

    print(f"Saved {len(rs)} results to '{output_folder}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vector Search")

    parser.add_argument("--database", type=str, help="Path to database", default="~/.lancedb")
    parser.add_argument("--table_name", type=str, help="Table name", required=True)
    parser.add_argument("--schema", type=str, help="Schema name", default="Myntra")
    parser.add_argument("--search_query", type=str, help="Search query", required=True)
    parser.add_argument("--limit", type=int, default=6, help="Result limit")
    parser.add_argument("--output_folder", type=str, default="output", help="Output folder")

    args = parser.parse_args()

    schema = get_schema_by_name(args.schema)
    if schema is None:
        raise ValueError(f"Unknown schema: {args.schema}")

    run_vector_search(
        args.database,
        args.table_name,
        schema,
        args.search_query,
        args.limit,
        args.output_folder,
    )
