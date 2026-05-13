from pathlib import Path
from typing import Any, List, Optional
from types import SimpleNamespace
from PIL import Image
import logging
import lancedb
import pandas as pd
import numpy as np

from .schema import get_schema_by_name
from .embeddings import embed_image, embed_text, embed_text_clip, cosine_similarity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CSV metadata cache — loaded once, keyed by image filename stem (integer)
# ---------------------------------------------------------------------------
_metadata_df: Optional[pd.DataFrame] = None
_METADATA_CSV = Path(__file__).parent.parent / "dataset" / "data.csv"


def _get_metadata_df() -> Optional[pd.DataFrame]:
    """Load and cache the Fashion Dataset CSV (keyed by integer stem)."""
    global _metadata_df
    if _metadata_df is not None:
        return _metadata_df
    if not _METADATA_CSV.exists():
        logger.warning(f"Metadata CSV not found at {_METADATA_CSV}. Metadata will not be corrected.")
        return None
    try:
        df = pd.read_csv(_METADATA_CSV).fillna("")
        _metadata_df = df
        logger.info(f"Loaded metadata CSV: {len(df)} rows.")
        return _metadata_df
    except Exception as e:
        logger.error(f"Failed to load metadata CSV: {e}")
        return None


def _enrich_result(result: Any) -> Any:
    """Override stored metadata with freshly looked-up CSV row by image filename.

    This corrects the mis-alignment that happens when the table was built
    with sampled images whose positional index doesn't match the CSV row.
    """
    df = _get_metadata_df()
    if df is None:
        return result

    try:
        image_name = Path(result.image_uri).name
        if "image" in df.columns:
            matched = df[df["image"] == image_name]
            if matched.empty:
                return result
            row = matched.iloc[0]
        else:
            stem = Path(result.image_uri).stem
            if not stem.isdigit():
                return result
            row_idx = int(stem)
            if row_idx not in df.index:
                return result
            row = df.iloc[row_idx] if row_idx < len(df) else None
            if row is None:
                return result

        if row is None:
            return result

        # Patch the pydantic object's fields in-place
        result.name = str(row.get("display name", row.get("name", "")) or "")
        result.brand = str(row.get("brand", "") or "")
        result.color = str(row.get("colour", row.get("color", "")) or "")
        result.description = str(row.get("description", "") or "")
        result.attributes = str(row.get("category", row.get("p_attributes", "")) or "")
        try:
            result.price = float(row.get("price", 0) or 0)
        except (ValueError, TypeError):
            result.price = 0.0
    except Exception as e:
        logger.debug(f"Could not enrich result {result.image_uri}: {e}")

    return result


# ---------------------------------------------------------------------------
# Metadata filters
# ---------------------------------------------------------------------------
def _apply_filters(results: List[Any], filters: dict) -> List[Any]:
    """Apply metadata filters to search results."""
    if not filters:
        return results

    filtered = []
    for result in results:
        match = True
        for key, value in filters.items():
            if key == "price_min" and hasattr(result, "price") and result.price is not None:
                if result.price < value:
                    match = False
                    break
            elif key == "price_max" and hasattr(result, "price") and result.price is not None:
                if result.price > value:
                    match = False
                    break
            elif hasattr(result, key):
                attr_value = getattr(result, key, "")
                if isinstance(attr_value, str):
                    if value.lower() not in attr_value.lower():
                        match = False
                        break
                elif attr_value != value:
                    match = False
                    break
        if match:
            filtered.append(result)
    return filtered


def _to_row_object(row: pd.Series) -> Any:
    """Convert a pandas row from LanceDB into an attribute-style object."""
    vector = row.get("vector", None)
    text_embedding = row.get("text_embedding", None)

    if vector is not None:
        vector = np.asarray(vector, dtype=np.float32)
    if text_embedding is not None:
        text_embedding = np.asarray(text_embedding, dtype=np.float32)

    return SimpleNamespace(
        id=row.get("id", None),
        image_uri=str(row.get("image_uri", "") or ""),
        name=str(row.get("name", "") or ""),
        brand=str(row.get("brand", "") or ""),
        price=float(row.get("price", 0.0) or 0.0),
        color=str(row.get("color", "") or ""),
        description=str(row.get("description", "") or ""),
        attributes=str(row.get("attributes", "") or ""),
        vector=vector,
        text_embedding=text_embedding,
    )


def _search_to_objects(search_builder: Any) -> List[Any]:
    """Execute Lance search and convert rows to objects safely."""
    df = search_builder.to_pandas()
    if df is None or df.empty:
        return []
    return [_to_row_object(row) for _, row in df.iterrows()]


# ---------------------------------------------------------------------------
# Standard vector search
# ---------------------------------------------------------------------------
def search(
    database: str,
    table_name: str,
    schema_name: str,
    search_query: Any,
    limit: int = 6,
    filters: dict = None,
) -> List[Any]:
    """Perform a multimodal vector search and return result objects.

    Args:
        database: Path to the LanceDB database.
        table_name: Name of the LanceDB table.
        schema_name: Name of the schema to use for results.
        search_query: Text string, PIL Image, or path to an image file.
        limit: Number of results to return.
        filters: Optional metadata filters
                 (e.g. {"brand": "Nike", "price_min": 100, "price_max": 500}).

    Returns:
        A list of Pydantic model objects with corrected metadata.
    """
    schema = get_schema_by_name(schema_name)
    if schema is None:
        raise ValueError(f"Unknown schema: {schema_name}")

    # Table was built with raw numpy vectors — must pre-embed every query.
    if isinstance(search_query, Image.Image):
        query_vec = embed_image(search_query)

    elif isinstance(search_query, str) and search_query.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):
        query_vec = embed_image(Image.open(search_query))

    elif isinstance(search_query, str):
        # Use CLIP text encoder (prompt-ensemble) so it's in the same space
        # as the stored image vectors.
        query_vec = embed_text_clip(search_query)

    else:
        raise ValueError(f"Unsupported search query type: {type(search_query)}")

    db = lancedb.connect(database)
    table = db.open_table(table_name)
    results = _search_to_objects(
        table.search(query_vec, vector_column_name="vector")
        .metric("cosine")
        .limit(limit)
    )

    # Fix metadata mis-alignment: re-read from CSV by image filename
    results = [_enrich_result(r) for r in results]

    # Apply optional metadata filters
    if filters:
        logger.debug(f"Applying filters: {filters}")
        results = _apply_filters(results, filters)

    return results


# ---------------------------------------------------------------------------
# Hybrid search (image CLIP + text SentenceTransformers)
# ---------------------------------------------------------------------------
def hybrid_search(
    database: str,
    table_name: str,
    schema_name: str,
    search_query: Any,
    limit: int = 6,
    image_weight: float = 0.6,
    text_weight: float = 0.4,
    filters: dict = None,
) -> List[Any]:
    """Hybrid search: weighted combination of image-CLIP and text similarity.

    Args:
        database: Path to LanceDB database.
        table_name: Name of the LanceDB table.
        schema_name: Name of the schema to use.
        search_query: Text string, PIL Image, or image path.
        limit: Number of results to return.
        image_weight: Weight for CLIP image similarity (0-1).
        text_weight: Weight for SentenceTransformers text similarity (0-1).
        filters: Optional metadata filters.

    Returns:
        List of result objects sorted by hybrid score with corrected metadata.
    """
    schema = get_schema_by_name(schema_name)
    if schema is None:
        raise ValueError(f"Unknown schema: {schema_name}")

    db = lancedb.connect(database)
    table = db.open_table(table_name)

    # Normalise weights
    total_weight = image_weight + text_weight
    if total_weight == 0:
        image_weight, text_weight = 0.5, 0.5
    else:
        image_weight /= total_weight
        text_weight /= total_weight

    # Pre-embed the query ONCE (not inside the per-row loop)
    query_image_vec = None
    query_text_vec = None

    if isinstance(search_query, Image.Image):
        query_image_vec = embed_image(search_query)
    elif isinstance(search_query, str) and search_query.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):
        query_image_vec = embed_image(Image.open(search_query))
    elif isinstance(search_query, str):
        query_text_vec = embed_text(search_query)
    else:
        raise ValueError(f"Unsupported search query type: {type(search_query)}")

    # Fetch all rows to score them
    all_df = table.to_pandas()
    all_results = [] if all_df is None or all_df.empty else [
        _to_row_object(row) for _, row in all_df.iterrows()
    ]

    scores = []
    for result in all_results:
        image_sim = (
            cosine_similarity(query_image_vec, result.vector)
            if query_image_vec is not None
            else 0.0
        )
        text_sim = 0.0
        if query_text_vec is not None and result.text_embedding is not None:
            text_sim = cosine_similarity(query_text_vec, result.text_embedding)
        combined = image_weight * image_sim + text_weight * text_sim
        scores.append((result, combined))

    scores.sort(key=lambda x: x[1], reverse=True)
    top_results = [result for result, _ in scores[:limit]]

    # Fix metadata mis-alignment
    top_results = [_enrich_result(r) for r in top_results]

    if filters:
        logger.debug(f"Applying filters: {filters}")
        top_results = _apply_filters(top_results, filters)

    logger.info(f"Hybrid search returned {len(top_results)} results.")
    return top_results
