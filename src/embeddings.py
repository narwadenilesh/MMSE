"""Embedding utilities for multimodal search."""

import logging
from typing import Union
import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

# Set up logging
logger = logging.getLogger(__name__)

# Lazy-loaded cached models (loaded once, reused across calls)
_open_clip_components = None  # (model, preprocess, tokenizer, device)
_text_model = None


def get_open_clip_components():
    """Get or initialize the OpenCLIP model, preprocess, tokenizer and device.

    Loads once and caches globally to avoid re-loading on every call.
    """
    global _open_clip_components
    if _open_clip_components is not None:
        return _open_clip_components

    import open_clip
    logger.info("Loading OpenCLIP ViT-B-32 model (first call — loading once)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    _open_clip_components = (model, preprocess, tokenizer, device)
    logger.info(f"OpenCLIP model loaded on {device}.")
    return _open_clip_components


def get_clip_model():
    """Get or initialize the CLIP model (LanceDB registry version)."""
    from .embedding_model import register_model
    return register_model("open-clip")


def get_text_embedding_model():
    """Get or initialize the SentenceTransformers text embedding model."""
    global _text_model
    if _text_model is not None:
        return _text_model

    try:
        _text_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("SentenceTransformers model loaded successfully.")
        return _text_model
    except Exception as e:
        logger.error(f"Failed to load text embedding model: {e}")
        raise RuntimeError(f"Text embedding model initialization failed: {e}") from e


def embed_image(image: Union[str, Image.Image]) -> np.ndarray:
    """Embed a single image using OpenCLIP (cached model).

    Args:
        image: PIL Image or path to image file.

    Returns:
        512-dim L2-normalised embedding vector (numpy array).
    """
    if isinstance(image, str):
        image = Image.open(image)

    model, preprocess, _, device = get_open_clip_components()
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = model.encode_image(img_tensor).cpu().numpy()[0]
    # L2-normalise — standard practice for CLIP retrieval
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def embed_text_clip(text: str) -> np.ndarray:
    """Embed a text string using OpenCLIP with prompt ensemble.

    Uses multiple fashion-specific prompt templates and averages the
    embeddings — the standard CLIP technique for improved retrieval
    accuracy over bare keywords.

    Args:
        text: Text query (e.g. "tshirt", "blue jeans").

    Returns:
        512-dim averaged & normalised embedding vector (numpy array).
    """
    # Fashion-specific prompt templates (CLIP paper ensemble technique)
    PROMPT_TEMPLATES = [
        "a photo of a {}",
        "a photo of a {} for sale",
        "a product photo of a {}",
        "a fashion photo of a {}",
        "a {} worn by a model",
        "a close up of a {}",
        "a {} on a white background",
        "this is a {}",
    ]

    model, _, tokenizer, device = get_open_clip_components()

    all_vecs = []
    for template in PROMPT_TEMPLATES:
        prompt = template.format(text.strip())
        tokens = tokenizer([prompt]).to(device)
        with torch.no_grad():
            vec = model.encode_text(tokens).cpu().numpy()[0]
        all_vecs.append(vec)

    # Average then normalise — matches how CLIP zero-shot classifiers work
    mean_vec = np.mean(all_vecs, axis=0)
    norm = np.linalg.norm(mean_vec)
    return mean_vec / norm if norm > 0 else mean_vec


def embed_text(text: str) -> np.ndarray:
    """Embed a text string using SentenceTransformers (384-dim).

    Used for the 'text_embedding' column (description/metadata similarity).

    Args:
        text: Text to embed.

    Returns:
        384-dim embedding vector (numpy array).
    """
    if not text or not text.strip():
        return np.zeros(384, dtype=np.float32)  # Default zero vector for empty text

    model = get_text_embedding_model()
    embeddings = model.encode([text], convert_to_numpy=True)
    return np.asarray(embeddings[0], dtype=np.float32)


def embed_texts(texts: list) -> np.ndarray:
    """Embed multiple texts at once using SentenceTransformers.

    Args:
        texts: List of text strings.

    Returns:
        Embedding matrix (numpy array, shape: [n_texts, 384]).
    """
    texts = [t if t and t.strip() else "" for t in texts]
    model = get_text_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    return np.asarray(embeddings, dtype=np.float32)


def normalize_embedding(vec: np.ndarray) -> np.ndarray:
    """Normalize an embedding vector to unit length."""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two embedding vectors."""
    vec1 = normalize_embedding(vec1)
    vec2 = normalize_embedding(vec2)
    return float(np.dot(vec1, vec2))
