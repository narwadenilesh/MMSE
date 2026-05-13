from typing import Any, Optional

from lancedb.pydantic import LanceModel, Vector
from PIL import Image

from .embedding_model import register_model

# Register CLIP model
clip = register_model("open-clip")


class Myntra(LanceModel):
    """
    Schema for Myntra products.
    """

    # Vector embedding column
    vector: Vector(clip.ndims()) = clip.VectorField()

    # Image source path
    image_uri: str = clip.SourceField()

    # Metadata fields — must be declared here so to_pydantic() populates them
    name: str = ""
    brand: str = ""
    price: float = 0.0
    color: str = ""
    description: str = ""
    attributes: str = ""

    @property
    def image(self):
        return Image.open(self.image_uri)


def get_schema_by_name(schema_name: str) -> Any:
    """
    Return schema class from schema name.
    """

    schema_map = {
        "Myntra": Myntra,
    }

    return schema_map.get(schema_name)
