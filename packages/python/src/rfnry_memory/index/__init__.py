from typing import Annotated

from pydantic import Field

from rfnry_memory.index import hnsw, ivfflat
from rfnry_memory.index.hnsw import HNSW
from rfnry_memory.index.ivfflat import IVFFlat

VectorIndex = Annotated[HNSW | IVFFlat, Field(discriminator="method")]

METHODS = {"hnsw": hnsw, "ivfflat": ivfflat}

__all__ = ["HNSW", "METHODS", "IVFFlat", "VectorIndex", "hnsw", "ivfflat"]
