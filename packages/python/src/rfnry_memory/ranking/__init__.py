from typing import Annotated

from pydantic import Field

from rfnry_memory.ranking import bm25, cover_density
from rfnry_memory.ranking.bm25 import BM25
from rfnry_memory.ranking.cover_density import CoverDensity

TextRanking = Annotated[BM25 | CoverDensity, Field(discriminator="method")]

METHODS = {"bm25": bm25, "cover_density": cover_density}

__all__ = ["BM25", "METHODS", "CoverDensity", "TextRanking", "bm25", "cover_density"]
