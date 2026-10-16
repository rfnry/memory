from typing import Annotated

from pydantic import Field

from rfnry_memory.fusion import relative_score, rrf
from rfnry_memory.fusion.relative_score import RelativeScore
from rfnry_memory.fusion.rrf import RRF

Fusion = Annotated[RRF | RelativeScore, Field(discriminator="method")]

METHODS = {"relative_score": relative_score, "rrf": rrf}

__all__ = ["METHODS", "RRF", "Fusion", "RelativeScore", "relative_score", "rrf"]
