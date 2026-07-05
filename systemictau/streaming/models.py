from pydantic import BaseModel, Field
from typing import List

class StreamPayload(BaseModel):
    tenant_id: str = Field(..., description="Unique tenant identifier for data isolation.")
    vector: List[float] = Field(..., description="The multi-dimensional data point for this timestep.")
