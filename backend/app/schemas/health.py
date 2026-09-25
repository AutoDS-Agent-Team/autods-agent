from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["healthy"]
    service: Literal["autods-backend"]
    
class ReadinessResponse(HealthResponse):
    dependencies: dict[str, str]

