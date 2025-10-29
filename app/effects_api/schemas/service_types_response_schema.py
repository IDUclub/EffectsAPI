from typing import List, Optional

from pydantic import BaseModel, Field

from app.common.dto.models import ServiceType


class ServiceTypesResponse(BaseModel):
    """
    List of service types available before and/or after scenario transformation.
    Both lists may be present, and `after` may be empty or populated depending on scenario changes.
    """
    before: List[ServiceType] = Field(
        ...,
        description="Service types in the base (before) scenario"
    )
    after: Optional[List[ServiceType]] = Field(
        None,
        description="Service types in the transformed (after) scenario; may be empty or identical to 'before'"
    )

