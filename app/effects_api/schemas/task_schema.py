from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel


class TaskStatusSchema(BaseModel):

    task_status: Literal["pending", "success", "error", ]
    error_info: Optional[str] = None

class TaskInfoSchema(BaseModel):

    project_id: int
    base_scenario_id: int
    lust_update: datetime


class TaskSchema(BaseModel):
    task_status: TaskStatusSchema
    target_scenario_id: int
    task_info: Optional[TaskInfoSchema] = None
