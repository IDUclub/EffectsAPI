from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class EffectType(Enum):
  TRANSPORT='Транспорт'
  PROVISION='Обеспеченность'
  CONNECTIVITY='Связность'


class ScaleType(Enum):
  PROJECT='Проект'
  CONTEXT='Контекст'


class ScaleTypeModel(BaseModel):
  scale_type: ScaleType = Field(...)


class ChartData(BaseModel):
  name : str
  before : float
  after : float
  delta : float