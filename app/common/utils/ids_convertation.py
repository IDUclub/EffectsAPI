import re
from typing import Any, Dict, Optional

import pandas as pd
from blocksnet.enums import LandUse
from loguru import logger

from app.clients.urban_api_client import UrbanAPIClient


class EffectsUtils:
    def __init__(
        self,
        urban_api_client: UrbanAPIClient,
    ):
        self.__name__ = "EffectsUtils"
        self.urban_api_client = urban_api_client

    def truthy_is_based(self, v: Any) -> bool:
        return v is True or v == 1 or (isinstance(v, str) and v.lower() == "true")

    def parent_id(self, s: Dict[str, Any]) -> Optional[int]:
        p = s.get("parent_scenario")
        return p.get("id") if isinstance(p, dict) else p

    def sid(self, s: Dict[str, Any]) -> Optional[int]:
        try:
            return int(s.get("scenario_id"))
        except Exception:
            return None

    async def resolve_base_id(self, token: str, scenario_id: int) -> int:
        info = await self.urban_api_client.get_scenario_info(scenario_id, token)
        project_id = (info.get("project") or {}).get("project_id")
        regional_id = (info.get("parent_scenario") or {}).get("id")

        if not project_id or not regional_id:
            return scenario_id

        scenarios = await self.urban_api_client.get_project_scenarios(project_id, token)
        matches = [
            s
            for s in scenarios
            if self.truthy_is_based(s.get("is_based"))
               and self.parent_id(s) == regional_id
               and self.sid(s) is not None
        ]
        if not matches:
            only_based = [
                s
                for s in scenarios
                if self.truthy_is_based(s.get("is_based")) and self.sid(s) is not None
            ]
            if not only_based:
                return scenario_id
            matches = only_based

        matches.sort(
            key=lambda x: (x.get("updated_at") is not None, x.get("updated_at")),
            reverse=True,
        )
        return self.sid(matches[0]) or scenario_id

    def coerce_land_use_enum(self, df: pd.DataFrame, col: str = "land_use") -> pd.DataFrame:
        """
        Normalize 'land_use' column to LandUse enum:
        - Accept LandUse enum → keep
        - Accept 'LandUse.NAME' → strip prefix, use NAME
        - Accept 'name' values → use by value ('residential', ...)
        - Accept 'NAME' values → use by name ('RESIDENTIAL', ...)
        - None/NaN → keep None
        Unknown values → None
        """
        if col not in df.columns:
            return df

        def _to_enum(v):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return None
            if isinstance(v, LandUse):
                return v
            if isinstance(v, str):
                s = v.strip()
                m = re.match(r"^(?:LandUse\.)?([A-Za-z_]+)$", s)
                if m:
                    key = m.group(1)
                    try:
                        return LandUse[key.upper()]
                    except KeyError:
                        pass
                    try:
                        return LandUse(key.lower())
                    except ValueError:
                        logger.warning("Unknown land_use value: %r -> set to None", v)
                        return None
            logger.warning("Unsupported land_use type: %r -> set to None", type(v).__name__)
            return None

        df[col] = df[col].map(_to_enum)
        return df
