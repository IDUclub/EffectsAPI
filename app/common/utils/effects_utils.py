import re
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from blocksnet.optimization.services import AreaSolution, Facade
import geopandas as gpd

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

    def clean_number(self, v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        try:
            if isinstance(v, (np.floating, float, np.integer, int)) and not np.isfinite(float(v)):
                return None
        except Exception:
            pass
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        return v

    def build_facade(
        self,
        after_blocks: gpd.GeoDataFrame,
        acc_mx: pd.DataFrame,
        service_types: pd.DataFrame,
    ) -> Facade:
        blocks_lus = after_blocks.loc[after_blocks["is_project"], "land_use"]
        blocks_lus = blocks_lus[~blocks_lus.isna()].to_dict()

        var_adapter = AreaSolution(blocks_lus)

        facade = Facade(
            blocks_lu=blocks_lus,
            blocks_df=after_blocks,
            accessibility_matrix=acc_mx,
            var_adapter=var_adapter,
        )

        for st_id, row in service_types.iterrows():
            st_name = row["name"]
            st_weight = row["infrastructure_weight"]
            st_column = f"capacity_{st_name}"

            if st_column in after_blocks.columns:
                df = after_blocks.rename(columns={st_column: "capacity"})[
                    ["capacity"]
                ].fillna(0)
            else:
                df = after_blocks[[]].copy()
                df["capacity"] = 0
            facade.add_service_type(st_name, st_weight, df)

        return facade
