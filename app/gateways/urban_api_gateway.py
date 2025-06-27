import json
import geopandas as gpd
import pandas as pd
import shapely
from typing import Any, Dict, Literal

from sqlalchemy.util import await_only

from app.common.api_handlers.json_api_handler import JSONAPIHandler
from app.common.exceptions.http_exception_wrapper import http_exception



class UrbanAPIGateway:

    def __init__(self, base_url: str) -> None:
        self.json_handler = JSONAPIHandler(base_url)
        
    # TODO context
    async def get_physical_objects(
        self,
        project_id: int,
        **kwargs: Any
    ) -> gpd.GeoDataFrame:
        res = await self.json_handler.get(
            f"/api/v1/projects/{project_id}/context/physical_objects_with_geometry",
            params=kwargs
        )
        features = res["features"]
        return (
            gpd.GeoDataFrame.from_features(features, crs=4326)
            .set_index("physical_object_id")
        )

    async def get_services(
            self,
            project_id: int,
            **kwargs: Any
    ) -> gpd.GeoDataFrame:
        res = await self.json_handler.get(
            f"/api/v1/projects/{project_id}/context/services_with_geometry",
            params=kwargs
        )
        features = res["features"]
        return (
            gpd.GeoDataFrame.from_features(features, crs=4326)
            .set_index("service_id")
        )

    async def get_functional_zones_sources(
            self,
            project_id: int
    ) -> pd.DataFrame:
        res = await self.json_handler.get(
            f"/api/v1/projects/{project_id}/context/functional_zone_sources"
        )
        return pd.DataFrame(res)

    async def get_functional_zones(
            self,
            project_id: int,
            year:       int,
            source:     int
    ) -> gpd.GeoDataFrame:
        res = await self.json_handler.get(
            f"/api/v1/projects/{project_id}/context/functional_zones",
            params={"year": year, "source": source}
        )
        features = res["features"]
        return (
            gpd.GeoDataFrame.from_features(features, crs=4326)
            .set_index("functional_zone_id")
        )


    async def get_project(self, project_id: int) -> Dict[str, Any]:
        res = await self.json_handler.get(f"/api/v1/projects/{project_id}")
        return res

    async def get_project_geometry(self, project_id: int):
        res = await self.json_handler.get(
            f"/api/v1/projects/{project_id}/territory"
        )
        geometry_json = json.dumps(res["geometry"])
        return shapely.from_geojson(geometry_json)

    # TODO scenario
    async def get_scenario(self, scenario_id: int) -> Dict[str, Any]:
        res = await self.json_handler.get(f"/api/v1/scenarios/{scenario_id}")
        return res

    async def get_functional_zones_sources_scenario(
            self,
            scenario_id: int,
            token: str,
            source: str,
            year: int
    ) -> pd.DataFrame:
        headers = {"Authorization": f"Bearer {token}"}
        res = await self.json_handler.get(
            f"/api/v1/scenarios/{scenario_id}/functional_zone_sources",
            headers=headers,
            params={
                "year": year,
                "source": source,
            }
        )
        return pd.DataFrame(res)

    async def get_functional_zones_scenario(
            self,
            scenario_id: int,
            year:       int,
            source:     str
    ) -> gpd.GeoDataFrame:
        res = await self.json_handler.get(
            f"/api/v1/scenarios/{scenario_id}/functional_zones",
            params={"year": year, "source": source}
        )
        features = res["features"]
        return (
            gpd.GeoDataFrame.from_features(features, crs=4326)
            .set_index("functional_zone_id")
        )

    async def get_physical_objects_scenario(
            self,
            scenario_id: int,
            **kwargs: Any
    ) -> gpd.GeoDataFrame:
        params = {
            k: (str(v).lower() if isinstance(v, bool) else v)
            for k, v in kwargs.items()
        }
        res = await self.json_handler.get(
            f"/api/v1/scenarios/{scenario_id}/physical_objects_with_geometry",
            params=params
        )
        features = res["features"]
        return (
            gpd.GeoDataFrame.from_features(features, crs=4326)
            .set_index("physical_object_id")
        )

    async def get_services_scenario(
            self,
            scenario_id: int,
            **kwargs: Any
    ) -> gpd.GeoDataFrame:
        res = await self.json_handler.get(
            f"/api/v1/scenarios/{scenario_id}/services_with_geometry",
            params=kwargs
        )
        features = res["features"]
        return (
            gpd.GeoDataFrame.from_features(features, crs=4326)
            .set_index("service_id")
        )

    async def get_optimap_func_zone_request_data(
            self,
            token: str,
            scenario_id: int,
            source: Literal["PZZ", "OSM", "User"] | None,
            year: int | None,
            project: bool = True
    ) -> tuple[str, int]:
        """
        Function retrieves best matching zone sources based on given source and year.
        Args:
            token (str): user token to access data in Urban API.
            scenario_id (int): id of scenario to retrieve data by.
            source (Literal["PZZ", "OSM", "User"] | None): Functional zone source from urban api. If None in order
            User -> PZZ -> OSM priority is retrieved.
            year (int | None): year to retrieve zones for. If None retrieves latest available year.
            project (bool): If True retrieves with User source.
        Returns:
            tuple[str, int]: Tuple with source and year.
        """

        async def get_optimal_source(
                sources_data: pd.DataFrame,
                target_year: int | None,
                is_project: bool
        ) -> tuple[str, int]:
            """
            Function estimates the best source and year
            Args:
                sources_data (pd.DataFrame): DataFrame containing functional zone sources
                target_year (int): year to retrieve zones for. If None retrieves latest available year.
                is_project (bool): If True retrieves with User source
            Returns:
                tuple[str, int]: Tuple with source and year.
            Raises:
                Any from Urban API
                404, if function couldn't match optimal source
            """

            if project:
                sources_priority = ["User", "PZZ", "OSM"]
            else:
                sources_priority = ["PZZ", "OSM"]
            for i in sources_priority:
                if i in sources_data["source"].unique():
                    sources_data = sources_data[sources_data["source"] == i]
                    source_name = sources_data["source"].iloc[0]
                    if year:
                        source_year = sources_data[sources_data["year"] == target_year].iloc[0]
                    else:
                        source_year = sources_data["year"].max()
                    return source_name, source_year
            raise http_exception(
                404,
                "No source found",
                _input={
                    "source": source,
                    "year": year,
                    "is_project": is_project,
                }
            )

        if not project and source == "User":
            raise http_exception(
                500,
                "Unreachable functional zones source for non-project data",
                _input={
                    "source": source,
                    "year": year,
                    "project": project,
                },
                _detail={
                    "available_sources": ["PZZ", "OSM"],
                }
            )
        headers = {"Authorization": f"Bearer {token}"}
        available_sources = await self.json_handler.get(
            f"/api/v1/{scenario_id}/available_sources",
            headers=headers
        )
        sources_df = pd.DataFrame.from_records(available_sources)
        if not source:
            return await get_optimal_source(sources_df, year, project)
        else:
            source_df = sources_df[sources_df["source"] == source]
            return await get_optimal_source(source_df, year, project)


