import json
import geopandas as gpd
import pandas as pd
import shapely
from typing import Any, Dict
from app.common.exceptions.http_exception_wrapper import http_exception
from app.dependencies import urban_api_handler



class UrbanAPIGateway:
    # TODO context
    @staticmethod
    async def get_physical_objects(
        project_id: int,
        **kwargs: Any
    ) -> gpd.GeoDataFrame:
        res = await urban_api_handler.get(
            f"/api/v1/projects/{project_id}/context/physical_objects_with_geometry",
            params=kwargs
        )
        features = res["features"]
        return (
            gpd.GeoDataFrame.from_features(features, crs=4326)
            .set_index("physical_object_id")
        )

    @staticmethod
    async def get_services(
        project_id: int,
        **kwargs: Any
    ) -> gpd.GeoDataFrame:
        res = await urban_api_handler.get(
            f"/api/v1/projects/{project_id}/context/services_with_geometry",
            params=kwargs
        )
        features = res["features"]
        return (
            gpd.GeoDataFrame.from_features(features, crs=4326)
            .set_index("service_id")
        )

    @staticmethod
    async def get_functional_zones_sources(
        project_id: int
    ) -> pd.DataFrame:
        res = await urban_api_handler.get(
            f"/api/v1/projects/{project_id}/context/functional_zone_sources"
        )
        return pd.DataFrame(res)

    @staticmethod
    async def get_functional_zones(
        project_id: int,
        year:       int,
        source:     int
    ) -> gpd.GeoDataFrame:
        res = await urban_api_handler.get(
            f"/api/v1/projects/{project_id}/context/functional_zones",
            params={"year": year, "source": source}
        )
        features = res["features"]
        return (
            gpd.GeoDataFrame.from_features(features, crs=4326)
            .set_index("functional_zone_id")
        )

    @staticmethod
    async def get_project(project_id: int) -> Dict[str, Any]:
        res = await urban_api_handler.get(f"/api/v1/projects/{project_id}")
        return res

    @staticmethod
    async def get_project_geometry(project_id: int):
        res = await urban_api_handler.get(
            f"/api/v1/projects/{project_id}/territory"
        )
        geometry_json = json.dumps(res["geometry"])
        return shapely.from_geojson(geometry_json)

    # TODO scenario
    @staticmethod
    async def get_scenario(scenario_id: int) -> Dict[str, Any]:
        res = await urban_api_handler.get(f"/api/v1/scenarios/{scenario_id}")
        return res

    @staticmethod
    async def get_functional_zones_sources_scenario(
        scenario_id: int
    ) -> pd.DataFrame:
        res = await urban_api_handler.get(
            f"/api/v1/scenarios/{scenario_id}/functional_zone_sources"
        )
        return pd.DataFrame(res)

    @staticmethod
    async def get_functional_zones_scenario(
        scenario_id: int,
        year:       int,
        source:     str
    ) -> gpd.GeoDataFrame:
        res = await urban_api_handler.get(
            f"/api/v1/scenarios/{scenario_id}/functional_zones",
            params={"year": year, "source": source}
        )
        features = res["features"]
        return (
            gpd.GeoDataFrame.from_features(features, crs=4326)
            .set_index("functional_zone_id")
        )

    @staticmethod
    async def get_physical_objects_scenario(
            scenario_id: int,
            **kwargs: Any
    ) -> gpd.GeoDataFrame:
        params = {
            k: (str(v).lower() if isinstance(v, bool) else v)
            for k, v in kwargs.items()
        }
        res = await urban_api_handler.get(
            f"/api/v1/scenarios/{scenario_id}/physical_objects_with_geometry",
            params=params
        )
        features = res["features"]
        return (
            gpd.GeoDataFrame.from_features(features, crs=4326)
            .set_index("physical_object_id")
        )

    @staticmethod
    async def get_services_scenario(
        scenario_id: int,
        **kwargs: Any
    ) -> gpd.GeoDataFrame:
        res = await urban_api_handler.get(
            f"/api/v1/scenarios/{scenario_id}/services_with_geometry",
            params=kwargs
        )
        features = res["features"]
        return (
            gpd.GeoDataFrame.from_features(features, crs=4326)
            .set_index("service_id")
        )

# Экземпляр для удобства
UrbanAPIGateway = UrbanAPIGateway()