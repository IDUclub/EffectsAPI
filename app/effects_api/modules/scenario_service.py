import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from blocksnet.blocks.cutting import cut_urban_blocks, preprocess_urban_objects
from blocksnet.preprocessing.imputing import impute_buildings, impute_services
from loguru import logger

from app.clients.urban_api_client import UrbanAPIClient
from app.common.exceptions.http_exception_wrapper import http_exception
from app.effects_api.constants.const import LIVING_BUILDINGS_ID, ROADS_ID, WATER_ID
from app.effects_api.modules.buildings_service import adapt_buildings
from app.effects_api.modules.functional_sources_service import adapt_functional_zones
from app.effects_api.modules.services_service import adapt_services

SOURCES_PRIORITY = ["PZZ", "OSM", "User"]


def close_gaps(gdf, tolerance):  # taken from momepy
    geom = gdf.geometry.array
    coords = shapely.get_coordinates(geom)
    indices = shapely.get_num_coordinates(geom)

    edges = [0]
    i = 0
    for ind in indices:
        ix = i + ind
        edges.append(ix - 1)
        edges.append(ix)
        i = ix
    edges = edges[:-1]
    points = shapely.points(np.unique(coords[edges], axis=0))

    buffered = shapely.buffer(points, tolerance / 2)
    dissolved = shapely.union_all(buffered)
    exploded = [
        shapely.get_geometry(dissolved, i)
        for i in range(shapely.get_num_geometries(dissolved))
    ]
    centroids = shapely.centroid(exploded)
    snapped = shapely.snap(geom, shapely.union_all(centroids), tolerance)
    return gpd.GeoSeries(snapped, crs=gdf.crs)


class ScenarioService:

    def __init__(self, urban_api_client: UrbanAPIClient):
        self.client = urban_api_client

    async def _get_project_boundaries(
        self, project_id: int, token: str
    ) -> gpd.GeoDataFrame:
        geom = await self.client.get_project_geometry(project_id, token)
        return gpd.GeoDataFrame(geometry=[geom], crs=4326)

    async def _get_scenario_roads(self, scenario_id: int, token: str):
        gdf = await self.client.get_physical_objects_scenario(
            scenario_id, token, physical_object_function_id=ROADS_ID
        )
        if gdf is None:
            return None
        return gdf[["geometry"]].reset_index(drop=True)

    async def _get_scenario_water(self, scenario_id: int, token: str):
        gdf = await self.client.get_physical_objects_scenario(
            scenario_id, token, physical_object_function_id=WATER_ID
        )
        if gdf is None:
            return None
        return gdf[["geometry"]].reset_index(drop=True)

    async def _get_scenario_blocks(
        self,
        user_scenario_id: int,
        base_scenario_id: int,
        boundaries: gpd.GeoDataFrame,
        token: str,
    ) -> gpd.GeoDataFrame:
        crs = boundaries.crs
        boundaries.geometry = boundaries.buffer(-1)

        water = await self._get_scenario_water(user_scenario_id, token)
        if water is not None and not water.empty:
            water = water.to_crs(crs)
            water = water.explode()

        user_roads = await self._get_scenario_roads(user_scenario_id, token)
        if user_roads is not None and not user_roads.empty:
            user_roads = user_roads.to_crs(crs)
            user_roads = user_roads.explode()

        base_roads = await self._get_scenario_roads(base_scenario_id, token)
        if base_roads is not None and not base_roads.empty:
            base_roads = base_roads.to_crs(crs)
            base_roads = base_roads.explode()

        if (
            base_roads is not None
            and not base_roads.empty
            and user_roads is not None
            and not user_roads.empty
        ):
            roads = pd.concat([user_roads, base_roads]).reset_index(drop=True)
            roads.geometry = close_gaps(roads, 1)
            roads = roads.explode(column="geometry")
        else:
            raise http_exception(404, "No objects found for polygons cutting")

        lines, polygons = preprocess_urban_objects(roads, None, water)
        blocks = cut_urban_blocks(boundaries, lines, polygons)
        return blocks

    async def _get_scenario_info(self, scenario_id: int, token: str) -> tuple[int, int]:
        scenario = await self.client.get_scenario(scenario_id, token)
        project_id = scenario["project"]["project_id"]
        project = await self.client.get_project(project_id, token)
        base_scenario_id = project["base_scenario"]["id"]
        return project_id, base_scenario_id

    async def _get_best_functional_zones_source(
        self,
        sources_df: pd.DataFrame,
        source: str | None = None,
        year: int | None = None,
    ) -> tuple[int | None, str | None]:
        """Выбор (year, source) по приоритетам."""
        if source and year:
            row = sources_df.query("source == @source and year == @year")
            if not row.empty:
                return year, source
            year = int(year)
            source = None

        if source and year is None:
            rows = sources_df.query("source == @source")
            if not rows.empty:
                return int(rows["year"].max()), source
            source = None

        if year is not None and source is None:
            for s in SOURCES_PRIORITY:
                row = sources_df.query("source == @s and year == @year")
                if not row.empty:
                    return year, s

        for s in SOURCES_PRIORITY:
            rows = sources_df.query("source == @s")
            if not rows.empty:
                return int(rows["year"].max()), s

    async def get_scenario_blocks(
        self, user_scenario_id: int, token: str
    ) -> gpd.GeoDataFrame:
        project_id, base_scenario_id = await self._get_scenario_info(
            user_scenario_id, token
        )
        project_boundaries = await self._get_project_boundaries(project_id, token)
        crs = project_boundaries.estimate_utm_crs()
        project_boundaries = project_boundaries.to_crs(crs)
        return await self._get_scenario_blocks(
            user_scenario_id, base_scenario_id, project_boundaries, token
        )

    async def get_scenario_functional_zones(
        self,
        scenario_id: int,
        token: str,
        source: str | None = None,
        year: int | None = None,
    ) -> gpd.GeoDataFrame:
        functional_zones = await self.client.get_functional_zones_scenario(
            scenario_id, token, year, source
        )
        functional_zones = functional_zones.loc[
            functional_zones.geometry.geom_type.isin({"Polygon", "MultiPolygon"})
        ].reset_index(drop=True)
        return adapt_functional_zones(functional_zones)

    async def get_scenario_buildings(self, scenario_id: int, token: str):
        try:
            gdf = await self.client.get_physical_objects_scenario(
                scenario_id,
                token,
                physical_object_type_id=LIVING_BUILDINGS_ID,
                centers_only=True,
            )
            if gdf is None:
                return None
            gdf = adapt_buildings(gdf.reset_index(drop=True))
            crs = gdf.estimate_utm_crs()
            return impute_buildings(gdf.to_crs(crs)).to_crs(4326)
        except Exception as e:
            logger.exception(e)
            raise http_exception(
                404,
                f"No buildings found for scenario {scenario_id}",
                _input={"scenario_id": scenario_id},
                _detail={"error": repr(e)},
            ) from e

    async def get_scenario_services(
        self, scenario_id: int, service_types: pd.DataFrame, token: str
    ):
        try:
            gdf = await self.client.get_services_scenario(
                scenario_id, centers_only=True, token=token
            )
            gdf = gdf.to_crs(gdf.estimate_utm_crs())
            gdfs = adapt_services(gdf.reset_index(drop=True), service_types)
            return {st: impute_services(gdf, st) for st, gdf in gdfs.items()}
        except Exception as e:
            logger.exception(e)
            raise http_exception(
                404,
                f"No services found for scenario {scenario_id}",
                _input={"scenario_id": scenario_id},
                _detail={"error": repr(e)},
            ) from e
