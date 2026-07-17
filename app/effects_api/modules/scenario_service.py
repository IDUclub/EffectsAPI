import asyncio

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from blocksnet.blocks.aggregation import aggregate_objects
from blocksnet.blocks.assignment import assign_land_use
from blocksnet.blocks.cutting import cut_urban_blocks, preprocess_urban_objects
from blocksnet.preprocessing.imputing import impute_buildings, impute_services
from loguru import logger

from app.clients.urban_api_client import UrbanAPIClient
from app.common.exceptions.http_exception_wrapper import http_exception
from app.effects_api.constants.const import (
    LAND_USE_RULES,
    LIVING_BUILDINGS_ID,
    ROADS_ID,
    WATER_ID,
)
from app.effects_api.modules.buildings_service import adapt_buildings
from app.effects_api.modules.functional_sources_service import adapt_functional_zones
from app.effects_api.modules.service_type_service import adapt_service_types
from app.effects_api.modules.services_service import adapt_services


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

    async def _get_project_boundaries(self, project_id: int) -> gpd.GeoDataFrame:
        geom = await self.client.get_project_geometry(project_id)
        return gpd.GeoDataFrame(geometry=[geom], crs=4326)

    async def _get_scenario_roads(self, scenario_id: int):
        gdf = await self.client.get_physical_objects_scenario(
            scenario_id, physical_object_function_id=ROADS_ID
        )
        if gdf is None:
            return None
        return gdf[["geometry"]].reset_index(drop=True)

    async def _get_scenario_water(self, scenario_id: int):
        gdf = await self.client.get_physical_objects_scenario(
            scenario_id, physical_object_function_id=WATER_ID
        )
        if gdf is None:
            return None
        return gdf[["geometry"]].reset_index(drop=True)

    async def _get_scenario_blocks(
        self,
        user_scenario_id: int,
        boundaries: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        crs = boundaries.crs
        boundaries.geometry = boundaries.buffer(-1)

        (
            water,
            user_roads,
        ) = await asyncio.gather(
            self._get_scenario_water(user_scenario_id),
            self._get_scenario_roads(user_scenario_id),
        )

        if water is not None and not water.empty:
            water = water.to_crs(crs).explode().reset_index(drop=True)
            water_geoms = ['Polygon', 'MultiPolygon', 'LineString', 'MultiLineString']
            water = water[water.geom_type.isin(water_geoms)].reset_index(drop=True)

        if user_roads is not None and not user_roads.empty:
            user_roads = user_roads.to_crs(crs).explode().reset_index(drop=True)

        if user_roads is not None and not user_roads.empty:
            user_roads.geometry = close_gaps(user_roads, 1)
            roads = user_roads.explode(column="geometry")
            roads_geoms = ['LineString', 'MultiLineString']

            roads = roads[roads.geom_type.isin(roads_geoms)].reset_index(drop=True)

        else:
            roads = gpd.GeoDataFrame(geometry=[], crs=boundaries.crs)
            water = None

        lines, polygons = preprocess_urban_objects(roads, None, water)
        blocks = cut_urban_blocks(boundaries, lines, polygons)
        return blocks

    async def _get_scenario_info(self, scenario_id: int) -> tuple[int, int]:
        scenario = await self.client.get_scenario(scenario_id)
        project_id = scenario["project"]["project_id"]
        project = await self.client.get_project(project_id)
        base_scenario_id = project["base_scenario"]["id"]
        return project_id, base_scenario_id

    async def get_scenario_blocks(
        self, user_scenario_id: int) -> gpd.GeoDataFrame:
        project_id, base_scenario_id = await self._get_scenario_info(
            user_scenario_id)
        project_boundaries = await self._get_project_boundaries(project_id)
        crs = project_boundaries.estimate_utm_crs()
        project_boundaries = project_boundaries.to_crs(crs)
        return await self._get_scenario_blocks(
            user_scenario_id, project_boundaries)

    async def get_scenario_functional_zones(
        self,
        scenario_id: int,
        source: str | None = None,
        year: int | None = None) -> gpd.GeoDataFrame:
        functional_zones = await self.client.get_functional_zones_scenario(
            scenario_id, year, source
        )
        functional_zones = functional_zones.loc[
            functional_zones.geometry.geom_type.isin({"Polygon", "MultiPolygon"})
        ].reset_index(drop=True)
        return adapt_functional_zones(functional_zones)

    async def get_scenario_buildings(self, scenario_id: int):
        try:
            gdf = await self.client.get_physical_objects_scenario(
                scenario_id,
                physical_object_type_id=LIVING_BUILDINGS_ID,
                centers_only=False)
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
        self, scenario_id: int, service_types: pd.DataFrame):
        try:
            res = await self.client.get_services_scenario(
                scenario_id, centers_only=True)
            features = res.get("features") or []

            if not features:
                logger.info(
                    f"Scenario {scenario_id}: no services (features=[]) -> returning empty dict"
                )
                return {}

            gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326").set_index(
                "service_id", drop=False
            )
            gdf = gdf.to_crs(gdf.estimate_utm_crs())

            gdfs = adapt_services(gdf.reset_index(drop=True), service_types)
            return {st: impute_services(g, st) for st, g in gdfs.items()}

        except Exception as e:
            logger.exception(
                f"Failed to fetch/process services for scenario {scenario_id}: {str(e)}"
            )
            raise http_exception(
                404,
                f"No services found for scenario {scenario_id}",
                _input={"scenario_id": scenario_id},
                _detail={"error": repr(e)},
            ) from e

    async def load_blocks_scenario(
        self, scenario_id: int) -> gpd.GeoDataFrame:
        gdf = await self.get_scenario_blocks(scenario_id)
        gdf["site_area"] = gdf.area
        return gdf

    async def assign_land_use_to_blocks_scenario(
        self,
        blocks: gpd.GeoDataFrame,
        scenario_id: int,
        source: str | None,
        year: int | None) -> gpd.GeoDataFrame:
        fzones = await self.get_scenario_functional_zones(
            scenario_id, source, year
        )
        fzones = fzones.to_crs(blocks.crs)
        lu = assign_land_use(blocks, fzones, LAND_USE_RULES)
        return blocks.join(lu.drop(columns=["geometry"]))

    async def enrich_with_buildings_scenario(
        self, blocks: gpd.GeoDataFrame, scenario_id: int) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame | None]:
        buildings = await self.get_scenario_buildings(scenario_id)
        if buildings is None:
            blocks["count_buildings"] = 0
            return blocks, None

        buildings = buildings.to_crs(blocks.crs)
        blocks_bld, _ = aggregate_objects(blocks, buildings)

        blocks = blocks.join(
            blocks_bld.drop(columns=["geometry"]).rename(
                columns={"count": "count_buildings"}
            )
        )
        blocks["count_buildings"] = blocks["count_buildings"].fillna(0).astype(int)
        if "is_living" not in blocks.columns:
            blocks["is_living"] = None
        return blocks, buildings

    async def enrich_with_services_scenario(
        self, blocks: gpd.GeoDataFrame, scenario_id: int) -> gpd.GeoDataFrame:
        stypes = await self.client.get_service_types()
        stypes = await adapt_service_types(stypes, self.client)
        sdict = await self.get_scenario_services(scenario_id, stypes)

        for stype, services in sdict.items():
            services = services.to_crs(blocks.crs)
            b_srv, _ = aggregate_objects(blocks, services)
            b_srv[["capacity", "count"]] = (
                b_srv[["capacity", "count"]].fillna(0).astype(int)
            )
            blocks = blocks.join(
                b_srv.drop(columns=["geometry"]).rename(
                    columns={"capacity": f"capacity_{stype}", "count": f"count_{stype}"}
                )
            )
        return blocks

    async def aggregate_blocks_layer_scenario(
        self,
        scenario_id: int,
        source: str | None = None,
        year: int | None = None) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame | None]:

        logger.info(f"[Scenario {scenario_id}] load blocks")
        blocks = await self.load_blocks_scenario(scenario_id)

        logger.info("Assigning land-use for scenario")
        blocks = await self.assign_land_use_to_blocks_scenario(
            blocks, scenario_id, source, year)

        logger.info("Aggregating buildings for scenario")
        blocks, buildings = await self.enrich_with_buildings_scenario(
            blocks, scenario_id)

        logger.info("Aggregating services for scenario")
        blocks = await self.enrich_with_services_scenario(blocks, scenario_id)

        blocks["is_project"] = True
        logger.success(f"[scenario {scenario_id}] blocks layer ready")

        return blocks, buildings
