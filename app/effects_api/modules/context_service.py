import asyncio
from pathlib import Path
from typing import Dict, Tuple

import geopandas as gpd
import pandas as pd
from blocksnet.blocks.aggregation import aggregate_objects
from blocksnet.blocks.assignment import assign_land_use
from blocksnet.blocks.cutting import cut_urban_blocks, preprocess_urban_objects
from blocksnet.preprocessing.imputing import impute_buildings, impute_services
from blocksnet.relations import get_accessibility_context
from loguru import logger

from app.clients.urban_api_client import UrbanAPIClient
from app.common.caching.caching_service import FileCache
from app.common.utils.geodata import get_best_functional_zones_source
from app.effects_api.constants.const import (
    LAND_USE_RULES,
    LIVING_BUILDINGS_ID,
    ROADS_ID,
    SOCIAL_INDICATORS_MAPPING,
    WATER_ID,
)
from app.effects_api.modules.buildings_service import adapt_buildings
from app.effects_api.modules.functional_sources_service import adapt_functional_zones
from app.effects_api.modules.scenario_service import close_gaps
from app.effects_api.modules.service_type_service import (
    adapt_service_types,
    adapt_social_service_types_df,
)
from app.effects_api.modules.services_service import adapt_services


class ContextService:
    """Context layer orchestration (blocks, buildings, services, fzones)."""

    def __init__(self, urban_api_client: UrbanAPIClient, cache: FileCache):
        self.client = urban_api_client
        self.cache = cache

    async def _get_project_boundaries(self, project_id: int) -> gpd.GeoDataFrame:
        """Return project boundary polygon as GeoDataFrame (EPSG:4326)."""
        geom = await self.client.get_project_geometry(project_id)
        return gpd.GeoDataFrame(geometry=[geom], crs=4326)

    async def _get_context_boundaries(self, project_id: int) -> gpd.GeoDataFrame:
        """Return union of context territories as GeoDataFrame (EPSG:4326)."""
        project = await self.client.get_project(project_id)
        context_ids = project["properties"]["context"]
        geometries = [
            await self.client.get_territory_geometry(tid) for tid in context_ids
        ]
        return gpd.GeoDataFrame(geometry=geometries, crs=4326)

    async def _get_context_roads(self, scenario_id: int) -> gpd.GeoDataFrame | None:
        """Return roads geometry for context cut (only geometry column)."""
        gdf = await self.client.get_physical_objects(
            scenario_id, physical_object_function_id=ROADS_ID
        )
        if gdf is None:
            return None
        return gdf[["geometry"]].reset_index(drop=True)

    async def _get_context_water(self, scenario_id: int) -> gpd.GeoDataFrame | None:
        """Return water geometry for context cut (only geometry column)."""
        gdf = await self.client.get_physical_objects(
            scenario_id, physical_object_function_id=WATER_ID
        )
        if gdf is None:
            return None
        return gdf[["geometry"]].reset_index(drop=True)

    async def _get_context_blocks(
        self, scenario_id: int, boundaries: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """Construct context blocks by cutting boundaries with roads/water."""
        crs = boundaries.crs
        boundaries.geometry = boundaries.buffer(-1)

        water, roads = await asyncio.gather(
            self._get_context_water(scenario_id),
            self._get_context_roads(scenario_id),
        )

        if water is not None and not water.empty:
            water = water.to_crs(crs).explode().reset_index(drop=True)
            water_geoms = ['Polygon', 'MultiPolygon', 'LineString', 'MultiLineString']
            water = water[water.geom_type.isin(water_geoms)].reset_index(drop=True)

        if roads is not None and not roads.empty:
            roads = roads.to_crs(crs).explode().reset_index(drop=True)
            roads.geometry = close_gaps(roads, 1)
            roads = roads.explode(column="geometry")
            roads_geoms = ['LineString', 'MultiLineString']

            roads = roads[roads.geom_type.isin(roads_geoms)].reset_index(drop=True)
        else:
            roads = gpd.GeoDataFrame(geometry=[], crs=boundaries.crs)
            water = None

        lines, polygons = preprocess_urban_objects(roads, None, water.reset_index(drop=True))
        blocks = cut_urban_blocks(boundaries, lines, polygons)
        return blocks

    async def get_context_blocks(
        self, project_id: int, scenario_id: int
    ) -> gpd.GeoDataFrame:
        """
        Build context blocks (outside project boundary but inside context territories).
        """
        project_boundaries, context_boundaries = await asyncio.gather(
            self._get_project_boundaries(project_id),
            self._get_context_boundaries(project_id),
        )

        crs = context_boundaries.estimate_utm_crs()
        context_boundaries = context_boundaries.to_crs(crs)
        project_boundaries = project_boundaries.to_crs(crs)

        context_boundaries = context_boundaries.overlay(
            project_boundaries, how="difference"
        )
        return await self._get_context_blocks(scenario_id, context_boundaries)

    async def get_context_functional_zones(
        self, scenario_id: int, source: str | None, year: int | None
    ) -> gpd.GeoDataFrame:
        """
        Fetch + adapt functional zones for context by best source/year if not given.
        """
        sources_df = await self.client.get_functional_zones_sources(scenario_id)
        year, source = await get_best_functional_zones_source(sources_df, source, year)
        functional_zones = await self.client.get_functional_zones(
            scenario_id, year, source
        )
        functional_zones = functional_zones.loc[
            functional_zones.geometry.geom_type.isin({"Polygon", "MultiPolygon"})
        ].reset_index(drop=True)
        return adapt_functional_zones(functional_zones)

    async def get_context_buildings(self, scenario_id: int) -> gpd.GeoDataFrame | None:
        """
        Fetch, adapt and impute living buildings for context.
        Returns EPSG:4326 GeoDataFrame or None if not found.
        """
        gdf = await self.client.get_physical_objects(
            scenario_id, physical_object_type_id=LIVING_BUILDINGS_ID, centers_only=True
        )
        if gdf is None or gdf.empty:
            return None

        gdf = adapt_buildings(gdf.reset_index(drop=True))
        crs = gdf.estimate_utm_crs()
        return impute_buildings(gdf.to_crs(crs)).to_crs(4326)

    async def get_context_services(
        self, scenario_id: int, service_types: pd.DataFrame
    ) -> Dict[str, gpd.GeoDataFrame]:
        """
        Fetch and adapt services by service type (dict of GeoDataFrames).
        """
        gdf = await self.client.get_services(scenario_id, centers_only=True)
        gdf = gdf.to_crs(gdf.estimate_utm_crs())
        gdfs = adapt_services(gdf.reset_index(drop=True), service_types)
        return {st: impute_services(gdf, st) for st, gdf in gdfs.items()}

    async def get_context_territories(self, project_id: int) -> gpd.GeoDataFrame:
        """
        Return context territories as polygons with column 'parent' = territory_id (EPSG:4326).
        """
        project = await self.client.get_all_project_info(project_id)
        context_ids = project["properties"]["context"]
        data = [
            {
                "parent": territory_id,
                "geometry": await self.client.get_territory_geometry(territory_id),
            }
            for territory_id in context_ids
        ]
        return gpd.GeoDataFrame(data=data, crs=4326)

    async def load_context_blocks(
        self, scenario_id: int
    ) -> Tuple[gpd.GeoDataFrame, int]:
        """
        Load raw context blocks and compute site_area.
        """
        project_id = await self.client.get_project_id(scenario_id)
        blocks = await self.get_context_blocks(project_id, scenario_id)
        blocks["site_area"] = blocks.area
        return blocks, project_id

    async def assign_land_use_context(
        self,
        blocks: gpd.GeoDataFrame,
        scenario_id: int,
        source: str | None,
        year: int | None,
    ) -> gpd.GeoDataFrame:
        """
        Assign land use to blocks via functional zones and LAND_USE_RULES.
        """
        fzones = await self.get_context_functional_zones(scenario_id, source, year)
        fzones = fzones.to_crs(blocks.crs)
        lu = assign_land_use(blocks, fzones, LAND_USE_RULES)
        return blocks.join(lu.drop(columns=["geometry"]))

    async def enrich_with_context_buildings(
        self, blocks: gpd.GeoDataFrame, scenario_id: int
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame | None]:
        """
        Aggregate living buildings on blocks (count_buildings), keep 'is_living' column.
        """
        buildings = await self.get_context_buildings(scenario_id)
        if buildings is None:
            blocks["count_buildings"] = 0
            blocks["is_living"] = None
            return blocks, None

        buildings = buildings.to_crs(blocks.crs)
        agg, _ = aggregate_objects(blocks, buildings)

        blocks = blocks.join(
            agg.drop(columns=["geometry"]).rename(columns={"count": "count_buildings"})
        )
        blocks["count_buildings"] = blocks["count_buildings"].fillna(0).astype(int)
        if "is_living" not in blocks.columns:
            blocks["is_living"] = None

        return blocks, buildings

    async def enrich_with_context_services(
        self, blocks: gpd.GeoDataFrame, scenario_id: int
    ) -> gpd.GeoDataFrame:
        """
        Aggregate services on blocks: add capacity_{st} / count_{st} columns.
        """
        stypes = await self.client.get_service_types()
        stypes = await adapt_service_types(stypes, self.client)

        sdict = await self.get_context_services(scenario_id, stypes)
        if not sdict:
            logger.info(
                f"No context services to aggregate for scenario_id={scenario_id}"
            )
            return blocks

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

    async def aggregate_blocks_layer_context(
        self, scenario_id: int, source: str | None = None, year: int | None = None
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame | None]:
        """
        Build full context blocks layer:
          1) load blocks
          2) assign land use
          3) enrich with buildings
          4) enrich with services
        """
        logger.info(f"[Context {scenario_id}] load blocks")
        blocks, _project_id = await self.load_context_blocks(scenario_id)

        logger.info("Assigning land-use for context")
        blocks = await self.assign_land_use_context(blocks, scenario_id, source, year)

        logger.info("Aggregating buildings for context")
        blocks, buildings = await self.enrich_with_context_buildings(
            blocks, scenario_id
        )

        logger.info("Aggregating services for context")
        blocks = await self.enrich_with_context_services(blocks, scenario_id)

        logger.success(f"[Context {scenario_id}] blocks layer ready", scenario_id)
        return blocks, buildings

    async def get_accessibility_context(
        self, blocks: pd.DataFrame, acc_mx: pd.DataFrame, accessibility: float
    ) -> list[int]:
        blocks["population"] = blocks["population"].fillna(0)
        project_blocks = blocks.copy()
        context_blocks = get_accessibility_context(
            acc_mx, project_blocks, accessibility, out=False, keep=True
        )
        return list(context_blocks.index)

    async def get_shared_context(
        self, project_id: int
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame]:
        """
        Get cached context (blocks, territories, service_types: artifacts) by project_id,
        or build and cache if missing/corrupted.

        JSON cache stores only paths to artifacts. If any artifact file is missing,
        the cache is treated as stale and context is recomputed.
        """
        method = "shared_context"
        params = {"project_id": int(project_id)}
        phash = self.cache.params_hash(params)

        cached = self.cache.load(method, project_id, phash)
        if cached:
            logger.info(f"Shared context cache hit for project_id={project_id}")
            data = cached["data"]

            try:
                ctx_blocks = self.cache.load_gdf_artifact(Path(data["context_blocks_path"]))
                ctx_territories = self.cache.load_gdf_artifact(Path(data["context_territories_path"]))
                service_types = self.cache.load_df_artifact(Path(data["service_types_path"]))
                return ctx_blocks, ctx_territories, service_types

            except (FileNotFoundError, OSError, KeyError) as exc:
                # KeyError — если в JSON вдруг нет нужного ключа
                logger.warning(
                    f"Shared context cache is corrupted/stale for project_id={project_id}. "
                    f"Rebuilding. Reason: {exc}"
                )
                # optional: если у тебя есть метод точечной инвалидции:
                # self.cache.invalidate(method, project_id, phash)

        logger.info(f"Shared context cache miss for project_id={project_id} — is building")

        territory_id = (await self.client.get_all_project_info(project_id))[
            "territory"
        ]["id"]
        base_sid = await self.client.get_base_scenario_id(project_id)
        ctx_src, ctx_year = await self.client.get_optimal_func_zone_request_data(
            data_id=base_sid, source=None, year=None, project=False
        )

        normatives = (await self.client.get_territory_normatives(territory_id))[
            [
                "radius_availability_meters",
                "time_availability_minutes",
                "services_per_1000_normative",
                "services_capacity_per_1000_normative",
            ]
        ].copy()

        service_types = await self.client.get_service_types()
        service_types = await adapt_service_types(service_types, self.client)
        service_types = service_types[service_types["infrastructure_type"].notna()].copy()
        service_types = adapt_social_service_types_df(
            service_types, SOCIAL_INDICATORS_MAPPING
        ).join(normatives)

        ctx_blocks, _ = await self.aggregate_blocks_layer_context(
            base_sid, ctx_src, ctx_year
        )
        ctx_territories = await self.get_context_territories(project_id)

        ctx_blocks_path = self.cache.save_gdf_artifact(
            ctx_blocks,
            method=method,
            owner_id=project_id,
            params=params,
            name="context_blocks",
            fmt="pkl",
        )
        ctx_territories_path = self.cache.save_gdf_artifact(
            ctx_territories,
            method=method,
            owner_id=project_id,
            params=params,
            name="context_territories",
            fmt="parquet",
        )
        service_types_path = self.cache.save_df_artifact(
            service_types,
            method=method,
            owner_id=project_id,
            params=params,
            name="service_types",
            fmt="parquet",
        )

        self.cache.save(
            method,
            project_id,
            params,
            {
                "context_blocks_path": str(ctx_blocks_path),
                "context_territories_path": str(ctx_territories_path),
                "service_types_path": str(service_types_path),
            },
        )
        return ctx_blocks, ctx_territories, service_types
