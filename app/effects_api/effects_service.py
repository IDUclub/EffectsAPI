import json
from typing import Any, Dict, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from blocksnet.analysis.indicators import calculate_development_indicators
from blocksnet.analysis.provision import competitive_provision, provision_strong_total
from blocksnet.blocks.aggregation import aggregate_objects
from blocksnet.blocks.assignment import assign_land_use
from blocksnet.config import service_types_config
from blocksnet.enums import LandUse
from blocksnet.machine_learning.regression import DensityRegressor, SocialRegressor
from blocksnet.optimization.services import (
    AreaSolution,
    Facade,
    SimpleChooser,
    TPEOptimizer,
    WeightedConstraints,
    WeightedObjective,
)
from blocksnet.relations import (
    calculate_accessibility_matrix,
    generate_adjacency_graph,
    get_accessibility_context,
    get_accessibility_graph,
)
from loguru import logger

from app.effects_api.modules.scenario_service import ScenarioService
from app.effects_api.modules.service_type_service import (
    adapt_service_types,
    build_en_to_ru_map,
    remap_properties_keys_in_geojson,
)

from ..clients.urban_api_client import UrbanAPIClient
from ..common.caching.caching_service import FileCache
from ..common.dto.models import SourceYear
from ..common.exceptions.http_exception_wrapper import http_exception
from ..common.utils.geodata import fc_to_gdf, gdf_to_ru_fc_rounded, is_fc, round_coords
from .constants.const import (
    INFRASTRUCTURES_WEIGHTS,
    LAND_USE_RULES,
    MAX_EVALS,
    MAX_RUNS,
)
from .dto.development_dto import (
    ContextDevelopmentDTO,
    DevelopmentDTO,
)
from .dto.socio_economic_project_dto import (
    SocioEconomicByProjectComputedDTO,
    SocioEconomicByProjectDTO,
)
from .dto.socio_economic_scenario_dto import SocioEconomicByScenarioDTO
from .dto.transformation_effects_dto import TerritoryTransformationDTO
from .modules.context_service import (
    get_context_blocks,
    get_context_buildings,
    get_context_functional_zones,
    get_context_services,
)
from .schemas.development_response_schema import DevelopmentResponseSchema
from .schemas.socio_economic_response_schema import (
    SocioEconomicResponseSchema,
    SocioEconomicSchema,
)


class EffectsService:
    def __init__(
        self,
        urban_api_client: UrbanAPIClient,
        cache: FileCache,
        scenario_service: ScenarioService,
    ):
        self.__name__ = "EffectsService"
        self.bn_social_regressor: SocialRegressor = SocialRegressor()
        self.urban_api_client = urban_api_client
        self.cache = cache
        self.scenario = scenario_service

    async def build_hash_params(
        self,
        params: ContextDevelopmentDTO | DevelopmentDTO,
        token: str,
    ) -> dict:
        project_id = (
            await self.urban_api_client.get_scenario_info(params.scenario_id, token)
        )["project"]["project_id"]
        base_scenario_id = await self.urban_api_client.get_base_scenario_id(project_id)
        base_src, base_year = (
            await self.urban_api_client.get_optimal_func_zone_request_data(
                token, base_scenario_id, None, None
            )
        )
        p = params.model_dump()
        p.pop("force", None)
        return p | {
            "base_func_zone_source": base_src,
            "base_func_zone_year": base_year,
        }

    async def get_optimal_func_zone_data(
        self,
        params: (
            DevelopmentDTO
            | ContextDevelopmentDTO
            | SocioEconomicByProjectDTO
            | TerritoryTransformationDTO
        ),
        token: str,
    ) -> DevelopmentDTO:
        """
        Get optimal functional zone source and year for the project scenario.
        If not provided, fetches the best available source and year.

        Params:
            params (DevelopmentDTO): DTO with scenario ID and optional
        Returns:
            DevelopmentDTO: DTO with updated functional zone source and year.
        """

        if not params.proj_func_zone_source or not params.proj_func_source_year:
            (params.proj_func_zone_source, params.proj_func_source_year) = (
                await self.urban_api_client.get_optimal_func_zone_request_data(
                    token,
                    params.scenario_id,
                    params.proj_func_zone_source,
                    params.proj_func_source_year,
                )
            )
            if isinstance(params, ContextDevelopmentDTO):
                if (
                    not params.context_func_zone_source
                    or not params.context_func_source_year
                ):
                    (
                        params.context_func_zone_source,
                        params.context_func_source_year,
                    ) = await self.urban_api_client.get_optimal_func_zone_request_data(
                        token,
                        params.scenario_id,
                        params.context_func_zone_source,
                        params.context_func_source_year,
                        project=False,
                    )
            return params
        return params

    async def load_blocks_scenario(
        self, scenario_id: int, token: str
    ) -> gpd.GeoDataFrame:
        gdf = await self.scenario.get_scenario_blocks(scenario_id, token)
        gdf["site_area"] = gdf.area
        return gdf

    async def assign_land_use_to_blocks_scenario(
        self,
        blocks: gpd.GeoDataFrame,
        scenario_id: int,
        source: str | None,
        year: int | None,
        token: str,
    ) -> gpd.GeoDataFrame:
        fzones = await self.scenario.get_scenario_functional_zones(
            scenario_id, token, source, year
        )
        fzones = fzones.to_crs(blocks.crs)
        lu = assign_land_use(blocks, fzones, LAND_USE_RULES)
        return blocks.join(lu.drop(columns=["geometry"]))

    async def enrich_with_buildings_scenario(
        self, blocks: gpd.GeoDataFrame, scenario_id: int, token: str
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame | None]:
        buildings = await self.scenario.get_scenario_buildings(scenario_id, token)
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
        self, blocks: gpd.GeoDataFrame, scenario_id: int, token: str
    ) -> gpd.GeoDataFrame:
        stypes = await self.urban_api_client.get_service_types()
        stypes = await adapt_service_types(stypes, self.urban_api_client)
        sdict = await self.scenario.get_scenario_services(scenario_id, stypes, token)

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
        year: int | None = None,
        token: str | None = None,
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame | None]:

        logger.info(f"[Scenario {scenario_id}] load blocks")
        blocks = await self.load_blocks_scenario(scenario_id, token)

        logger.info("Assigning land-use for scenario")
        blocks = await self.assign_land_use_to_blocks_scenario(
            blocks, scenario_id, source, year, token
        )

        logger.info("Aggregating buildings for scenario")
        blocks, buildings = await self.enrich_with_buildings_scenario(
            blocks, scenario_id, token
        )

        logger.info("Aggregating services for scenario")
        blocks = await self.enrich_with_services_scenario(blocks, scenario_id, token)

        blocks["is_project"] = True
        logger.success(f"[scenario {scenario_id}] blocks layer ready")

        return blocks, buildings

    async def load_context_blocks(
        self, scenario_id: int, token: str
    ) -> tuple[gpd.GeoDataFrame, int]:
        project_id = await self.urban_api_client.get_project_id(scenario_id, token)
        blocks = await get_context_blocks(
            project_id, scenario_id, token, self.urban_api_client
        )
        blocks["site_area"] = blocks.area
        return blocks, project_id

    async def assign_land_use_context(
        self,
        blocks: gpd.GeoDataFrame,
        scenario_id: int,
        source: str | None,
        year: int | None,
        token: str,
    ) -> gpd.GeoDataFrame:
        fzones = await get_context_functional_zones(
            scenario_id, source, year, token, self.urban_api_client
        )
        fzones = fzones.to_crs(blocks.crs)
        lu = assign_land_use(blocks, fzones, LAND_USE_RULES)
        return blocks.join(lu.drop(columns=["geometry"]))

    async def enrich_with_context_buildings(
        self, blocks: gpd.GeoDataFrame, scenario_id: int, token: str
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame | None]:

        buildings = await get_context_buildings(
            scenario_id, token, self.urban_api_client
        )
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
        self, blocks: gpd.GeoDataFrame, scenario_id: int, token: str
    ) -> gpd.GeoDataFrame:

        stypes = await self.urban_api_client.get_service_types()
        stypes = await adapt_service_types(stypes, self.urban_api_client)
        sdict = await get_context_services(
            scenario_id, stypes, token, self.urban_api_client
        )

        for stype, services in sdict.items():
            services = services.to_crs(blocks.crs)
            b_srv, _ = aggregate_objects(blocks, services)
            b_srv[["capacity", "count"]] = (
                b_srv[["capacity", "count"]].fillna(0).astype(int)
            )

            blocks = blocks.join(
                b_srv.drop(columns=["geometry"]).rename(
                    columns={
                        "capacity": f"capacity_{stype}",
                        "count": f"count_{stype}",
                    }
                )
            )
        return blocks

    async def aggregate_blocks_layer_context(
        self,
        scenario_id: int,
        source: str | None = None,
        year: int | None = None,
        token: str | None = None,
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame | None]:

        logger.info(f"[Context {scenario_id}] load blocks")
        blocks, project_id = await self.load_context_blocks(scenario_id, token)

        logger.info("Assigning land-use for context")
        blocks = await self.assign_land_use_context(
            blocks, scenario_id, source, year, token
        )

        logger.info("Aggregating buildings for context")
        blocks, buildings = await self.enrich_with_context_buildings(
            blocks, scenario_id, token
        )

        logger.info("Aggregating services for context")
        blocks = await self.enrich_with_context_services(blocks, scenario_id, token)

        logger.success(f"[Context {scenario_id}] blocks layer ready")
        return blocks, buildings

    async def get_services_layer(self, scenario_id: int, token: str):
        """
        Fetch every service layer for a scenario, aggregate counts/capacities
        into the scenario blocks and return the resulting block layer.

        Params:
        scenario_id : int
            Scenario whose services are queried and aggregated.

        Returns:
        gpd.GeoDataFrame
            Scenario block layer with additional columns
            `capacity_<service_type>` and `count_<service_type>` for each
            detected service category.
        """
        blocks = await self.scenario.get_scenario_blocks(scenario_id, token)
        blocks_crs = blocks.crs
        logger.info(
            f"{len(blocks)} START blocks layer scenario{scenario_id}, CRS: {blocks.crs}"
        )
        service_types = await self.urban_api_client.get_service_types()
        logger.info(f"{service_types}")
        services_dict = await self.scenario.get_scenario_services(
            scenario_id, service_types, token
        )

        for service_type, services in services_dict.items():
            services = services.to_crs(blocks_crs)
            blocks_services, _ = aggregate_objects(blocks, services)
            blocks_services["capacity"] = (
                blocks_services["capacity"].fillna(0).astype(int)
            )
            blocks_services["objects_count"] = (
                blocks_services["objects_count"].fillna(0).astype(int)
            )
            blocks = blocks.join(
                blocks_services.drop(columns=["geometry"]).rename(
                    columns={
                        "capacity": f"capacity_{service_type}",
                        "objects_count": f"count_{service_type}",
                    }
                )
            )
        logger.info(
            f"{len(blocks)} SERVICES blocks layer scenario {scenario_id}, CRS: {blocks.crs}"
        )
        return blocks

    @staticmethod
    async def run_development_parameters(
        blocks_gdf: gpd.GeoDataFrame,
    ) -> pd.DataFrame:
        """
        Compute core *development* indicators (FSI, GSI, MXI, etc.) for each
        block and derive population estimates.

        The routine:
        1. Clips every land-use share to [0, 1].
        2. Generates an adjacency graph (10 m tolerance).
        3. Uses DensityRegressor to predict density indices.
        4. Converts indices into built-area, footprint, living area, etc.
        5. Estimates population by living_area // 20.

        Params:
        blocks_gdf : gpd.GeoDataFrame
            Block layer already containing per-land-use **shares**
            (0 ≤ share ≤ 1) and `site_area`.

        Returns:
        pd.DataFrame with added columns:
            `build_floor_area`, `footprint_area`, `living_area`,
            `non_living_area`, `population`, plus the original density indices.
        """
        for lu in LandUse:
            blocks_gdf[lu.value] = blocks_gdf[lu.value].apply(lambda v: min(v, 1))

        adjacency_graph = generate_adjacency_graph(blocks_gdf, 10)
        dr = DensityRegressor()

        density_df = dr.evaluate(blocks_gdf, adjacency_graph)
        density_df.loc[density_df["fsi"] < 0, "fsi"] = 0

        density_df.loc[density_df["gsi"] < 0, "gsi"] = 0
        density_df.loc[density_df["gsi"] > 1, "gsi"] = 1

        density_df.loc[density_df["mxi"] < 0, "mxi"] = 0
        density_df.loc[density_df["mxi"] > 1, "mxi"] = 1

        density_df.loc[blocks_gdf["residential"] == 0, "mxi"] = 0
        density_df["site_area"] = blocks_gdf["site_area"]

        development_df = calculate_development_indicators(density_df)
        development_df["population"] = development_df["living_area"] // 20

        return development_df

    async def run_social_reg_prediction(
        self,
        blocks: gpd.GeoDataFrame,
        data_input: pd.DataFrame,
    ):
        """
        Function runs social regression from blocksnet
        Args:
            blocks (gpd.GeoDataFrame): Block layer already containing per-land-use **shares**
            data_input (pd.DataFrame): Data to run regression on
        Returns:
            SocioEconomicSchema: SocioEconomicSchema from schemas to return result generation
        """

        data_input["latitude"] = blocks.geometry.union_all().centroid.x
        data_input["longitude"] = blocks.geometry.union_all().centroid.y
        data_input["buildings_count"] = data_input["count_buildings"]
        y_pred, pi_lower, pi_upper = self.bn_social_regressor.evaluate(data_input)
        iloc = 0
        result_data = {
            "pred": y_pred.apply(round).astype(int).iloc[iloc].to_dict(),
            "lower": pi_lower.iloc[iloc].to_dict(),
            "upper": pi_upper.iloc[iloc].to_dict(),
        }
        result_df = pd.DataFrame.from_dict(result_data)
        result_df["is_interval"] = (result_df["pred"] <= result_df["upper"]) & (
            result_df["pred"] >= result_df["lower"]
        )
        res = result_df.to_dict(orient="index")
        return SocioEconomicSchema(socio_economic_prediction=res)

    async def evaluate_master_plan_by_project(
        self, params: SocioEconomicByProjectDTO, token: str
    ) -> SocioEconomicResponseSchema:
        logger.info(
            f"[Effects] project mode: project_id={params.project_id}, regional_scenario_id={params.regional_scenario_id}"
        )

        project_info = await self.urban_api_client.get_all_project_info(
            params.project_id, token
        )
        context_territories = project_info.get("properties", {}).get("context") or []

        base_sid = await self.urban_api_client.get_base_scenario_id(params.project_id)
        ctx_src, ctx_year = (
            await self.urban_api_client.get_optimal_func_zone_request_data(
                token=token, data_id=base_sid, source=None, year=None, project=False
            )
        )
        context_blocks, _ = await self.aggregate_blocks_layer_context(
            base_sid, ctx_src, ctx_year, token
        )

        context_split: Optional[Dict[int, SocioEconomicSchema]] = None
        if params.split and context_territories:
            context_split = {}
            for tid in context_territories:
                territory = gpd.GeoDataFrame(
                    geometry=[await self.urban_api_client.get_territory_geometry(tid)],
                    crs=4326,
                )
                ter_blocks = (
                    context_blocks.sjoin(
                        territory.to_crs(territory.estimate_utm_crs()), how="left"
                    )
                    .dropna(subset="index_right")
                    .drop(columns="index_right")
                )
                ter_data = [
                    ter_blocks.drop(columns=["land_use", "geometry"]).sum().to_dict()
                ]
                ter_input = pd.DataFrame(ter_data)
                context_split[tid] = await self.run_social_reg_prediction(
                    ter_blocks, ter_input
                )

        scenarios = await self.urban_api_client.get_project_scenarios(
            params.project_id, token
        )
        target = [
            s
            for s in scenarios
            if (s.get("parent_scenario") or {}).get("id") == params.regional_scenario_id
        ]
        logger.info(
            f"[Effects] matched {len(target)} scenarios in project {params.project_id} (parent={params.regional_scenario_id})"
        )

        landuse_cols = [
            "residential",
            "business",
            "recreation",
            "industrial",
            "transport",
            "special",
            "agriculture",
        ]
        results_by_scenario: Dict[int, Dict[str, Any]] = {}
        project_sources: Dict[int, SourceYear] = {}

        for s in target:
            sid = s["scenario_id"]
            proj_src, proj_year = (
                await self.urban_api_client.get_optimal_func_zone_request_data(
                    token=token, data_id=sid, source=None, year=None, project=True
                )
            )
            project_sources[sid] = SourceYear(source=proj_src, year=proj_year)
            scenario_blocks, _ = await self.aggregate_blocks_layer_scenario(
                sid, proj_src, proj_year, token
            )
            scenario_blocks = scenario_blocks.to_crs(context_blocks.crs)

            blocks = gpd.GeoDataFrame(
                pd.concat([context_blocks, scenario_blocks], ignore_index=True),
                crs=context_blocks.crs,
            )
            blocks[landuse_cols] = blocks[landuse_cols].clip(upper=1)
            development_df = await self.run_development_parameters(blocks)

            add_cols = [
                "build_floor_area",
                "footprint_area",
                "living_area",
                "non_living_area",
                "population",
            ]
            blocks[add_cols] = development_df[add_cols].values

            for lu in LandUse:
                blocks[lu.value] = blocks[lu.value] * blocks["site_area"]

            main_data = [blocks.drop(columns=["land_use", "geometry"]).sum().to_dict()]
            main_input = pd.DataFrame(main_data)
            main_res: SocioEconomicSchema = await self.run_social_reg_prediction(
                blocks, main_input
            )

            results_by_scenario[sid] = main_res.socio_economic_prediction

        computed_params = SocioEconomicByProjectComputedDTO(
            project_id=params.project_id,
            regional_scenario_id=params.regional_scenario_id,
            split=params.split,
            context_func_zone_source=ctx_src,
            context_func_source_year=ctx_year,
            project_sources=project_sources,
        )

        return SocioEconomicResponseSchema(
            socio_economic_prediction=results_by_scenario,
            split_prediction=context_split or None,
            params_data=computed_params,
        )

    async def evaluate_master_plan_by_scenario(
        self, params: SocioEconomicByScenarioDTO, token: str
    ) -> SocioEconomicResponseSchema:
        sid = params.scenario_id
        logger.info(f"[Effects] legacy mode: scenario_id={sid}")

        project_id = await self.urban_api_client.get_project_id(sid, token)
        project_info = await self.urban_api_client.get_all_project_info(
            project_id, token
        )
        context_territories = project_info.get("properties", {}).get("context") or []
        params = await self.get_optimal_func_zone_data(params, token)

        context_blocks, _ = await self.aggregate_blocks_layer_context(
            sid, params.context_func_zone_source, params.proj_func_source_year, token
        )

        scenario_blocks, _ = await self.aggregate_blocks_layer_scenario(
            sid, params.proj_func_zone_source, params.proj_func_source_year, token
        )

        scenario_blocks = scenario_blocks.to_crs(context_blocks.crs)

        blocks = gpd.GeoDataFrame(
            pd.concat([context_blocks, scenario_blocks], ignore_index=True),
            crs=context_blocks.crs,
        )

        landuse_cols = [
            "residential",
            "business",
            "recreation",
            "industrial",
            "transport",
            "special",
            "agriculture",
        ]
        blocks[landuse_cols] = blocks[landuse_cols].clip(upper=1)
        development_df = await self.run_development_parameters(blocks)

        add_cols = [
            "build_floor_area",
            "footprint_area",
            "living_area",
            "non_living_area",
            "population",
        ]
        blocks[add_cols] = development_df[add_cols].values

        for lu in LandUse:
            blocks[lu.value] = blocks[lu.value] * blocks["site_area"]

        main_data = [blocks.drop(columns=["land_use", "geometry"]).sum().to_dict()]
        main_input = pd.DataFrame(main_data)
        main_res: SocioEconomicSchema = await self.run_social_reg_prediction(
            blocks, main_input
        )

        split_results: Optional[Dict[int, SocioEconomicSchema]] = None
        if params.split and context_territories:
            split_results = {}
            for tid in context_territories:
                territory = gpd.GeoDataFrame(
                    geometry=[await self.urban_api_client.get_territory_geometry(tid)],
                    crs=4326,
                )
                ter_blocks = (
                    blocks.sjoin(
                        territory.to_crs(territory.estimate_utm_crs()), how="left"
                    )
                    .dropna(subset="index_right")
                    .drop(columns="index_right")
                )
                ter_data = [
                    ter_blocks.drop(columns=["land_use", "geometry"]).sum().to_dict()
                ]
                ter_input = pd.DataFrame(ter_data)
                split_results[tid] = await self.run_social_reg_prediction(
                    ter_blocks, ter_input
                )

        return SocioEconomicResponseSchema(
            socio_economic_prediction={sid: main_res.socio_economic_prediction},
            split_prediction=split_results or None,
            params_data=params,
        )

    async def calc_project_development(
        self, token: str, params: DevelopmentDTO
    ) -> DevelopmentResponseSchema:
        """
        Function calculates development only for project with blocksnet
        Args:
            token (str): User token to access data from Urban API
            params (DevelopmentDTO): development request params
        Returns:
            DevelopmentResponseSchema: Response schema with development indicators
        """

        params = await self.get_optimal_func_zone_data(params, token)
        blocks, buildings = await self.aggregate_blocks_layer_scenario(
            params.scenario_id,
            params.proj_func_zone_source,
            params.proj_func_source_year,
            token,
        )
        res = await self.run_development_parameters(blocks)
        res = res.to_dict(orient="list")
        res.update({"params_data": params.model_dump()})
        return DevelopmentResponseSchema(**res)

    async def calc_context_development(
        self, token: str, params: ContextDevelopmentDTO
    ) -> DevelopmentResponseSchema:
        """
        Function calculates development for context  with project with blocksnet
        Args:
            token (str): User token to access data from Urban API
            params (DevelopmentDTO):
        Returns:
            DevelopmentResponseSchema: Response schema with development indicators
        """

        params = await self.get_optimal_func_zone_data(params, token)
        context_blocks, context_buildings = await self.aggregate_blocks_layer_context(
            params.scenario_id,
            params.context_func_zone_source,
            params.context_func_source_year,
            token,
        )
        scenario_blocks, scenario_buildings = (
            await self.aggregate_blocks_layer_scenario(
                params.scenario_id,
                params.proj_func_zone_source,
                params.proj_func_source_year,
                token,
            )
        )
        blocks = pd.concat([context_blocks, scenario_blocks]).reset_index(drop=True)
        res = await self.run_development_parameters(blocks)
        res = res.to_dict(orient="list")
        res.update({"params_data": params.model_dump()})
        return DevelopmentResponseSchema(**res)

    async def _get_accessibility_context(
        self, blocks: pd.DataFrame, acc_mx: pd.DataFrame, accessibility: float
    ) -> list[int]:
        blocks["population"] = blocks["population"].fillna(0)
        project_blocks = blocks.copy()
        context_blocks = get_accessibility_context(
            acc_mx, project_blocks, accessibility, out=False, keep=True
        )
        return list(context_blocks.index)

    async def _assess_provision(
        self, blocks: pd.DataFrame, acc_mx: pd.DataFrame, service_type: str
    ) -> gpd.GeoDataFrame:
        _, demand, accessibility = service_types_config[service_type].values()
        blocks["is_project"] = blocks["is_project"].fillna(False).astype(bool)
        context_ids = await self._get_accessibility_context(
            blocks, acc_mx, accessibility
        )
        capacity_column = f"capacity_{service_type}"
        if capacity_column in blocks.columns:
            blocks_df = (
                blocks[["geometry", "population", capacity_column]]
                .rename(columns={capacity_column: "capacity"})
                .fillna(0)
            )
        else:
            blocks_df = blocks[["geometry", "population"]].copy().fillna(0)
            blocks_df["capacity"] = 0
        prov_df, _ = competitive_provision(blocks_df, acc_mx, accessibility, demand)
        prov_df = prov_df.loc[context_ids].copy()
        return blocks[["geometry"]].join(prov_df, how="right")

    async def calculate_provision_totals(
        self,
        provision_gdfs_dict: dict[str, gpd.GeoDataFrame],
        ndigits: int = 2,
    ) -> dict[str, float | None]:
        prov_totals: dict[str, float | None] = {}
        for st_name, prov_gdf in provision_gdfs_dict.items():
            if prov_gdf.demand.sum() == 0:
                prov_totals[st_name] = None
            else:
                total = float(provision_strong_total(prov_gdf))
                prov_totals[st_name] = round(total, ndigits)
        return prov_totals

    async def territory_transformation_scenario_before(
        self,
        token: str,
        params: ContextDevelopmentDTO,
        context_blocks: gpd.GeoDataFrame = None,
    ):
        method_name = "territory_transformation"

        info = await self.urban_api_client.get_scenario_info(params.scenario_id, token)
        updated_at = info["updated_at"]
        project_id = info["project"]["project_id"]
        base_scenario_id = await self.urban_api_client.get_base_scenario_id(project_id)

        params = await self.get_optimal_func_zone_data(params, token)

        params_for_hash = await self.build_hash_params(params, token)
        phash = self.cache.params_hash(params_for_hash)

        force = getattr(params, "force", False)
        cached = (
            None if force else self.cache.load(method_name, params.scenario_id, phash)
        )
        if (
            cached
            and cached["meta"]["scenario_updated_at"] == updated_at
            and "before" in cached["data"]
        ):
            return {
                n: fc_to_gdf(fc)
                for n, fc in cached["data"]["before"].items()
                if is_fc(fc)
            }

        logger.info("Cache stale, missing or forced: calculating BEFORE")

        service_types = await self.urban_api_client.get_service_types()
        service_types = await adapt_service_types(service_types, self.urban_api_client)
        service_types = service_types[
            ~service_types["infrastructure_type"].isna()
        ].copy()

        params = await self.get_optimal_func_zone_data(params, token)
        base_src, base_year = (
            await self.urban_api_client.get_optimal_func_zone_request_data(
                token, base_scenario_id, None, None
            )
        )

        base_scenario_blocks, base_scenario_buildings = (
            await self.aggregate_blocks_layer_scenario(
                base_scenario_id, base_src, base_year, token
            )
        )

        before_blocks = pd.concat([context_blocks, base_scenario_blocks]).reset_index(
            drop=True
        )

        if "is_project" not in before_blocks.columns:
            before_blocks["is_project"] = False
        else:
            before_blocks["is_project"] = (
                before_blocks["is_project"].fillna(False).astype(bool)
            )

        graph = get_accessibility_graph(before_blocks, "intermodal")
        acc_mx = calculate_accessibility_matrix(before_blocks, graph)

        prov_gdfs_before = {}
        for st_id in service_types.index:
            st_name = service_types.loc[st_id, "name"]
            _, demand, accessibility = service_types_config[st_name].values()
            prov_gdf = await self._assess_provision(before_blocks, acc_mx, st_name)
            prov_gdf = prov_gdf.join(
                before_blocks[["is_project"]].reindex(prov_gdf.index), how="left"
            )
            prov_gdf["is_project"] = prov_gdf["is_project"].fillna(False).astype(bool)
            prov_gdf = prov_gdf.to_crs(4326)
            prov_gdf = prov_gdf.drop(axis="columns", columns="provision_weak")
            prov_gdfs_before[st_name] = prov_gdf

        prov_totals = await self.calculate_provision_totals(prov_gdfs_before)

        existing_data = cached["data"] if cached else {}
        existing_data["before"] = {
            name: await gdf_to_ru_fc_rounded(gdf, ndigits=6)
            for name, gdf in prov_gdfs_before.items()
        }
        existing_data["before"]["provision_total_before"] = prov_totals

        self.cache.save(
            method_name,
            params.scenario_id,
            params_for_hash,
            existing_data,
            scenario_updated_at=updated_at,
        )

        return prov_gdfs_before

    def _build_facade(
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

    async def territory_transformation_scenario_after(
        self,
        token,
        params: ContextDevelopmentDTO | DevelopmentDTO,
        context_blocks: gpd.GeoDataFrame,
        save_cache: bool = True,
    ):
        # provision after
        method_name = "territory_transformation"

        info = await self.urban_api_client.get_scenario_info(params.scenario_id, token)
        updated_at = info["updated_at"]
        is_based = info["is_based"]

        if is_based:
            raise http_exception(
                400, "base scenario has no 'after' layer needed for calculation"
            )

        params = await self.get_optimal_func_zone_data(params, token)

        params_for_hash = await self.build_hash_params(params, token)
        phash = self.cache.params_hash(params_for_hash)

        force = getattr(params, "force", False)
        cached = (
            None if force else self.cache.load(method_name, params.scenario_id, phash)
        )
        if (
            cached
            and cached["meta"]["scenario_updated_at"] == updated_at
            and "after" in cached["data"]
        ):
            return {
                n: gpd.GeoDataFrame.from_features(fc["features"], crs="EPSG:4326")
                for n, fc in cached["data"]["after"].items()
            }

        logger.info("Cache stale, missing or forced: calculating AFTER")

        service_types = await self.urban_api_client.get_service_types()
        service_types = await adapt_service_types(service_types, self.urban_api_client)
        service_types = service_types[
            ~service_types["infrastructure_type"].isna()
        ].copy()

        scenario_blocks, _ = await self.aggregate_blocks_layer_scenario(
            params.scenario_id,
            params.proj_func_zone_source,
            params.proj_func_source_year,
            token,
        )

        after_blocks = pd.concat([context_blocks, scenario_blocks]).reset_index(
            drop=True
        )

        after_blocks["is_project"] = (
            after_blocks["is_project"].fillna(False).astype(bool)
        )
        try:
            graph = get_accessibility_graph(after_blocks, "intermodal")
        except Exception as e:
            raise http_exception(
                500, "Error generating territory graph", _detail=str(e)
            )

        acc_mx = calculate_accessibility_matrix(after_blocks, graph)

        service_types["infrastructure_weight"] = (
            service_types["infrastructure_type"].map(INFRASTRUCTURES_WEIGHTS)
            * service_types["infrastructure_weight"]
        )

        if (
            "population" not in after_blocks.columns
            or after_blocks["population"].isna().any()
        ):
            dev_df = await self.run_development_parameters(after_blocks)
            after_blocks["population"] = pd.to_numeric(
                dev_df["population"], errors="coerce"
            ).fillna(0)
        else:
            after_blocks["population"] = pd.to_numeric(
                after_blocks["population"], errors="coerce"
            ).fillna(0)
        facade = self._build_facade(after_blocks, acc_mx, service_types)

        services_weights = service_types.set_index("name")[
            "infrastructure_weight"
        ].to_dict()

        objective = WeightedObjective(
            num_params=facade.num_params,
            facade=facade,
            weights=services_weights,
            max_evals=MAX_EVALS,
        )
        constraints = WeightedConstraints(num_params=facade.num_params, facade=facade)
        tpe_optimizer = TPEOptimizer(
            objective=objective,
            constraints=constraints,
            vars_chooser=SimpleChooser(facade),
        )

        best_x, best_val, perc, func_evals = tpe_optimizer.run(
            max_runs=MAX_RUNS, timeout=60000, initial_runs_num=1
        )

        prov_gdfs_after = {}
        for st_id in service_types.index:
            st_name = service_types.loc[st_id, "name"]
            if st_name in facade._chosen_service_types:
                prov_df = facade._provision_adapter.get_last_provision_df(st_name)
                prov_gdf = (
                    after_blocks[["geometry", "is_project"]]
                    .join(prov_df, how="left")
                    .drop(columns="provision_weak", errors="ignore")
                )

                if getattr(prov_gdf, "crs", None) is None:
                    prov_gdf = gpd.GeoDataFrame(
                        prov_gdf, geometry="geometry", crs=after_blocks.crs
                    )
                prov_gdf = prov_gdf.to_crs(4326)

                prov_gdf["is_project"] = (
                    prov_gdf["is_project"].fillna(False).astype(bool)
                )
                num_cols = [
                    c
                    for c in prov_gdf.select_dtypes(include=["number"]).columns
                    if c != "is_project"
                ]
                if num_cols:
                    prov_gdf[num_cols] = prov_gdf[num_cols].fillna(0)

                prov_gdfs_after[st_name] = gpd.GeoDataFrame(
                    prov_gdf, geometry="geometry", crs="EPSG:4326"
                )

        prov_totals = await self.calculate_provision_totals(prov_gdfs_after)

        after_fc = {
            name: await gdf_to_ru_fc_rounded(gdf, ndigits=6)
            for name, gdf in prov_gdfs_after.items()
        }
        after_fc["provision_total_after"] = prov_totals

        from_cache = cached.get("data", {}).copy() if cached else {}
        from_cache["after"] = after_fc
        from_cache["opt_context"] = {"best_x": best_x}

        if save_cache:
            self.cache.save(
                "territory_transformation",
                params.scenario_id,
                params_for_hash,
                from_cache,  # <-- сохраняем объединённые данные
                scenario_updated_at=updated_at,
            )

        return {
            "best_x": best_x,
            "prov_totals": prov_totals,
            "prov_gdfs_after": prov_gdfs_after,
        }

    async def territory_transformation(
        self,
        token: str,
        params: ContextDevelopmentDTO,
    ) -> dict[str, Any] | dict[str, dict[str, Any]]:

        info = await self.urban_api_client.get_scenario_info(params.scenario_id, token)
        is_based = info["is_based"]
        updated_at = info["updated_at"]

        context_blocks, _ = await self.aggregate_blocks_layer_context(
            params.scenario_id,
            params.context_func_zone_source,
            params.context_func_source_year,
            token,
        )
        prov_before = await self.territory_transformation_scenario_before(
            token, params, context_blocks
        )
        if is_based:
            return prov_before

        params_for_hash = await self.build_hash_params(params, token)
        phash = self.cache.params_hash(params_for_hash)

        cached = self.cache.load("territory_transformation", params.scenario_id, phash)
        if (
            cached
            and cached["meta"]["scenario_updated_at"] == updated_at
            and "after" in cached["data"]
        ):
            prov_after = {
                name: gpd.GeoDataFrame.from_features(fc["features"], crs="EPSG:4326")
                for name, fc in cached["data"]["after"].items()
            }
            return {"before": prov_before, "after": prov_after}

        prov_after = await self.territory_transformation_scenario_after(
            token, params, context_blocks
        )
        return {"before": prov_before, "after": prov_after}

    async def values_transformation(
        self,
        token: str,
        params: TerritoryTransformationDTO,
    ) -> dict:
        opt_method = "territory_transformation_opt"

        params = await self.get_optimal_func_zone_data(params, token)

        params_for_hash = await self.build_hash_params(params, token)
        phash = self.cache.params_hash(params_for_hash)
        force = getattr(params, "force", False)

        info = await self.urban_api_client.get_scenario_info(params.scenario_id, token)
        updated_at = info["updated_at"]

        context_blocks, _ = await self.aggregate_blocks_layer_context(
            params.scenario_id,
            params.context_func_zone_source,
            params.context_func_source_year,
            token,
        )

        opt_cached = (
            None if force else self.cache.load(opt_method, params.scenario_id, phash)
        )
        need_refresh = (
            force
            or not opt_cached
            or opt_cached["meta"]["scenario_updated_at"] != updated_at
            or "best_x" not in opt_cached["data"]
        )
        if need_refresh:
            res = await self.territory_transformation_scenario_after(
                token, params, context_blocks, save_cache=False
            )
            best_x_val = res["best_x"]

            self.cache.save(
                opt_method,
                params.scenario_id,
                params_for_hash,
                {"best_x": best_x_val},
                scenario_updated_at=updated_at,
            )
            opt_cached = self.cache.load(opt_method, params.scenario_id, phash)

        best_x = opt_cached["data"]["best_x"]

        scenario_blocks, _ = await self.aggregate_blocks_layer_scenario(
            params.scenario_id,
            params.proj_func_zone_source,
            params.proj_func_source_year,
            token,
        )

        after_blocks = pd.concat([context_blocks, scenario_blocks], ignore_index=False)
        if "block_id" in after_blocks.columns:
            after_blocks["block_id"] = after_blocks["block_id"].astype(int)
            if after_blocks.index.name == "block_id":
                after_blocks = after_blocks.reset_index(drop=True)
            after_blocks = (
                after_blocks.drop_duplicates(subset="block_id", keep="last")
                .set_index("block_id")
                .sort_index()
            )
        else:
            after_blocks.index = after_blocks.index.astype(int)
            after_blocks = after_blocks[
                ~after_blocks.index.duplicated(keep="last")
            ].sort_index()
        after_blocks.index.name = "block_id"

        if "is_project" in after_blocks.columns:
            after_blocks["is_project"] = (
                after_blocks["is_project"].fillna(False).astype(bool)
            )
        else:
            after_blocks["is_project"] = False

        try:
            graph = get_accessibility_graph(after_blocks, "intermodal")
        except Exception as e:
            raise http_exception(
                500, "Error generating territory graph", _detail=str(e)
            )

        acc_mx = calculate_accessibility_matrix(after_blocks, graph)

        service_types = await self.urban_api_client.get_service_types()
        service_types = await adapt_service_types(service_types, self.urban_api_client)
        service_types = service_types[
            ~service_types["infrastructure_type"].isna()
        ].copy()
        service_types["infrastructure_weight"] = (
            service_types["infrastructure_type"].map(INFRASTRUCTURES_WEIGHTS)
            * service_types["infrastructure_weight"]
        )

        facade = self._build_facade(after_blocks, acc_mx, service_types)
        test_blocks: gpd.GeoDataFrame = after_blocks.loc[
            list(facade._blocks_lu.keys())
        ].copy()
        test_blocks.index = test_blocks.index.astype(int)

        solution_df = facade.solution_to_services_df(best_x).copy()
        solution_df["block_id"] = solution_df["block_id"].astype(int)
        metrics = [
            c
            for c in ["site_area", "build_floor_area", "capacity", "count"]
            if c in solution_df.columns
        ]
        zero_dict = {m: 0 for m in metrics}

        if len(metrics):
            agg = (
                solution_df.groupby(["block_id", "service_type"])[metrics]
                .sum()
                .sort_index()
            )
        else:
            agg = (
                solution_df.groupby(["block_id", "service_type"])
                .size()
                .to_frame(name="__dummy__")
                .drop(columns="__dummy__")
            )

        def _row_to_dict(s: pd.Series) -> dict:
            d = {m: (0 if pd.isna(s.get(m)) else s.get(m)) for m in metrics}
            for k, v in d.items():
                try:
                    fv = float(v)
                    d[k] = int(fv) if fv.is_integer() else fv
                except Exception:
                    pass
            return d

        cells = (
            agg.apply(_row_to_dict, axis=1)
            if len(metrics)
            else agg.apply(lambda _: {}, axis=1)
        )
        wide = cells.unstack("service_type").reindex(index=test_blocks.index)

        all_services = sorted(solution_df["service_type"].dropna().unique().tolist())
        for s in all_services:
            if s not in wide.columns:
                wide[s] = np.nan

        def _fill_cell(x):
            return x if isinstance(x, dict) else zero_dict.copy()

        wide = wide.applymap(_fill_cell)
        wide = wide[all_services]
        test_blocks_with_services: gpd.GeoDataFrame = test_blocks.join(wide, how="left")

        logger.info("Values transformed complete")

        geom_col = test_blocks_with_services.geometry.name
        service_cols = all_services
        base_cols = [
            c for c in ["is_project"] if c in test_blocks_with_services.columns
        ]

        gdf_out = test_blocks_with_services[base_cols + service_cols + [geom_col]]
        gdf_out = gdf_out.to_crs(crs="EPSG:4326")
        gdf_out.geometry = round_coords(gdf_out.geometry, 6)
        geojson = json.loads(gdf_out.to_json())
        service_types = await self.urban_api_client.get_service_types()
        en2ru = await build_en_to_ru_map(service_types)
        geojson = await remap_properties_keys_in_geojson(geojson, en2ru)

        self.cache.save(
            "values_transformation",
            params.scenario_id,
            params_for_hash,
            geojson,
            scenario_updated_at=updated_at,
        )

        return geojson

    def _get_value_level(self, provisions: list[float | None]) -> float:
        vals = [p for p in provisions if p is not None]
        return float(np.mean(vals)) if vals else np.nan

    async def values_oriented_requirements(
        self,
        token: str,
        params: TerritoryTransformationDTO,
    ):
        method_name = "values_oriented_requirements"

        params = await self.get_optimal_func_zone_data(params, token)
        params_for_hash = await self.build_hash_params(params, token)
        phash = self.cache.params_hash(params_for_hash)
        force = getattr(params, "force", False)

        info = await self.urban_api_client.get_scenario_info(params.scenario_id, token)
        updated_at = info["updated_at"]

        cached = (
            None if force else self.cache.load(method_name, params.scenario_id, phash)
        )
        if cached and cached["meta"].get("scenario_updated_at") == updated_at:
            if "result" in cached["data"]:
                payload = cached["data"]["result"]
                result_df = pd.DataFrame(
                    data=payload["data"],
                    index=payload["index"],
                    columns=payload["columns"],
                )
                result_df.index.name = payload.get("index_name", None)
                return result_df

        context_blocks, _ = await self.aggregate_blocks_layer_context(
            params.scenario_id,
            params.context_func_zone_source,
            params.context_func_source_year,
            token,
        )

        scenario_blocks, _ = await self.aggregate_blocks_layer_scenario(
            params.scenario_id,
            params.proj_func_zone_source,
            params.proj_func_source_year,
            token,
        )

        scenario_blocks = scenario_blocks.to_crs(context_blocks.crs)

        cap_cols = [c for c in scenario_blocks.columns if c.startswith("capacity_")]
        scenario_blocks.loc[
            scenario_blocks["is_project"], ["population"] + cap_cols
        ] = 0
        if "capacity" in scenario_blocks.columns:
            scenario_blocks = scenario_blocks.drop(columns="capacity")

        blocks = gpd.GeoDataFrame(
            pd.concat([context_blocks, scenario_blocks], ignore_index=True),
            crs=context_blocks.crs,
        )

        service_types = await self.urban_api_client.get_service_types()
        service_types = await adapt_service_types(service_types, self.urban_api_client)
        service_types = service_types[~service_types["social_values"].isna()].copy()

        graph = get_accessibility_graph(blocks, "intermodal")
        acc_mx = calculate_accessibility_matrix(blocks, graph)

        prov_gdfs: dict[str, gpd.GeoDataFrame] = {}
        if (
            cached
            and cached["meta"].get("scenario_updated_at") == updated_at
            and "provision" in cached["data"]
        ):
            for st_name, fc in cached["data"]["provision"].items():
                prov_gdfs[st_name] = gpd.GeoDataFrame.from_features(
                    fc["features"], crs="EPSG:4326"
                )
        else:
            for st_id in service_types.index:
                st_name = service_types.loc[st_id, "name"]
                prov_gdf = await self._assess_provision(blocks, acc_mx, st_name)
                prov_gdf = prov_gdf.to_crs(4326).drop(
                    columns="provision_weak", errors="ignore"
                )
                num_cols = prov_gdf.select_dtypes(include="number").columns
                prov_gdf[num_cols] = prov_gdf[num_cols].fillna(0)
                prov_gdfs[st_name] = prov_gdf

        social_values_provisions: dict[str, list[float | None]] = {}
        for st_id in service_types.index:
            st_name = service_types.loc[st_id, "name"]
            social_values = service_types.loc[st_id, "social_values"]

            prov_gdf = prov_gdfs.get(st_name)
            if prov_gdf is None or prov_gdf.empty:
                continue

            if prov_gdf["demand"].sum() == 0:
                prov_total = None
            else:
                prov_total = float(provision_strong_total(prov_gdf))

            for sv in social_values:
                social_values_provisions.setdefault(sv, []).append(prov_total)

        soc_values_map = await self.urban_api_client.get_social_values_info()

        index = list(social_values_provisions.keys())
        result_df = pd.DataFrame(
            data=[self._get_value_level(social_values_provisions[sv]) for sv in index],
            index=index,
            columns=["social_value_level"],
        )

        values_table = {
            int(sv_id): {
                "name": soc_values_map.get(sv_id, str(sv_id)),
                "value": round(float(val), 2) if val else 0.0,
            }
            for sv_id, val in result_df["social_value_level"].to_dict().items()
        }

        self.cache.save(
            method_name,
            params.scenario_id,
            params_for_hash,
            {
                "provision": {
                    name: await gdf_to_ru_fc_rounded(gdf, ndigits=6)
                    for name, gdf in prov_gdfs.items()
                },
                "result": values_table,
            },
            scenario_updated_at=updated_at,
        )

        return result_df
