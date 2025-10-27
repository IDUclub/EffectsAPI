import json
from typing import Any, Dict, Literal, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from blocksnet.analysis.indicators.socio_economic import calculate_general_indicators, calculate_demographic_indicators, \
    calculate_transport_indicators, calculate_engineering_indicators, calculate_social_indicators
from blocksnet.analysis.land_use.prediction import SpatialClassifier
from blocksnet.analysis.provision import competitive_provision, provision_strong_total
from blocksnet.blocks.aggregation import aggregate_objects
from blocksnet.blocks.assignment import assign_land_use, assign_objects
from blocksnet.config import service_types_config
from blocksnet.enums import LandUse
from blocksnet.machine_learning.regression import SocialRegressor
from blocksnet.optimization.services import (
    AreaSolution,
    Facade,
    TPEOptimizer,
    WeightedConstraints,
    WeightedObjective, GradientChooser,
)
from blocksnet.relations import (
    calculate_accessibility_matrix,
    get_accessibility_context,
    get_accessibility_graph, calculate_distance_matrix,
)
from loguru import logger

from app.effects_api.modules.scenario_service import ScenarioService
from app.effects_api.modules.service_type_service import (
    adapt_service_types,
    build_en_to_ru_map,
    remap_properties_keys_in_geojson, adapt_social_service_types_df, generate_blocksnet_columns, ensure_missing_id_and_name_columns,
)

from ..clients.urban_api_client import UrbanAPIClient
from ..common.caching.caching_service import FileCache
from ..common.dto.models import SourceYear
from ..common.exceptions.http_exception_wrapper import http_exception
from ..common.utils.geodata import fc_to_gdf, gdf_to_ru_fc_rounded, is_fc, round_coords, _ensure_block_index
from .constants.const import (
    INFRASTRUCTURES_WEIGHTS,
    LAND_USE_RULES,
    MAX_EVALS,
    MAX_RUNS,
    PRED_VALUE_RU,
    PROB_COLS_EN_TO_RU, SOCIAL_INDICATORS_MAPPING, ROADS_ID, INDICATORS_MAPPING, SERVICE_TYPES_MAPPING,
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
    get_context_services, get_context_territories,
)
from .schemas.development_response_schema import DevelopmentResponseSchema
from .schemas.socio_economic_response_schema import (
    SocioEconomicResponseSchema,
    SocioEconomicSchema,
)
from ..common.utils.effects_utils import EffectsUtils


class EffectsService:
    def __init__(
        self,
        urban_api_client: UrbanAPIClient,
        cache: FileCache,
        scenario_service: ScenarioService,
        effects_utils: EffectsUtils
    ):
        self.__name__ = "EffectsService"
        self.bn_social_regressor: SocialRegressor = SocialRegressor()
        self.urban_api_client = urban_api_client
        self.cache = cache
        self.scenario = scenario_service
        self.effects_utils = effects_utils

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
            blocks_services["count"] = (
                blocks_services["count"].fillna(0).astype(int)
            )
            blocks = blocks.join(
                blocks_services.drop(columns=["geometry"]).rename(
                    columns={
                        "capacity": f"capacity_{service_type}",
                        "count": f"count_{service_type}",
                    }
                )
            )
        logger.info(
            f"{len(blocks)} SERVICES blocks layer scenario {scenario_id}, CRS: {blocks.crs}"
        )
        return blocks

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
        try:
            graph = get_accessibility_graph(before_blocks, "intermodal")
        except Exception as e:
            raise http_exception(
                500, "Error generating territory graph", _detail=str(e)
            )
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
            gdfs_after = {
                n: fc_to_gdf(fc)
                for n, fc in cached["data"]["after"].items()
                if is_fc(fc)
            }
            totals = cached["data"]["after"].get("provision_total_after")
            opt_ctx = cached.get("data", {}).get("opt_context") or {}
            return {"prov_gdfs_after": gdfs_after, "prov_totals": totals, **opt_ctx}

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
            vars_chooser=GradientChooser(facade, facade.num_params, num_top=5),
        )

        best_x, best_val, perc, func_evals = tpe_optimizer.run(
            max_runs=MAX_RUNS, timeout=4*60, initial_runs_num=1
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
                from_cache,
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
                name: fc_to_gdf(fc)
                for name, fc in cached["data"]["after"].items()
                if is_fc(fc)
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

        try:
            logger.info("Running land-use prediction on 'after_blocks'")

            ab = after_blocks[after_blocks.geometry.notna() & ~after_blocks.geometry.is_empty].copy()
            ab.geometry = ab.geometry.buffer(0)

            try:
                utm_crs = ab.estimate_utm_crs()
                ab = ab.to_crs(utm_crs)
            except Exception:
                ab = ab.to_crs("EPSG:3857")

            clf = SpatialClassifier.default()
            lu = clf.run(ab)

            lu = lu.drop(columns=["category"], errors="ignore")

            keep_cols = ["pred_name", "prob_urban", "prob_non_urban", "prob_industrial"]
            for c in keep_cols:
                if c not in lu.columns:
                    lu[c] = np.nan
            lu = lu[keep_cols]

            lu = _ensure_block_index(lu)
            gdf_out = _ensure_block_index(gdf_out)
            gdf_out = gdf_out.join(lu, how="left")

            logger.info("Attached land-use predictions to gdf_out (cols: {})", keep_cols)

            if "pred_name" in gdf_out.columns:
                gdf_out["Предсказанный вид использования"] = (
                    gdf_out["pred_name"]
                    .str.lower()
                    .map(PRED_VALUE_RU)
                    .fillna(gdf_out["pred_name"])
                )
                gdf_out = gdf_out.drop(columns=["pred_name"])

            prob_cols = [c for c in ["prob_urban", "prob_non_urban", "prob_industrial"] if c in gdf_out.columns]
            for col in prob_cols:
                gdf_out[col] = gdf_out[col].astype(float).round(1)

            rename_map = {k: v for k, v in PROB_COLS_EN_TO_RU.items() if k in gdf_out.columns}
            gdf_out = gdf_out.rename(columns=rename_map)

        except Exception as e:
            raise http_exception(500, "Failed to attach land-use predictions: {}", e)

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

        logger.info("Values transformed complete (with land-use predictions)")
        return geojson

    def _get_value_level(self, provisions: list[float | None]) -> float:
        vals = [p for p in provisions if p is not None]
        return float(np.mean(vals)) if vals else np.nan

    async def values_oriented_requirements(
        self,
        token: str,
        params: TerritoryTransformationDTO | DevelopmentDTO,
        persist: Literal["full", "table_only"] = "full",
    ):

        method_name = "values_oriented_requirements"

        info_curr = await self.urban_api_client.get_scenario_info(
            params.scenario_id, token
        )
        updated_at_curr = info_curr["updated_at"]
        project_id = (info_curr.get("project") or {}).get("project_id")
        regional_id = (info_curr.get("parent_scenario") or {}).get("id")

        force: bool = bool(getattr(params, "force", False))

        base_id: Optional[int] = None
        if project_id and regional_id:
            proj_scenarios = await self.urban_api_client.get_project_scenarios(
                project_id, token
            )

            matches = [
                s
                for s in proj_scenarios
                if self.effects_utils.truthy_is_based(s)
                   and self.effects_utils.parent_id(s) == regional_id
                   and self.effects_utils.sid(s) is not None
            ]
            if not matches:
                only_based = [
                    s
                    for s in proj_scenarios
                    if self.effects_utils.truthy_is_based(s) and self.effects_utils.sid(s) is not None
                ]
                if only_based:
                    only_based.sort(
                        key=lambda x: (
                            x.get("updated_at") is not None,
                            x.get("updated_at"),
                        ),
                        reverse=True,
                    )
                    matches = [only_based[0]]
            if matches:
                matches.sort(
                    key=lambda x: (
                        x.get("updated_at") is not None,
                        x.get("updated_at"),
                    ),
                    reverse=True,
                )
                base_id = self.effects_utils.sid(matches[0])

        if base_id is None:
            base_id = params.scenario_id

        params_base = params.model_copy(
            update={
                "scenario_id": base_id,
                "proj_func_zone_source": None,
                "proj_func_source_year": None,
                "context_func_zone_source": None,
                "context_func_source_year": None,
            }
        )
        params_base = await self.get_optimal_func_zone_data(params_base, token)

        params_for_hash_base = await self.build_hash_params(params_base, token)
        phash_base = self.cache.params_hash(params_for_hash_base)
        info_base = await self.urban_api_client.get_scenario_info(base_id, token)
        updated_at_base = info_base["updated_at"]

        def _result_to_df(payload: Any) -> pd.DataFrame:
            if isinstance(payload, dict) and "data" not in payload:
                items = sorted(
                    ((int(k), v.get("value", 0.0)) for k, v in payload.items()),
                    key=lambda t: t[0],
                )
                idx = [k for k, _ in items]
                vals = [float(v) if v is not None else 0.0 for _, v in items]
                return pd.DataFrame({"social_value_level": vals}, index=idx)
            df = pd.DataFrame(
                data=payload["data"], index=payload["index"], columns=payload["columns"]
            )
            df.index.name = payload.get("index_name", None)
            return df

        if not force:
            cached_base = self.cache.load(method_name, base_id, phash_base)
            if (
                cached_base
                and cached_base["meta"].get("scenario_updated_at") == updated_at_base
                and "result" in cached_base["data"]
            ):
                return _result_to_df(cached_base["data"]["result"])

        context_blocks, _ = await self.aggregate_blocks_layer_context(
            params.scenario_id,
            params_base.context_func_zone_source,
            params_base.context_func_source_year,
            token,
        )

        scenario_blocks, _ = await self.aggregate_blocks_layer_scenario(
            params_base.scenario_id,
            params_base.proj_func_zone_source,
            params_base.proj_func_source_year,
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

        try:
            graph = get_accessibility_graph(blocks, "intermodal")
        except Exception as e:
            raise http_exception(
                500, "Error generating territory graph", _detail=str(e)
            )
        acc_mx = calculate_accessibility_matrix(blocks, graph)

        prov_gdfs: Dict[str, gpd.GeoDataFrame] = {}
        for st_id in service_types.index:
            st_name = service_types.loc[st_id, "name"]
            prov_gdf = await self._assess_provision(blocks, acc_mx, st_name)
            prov_gdf = prov_gdf.to_crs(4326).drop(
                columns="provision_weak", errors="ignore"
            )
            num_cols = prov_gdf.select_dtypes(include="number").columns
            prov_gdf[num_cols] = prov_gdf[num_cols].fillna(0)
            prov_gdfs[st_name] = prov_gdf

        social_values_provisions: Dict[str, list[float | None]] = {}
        for st_id in service_types.index:
            st_name = service_types.loc[st_id, "name"]
            social_values = service_types.loc[st_id, "social_values"]
            prov_gdf = prov_gdfs.get(st_name)
            if prov_gdf is None or prov_gdf.empty:
                continue
            prov_total = (
                None
                if prov_gdf["demand"].sum() == 0
                else float(provision_strong_total(prov_gdf))
            )
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

        raw_services_df = await self.urban_api_client.get_service_types()
        en2ru = await build_en_to_ru_map(raw_services_df)

        demand_left_col = "demand_left"
        social_values_table: list[dict] = []

        for st_id in service_types.index:
            st_en = service_types.loc[st_id, "name"]
            st_ru = en2ru.get(st_en, st_en)

            linked_ids = list(
                map(int, (service_types.loc[st_id, "social_values"] or []))
            )
            linked_ru = [soc_values_map.get(sv_id, str(sv_id)) for sv_id in linked_ids]

            gdf = prov_gdfs.get(st_en)
            total_unsatisfied = 0.0
            if gdf is not None and not gdf.empty:
                if demand_left_col not in gdf.columns:
                    raise RuntimeError(
                        f"Колонка '{demand_left_col}' отсутствует для сервиса '{st_en}'"
                    )
                total_unsatisfied = float(gdf[demand_left_col].sum())

            social_values_table.append(
                {
                    "service": st_ru,
                    "unsatisfied_demand_sum": round(total_unsatisfied, 2),
                    "social_values": linked_ru,
                }
            )

        if persist == "full":
            payload = {
                "provision": {
                    name: await gdf_to_ru_fc_rounded(gdf, ndigits=6)
                    for name, gdf in prov_gdfs.items()
                },
                "result": values_table,
                "social_values_table": social_values_table,
                "services_type_deficit": social_values_table,
            }
        else:
            payload = {
                "result": values_table,
                "social_values_table": social_values_table,
                "services_type_deficit": social_values_table,
            }

        self.cache.save(
            method_name,
            base_id,
            params_for_hash_base,
            payload,
            scenario_updated_at=updated_at_base,
        )

        return result_df

    async def _get_project_scenarios_by_parent(
            self, project_id: int, regional_scenario_id: int, token: str
    ) -> list[int]:
        """Return scenario_ids for a given project that have the given regional_scenario_id as parent."""
        all_sc = await self.urban_api_client.get_project_scenarios(project_id, token)
        return [
            int(s["scenario_id"])
            for s in all_sc
            if (s.get("parent_scenario") or {}).get("id") == int(regional_scenario_id)
        ]

    async def _build_shared_context(
            self, project_id: int, token: str
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame]:
        """
        Build once per project:
          - context blocks (base scenario + sources)
          - context territories layer
          - prepared service_types_df (adapted + joined with normatives)
        """
        territory_id = (await self.urban_api_client.get_all_project_info(project_id, token))["territory"]["id"]
        base_sid = await self.urban_api_client.get_base_scenario_id(project_id)
        ctx_src, ctx_year = await self.urban_api_client.get_optimal_func_zone_request_data(
            token=token, data_id=base_sid, source=None, year=None, project=False
        )

        normatives = (await self.urban_api_client.get_territory_normatives(territory_id))[[
            "radius_availability_meters",
            "time_availability_minutes",
            "services_per_1000_normative",
            "services_capacity_per_1000_normative",
        ]].copy()

        service_types = await self.urban_api_client.get_service_types()
        service_types = await adapt_service_types(service_types, self.urban_api_client)
        service_types = service_types[service_types["infrastructure_type"].notna()].copy()
        service_types = adapt_social_service_types_df(service_types, SOCIAL_INDICATORS_MAPPING).join(normatives)

        context_blocks, _ = await self.aggregate_blocks_layer_context(base_sid, ctx_src, ctx_year, token)
        context_territories_gdf = await get_context_territories(project_id, token, self.urban_api_client)

        return context_blocks, context_territories_gdf, service_types

    async def _compute_for_single_scenario(
            self,
            scenario_id: int,
            *,
            context_blocks: gpd.GeoDataFrame,
            context_territories_gdf: gpd.GeoDataFrame,
            service_types_df: pd.DataFrame,
            proj_src: str,
            proj_year: int,
            token: str,
            only_parent_ids: set[int] | None = None,
    ) -> list[dict]:
        """
        Compute indicators for ONE scenario with shared context.
        Returns JSON-serializable list of records: [{territory_id, indicator_id, value}, ...]
        """
        logger.info(f"Computing indicators for scenario_id={scenario_id}")

        scenario_blocks, _ = await self.aggregate_blocks_layer_scenario(
            scenario_id, proj_src, proj_year, token
        )
        before_blocks = pd.concat([context_blocks, scenario_blocks], ignore_index=True)

        svc_cols = [c for c in before_blocks.columns if c.startswith(("count_", "capacity_"))]
        if svc_cols:
            before_blocks[svc_cols] = (
                before_blocks[svc_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype("int64")
            )

        context_territories_gdf = context_territories_gdf.to_crs(before_blocks.crs)
        assigned = assign_objects(before_blocks, context_territories_gdf.rename(columns={"parent": "name"}))
        before_blocks["parent"] = assigned["name"].astype(int)

        if only_parent_ids:
            before_blocks = before_blocks[before_blocks["parent"].isin(only_parent_ids)].copy()

        before_blocks = generate_blocksnet_columns(before_blocks, service_types_df)
        before_blocks = ensure_missing_id_and_name_columns(before_blocks)
        if "population" in before_blocks.columns:
            s = pd.to_numeric(before_blocks["population"], errors="coerce").fillna(0)
            if pd.api.types.is_float_dtype(s):
                s = s.round()
            before_blocks["population"] = s.astype("int64")
        else:
            before_blocks["population"] = 0

        roads_gdf = await self.urban_api_client.get_physical_objects_scenario(
            scenario_id, token=token, physical_object_function_id=ROADS_ID
        )
        roads_gdf = roads_gdf.to_crs(before_blocks.crs).overlay(before_blocks)

        graph = get_accessibility_graph(before_blocks, "drive")
        acc_mx = calculate_accessibility_matrix(before_blocks, graph)
        dist_mx = calculate_distance_matrix(before_blocks)

        st_for_social = service_types_df[
            service_types_df["infrastructure_type"].notna()
            & service_types_df["blocksnet"].notna()
            ].copy()

        general = calculate_general_indicators(before_blocks)
        demo = calculate_demographic_indicators(before_blocks)
        transp = calculate_transport_indicators(before_blocks, acc_mx, roads_gdf)
        eng = calculate_engineering_indicators(before_blocks)
        sc, sp = calculate_social_indicators(before_blocks, acc_mx, dist_mx, st_for_social)

        indicators_df = pd.concat([general, demo, transp, eng, sc, sp])

        long_df = (
            indicators_df.reset_index()
            .rename(columns={"index": "indicator"})
            .melt(id_vars=["indicator"], var_name="territory_id", value_name="value")
        )
        long_df = long_df[long_df["territory_id"] != "total"].copy()
        long_df["indicator_id"] = long_df["indicator"].map(INDICATORS_MAPPING)


        long_df["territory_id"] = pd.to_numeric(long_df["territory_id"], errors="coerce").apply(self.effects_utils.clean_number)
        long_df["indicator_id"] = long_df["indicator_id"].apply(self.effects_utils.clean_number)
        long_df["value"] = long_df["value"].apply(self.effects_utils.clean_number)
        long_df = long_df[long_df["indicator_id"].notna() & long_df["territory_id"].notna()].fillna(0)

        return long_df[["territory_id", "indicator_id", "value"]].to_dict(orient="records")

    async def evaluate_social_economical_metrics(
            self,
            params: SocioEconomicByProjectDTO,
            token: str,
    ):
        """
        Project-level multi-scenario calculation with a shared context.
        Return: {scenario_id: [{territory_id, indicator_id, value}, ...]}
        """

        project_id = int(params.project_id)
        parent_id = int(params.regional_scenario_id)

        base_sid = await self.urban_api_client.get_base_scenario_id(project_id)
        ctx_src, ctx_year = await self.urban_api_client.get_optimal_func_zone_request_data(
            token=token, data_id=base_sid, source=None, year=None, project=False
        )
        context_blocks, _ = await self.aggregate_blocks_layer_context(
            base_sid, ctx_src, ctx_year, token
        )
        context_territories_gdf = await get_context_territories(project_id, token, self.urban_api_client)

        territory_id = (await self.urban_api_client.get_all_project_info(project_id, token))["territory"]["id"]
        normatives = (await self.urban_api_client.get_territory_normatives(territory_id))[[
            "radius_availability_meters",
            "time_availability_minutes",
            "services_per_1000_normative",
            "services_capacity_per_1000_normative",
        ]].copy()

        service_types = await self.urban_api_client.get_service_types()
        service_types = await adapt_service_types(service_types, self.urban_api_client)
        service_types = service_types[service_types["infrastructure_type"].notna()].copy()
        service_types = adapt_social_service_types_df(service_types, SOCIAL_INDICATORS_MAPPING).join(normatives)

        scenarios = await self.urban_api_client.get_project_scenarios(project_id, token)
        target = [
            s for s in scenarios
            if (s.get("parent_scenario") or {}).get("id") == parent_id
        ]
        logger.info(
            f"[Effects] matched {len(target)} scenarios in project {project_id} (parent={parent_id})"
        )

        only_parent_ids = {int(x) for x in getattr(params, "territory_ids", [])} or None

        results: dict[int, list[dict]] = {}

        for s in target:
            sid = int(s["scenario_id"])
            proj_src, proj_year = await self.urban_api_client.get_optimal_func_zone_request_data(
                token=token, data_id=sid, source=None, year=None, project=True
            )

            records = await self._compute_for_single_scenario(
                sid,
                context_blocks=context_blocks,
                context_territories_gdf=context_territories_gdf,
                service_types_df=service_types,
                proj_src=proj_src,
                proj_year=proj_year,
                token=token,
                only_parent_ids=only_parent_ids,
            )
            results[sid] = records

        return results


