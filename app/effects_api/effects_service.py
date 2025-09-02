import json
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from blocksnet.analysis.indicators import calculate_development_indicators
from blocksnet.analysis.provision import competitive_provision
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

from app.dependencies import urban_api_gateway
from app.effects_api.modules.scenario_service import (
    get_scenario_blocks,
    get_scenario_buildings,
    get_scenario_functional_zones,
    get_scenario_services,
)
from app.effects_api.modules.service_type_service import adapt_service_types
from .dto.values_development_dto import ValuesDevelopmentDTO

from ..common.caching.caching_service import cache
from ..common.exceptions.http_exception_wrapper import http_exception
from ..common.utils.geodata import _fc_to_gdf, _gdf_to_ru_fc
from .constants.const import INFRASTRUCTURES_WEIGHTS, LAND_USE_RULES
from .dto.development_dto import (
    ContextDevelopmentDTO,
    DevelopmentDTO,
    SocioEconomicPredictionDTO,
)
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
    ):
        self.__name__ = "EffectsService"
        self.bn_social_regressor: SocialRegressor = SocialRegressor()

    async def build_hash_params(
        self,
        params: ContextDevelopmentDTO | DevelopmentDTO,
        token: str,
    ) -> dict:
        project_id = (
            await urban_api_gateway.get_scenario_info(params.scenario_id, token)
        )["project"]["project_id"]
        base_scenario_id = await urban_api_gateway.get_base_scenario_id(project_id)
        base_src, base_year = (
            await urban_api_gateway.get_optimal_func_zone_request_data(
                token, base_scenario_id, None, None
            )
        )
        return params.model_dump() | {
            "base_func_zone_source": base_src,
            "base_func_zone_year": base_year,
        }

    @staticmethod
    async def get_optimal_func_zone_data(
        params: DevelopmentDTO | ContextDevelopmentDTO | SocioEconomicPredictionDTO | ValuesDevelopmentDTO,
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
                await urban_api_gateway.get_optimal_func_zone_request_data(
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
                    ) = await urban_api_gateway.get_optimal_func_zone_request_data(
                        token,
                        params.scenario_id,
                        params.context_func_zone_source,
                        params.context_func_source_year,
                        project=False,
                    )
            return params
        return params

    @staticmethod
    async def load_blocks_scenario(scenario_id: int, token: str) -> gpd.GeoDataFrame:
        gdf = await get_scenario_blocks(scenario_id, token)
        gdf["site_area"] = gdf.area
        return gdf

    @staticmethod
    async def assign_land_use_to_blocks_scenario(
        blocks: gpd.GeoDataFrame,
        scenario_id: int,
        source: str | None,
        year: int | None,
        token: str,
    ) -> gpd.GeoDataFrame:

        fzones = await get_scenario_functional_zones(scenario_id, token, source, year)
        fzones = fzones.to_crs(blocks.crs)

        lu = assign_land_use(blocks, fzones, LAND_USE_RULES)
        return blocks.join(lu.drop(columns=["geometry"]))

    @staticmethod
    async def enrich_with_buildings_scenario(
        blocks: gpd.GeoDataFrame, scenario_id: int, token: str
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame | None]:

        buildings = await get_scenario_buildings(scenario_id, token)
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

    @staticmethod
    async def enrich_with_services_scenario(
        blocks: gpd.GeoDataFrame, scenario_id: int, token: str
    ) -> gpd.GeoDataFrame:

        stypes = await urban_api_gateway.get_service_types()
        stypes = await adapt_service_types(stypes)
        sdict = await get_scenario_services(scenario_id, stypes, token)

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

    @staticmethod
    async def load_context_blocks(
        scenario_id: int, token: str
    ) -> tuple[gpd.GeoDataFrame, int]:
        project_id = await urban_api_gateway.get_project_id(scenario_id, token)
        blocks = await get_context_blocks(project_id, scenario_id)
        blocks["site_area"] = blocks.area
        return blocks, project_id

    @staticmethod
    async def assign_land_use_context(
        blocks: gpd.GeoDataFrame,
        scenario_id: int,
        source: str | None,
        year: int | None,
        token: str,
    ) -> gpd.GeoDataFrame:
        fzones = await get_context_functional_zones(scenario_id, source, year, token)
        fzones = fzones.to_crs(blocks.crs)
        lu = assign_land_use(blocks, fzones, LAND_USE_RULES)
        return blocks.join(lu.drop(columns=["geometry"]))

    @staticmethod
    async def enrich_with_context_buildings(
        blocks: gpd.GeoDataFrame, scenario_id: int
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame | None]:

        buildings = await get_context_buildings(scenario_id)
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

    @staticmethod
    async def enrich_with_context_services(
        blocks: gpd.GeoDataFrame, scenario_id: int, token: str
    ) -> gpd.GeoDataFrame:

        stypes = await urban_api_gateway.get_service_types()
        stypes = await adapt_service_types(stypes)
        sdict = await get_context_services(scenario_id, stypes)

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
        blocks, buildings = await self.enrich_with_context_buildings(blocks, scenario_id)

        logger.info("Aggregating services for context")
        blocks = await self.enrich_with_context_services(blocks, scenario_id, token)

        logger.success(f"[Context {scenario_id}] blocks layer ready")
        return blocks, buildings

    @staticmethod
    async def get_services_layer(scenario_id: int, token: str):
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
        blocks = await get_scenario_blocks(scenario_id, token)
        blocks_crs = blocks.crs
        logger.info(
            f"{len(blocks)} START blocks layer scenario{scenario_id}, CRS: {blocks.crs}"
        )
        service_types = await urban_api_gateway.get_service_types()
        logger.info(f"{service_types}")
        services_dict = await get_scenario_services(scenario_id, service_types, token)

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

    async def evaluate_master_plan(
        self, params: SocioEconomicPredictionDTO, token: str = None
    ) -> SocioEconomicSchema:
        """
        End-to-end pipeline that fuses *project* and *context* blocks, enriches
        them with development parameters and produces socio-economic forecasts
        via ``SocialRegressor``.

        Params:
        params (ContextDevelopmentDTO): dto class containing following parameters:
            scenario_id : int
                Scenario to evaluate.
            functional_zone_source, functional_zone_year : str | int | None
                Source and year for **project** functional zones.
            context_functional_zone_source, context_functional_zone_year : str | int | None
                Source and year for **context** functional zones.
        token : str | None, default None
            Optional bearer token for Urban API.

        Returns:
        SocioEconomicResponseSchema:
            pd.DataFrame.to_dict(orient="index") representation as schema with additional params keys:
            `pred`, `lower`, `upper`, `is_interval`
            – predicted socio-economic indicator and its prediction interval.

        Workflow:
        1. Aggregate context blocks and project blocks.
        2. Merge them, clip land-use shares to 1.
        3. Compute development parameters (`run_development_parameters`).
        4. Feed summarised indicators into SocialRegressor.
        """

        logger.info(f"Evaluating master plan effects with {params.model_dump()}")
        params = await self.get_optimal_func_zone_data(params, token)
        project_id = await urban_api_gateway.get_project_id(params.scenario_id, token)
        project_info = await urban_api_gateway.get_all_project_info(project_id, token)
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

        scenario_blocks = scenario_blocks.to_crs(context_blocks.crs)

        blocks = gpd.GeoDataFrame(
            pd.concat([context_blocks, scenario_blocks], ignore_index=True),
            crs=context_blocks.crs,
        )
        cols = [
            "residential",
            "business",
            "recreation",
            "industrial",
            "transport",
            "special",
            "agriculture",
        ]

        blocks[cols] = blocks[cols].clip(upper=1)
        development_df = await self.run_development_parameters(blocks)

        cols = [
            "build_floor_area",
            "footprint_area",
            "living_area",
            "non_living_area",
            "population",
        ]
        blocks[cols] = development_df[cols].values
        for lu in LandUse:
            blocks[lu.value] = blocks[lu.value] * blocks["site_area"]
        main_data = [blocks.drop(columns=["land_use", "geometry"]).sum().to_dict()]
        main_input = pd.DataFrame(main_data)
        main_res = await self.run_social_reg_prediction(blocks, main_input)
        context_results = {}
        if params.split:
            for context_ter_id in project_info["properties"]["context"]:
                territory = gpd.GeoDataFrame(
                    geometry=[
                        await urban_api_gateway.get_territory_geometry(context_ter_id)
                    ],
                    crs=4326,
                )
                ter_blocks = (
                    blocks.sjoin(territory.to_crs(blocks.crs), how="left")
                    .dropna(subset="index_right")
                    .drop(columns="index_right")
                )
                ter_data = [
                    ter_blocks.drop(columns=["land_use", "geometry"]).sum().to_dict()
                ]
                ter_input = pd.DataFrame(ter_data)
                context_results[context_ter_id] = await self.run_social_reg_prediction(
                    ter_blocks, ter_input
                )

        logger.info(
            f"Finished evaluating master plan effects with {params.model_dump()}"
        )
        return SocioEconomicResponseSchema(
            socio_economic_prediction=main_res.socio_economic_prediction,
            split_prediction=context_results if context_results else None,
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

        logger.info(f"Calculating development for project {params.model_dump()}")
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
        logger.info(
            f"Finished calculating development for project {params.model_dump()}"
        )
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

        logger.info(f"Calculating development for context {params.model_dump()}")
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
        logger.info(
            f"Finished calculating development for context {params.model_dump()}"
        )
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

    async def territory_transformation_scenario_before(
        self, token: str, params: ContextDevelopmentDTO
    ):
        method_name = "territory_transformation"

        info = await urban_api_gateway.get_scenario_info(params.scenario_id, token)
        updated_at = info["updated_at"]
        project_id = info["project"]["project_id"]
        base_scenario_id = await urban_api_gateway.get_base_scenario_id(project_id)

        params = await self.get_optimal_func_zone_data(params, token)

        base_src, base_year = (
            await urban_api_gateway.get_optimal_func_zone_request_data(
                token, base_scenario_id, None, None
            )
        )

        params_for_hash = params.model_dump() | {
            "base_func_zone_source": base_src,
            "base_func_zone_year": base_year,
        }
        phash = cache.params_hash(params_for_hash)

        cached = cache.load(method_name, params.scenario_id, phash)
        if (
            cached
            and cached["meta"]["scenario_updated_at"] == updated_at
            and "before" in cached["data"]
        ):
            return {n: _fc_to_gdf(fc) for n, fc in cached["data"]["before"].items()}

        logger.info("Cache stale or missing: recalculating BEFORE")

        service_types = await urban_api_gateway.get_service_types()
        service_types = await adapt_service_types(service_types)
        service_types = service_types[
            ~service_types["infrastructure_type"].isna()
        ].copy()

        params = await self.get_optimal_func_zone_data(params, token)
        base_src, base_year = (
            await urban_api_gateway.get_optimal_func_zone_request_data(
                token, base_scenario_id, None, None
            )
        )

        context_blocks, context_buildings = await self.aggregate_blocks_layer_context(
            params.scenario_id,
            params.context_func_zone_source,
            params.context_func_source_year,
            token,
        )

        base_scenario_blocks, base_scenario_buildings = (
            await self.aggregate_blocks_layer_scenario(
                base_scenario_id, base_src, base_year, token
            )
        )

        before_blocks = pd.concat([context_blocks, base_scenario_blocks]).reset_index(
            drop=True
        )
        graph = get_accessibility_graph(before_blocks, "intermodal")
        acc_mx = calculate_accessibility_matrix(before_blocks, graph)

        prov_gdfs_before = {}
        for st_id in service_types.index:
            st_name = service_types.loc[st_id, "name"]
            _, demand, accessibility = service_types_config[st_name].values()
            prov_gdf = await self._assess_provision(before_blocks, acc_mx, st_name)
            prov_gdf = prov_gdf.to_crs(4326)
            prov_gdf = prov_gdf.drop(axis="columns", columns="provision_weak")
            prov_gdfs_before[st_name] = prov_gdf

        existing_data = cached["data"] if cached else {}
        existing_data["before"] = {
            n: _gdf_to_ru_fc(g) for n, g in prov_gdfs_before.items()
        }

        cache.save(
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
                df = after_blocks.rename(columns={st_column: "capacity"})[["capacity"]].fillna(0)
            else:
                df = after_blocks[[]].copy()
                df["capacity"] = 0
            facade.add_service_type(st_name, st_weight, df)

        return facade

    async def territory_transformation_scenario_after(
        self, token, params: ContextDevelopmentDTO | DevelopmentDTO
    ):
        # provision after
        method_name = "territory_transformation"

        info = await urban_api_gateway.get_scenario_info(params.scenario_id, token)
        updated_at = info["updated_at"]
        is_based = info["is_based"]

        if is_based:
            raise http_exception(400, "base scenario has no 'after' layer")

        params = await self.get_optimal_func_zone_data(params, token)

        params_for_hash = await self.build_hash_params(params, token)
        phash = cache.params_hash(params_for_hash)

        cached = cache.load(method_name, params.scenario_id, phash)
        if (
            cached
            and cached["meta"]["scenario_updated_at"] == updated_at
            and "after" in cached["data"]
        ):
            return {
                n: gpd.GeoDataFrame.from_features(fc["features"], crs="EPSG:4326")
                for n, fc in cached["data"]["after"].items()
            }

        logger.info("AFTER: cache stale or missing; recalculating")

        service_types = await urban_api_gateway.get_service_types()
        service_types = await adapt_service_types(service_types)
        service_types = service_types[
            ~service_types["infrastructure_type"].isna()
        ].copy()

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

        after_blocks = pd.concat([context_blocks, scenario_blocks]).reset_index(
            drop=True
        )

        after_blocks["is_project"] = (
            after_blocks["is_project"].fillna(False).astype(bool)
        )
        graph = get_accessibility_graph(after_blocks, "intermodal")
        acc_mx = calculate_accessibility_matrix(after_blocks, graph)

        service_types["infrastructure_weight"] = (
            service_types["infrastructure_type"].map(INFRASTRUCTURES_WEIGHTS)
            * service_types["infrastructure_weight"]
        )

        facade = self._build_facade(after_blocks, acc_mx, service_types)

        services_weights = service_types.set_index("name")[
            "infrastructure_weight"
        ].to_dict()

        objective = WeightedObjective(
            num_params=facade.num_params,
            facade=facade,
            weights=services_weights,
            max_evals=50,
        )
        constraints = WeightedConstraints(num_params=facade.num_params, facade=facade)
        tpe_optimizer = TPEOptimizer(
            objective=objective,
            constraints=constraints,
            vars_chooser=SimpleChooser(facade),
        )

        best_x, best_val, perc, func_evals = tpe_optimizer.run(
            max_runs=50, timeout=60000, initial_runs_num=1
        )

        prov_gdfs_after = {}
        for st_id in service_types.index:
            st_name = service_types.loc[st_id, "name"]
            if st_name in facade._chosen_service_types:
                prov_df = facade._provision_adapter.get_last_provision_df(st_name)
                prov_gdf = (
                    after_blocks[["geometry"]]
                    .join(prov_df, how="left")
                    .to_crs(4326)
                    .drop(columns="provision_weak", errors="ignore")
                )
                num_cols = prov_gdf.select_dtypes(include="number").columns
                prov_gdf[num_cols] = prov_gdf[num_cols].fillna(0)
                prov_gdfs_after[st_name] = gpd.GeoDataFrame(
                    prov_gdf, geometry="geometry", crs="EPSG:4326"
                )

        from_cache = cached["data"] if cached else {}
        from_cache["after"] = {n: _gdf_to_ru_fc(g) for n, g in prov_gdfs_after.items()}
        from_cache["opt_context"] = {
            "best_x": best_x,
        }

        cache.save(
            method_name,
            params.scenario_id,
            params_for_hash,
            from_cache,
            scenario_updated_at=updated_at,
        )

        return prov_gdfs_after

    async def territory_transformation(
        self,
        token: str,
        params: ContextDevelopmentDTO,
    ) -> dict[str, Any] | dict[str, dict[str, Any]]:

        info = await urban_api_gateway.get_scenario_info(params.scenario_id, token)
        is_based = info["is_based"]
        updated_at = info["updated_at"]

        prov_before = await self.territory_transformation_scenario_before(token, params)
        if is_based:
            return prov_before

        params_for_hash = await self.build_hash_params(params, token)
        phash = cache.params_hash(params_for_hash)

        cached = cache.load("territory_transformation", params.scenario_id, phash)
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

        prov_after = await self.territory_transformation_scenario_after(token, params)
        return {"before": prov_before, "after": prov_after}

    async def values_transformation(
            self,
            token: str,
            params: ValuesDevelopmentDTO,
    ) -> pd.DataFrame:

        method_name = "territory_transformation"
        params = await self.get_optimal_func_zone_data(params, token)

        params_for_hash = await self.build_hash_params(params, token)
        phash = cache.params_hash(params_for_hash)

        info = await urban_api_gateway.get_scenario_info(params.scenario_id, token)
        updated_at = info["updated_at"]

        cached = cache.load(method_name, params.scenario_id, phash)

        need_refresh = (
                not cached
                or cached["meta"]["scenario_updated_at"] != updated_at
                or "opt_context" not in cached["data"]
                or "best_x" not in cached["data"]["opt_context"]
        )
        if need_refresh:
            await self.territory_transformation_scenario_after(token, params)
            cached = cache.load(method_name, params.scenario_id, phash)

        best_x = cached["data"]["opt_context"]["best_x"]

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
        after_blocks = pd.concat([context_blocks, scenario_blocks]).reset_index(drop=True)
        after_blocks["is_project"] = after_blocks["is_project"].fillna(False).astype(bool)

        graph = get_accessibility_graph(after_blocks, "intermodal")
        acc_mx = calculate_accessibility_matrix(after_blocks, graph)

        service_types = await urban_api_gateway.get_service_types()
        service_types = await adapt_service_types(service_types)
        service_types = service_types[~service_types["infrastructure_type"].isna()].copy()
        service_types["infrastructure_weight"] = (
                service_types["infrastructure_type"].map(INFRASTRUCTURES_WEIGHTS)
                * service_types["infrastructure_weight"]
        )

        facade = self._build_facade(after_blocks, acc_mx, service_types)

        solution_df = facade.solution_to_services_df(best_x)

        if params.required_service:
            solution_df = solution_df.loc[solution_df["service_type"] == params.required_service]

        result = json.loads(
            solution_df.to_json(orient="records", date_format="iso")
        )
        return result

    def _get_value_level(self, provisions: list[float | None]) -> float:
        vals = [p for p in provisions if p is not None]
        return float(np.mean(vals)) if vals else np.nan

    async def values_oriented_requirements(
            self,
            token: str,
            params: ValuesDevelopmentDTO,
    ):
        method_name = "values_oriented_requirements"

        params = await self.get_optimal_func_zone_data(params, token)

        params_for_hash = await self.build_hash_params(params, token)
        phash = cache.params_hash(params_for_hash)

        info = await urban_api_gateway.get_scenario_info(params.scenario_id, token)
        updated_at = info["updated_at"]

        cached = cache.load(method_name, params.scenario_id, phash)
        if cached and cached["meta"].get("scenario_updated_at") == updated_at and "result" in cached["data"]:
            payload = cached["data"]["result"]
            result_df = pd.DataFrame(data=payload["data"], index=payload["index"], columns=payload["columns"])
            result_df.index.name = payload.get("index_name", None)
            return result_df

        project_id = await urban_api_gateway.get_project_id(params.scenario_id, token)

        context_blocks, context_buildings = await self.aggregate_blocks_layer_context(
            params.scenario_id,
            params.context_func_zone_source,
            params.context_func_source_year,
            token,
        )

        scenario_blocks, scenario_buildings = await self.aggregate_blocks_layer_scenario(
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

        social_values_provisions: dict[str, list[float | None]] = {}
        provisions_gdfs: dict[str, gpd.GeoDataFrame] = {}

        service_types = await urban_api_gateway.get_service_types()
        service_types = await adapt_service_types(service_types)
        service_types = service_types[~service_types["social_values"].isna()].copy()

        graph = get_accessibility_graph(blocks, "intermodal")
        acc_mx = calculate_accessibility_matrix(blocks, graph)

        for st_id in service_types.index:
            st_name = service_types.loc[st_id, "name"]
            social_values = service_types.loc[st_id, "social_values"]

            prov_gdf = await self._assess_provision(blocks, acc_mx, st_name)

            if "provision_strong" in prov_gdf.columns:
                prov_total: float | None = float(prov_gdf["provision_strong"].sum())
            elif "provision" in prov_gdf.columns:
                prov_total = float(prov_gdf["provision"].sum())
            else:
                prov_total = None

            provisions_gdfs[st_name] = prov_gdf

            for social_value in social_values:
                social_values_provisions.setdefault(social_value, []).append(prov_total)

        index = list(social_values_provisions.keys())
        result_df = pd.DataFrame(
            data=[self._get_value_level(social_values_provisions[sv_id]) for sv_id in index],
            index=index,
            columns=["social_value_level"],
        )
        result_df.index.name = "social_value_id"

        payload = {
            "columns": result_df.columns.tolist(),
            "index": result_df.index.tolist(),
            "data": result_df.values.tolist(),
            "index_name": result_df.index.name,
        }
        cache.save(
            method_name,
            params.scenario_id,
            params_for_hash,
            {"result": payload},
            scenario_updated_at=updated_at,
        )

        return result_df


effects_service = EffectsService()
