import json
from typing import Any, Dict, Literal, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from blocksnet.analysis.indicators import calculate_development_indicators
from blocksnet.analysis.indicators.socio_economic import calculate_general_indicators, calculate_demographic_indicators, \
    calculate_transport_indicators, calculate_engineering_indicators, calculate_social_indicators
from blocksnet.analysis.land_use.prediction import SpatialClassifier
from blocksnet.analysis.provision import competitive_provision, provision_strong_total
from blocksnet.blocks.assignment import assign_objects
from blocksnet.config import service_types_config
from blocksnet.enums import LandUse
from blocksnet.machine_learning.regression import SocialRegressor, DensityRegressor
from blocksnet.optimization.services import (
    TPEOptimizer,
    WeightedConstraints,
    WeightedObjective, GradientChooser,
)
from blocksnet.relations import (
 calculate_distance_matrix,
 generate_adjacency_graph,
)
from loguru import logger

from app.effects_api.modules.scenario_service import ScenarioService
from app.effects_api.modules.service_type_service import (
    adapt_service_types,
    build_en_to_ru_map,
    remap_properties_keys_in_geojson, generate_blocksnet_columns, ensure_missing_id_and_name_columns,
)
from .modules.context_service import ContextService
from ..clients.urban_api_client import UrbanAPIClient
from ..common.caching.caching_service import FileCache
from ..common.exceptions.http_exception_wrapper import http_exception
from ..common.utils.geodata import fc_to_gdf, gdf_to_ru_fc_rounded, is_fc, round_coords, _ensure_block_index, \
    get_accessibility_matrix
from .constants.const import (
    INFRASTRUCTURES_WEIGHTS,
    MAX_EVALS,
    MAX_RUNS,
    PRED_VALUE_RU,
    PROB_COLS_EN_TO_RU, ROADS_ID, INDICATORS_MAPPING, )
from .dto.development_dto import (
    ContextDevelopmentDTO,
    DevelopmentDTO,
)
from .dto.socio_economic_project_dto import (
    SocioEconomicByProjectDTO,
)
from .dto.transformation_effects_dto import TerritoryTransformationDTO
from ..common.utils.effects_utils import EffectsUtils


class EffectsService:
    def __init__(
        self,
        urban_api_client: UrbanAPIClient,
        cache: FileCache,
        scenario_service: ScenarioService,
        context_service: ContextService,
        effects_utils: EffectsUtils
    ):
        self.__name__ = "EffectsService"
        self.bn_social_regressor: SocialRegressor = SocialRegressor()
        self.urban_api_client = urban_api_client
        self.cache = cache
        self.scenario = scenario_service
        self.context = context_service
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

    async def _assess_provision(
        self, blocks: pd.DataFrame, acc_mx: pd.DataFrame, service_type: str
    ) -> gpd.GeoDataFrame:
        _, demand, accessibility = service_types_config[service_type].values()
        blocks["is_project"] = blocks["is_project"].fillna(False).astype(bool)
        context_ids = await self.context.get_accessibility_context(
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
            await self.scenario.aggregate_blocks_layer_scenario(
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
        acc_mx = get_accessibility_matrix(before_blocks)

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

        scenario_blocks, _ = await self.scenario.aggregate_blocks_layer_scenario(
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
        acc_mx = get_accessibility_matrix(after_blocks)

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
        facade = self.effects_utils.build_facade(after_blocks, acc_mx, service_types)

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
            max_runs=MAX_RUNS, timeout=10, initial_runs_num=1
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

        context_blocks, _ = await self.context.aggregate_blocks_layer_context(
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

        context_blocks, _ = await self.context.aggregate_blocks_layer_context(
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

        scenario_blocks, _ = await self.scenario.aggregate_blocks_layer_scenario(
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

        acc_mx = get_accessibility_matrix(after_blocks)

        service_types = await self.urban_api_client.get_service_types()
        service_types = await adapt_service_types(service_types, self.urban_api_client)
        service_types = service_types[
            ~service_types["infrastructure_type"].isna()
        ].copy()
        service_types["infrastructure_weight"] = (
            service_types["infrastructure_type"].map(INFRASTRUCTURES_WEIGHTS)
            * service_types["infrastructure_weight"]
        )

        facade = self.effects_utils.build_facade(after_blocks, acc_mx, service_types)
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

        gdf_out = gdf_out.to_crs("EPSG:4326")
        gdf_out.geometry = round_coords(gdf_out.geometry, 6)

        service_types = await self.urban_api_client.get_service_types()
        en2ru = await build_en_to_ru_map(service_types)
        rename_map = {k: v for k, v in en2ru.items() if k in gdf_out.columns}
        if rename_map:
            gdf_out = gdf_out.rename(columns=rename_map)

        geom_col = gdf_out.geometry.name
        non_geom = [c for c in gdf_out.columns if c != geom_col]

        pin_first = [c for c in ["is_project", "Предсказанный вид использования"] if c in non_geom]

        rest = [c for c in non_geom if c not in pin_first]
        rest_sorted = sorted(rest, key=lambda s: s.casefold())

        gdf_out = gdf_out[pin_first + rest_sorted + [geom_col]]

        geojson = json.loads(gdf_out.to_json())

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

        force: bool = bool(getattr(params, "force", False))

        base_id = await self.effects_utils.resolve_base_id(token, params.scenario_id)
        logger.info(
            f"Using base scenario_id={base_id} (requested={params.scenario_id})")

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

        context_blocks, _ = await self.context.aggregate_blocks_layer_context(
            params.scenario_id,
            params_base.context_func_zone_source,
            params_base.context_func_source_year,
            token,
        )

        scenario_blocks, _ = await self.scenario.aggregate_blocks_layer_scenario(
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

        acc_mx = get_accessibility_matrix(blocks)

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

    async def _compute_for_single_scenario(
            self,
            scenario_id: int,
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

        scenario_blocks, _ = await self.scenario.aggregate_blocks_layer_scenario(
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

        acc_mx = get_accessibility_matrix(before_blocks)
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
        long_df["value"] = long_df["value"].round(2)
        long_df = long_df[long_df["indicator_id"].notna() & long_df["territory_id"].notna()].fillna(0)

        return long_df[["territory_id", "indicator_id", "value"]].to_dict(orient="records")

    async def evaluate_social_economical_metrics(
            self,
            token: str,
            params: SocioEconomicByProjectDTO
    ):
        """
        Project-level multi-scenario calculation with a shared context.
        Return: {territory_id: {indicator_id: {scenario_id: value}}}
        """

        project_id = params.project_id
        parent_id = params.regional_scenario_id

        method_name = "social_economical_metrics"

        only_parent_ids = {int(x) for x in getattr(params, "territory_ids", [])} or None

        params_for_hash = {
            "project_id": project_id,
            "regional_scenario_id": parent_id,
            "territory_ids": sorted(list(only_parent_ids)) if only_parent_ids else [],
        }

        if not params.force:
            phash = self.cache.params_hash(params_for_hash)
            cached = self.cache.load(method_name, project_id, phash)
            if cached:
                logger.info(f"[Effects] cache hit for project {project_id}, returning cached data")
                return cached["results"]
        else:
            logger.info(f"[Effects] force=True, recalculating metrics for project {project_id}")

        context_blocks, context_territories_gdf, service_types = await self.context.get_shared_context(
            project_id, token
        )

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

        results = self.effects_utils.pivot_results_by_territory(results)

        project_info = await self.urban_api_client.get_project(project_id, token)
        updated_at = project_info.get("updated_at")

        self.cache.save(
            method_name,
            project_id,
            params_for_hash,
            {"results": results},
            scenario_updated_at=updated_at,
        )

        logger.success(f"[Effects] socio-economic metrics cached for project_id={project_id}")
        return results

