import geopandas as gpd
import pandas as pd
from blocksnet.blocks.aggregation import aggregate_objects
from loguru import logger


def _adapt_service_type(data: dict, service_types: pd.DataFrame) -> int:
    service_type_id = int(data["service_type_id"])
    if service_type_id in service_types.index:
        service_type_name = service_types.loc[service_type_id, "name"]
        return service_type_name
    return None


def adapt_services(
    buildings_gdf: gpd.GeoDataFrame, service_types: pd.DataFrame
) -> dict[int, gpd.GeoDataFrame]:
    """
    Convert the raw building GeoDataFrame into a dictionary where each key is a
    canonical service-type ID and the value is a GeoDataFrame of buildings of
    that service type.

    Parameters:
    buildings_gdf : gpd.GeoDataFrame
        Required columns:
          • geometry      – building footprint or centroid
          • capacity      – numeric design capacity
          • service_type  – raw service-type ID
    service_types : pd.DataFrame
        Lookup table used by the helper _adapt_service_type to map raw
        service_type IDs onto canonical IDs.

    Returns:
    dict[int, gpd.GeoDataFrame]
        Keys are canonical service-type IDs (int).
        Each value contains only geometry and capacity columns; the temporary
        service_type column is removed.
        Buildings whose service_type cannot be mapped are discarded.
    """
    gdf = buildings_gdf[["geometry", "capacity"]].copy()
    gdf["service_type"] = buildings_gdf["service_type"].apply(
        lambda st: _adapt_service_type(st, service_types)
    )
    gdf = gdf[~gdf["service_type"].isna()].copy()
    return {
        st: gdf[gdf["service_type"] == st].drop(columns=["service_type"])
        for st in sorted(gdf["service_type"].unique())
    }

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
