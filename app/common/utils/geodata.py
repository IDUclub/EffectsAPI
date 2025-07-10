import json

import geopandas as gpd

from app.effects_api.constants.const import COL_RU


def _gdf_to_ru_fc(gdf: gpd.GeoDataFrame) -> dict:
    if "provision_weak" in gdf.columns:
        gdf = gdf.drop(columns="provision_weak")
    gdf = gdf.rename(
        columns={k: v for k, v in COL_RU.items() if k in gdf.columns}, errors="raise"
    )
    return json.loads(gdf.to_crs(4326).to_json(drop_id=True))


def _fc_to_gdf(fc: dict) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame.from_features(fc["features"], crs="EPSG:4326")
