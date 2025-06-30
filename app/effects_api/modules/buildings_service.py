import geopandas as gpd
import pandas as pd

from app.effects_api.constants.const import BUILDINGS_RULES


def _parse(data: dict | None, *args):
    key = args[0]
    args = args[1:]
    if data is not None and key in data and data[key] is not None:
        if len(args) == 0:
            value = data[key]
            if isinstance(value, str):
                value = value.replace(",", ".")
            return value
        return _parse(data[key], *args)
    return None


def _adapt(data: dict, rules: list):
    for rule in rules:
        value = _parse(data, *rule)
        if value is not None:
            return value
    return None


def adapt_buildings(buildings_gdf: gpd.GeoDataFrame):
    gdf = buildings_gdf[["geometry"]].copy()
    gdf["is_living"] = buildings_gdf["physical_object_type"].apply(
        lambda pot: pot["physical_object_type_id"] == 4
    )
    for column, rules in BUILDINGS_RULES.items():
        series = buildings_gdf["building"].apply(lambda b: _adapt(b, rules))
        gdf[column] = pd.to_numeric(series, errors="coerce")
    return gdf
