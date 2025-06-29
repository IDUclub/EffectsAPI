import geopandas as gpd

def _adapt_functional_zone(data : dict):
    functional_zone_type_id = data['name']
    return functional_zone_type_id

def adapt_functional_zones(functional_zones_gdf : gpd.GeoDataFrame):
    gdf = functional_zones_gdf[['geometry']].copy()
    gdf['functional_zone'] = functional_zones_gdf['functional_zone_type'].apply(_adapt_functional_zone)
    return gdf
