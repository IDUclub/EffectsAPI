import geopandas as gpd
import shapely
import numpy as np
from blocksnet.blocks.cutting import preprocess_urban_objects, cut_urban_blocks

from app.effects_api.modules.urban_api_gateway import get_zones, get_project_geometry


def close_gaps(gdf, tolerance): # taken from momepy
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

LIVING_BUILDING_POT_ID = 4

def _get_project_boundaries(project_id : int):
    return gpd.GeoDataFrame(geometry=[projects.get_project_geometry(project_id)], crs=4326)

def _get_context_boundaries(project_id : int) -> gpd.GeoDataFrame:
    project = projects.get_project(project_id)
    context_ids = project['properties']['context']
    geometries = [territories.get_territory_geometry(territory_id) for territory_id in context_ids]
    return gpd.GeoDataFrame(geometry=geometries, crs=4326)

def _get_context_roads(project_id : int):
    gdf = projects.get_physical_objects(project_id, physical_object_function_id=ROADS_POF_ID)
    return gdf[['geometry']].reset_index(drop=True)

def _get_context_water(project_id : int):
    gdf = projects.get_physical_objects(project_id, physical_object_function_id=WATER_POF_ID)
    return gdf[['geometry']].reset_index(drop=True)

def _get_context_blocks(project_id : int, boundaries : gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    crs = boundaries.crs
    boundaries.geometry = boundaries.buffer(-1)

    water = _get_context_water(project_id).to_crs(crs)
    roads = _get_context_roads(project_id).to_crs(crs)
    roads.geometry = close_gaps(roads, 1)

    lines, polygons = preprocess_urban_objects(roads, None, water)
    blocks = cut_urban_blocks(boundaries, lines, polygons)
    return blocks


def get_context_blocks(project_id: int, token: str):
    project_boundaries = get_project_geometry(project_id, token)
    context_boundaries = _get_context_boundaries(project_id)

    crs = context_boundaries.estimate_utm_crs()
    context_boundaries = context_boundaries.to_crs(crs)
    project_boundaries = project_boundaries.to_crs(crs)

    context_boundaries = context_boundaries.overlay(project_boundaries, how='difference')
    return _get_context_blocks(project_id, context_boundaries)

def get_context_functional_zones(project_id : int, token: str) -> gpd.GeoDataFrame:
    sources_df = get_zones(project_id, token, is_context=True)
    year, source = _get_best_functional_zones_source(sources_df)
    functional_zones = projects.get_functional_zones(project_id, year, source)
    return adapt_functional_zones(functional_zones)

def get_context_buildings(project_id : int):
    gdf = projects.get_physical_objects(project_id, physical_object_type_id=1, centers_only=True)
    gdf = adapt_buildings(gdf.reset_index(drop=True))
    crs = gdf.estimate_utm_crs()
    return impute_buildings(gdf.to_crs(crs)).to_crs(4326)

def get_context_services(project_id : int, service_types : pd.DataFrame):
    gdf = projects.get_services(project_id, centers_only=True)
    gdfs = adapt_services(gdf.reset_index(drop=True), service_types)
    return {st:impute_services(gdf,st) for st,gdf in gdfs.items()}

