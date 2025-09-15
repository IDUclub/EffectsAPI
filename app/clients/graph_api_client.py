import json
from typing import Literal

from geopandas import GeoDataFrame
from loguru import logger
from networkx.classes import MultiDiGraph, set_node_attributes
from pandas import Series, notna
from shapely.geometry.base import BaseGeometry
from shapely.geometry.geo import mapping

from app.common.api_handlers.json_api_handler import JSONAPIHandler
from app.common.exceptions.http_exception_wrapper import http_exception

#TODO
class GraphAPIClient:
    def __init__(self, json_handler: JSONAPIHandler) -> None:
        self.json_handler = json_handler
        self.__name__ = "GraphAPIClient"

    async def get_graph(self, _type: Literal["intermodal", "drive", "walk"], geometry: BaseGeometry) -> MultiDiGraph:
        try:
            response = await self.json_handler.post(
                endpoint_url="/api/v1/graph/build",
                headers={"Content-Type": "application/json"},
                params={"id_or_name": 1, "type": _type},
                data=mapping(geometry),
            )
        except TimeoutError:
            raise http_exception(500, "Graph API didn't respond in time: TimeoutError")
        graph_attrs, edges, nodes = response["attributes"], response["edges"], response["nodes"]
        logger.info(f"Received {len(edges)} edges and {len(nodes)} nodes")

        gdf_edges = GeoDataFrame.from_features(edges, crs="EPSG:4326")
        gdf_edges["u"] = gdf_edges["u"].astype(int)
        gdf_edges["v"] = gdf_edges["v"].astype(int)
        gdf_edges["key"] = gdf_edges["key"].astype(int)
        gdf_edges.set_index(["u", "v", "key"], inplace=True)
        df_nodes = GeoDataFrame.from_features(nodes, crs="EPSG:4326").set_index('node_id')
        df_nodes[['x', 'y']] = df_nodes.to_crs(32636)['geometry'].apply(lambda x: Series([x.x, x.y])) #esimate_utm_crs
        logger.info(f"Data prepared")

        graph_attrs["crs"] = GeoDataFrame(geometry=[geometry], crs=4326).estimate_utm_crs().to_epsg()
        G = MultiDiGraph(**graph_attrs)

        attr_names = gdf_edges.columns.tolist()
        for (u, v, k), attr_vals in zip(gdf_edges.index, gdf_edges.to_numpy()):
            data_all = zip(attr_names, attr_vals)
            data = {name: val for name, val in data_all if isinstance(val, list) or notna(val)}
            G.add_edge(u, v, key=k, **data)
        logger.info("Added edges into graph")

        for col in df_nodes.columns:
            set_node_attributes(G, name=col, values=df_nodes[col].fillna(0))
        logger.info("Node attributes set")
        return G
