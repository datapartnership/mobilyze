import geopandas
import h3
import pandas as pd
from shapely.geometry import Polygon, mapping


def tessellate(gdf: geopandas.GeoDataFrame, columns=["shapeName"], resolution=7):
    """
    Tessellates the geometries into H3 indexes in a GeoDataFrame

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The GeoDataFrame ()containing the geometries to tessellate.
    column : str, optional
        The column name in `gdf` to use in the resulting GeoDataFrame.
        Default is "shapeName".
    resolution : int, optional
        The H3 resolution level for tessellation. Higher resolution results in smaller hexagons.
        Default is 7.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing the tessellated hexagons with the specified `index` and geometry.

    Raises
    ------
    Exception
        If a geometry type other than "Polygon" or "MultiPolygon" is encountered.

    Examples
    --------
    >>> import geopandas as gpd
    >>> from shapely.geometry import Polygon
    >>> gdf = gpd.GeoDataFrame({
    ...     'geometry': [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
    ...     'shapeName': ['A']
    ... })
    >>> tessellate(gdf)
          shapeName                                           geometry
    8a69a0f7fffffff      POLYGON ((0 0, 0.5 0, 1 0.5, 1 1, 0.5 1, 0 1, 0 0))
    """
    gdf = gdf.to_crs("EPSG:4326")
    mapper = dict()

    for idx, row in gdf.iterrows():
        geometry = row["geometry"]
        geom_type = geometry.geom_type

        if geom_type == "Polygon":
            hex_ids = h3.polyfill(
                mapping(geometry),
                resolution,
                geo_json_conformant=True,
            )
            mapper.update([(hex_id, row[columns]) for hex_id in hex_ids])

        if geom_type == "MultiPolygon":
            for x in geometry.geoms:
                hex_ids = h3.polyfill(
                    mapping(x),
                    resolution,
                    geo_json_conformant=True,
                )

                mapper.update([(hex_id, row[columns]) for hex_id in hex_ids])

    # python>=3.10
    # match geometry.geom_type:
    #     case "Polygon":
    #         hex_ids = h3.polyfill(
    #             mapping(geometry),
    #             resolution,
    #             geo_json_conformant=True,
    #         )
    #         mapper.update([(hex_id, row[columns]) for hex_id in hex_ids])

    #     case "MultiPolygon":
    #         for x in geometry.geoms:
    #             hex_ids = h3.polyfill(
    #                 mapping(x),
    #                 resolution,
    #                 geo_json_conformant=True,
    #             )

    #             mapper.update([(hex_id, row[columns]) for hex_id in hex_ids])
    #     case _:
    #         raise (Exception)

    # Create dataframe containing `hex_id`
    df = pd.DataFrame.from_dict(mapper, orient="index", columns=columns)
    gdf = geopandas.GeoDataFrame(
        df,
        geometry=[Polygon(h3.h3_to_geo_boundary(idx, True)) for idx in df.index],
        crs="EPSG:4326",
    )

    return gdf
