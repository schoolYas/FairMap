import pytest
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon, Point
import geopandas as gpd
from backend import calculate_district_metrics  # replace with actual import


# Helper: create a GeoDataFrame from a list of geometries
def make_gdf(geoms, map_names=None):
    if map_names is None:
        map_names = [f"district_{i}" for i in range(len(geoms))]
    return gpd.GeoDataFrame({'map_name': map_names, 'geometry': geoms}, crs="EPSG:3857")


# --------------------------
# Test: basic polygons
# --------------------------
def test_square_polygon():
    square = Polygon([(0,0), (0,10), (10,10), (10,0)])
    gdf = make_gdf([square])
    df = calculate_district_metrics(gdf)
    assert 'polsby_popper' in df.columns
    assert np.isclose(df.loc[0, 'polsby_popper'], 4*np.pi*100/400**2)

def test_rectangle_polygon():
    rect = Polygon([(0,0),(0,5),(20,5),(20,0)])
    gdf = make_gdf([rect])
    df = calculate_district_metrics(gdf)
    assert df.loc[0, 'convex_ratio'] == 1.0

def test_circle_approximation():
    # approximate circle with 100-gon
    import math
    circle = Polygon([(math.cos(2*math.pi/100*i)*10, math.sin(2*math.pi/100*i)*10) for i in range(100)])
    gdf = make_gdf([circle])
    df = calculate_district_metrics(gdf)
    assert df.loc[0, 'schwartzberg'] > 0  # positive metric

# --------------------------
# Test: MultiPolygon
# --------------------------
def test_multipolygon():
    poly1 = Polygon([(0,0),(0,5),(5,5),(5,0)])
    poly2 = Polygon([(10,10),(10,15),(15,15),(15,10)])
    multi = MultiPolygon([poly1, poly2])
    gdf = make_gdf([multi])
    df = calculate_district_metrics(gdf)
    assert not np.isnan(df.loc[0, 'eig_ratio'])
    assert df.loc[0, 'convex_ratio'] <= 1.0

# --------------------------
# Test: invalid / empty geometries
# --------------------------
def test_empty_geometry():
    gdf = make_gdf([None])
    df = calculate_district_metrics(gdf)
    assert np.isnan(df.loc[0, 'polsby_popper'])

def test_point_geometry():
    pt = Point(0,0)
    gdf = make_gdf([pt])
    df = calculate_district_metrics(gdf)
    assert np.isnan(df.loc[0, 'polsby_popper'])
    assert np.isnan(df.loc[0, 'convex_ratio'])

# --------------------------
# Test: multiple metrics and return_gdf
# --------------------------
def test_multiple_metrics_return_gdf():
    square = Polygon([(0,0),(0,10),(10,10),(10,0)])
    gdf = make_gdf([square])
    df = calculate_district_metrics(gdf, metrics_to_compute=["polsby_popper","eig_ratio"], return_gdf=True)
    assert 'geometry' in df.columns
    assert 'polsby_popper' in df.columns
    assert 'eig_ratio' in df.columns

# --------------------------
# Test: parallel execution
# --------------------------
def test_parallel_execution():
    polys = [Polygon([(i,0),(i,1),(i+1,1),(i+1,0)]) for i in range(20)]
    gdf = make_gdf(polys)
    df = calculate_district_metrics(gdf, max_workers=8)
    assert len(df) == 20
    assert not df['polsby_popper'].isna().any()

# --------------------------
# Test: CRS transformation
# --------------------------
def test_crs_transformation():
    square = Polygon([(0,0),(0,0.001),(0.001,0.001),(0.001,0)])
    gdf = gpd.GeoDataFrame({'map_name': ['small'], 'geometry':[square]}, crs="EPSG:4326")
    df = calculate_district_metrics(gdf)
    assert 'polsby_popper' in df.columns
