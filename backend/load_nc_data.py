import geopandas as gpd

# Load NC shapefile
gdf = gpd.read_file("data/NC-shapefiles/NC_precincts_2020.shp")

# Inspect columns and first rows
print(gdf.columns)
print(gdf.head())
