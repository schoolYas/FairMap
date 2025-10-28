import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from sklearn.preprocessing import MinMaxScaler
from geopandas import GeoDataFrame


# -----------------------------
# Step 1: Load shapefile
# -----------------------------
gdf = gpd.read_file("../data/NC-shapefiles-master/NC_VTD/NC_VTD.shp")

# -----------------------------
# Step 2: Aggregate precincts to districts
# -----------------------------
# Only numeric columns, exclude 'geometry'
agg_cols = [col for col in gdf.columns if col != 'geometry' and np.issubdtype(gdf[col].dtype, np.number)]
districts = gdf.dissolve(by='CD', aggfunc='sum')[agg_cols]
districts['geometry'] = gdf.dissolve(by='CD')['geometry']
districts = GeoDataFrame(districts, geometry=districts['geometry'], crs=gdf.crs)

# -----------------------------
# Step 3: Geometry metrics
# -----------------------------
# Polsby-Popper
districts['PP'] = 4 * np.pi * districts['Shape_Area'] / (districts['Shape_Leng']**2)

# Reock: approximate via minimum rotated rectangle
def reock_score(geom):
    try:
        rect = geom.minimum_rotated_rectangle
        radius = rect.length / (2 * np.pi)
        circle_area = np.pi * radius**2
        return geom.area / circle_area
    except:
        return np.nan

districts['Reock'] = districts['geometry'].apply(reock_score)

# Perimeter/Area ratio
districts['PA_ratio'] = districts['Shape_Leng'] / districts['Shape_Area']

# -----------------------------
# Step 4: Partisan metrics
# -----------------------------
# Efficiency gap
def wasted_votes(row):
    dem_votes = row['EL16G_PR_D']
    rep_votes = row['EL16G_PR_R']
    total = dem_votes + rep_votes
    dem_wasted = dem_votes - (total // 2 + 1) if dem_votes > rep_votes else dem_votes
    rep_wasted = rep_votes - (total // 2 + 1) if rep_votes > dem_votes else rep_votes
    return dem_wasted, rep_wasted, total

districts[['dem_wasted','rep_wasted','total_votes']] = districts.apply(lambda row: pd.Series(wasted_votes(row)), axis=1)
districts['EG'] = (districts['dem_wasted'] - districts['rep_wasted']) / districts['total_votes']

# Mean-Median
dem_share = districts['EL16G_PR_D'] / (districts['EL16G_PR_D'] + districts['EL16G_PR_R'])
districts['MM'] = dem_share.mean() - dem_share.median()

# -----------------------------
# Step 5: Competitiveness
# -----------------------------
def competitiveness(row):
    dem = row['EL16G_PR_D']
    rep = row['EL16G_PR_R']
    total = dem + rep
    if total == 0:
        return 0
    margin = abs(dem - rep) / total  # 0 = tie, 1 = blowout
    return 1 - margin  # higher = more competitive

districts['Competitiveness'] = districts.apply(competitiveness, axis=1)

# -----------------------------
# Step 6: Demographic metrics
# -----------------------------
# Fractional minority (Black + Hispanic + American Indian)
minority_pop = districts['BVAP'] + districts['HVAP'] + districts['AMINVAP']
districts['Minority'] = minority_pop / districts['VAP']

# -----------------------------
# Step 7: Normalize all metrics
# -----------------------------
metrics = ['PP','Reock','PA_ratio','EG','MM','Competitiveness','Minority']
scaler = MinMaxScaler()
districts[metrics] = scaler.fit_transform(districts[metrics])

# -----------------------------
# Step 8: Compute weighted composite score
# -----------------------------
weights = {
    'geometry': 0.25,    # average of PP, Reock, PA_ratio
    'partisan': 0.4,     # average of EG, MM
    'competitiveness': 0.2,
    'demographics': 0.15
}

districts['geometry_score'] = districts[['PP','Reock','PA_ratio']].mean(axis=1)
districts['partisan_score'] = districts[['EG','MM']].mean(axis=1)

districts['composite_score'] = (
    weights['geometry'] * districts['geometry_score'] +
    weights['partisan'] * districts['partisan_score'] +
    weights['competitiveness'] * districts['Competitiveness'] +
    weights['demographics'] * districts['Minority']
)

# -----------------------------
# Step 9: Inspect results
# -----------------------------
print(districts[['geometry_score','partisan_score','Competitiveness','Minority','composite_score']])

# -----------------------------
# Step 10: Save weighted districts
# -----------------------------
output_path = "../data/NC-shapefiles-master/NC_VTD/NC_districts_weighted.geojson"
districts.to_file(output_path, driver='GeoJSON')
print(f"Weighted district file saved to {output_path}")
