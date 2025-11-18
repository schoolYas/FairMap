import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from geopandas import GeoDataFrame
import statsmodels.api as sm
import os

# ---------------------------------------------------------------------
# FUNCTION: compute metrics for any uploaded district map
# ---------------------------------------------------------------------
def compute_metrics_for_file(input_path):

    # Load the uploaded GeoJSON or Shapefile
    gdf = gpd.read_file(input_path)

    # -----------------------------
    # Aggregate by district
    # -----------------------------
    agg_cols = [c for c in gdf.columns if c != "geometry" and np.issubdtype(gdf[c].dtype, np.number)]

    districts = gdf.dissolve(by="CD", aggfunc="sum")[agg_cols]
    geometry = gdf.dissolve(by="CD").geometry
    districts["geometry"] = geometry
    districts = GeoDataFrame(districts, geometry=districts["geometry"], crs=gdf.crs)

    # -----------------------------
    # Geometry Metrics
    # -----------------------------
    districts["PP"] = 1 - (4 * np.pi * districts["Shape_Area"] / (districts["Shape_Leng"] ** 2))

    def reock_score(geom):
        try:
            rect = geom.minimum_rotated_rectangle
            radius = rect.length / (2 * np.pi)
            circle_area = np.pi * radius**2
            return 1 - geom.area / circle_area
        except:
            return np.nan

    districts["Reock"] = districts["geometry"].apply(reock_score)
    districts["PA_ratio"] = districts["Shape_Leng"] / districts["Shape_Area"]
    districts["ConvexHull_ratio"] = 1 - (districts["Shape_Area"] / districts["geometry"].convex_hull.area)
    districts["Schwartzberg"] = 1 - (np.sqrt(4 * np.pi * districts["Shape_Area"]) / districts["Shape_Leng"])

    # -----------------------------
    # Partisan Metrics
    # -----------------------------
    def wasted_votes(row):
        dem = row["EL16G_PR_D"]
        rep = row["EL16G_PR_R"]
        total = dem + rep
        if total == 0: return (0,0,1)

        dem_w = dem - (total//2 + 1) if dem > rep else dem
        rep_w = rep - (total//2 + 1) if rep > dem else rep

        return dem_w, rep_w, total

    districts[["dem_wasted","rep_wasted","total_votes"]] = districts.apply(
        lambda r: pd.Series(wasted_votes(r)), axis=1
    )
    districts["EG"] = abs(districts["dem_wasted"] - districts["rep_wasted"]) / districts["total_votes"].replace(0,1)

    # regression-adjusted EG
    districts["Black_frac"] = districts["BVAP"] / districts["VAP"]
    districts["Hispanic_frac"] = districts["HVAP"] / districts["VAP"]
    districts["White_frac"] = districts["WVAP"] / districts["VAP"]
    districts["Native_frac"] = districts["AMINVAP"] / districts["VAP"]

    X = districts[["Black_frac","Hispanic_frac","White_frac","Native_frac"]]
    X = sm.add_constant(X)
    y = districts["EL16G_PR_D"] / (districts["EL16G_PR_D"] + districts["EL16G_PR_R"])

    model = sm.OLS(y, X).fit()
    districts["expected_dem_share"] = model.predict(X)
    districts["residual_dem_share"] = y - districts["expected_dem_share"]

    def wasted_votes_regression(row):
        dem_share = row["residual_dem_share"] + 0.5
        rep_share = 1 - dem_share
        total = row["VAP"] if row["VAP"] > 0 else 1

        dem_v = dem_share * total
        rep_v = rep_share * total

        dem_w = dem_v - (total/2) if dem_v > rep_v else dem_v
        rep_w = rep_v - (total/2) if rep_v > dem_v else rep_v
        return dem_w, rep_w, total

    districts[["dem_wasted_reg","rep_wasted_reg","total_votes_reg"]] = districts.apply(
        lambda r: pd.Series(wasted_votes_regression(r)), axis=1
    )
    districts["EG_reg"] = (districts["dem_wasted_reg"] - districts["rep_wasted_reg"]) / districts["total_votes_reg"]

    # Mean–Median
    dem_share = districts["EL16G_PR_D"] / (districts["EL16G_PR_D"] + districts["EL16G_PR_R"])
    districts["MM"] = dem_share.mean() - dem_share.median()
    districts["Partisan_bias"] = dem_share - 0.5
    districts["Competitive_flag"] = ((dem_share > 0.45) & (dem_share < 0.55)).astype(float)

    # -----------------------------
    # Competitiveness
    # -----------------------------
    def competitiveness(r):
        dem, rep = r["EL16G_PR_D"], r["EL16G_PR_R"]
        total = dem + rep
        if total == 0: return 0.01
        margin = abs(dem - rep) / total
        return np.clip(1-margin, 0.01, 0.99)

    districts["Competitiveness"] = districts.apply(competitiveness, axis=1)
    districts["Competitiveness_std"] = 1 - abs(dem_share - dem_share.mean())
    districts["Competitiveness_std"] = districts["Competitiveness_std"].clip(0.01, 0.99)

    # -----------------------------
    # Demographics
    # -----------------------------
    minority = districts["BVAP"] + districts["HVAP"] + districts["AMINVAP"]
    districts["Minority"] = (minority / districts["VAP"]).clip(0.01, 0.99)

    districts["Diversity_index"] = 1 - (
        (districts["WVAP"]/districts["VAP"])**2 +
        (districts["BVAP"]/districts["VAP"])**2 +
        (districts["HVAP"]/districts["VAP"])**2 +
        (districts["AMINVAP"]/districts["VAP"])**2
    )
    districts["Diversity_index"] = districts["Diversity_index"].clip(0.01, 0.99)

    # -----------------------------
    # Metric Normalization
    # -----------------------------
    metrics = [
        'PP','Reock','PA_ratio','ConvexHull_ratio','Schwartzberg',
        'EG','EG_reg','MM','Partisan_bias','Competitive_flag',
        'Competitiveness','Competitiveness_std',
        'Minority','Hispanic_frac','Black_frac','Diversity_index'
    ]

    # 5–95% clipping
    for col in metrics:
        lo, hi = np.percentile(districts[col].dropna(), [5,95])
        districts[col] = districts[col].clip(lo,hi)

    # rank normalize to 0.01–0.99
    districts[metrics] = districts[metrics].rank(pct=True) * 0.98 + 0.01

    # -----------------------------
    # Composite Score
    # -----------------------------
    districts["geometry_score"] = districts[['PP','Reock','PA_ratio','ConvexHull_ratio','Schwartzberg']].mean(axis=1)
    districts["partisan_score"] = districts[['EG','EG_reg','MM','Partisan_bias','Competitive_flag']].mean(axis=1)
    districts["competitiveness_score"] = districts[['Competitiveness','Competitiveness_std']].mean(axis=1)
    districts["demographics_score"] = districts[['Minority','Hispanic_frac','Black_frac','Diversity_index']].mean(axis=1)

    districts["composite_score"] = (
        0.25 * districts["geometry_score"] +
        0.40 * districts["partisan_score"] +
        0.20 * districts["competitiveness_score"] +
        0.15 * districts["demographics_score"]
    )

    # -----------------------------
    # Save to a temporary output
    # -----------------------------
    output_path = input_path.replace(".geojson", "_weighted.geojson")
    districts.to_file(output_path, driver="GeoJSON")

    return output_path
