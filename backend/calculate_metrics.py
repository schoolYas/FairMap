import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from geopandas import GeoDataFrame
import statsmodels.api as sm
import os

# ---------------------------------------------------------------------
# CORE FUNCTION: compute metrics from an in-memory GeoDataFrame
# ---------------------------------------------------------------------
def compute_metrics_for_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Assumes gdf is precinct-level (or similar) with a 'CD' column for districts.
    Returns a district-level GeoDataFrame with all metric columns, including
    composite_score, ready for frontend or ensembling use.
    """

    # Work on a copy so we don't mutate the caller's data
    gdf = gdf.copy()

    # ------------------------------------------------------------------
    # 1) Make sure we have CD and TOTPOP so downstream code doesn't blow
    # ------------------------------------------------------------------
    if "CD" not in gdf.columns:
        gdf["CD"] = np.arange(len(gdf))

    if "TOTPOP" not in gdf.columns:
        gdf["TOTPOP"] = 1.0

    # -----------------------------
    # 2) Aggregate by district
    # -----------------------------
    agg_cols = [
        c for c in gdf.columns
        if c not in ("geometry", "CD") and np.issubdtype(gdf[c].dtype, np.number)
    ]

    dissolved = gdf.dissolve(by="CD", aggfunc="sum")

    if "geometry" not in dissolved.columns:
        dissolved = dissolved.set_geometry("geometry")

    districts = dissolved[agg_cols].copy()
    districts["geometry"] = dissolved.geometry
    districts = GeoDataFrame(districts, geometry="geometry", crs=gdf.crs)

    # equal-area CRS for geometry-based scores
    if districts.crs is None:
        # Fallback: use current coordinates as-is (OK for dummy test maps)
        metric_geom = districts
    else:
        metric_geom = districts.to_crs("EPSG:5070")  # US equal-area CRS (good enough baseline)

    # ------------------------------------------------------------------
    # 3) Ensure basic geometry columns exist (Shape_Area, Shape_Leng)
    # ------------------------------------------------------------------
    if "Shape_Area" not in districts.columns:
        districts["Shape_Area"] = metric_geom.geometry.area
    if "Shape_Leng" not in districts.columns:
        districts["Shape_Leng"] = metric_geom.geometry.length

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
        except Exception:
            return np.nan

    districts["Reock"] = metric_geom["geometry"].apply(reock_score)
    districts["PA_ratio"] = districts["Shape_Leng"] / districts["Shape_Area"]
    districts["ConvexHull_ratio"] = 1 - (
        districts["Shape_Area"] / metric_geom["geometry"].convex_hull.area
    )
    districts["Schwartzberg"] = 1 - (
        np.sqrt(4 * np.pi * districts["Shape_Area"]) / districts["Shape_Leng"]
    )

    # ------------------------------------------------------------------
    # 4) Ensure demographic + vote columns exist (for dummy files)
    # ------------------------------------------------------------------
    # VAP
    if "VAP" not in districts.columns:
        if "TOTPOP" in districts.columns:
            districts["VAP"] = districts["TOTPOP"]
        else:
            districts["VAP"] = 1.0

    # Race columns
    for col in ["BVAP", "HVAP", "WVAP", "AMINVAP"]:
        if col not in districts.columns:
            districts[col] = 0.0

    # If all race counts are 0, assume everyone is WVAP so Diversity_index isn't NaN
    race_sum = districts[["BVAP", "HVAP", "WVAP", "AMINVAP"]].sum(axis=1)
    mask_zero_race = race_sum == 0
    districts.loc[mask_zero_race, "WVAP"] = districts.loc[mask_zero_race, "VAP"]

    # Vote columns
    for col in ["EL16G_PR_D", "EL16G_PR_R"]:
        if col not in districts.columns:
            districts[col] = 0.0

    vote_totals = districts["EL16G_PR_D"] + districts["EL16G_PR_R"]
    has_any_votes = (vote_totals > 0).any()

    # -----------------------------
    # Partisan Metrics
    # -----------------------------
    def wasted_votes(row):
        dem = row["EL16G_PR_D"]
        rep = row["EL16G_PR_R"]
        total = dem + rep
        if total == 0:
            return (0, 0, 1)

        dem_w = dem - (total // 2 + 1) if dem > rep else dem
        rep_w = rep - (total // 2 + 1) if rep > dem else rep

        return dem_w, rep_w, total

    districts[["dem_wasted", "rep_wasted", "total_votes"]] = districts.apply(
        lambda r: pd.Series(wasted_votes(r)), axis=1
    )
    districts["EG"] = abs(districts["dem_wasted"] - districts["rep_wasted"]) / districts["total_votes"].replace(0, 1)

    # -----------------------------
    # Regression-adjusted EG (only if we actually have vote data)
    # -----------------------------
    if has_any_votes:
        districts["Black_frac"] = districts["BVAP"] / districts["VAP"]
        districts["Hispanic_frac"] = districts["HVAP"] / districts["VAP"]
        districts["White_frac"] = districts["WVAP"] / districts["VAP"]
        districts["Native_frac"] = districts["AMINVAP"] / districts["VAP"]

        X = districts[["Black_frac", "Hispanic_frac", "White_frac", "Native_frac"]]
        X = sm.add_constant(X)

        valid = vote_totals > 0
        y = pd.Series(0.5, index=districts.index)
        y.loc[valid] = districts.loc[valid, "EL16G_PR_D"] / vote_totals.loc[valid]

        try:
            model = sm.OLS(y.loc[valid], X.loc[valid]).fit()
            districts["expected_dem_share"] = model.predict(X)
        except Exception:
            districts["expected_dem_share"] = 0.5

        districts["residual_dem_share"] = y - districts["expected_dem_share"]

        def wasted_votes_regression(row):
            dem_share = row["residual_dem_share"] + 0.5
            rep_share = 1 - dem_share
            total = row["VAP"] if row["VAP"] > 0 else 1

            dem_v = dem_share * total
            rep_v = rep_share * total

            dem_w = dem_v - (total / 2) if dem_v > rep_v else dem_v
            rep_w = rep_v - (total / 2) if rep_v > dem_v else rep_v
            return dem_w, rep_w, total

        districts[["dem_wasted_reg", "rep_wasted_reg", "total_votes_reg"]] = districts.apply(
            lambda r: pd.Series(wasted_votes_regression(r)), axis=1
        )
        districts["EG_reg"] = (districts["dem_wasted_reg"] - districts["rep_wasted_reg"]) / districts["total_votes_reg"]

        dem_share = y
    else:
        # Neutral defaults when there is no vote data at all (like your test_map.geojson)
        districts["Black_frac"] = 0.0
        districts["Hispanic_frac"] = 0.0
        districts["White_frac"] = 1.0
        districts["Native_frac"] = 0.0

        districts["expected_dem_share"] = 0.5
        districts["residual_dem_share"] = 0.0
        districts["dem_wasted_reg"] = 0.0
        districts["rep_wasted_reg"] = 0.0
        districts["total_votes_reg"] = 1.0
        districts["EG_reg"] = 0.0

        dem_share = pd.Series(0.5, index=districts.index)

    # Mean–Median, Partisan bias, Competitive_flag
    districts["MM"] = float(dem_share.mean() - dem_share.median())
    districts["Partisan_bias"] = dem_share - 0.5
    districts["Competitive_flag"] = ((dem_share > 0.45) & (dem_share < 0.55)).astype(float)

    # -----------------------------
    # Competitiveness
    # -----------------------------
    def competitiveness(r):
        dem, rep = r["EL16G_PR_D"], r["EL16G_PR_R"]
        total = dem + rep
        if total == 0:
            return 0.5
        margin = abs(dem - rep) / total
        return np.clip(1 - margin, 0.01, 0.99)

    districts["Competitiveness"] = districts.apply(competitiveness, axis=1)
    districts["Competitiveness_std"] = 1 - abs(dem_share - dem_share.mean())
    districts["Competitiveness_std"] = districts["Competitiveness_std"].clip(0.01, 0.99)

    # -----------------------------
    # Demographics
    # -----------------------------
    minority = districts["BVAP"] + districts["HVAP"] + districts["AMINVAP"]
    districts["Minority"] = (minority / districts["VAP"]).clip(0.01, 0.99)

    districts["Diversity_index"] = 1 - (
        (districts["WVAP"] / districts["VAP"])**2 +
        (districts["BVAP"] / districts["VAP"])**2 +
        (districts["HVAP"] / districts["VAP"])**2 +
        (districts["AMINVAP"] / districts["VAP"])**2
    )
    districts["Diversity_index"] = districts["Diversity_index"].clip(0.01, 0.99)

    # -----------------------------
    # Metric Normalization
    # -----------------------------
    metrics = [
        'PP', 'Reock', 'PA_ratio', 'ConvexHull_ratio', 'Schwartzberg',
        'EG', 'EG_reg', 'MM', 'Partisan_bias', 'Competitive_flag',
        'Competitiveness', 'Competitiveness_std',
        'Minority', 'Hispanic_frac', 'Black_frac', 'Diversity_index'
    ]

    for col in metrics:
        col_data = districts[col].dropna()
        if len(col_data) == 0:
            continue
        lo, hi = np.percentile(col_data, [5, 95])
        districts[col] = districts[col].clip(lo, hi)

    districts[metrics] = districts[metrics].rank(pct=True) * 0.98 + 0.01

    # -----------------------------
    # Composite Score
    # -----------------------------
    districts["geometry_score"] = districts[['PP', 'Reock', 'PA_ratio', 'ConvexHull_ratio', 'Schwartzberg']].mean(axis=1)
    districts["partisan_score"] = districts[['EG', 'EG_reg', 'MM', 'Partisan_bias', 'Competitive_flag']].mean(axis=1)
    districts["competitiveness_score"] = districts[['Competitiveness', 'Competitiveness_std']].mean(axis=1)
    districts["demographics_score"] = districts[['Minority', 'Hispanic_frac', 'Black_frac', 'Diversity_index']].mean(axis=1)

    districts["composite_score"] = (
        0.25 * districts["geometry_score"] +
        0.40 * districts["partisan_score"] +
        0.20 * districts["competitiveness_score"] +
        0.15 * districts["demographics_score"]
    )

    return districts


# ---------------------------------------------------------------------
# HELPER: collapse district metrics to a single statewide composite
#         (for ensembling)
# ---------------------------------------------------------------------
def compute_state_composite(districts: gpd.GeoDataFrame, pop_weighted: bool = True) -> float:
    """
    Returns a single scalar composite score for the whole plan.
    Use this inside ensembling: run metrics, then call this.
    """
    if pop_weighted and "VAP" in districts.columns:
        total_pop = districts["VAP"].sum()
        if total_pop <= 0:
            return float(districts["composite_score"].mean())
        return float((districts["composite_score"] * districts["VAP"]).sum() / total_pop)
    else:
        return float(districts["composite_score"].mean())


# ---------------------------------------------------------------------
# FILE-BASED WRAPPER: keep existing behavior for the frontend
# ---------------------------------------------------------------------
def compute_metrics_for_file(input_path: str) -> str:
    """
    Existing entry point for your FastAPI endpoint.
    Reads file from disk, runs metrics, writes *_weighted.geojson,
    and returns the output path (unchanged API for the frontend).
    """
    gdf = gpd.read_file(input_path)
    districts = compute_metrics_for_gdf(gdf)

    output_path = input_path.replace(".geojson", "_weighted.geojson")
    districts.to_file(output_path, driver="GeoJSON")

    return output_path
