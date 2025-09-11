from load_shapefiles import load_tiger_cd_shapefiles
from calculate_metrics import calculate_district_metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import pandas as pd

# Step 1: Load shapefiles
shapefile_dir = "path_to/TIGER/2024/CD"
districts_gdf = load_tiger_cd_shapefiles(shapefile_dir)

# Step 2: Compute metrics
features_df = calculate_district_metrics(districts_gdf)

# Step 3: Scale metrics
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features_df[["polsby_popper", "convex_ratio", "schwartzberg", "reock", "eig_ratio"]])

# Step 4: Cluster
db = DBSCAN(eps=1.5, min_samples=5)
features_df["cluster"] = db.fit_predict(X_scaled)

# Step 5: Inspect outliers
print(features_df[features_df["cluster"] == -1])
