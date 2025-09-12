"""
Enterprise-ready district metrics computation:
- Supports projected CRS for correct area/length
- Computes polsby_popper, convex_ratio, schwartzberg, reock, eig_ratio
- Vectorized computation with GeoPandas and NumPy
- WKB serialization for audit/reproducibility
"""

from __future__ import annotations
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Sequence
import subprocess
from importlib import metadata as importlib_metadata

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry.base import BaseGeometry
from shapely import wkb

logger = logging.getLogger("calculate_metrics")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# ---------------- Constants -------------------------------------------------
DEFAULT_FEATURES = ["polsby_popper", "convex_ratio", "schwartzberg", "reock", "eig_ratio"]
DEFAULT_CRS = 3857  # Projected CRS for area/length calculations

# ---------------- CRS Utilities --------------------------------------------
def project_gdf(gdf: GeoDataFrame, target_crs: int = DEFAULT_CRS) -> GeoDataFrame:
    """Ensure GeoDataFrame is in projected CRS."""
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS defined")
    if gdf.crs.to_epsg() != target_crs:
        logger.info("Reprojecting GeoDataFrame to EPSG:%d", target_crs)
        gdf = gdf.to_crs(epsg=target_crs)
    return gdf

# ---------------- Metric Computation ---------------------------------------
def compute_vectorized(gdf: GeoDataFrame, metrics: Sequence[str] = DEFAULT_FEATURES) -> pd.DataFrame:
    """Vectorized computation of compactness metrics."""
    gdf = project_gdf(gdf)
    geom = gdf.geometry
    valid_mask = gdf.geometry.notna() & ~gdf.geometry.is_empty & gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    df = pd.DataFrame(index=gdf.index)

    if "polsby_popper" in metrics:
        df["polsby_popper"] = np.where(valid_mask, 4 * np.pi * geom.area / (geom.length ** 2), np.nan)
    if "convex_ratio" in metrics:
        df["convex_ratio"] = np.where(valid_mask, geom.area / geom.convex_hull.area, np.nan)
    if "schwartzberg" in metrics:
        df["schwartzberg"] = np.where(valid_mask, geom.length / (2 * np.sqrt(np.pi * geom.area)), np.nan)
    if "reock" in metrics:
        def reock_ratio(g: BaseGeometry) -> float:
            if g.is_empty: return np.nan
            min_circle = g.minimum_rotated_rectangle.minimum_rotated_rectangle
            return g.area / min_circle.area if min_circle.area > 0 else np.nan
        df["reock"] = np.where(valid_mask, geom.apply(reock_ratio), np.nan)
    if "eig_ratio" in metrics:
        def eig_ratio(g: BaseGeometry) -> float:
            if g.is_empty: return np.nan
            coords = np.array(g.exterior.coords) if hasattr(g, "exterior") else np.array(g.geoms[0].exterior.coords)
            cov = np.cov(coords.T)
            eigs = np.linalg.eigvals(cov)
            return np.nan if np.any(eigs <= 0) else eigs.min() / eigs.max()
        df["eig_ratio"] = np.where(valid_mask, geom.apply(eig_ratio), np.nan)

    return df

# ---------------- Clustering -------------------------------------------------
def cluster_metrics(metrics_df: pd.DataFrame, features: Optional[Sequence[str]] = None,
                    eps: float = 1.5, min_samples: int = 5) -> pd.DataFrame:
    """DBSCAN clustering of districts."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN

    if features is None:
        features = DEFAULT_FEATURES
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(metrics_df[features])
    db = DBSCAN(eps=eps, min_samples=min_samples)
    metrics_df["cluster"] = db.fit_predict(X_scaled)
    return metrics_df

# ---------------- Serialization Utilities ----------------------------------
def serialize_geom(g: BaseGeometry) -> bytes:
    try:
        return wkb.dumps(g, hex=False)
    except Exception:
        return b""

def deserialize_geom(b: bytes) -> Optional[BaseGeometry]:
    if not b:
        return None
    try:
        return wkb.loads(b)
    except Exception:
        return None

# ---------------- Audit Metadata -------------------------------------------
def audit_metadata() -> Dict[str, Any]:
    meta = {"timestamp": datetime.utcnow().isoformat() + "Z", "versions": {}}
    for pkg in ["numpy", "pandas", "geopandas", "shapely", "sklearn"]:
        try:
            meta["versions"][pkg] = importlib_metadata.version(pkg)
        except Exception:
            meta["versions"][pkg] = None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        meta["git_commit"] = commit
    except Exception:
        meta["git_commit"] = None
    return meta
