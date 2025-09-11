# --- Standard library imports ----------------------------------------------------------------
import asyncio                                      # Asynchronous execution for IO-bound tasks
import logging                                      # Logging framework
import json                                         # JSON serialization for structured logs
from pathlib import Path                            # Filesystem paths handling
from typing import List, Callable, Type             # Type hints for clarity

# --- Third-party imports ----------------------------------------------------------------------
import pandas as pd                                 # DataFrames for tabular data
import numpy as np                                  # Numerical operations
import yaml                                         # Optional YAML config parsing
import os                                           # For CPU count and other OS-level utilities
from pydantic import BaseModel, Field, ValidationError                  # Config validation
from prometheus_client import Counter, Histogram, start_http_server     # Metrics/monitoring
from sklearn.preprocessing import StandardScaler                        # Feature scaling for clustering
from sklearn.cluster import DBSCAN                                      # Density-based clustering algorithm

# ---- Optional / Conditional Imports -----------------------------------------------------------
try:
    import dask
    import dask.dataframe as dd
    from dask.distributed import Client, LocalCluster
    DASK_AVAILABLE = True                           # Flag indicating Dask is installed and usable
except ImportError:
    DASK_AVAILABLE = False

# ---- Local Imports -----------------------------------------------------------------------------
from load_shapefiles import load_tiger_cd_shapefiles  # Load TIGER shapefiles
from calculate_metrics import calculate_district_metrics  # Compute district metrics

# --- Logging Configuration ----------------------------------------------------------------------
class JsonFormatter(logging.Formatter):
    """
    Custom logging formatter to produce structured JSON logs
    for easier monitoring and ingestion into logging systems.
    """
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "lineNo": record.lineno
        }
        return json.dumps(log_record)

#--- Configure Logger ---------------------------------------------------------------------------
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger = logging.getLogger("enterprise_district_pipeline")
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# --- Prometheus Metrics -------------------------------------------------------------------------
PIPELINE_COUNTER = Counter(
    "district_pipeline_operations_total", 
    "Number of pipeline operations", 
    ["stage"]  # Label by pipeline stage
)
PIPELINE_LATENCY = Histogram(
    "district_pipeline_latency_seconds", 
    "Time taken for pipeline stages", 
    ["stage"]  # Label by stage
)

# --- Custom Exceptions ---------------------------------------------------------------------------
# Base pipeline error with a code attribute
class PipelineError(Exception): code = 1000
class CRSValidationError(PipelineError): code = 1001
class FeatureValidationError(PipelineError): code = 1002
class MetricValidationError(PipelineError): code = 1003
class ClusteringError(PipelineError): code = 1004
class ConfigValidationError(PipelineError): code = 1005

# ---- Pydantic Configuration ----------------------------------------------------------------------
class PipelineConfig(BaseModel):
    """
    Pydantic-based configuration schema for the pipeline
    with validation rules for each field.
    """
    shapefile_dir: str
    dbscan_eps: float = Field(..., gt=0)
    dbscan_min_samples: int = Field(..., gt=0)
    features: List[str]
    retry_attempts: int = Field(3, ge=1)
    retry_backoff_base: int = Field(2, ge=1)
    output_dir: str = "pipeline_outputs"
    use_dask: bool = False

# --- Default Configuration Instance ----------------------------------------------------------------
DEFAULT_CONFIG = PipelineConfig(
    shapefile_dir="path_to/TIGER/2024/CD",
    dbscan_eps=1.5,
    dbscan_min_samples=5,
    features=["polsby_popper", "convex_ratio", "schwartzberg", "reock", "eig_ratio"]
)
""" LOAD_CONFIG
    Load pipeline configuration from YAML file if it exists.
    Falls back to DEFAULT_CONFIG otherwise.
    Performs Pydantic validation.
"""
def load_config(path: str = "config.yml") -> PipelineConfig:
    
    if not Path(path).exists():
        logger.warning("Config file not found. Using default configuration.")
        return DEFAULT_CONFIG
    try:
        with open(path, "r") as f:
            cfg_dict = yaml.safe_load(f)
        config = PipelineConfig(**{**DEFAULT_CONFIG.dict(), **cfg_dict})
        return config
    except ValidationError as ve:
        logger.error("Configuration validation failed: %s", ve.json())
        raise ConfigValidationError(str(ve))

# -------------------- Retry Decorator --------------------------------------------------------
""" RETRY
    Decorator to retry async functions on transient errors.
    Implements exponential backoff.
"""
def retry(exceptions: Type[Exception] = PipelineError, max_attempts: int = 3, backoff_base: int = 2):
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < max_attempts:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    wait_time = backoff_base ** attempt
                    logger.warning(
                        "Attempt %d/%d failed for %s: %s. Retrying in %ds",
                        attempt, max_attempts, func.__name__, str(e), wait_time
                    )
                    await asyncio.sleep(wait_time)
            logger.error("All %d attempts failed for %s", max_attempts, func.__name__)
            raise
        return wrapper
    return decorator

# -------------------- Validation Utilities --------------------------------------------------
def validate_crs(gdf, expected_crs="EPSG:4326"):
    """Ensure GeoDataFrame has correct CRS, reproject if necessary."""
    if gdf.crs is None:
        raise CRSValidationError("GeoDataFrame has no CRS defined")
    if gdf.crs.to_string() != expected_crs:
        logger.info("Reprojecting GeoDataFrame to %s", expected_crs)
        gdf = gdf.to_crs(expected_crs)
    return gdf

def validate_features(metrics_df: pd.DataFrame, required_features: List[str]):
    """Check that required metric columns exist in the DataFrame."""
    missing = [f for f in required_features if f not in metrics_df.columns]
    if missing:
        raise FeatureValidationError(f"Missing required metric columns: {missing}")

def validate_metrics(metrics_df: pd.DataFrame, features: List[str]):
    """Check metric values for finiteness and flag extreme values."""
    for f in features:
        if not np.isfinite(metrics_df[f]).all():
            raise MetricValidationError(f"Non-finite values detected in feature '{f}'")
        if metrics_df[f].max() > 1e6 or metrics_df[f].min() < -1e6:
            logger.warning("Extreme values detected in feature '%s'", f)

# -------------------- Output Utilities -------------------------------------------------------
def save_output(metrics_df: pd.DataFrame, output_dir: str, prefix: str):
    """
    Save metrics DataFrame to CSV and JSON for downstream analysis.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(output_dir) / f"{prefix}.csv"
    json_path = Path(output_dir) / f"{prefix}.json"
    metrics_df.to_csv(csv_path, index=False)
    metrics_df.to_json(json_path, orient="records", indent=2)
    logger.info("Saved outputs: %s and %s", csv_path, json_path)

# -------------------- Core Pipeline Functions ------------------------------------------------
async def load_shapefiles_distributed(shapefile_dir: str):
    """
    Load shapefiles in parallel using Dask if available.
    Falls back to single-threaded async loading if Dask unavailable.
    """
    if DASK_AVAILABLE:
        logger.info("Using Dask for distributed shapefile loading")
        files = list(Path(shapefile_dir).glob("*.shp"))
        futures = [dask.delayed(load_tiger_cd_shapefiles)(str(f)) for f in files]
        gdfs = dask.compute(*futures)
        return pd.concat(gdfs, ignore_index=True)
    else:
        logger.info("Dask not available, using standard loading")
        return await asyncio.to_thread(load_tiger_cd_shapefiles, shapefile_dir)

@retry(max_attempts=DEFAULT_CONFIG.retry_attempts, backoff_base=DEFAULT_CONFIG.retry_backoff_base)
async def load_and_compute_metrics(shapefile_dir: str, features: List[str], use_dask: bool = False) -> pd.DataFrame:
    """
    Load shapefiles and compute metrics.
    Validates features and metrics, increments Prometheus counters.
    """
    try:
        with PIPELINE_LATENCY.labels(stage="load_and_compute").time():
            if use_dask:
                metrics_df = await load_shapefiles_distributed(shapefile_dir)
            else:
                gdf = await asyncio.to_thread(load_tiger_cd_shapefiles, shapefile_dir)
                gdf = validate_crs(gdf)
                metrics_df = await asyncio.to_thread(calculate_district_metrics, gdf)
            validate_features(metrics_df, features)
            validate_metrics(metrics_df, features)
            PIPELINE_COUNTER.labels(stage="load_and_compute").inc()
            return metrics_df
    except PipelineError as e:
        logger.error("Pipeline stage failed: %s", str(e))
        raise

async def cluster_districts(metrics_df: pd.DataFrame, features: List[str], eps: float, min_samples: int) -> pd.DataFrame:
    """
    Cluster districts using DBSCAN after scaling the features.
    Adds 'cluster' column to the DataFrame.
    """
    validate_features(metrics_df, features)
    validate_metrics(metrics_df, features)
    with PIPELINE_LATENCY.labels(stage="clustering").time():
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(metrics_df[features])
        db = DBSCAN(eps=eps, min_samples=min_samples)
        metrics_df["cluster"] = db.fit_predict(X_scaled)
        PIPELINE_COUNTER.labels(stage="clustering").inc()
    return metrics_df

def get_outliers(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Return districts labeled as outliers (-1) by DBSCAN."""
    outliers = metrics_df[metrics_df["cluster"] == -1]
    logger.info("Detected %d outlier districts", len(outliers))
    return outliers

# -------------------- Main Execution ---------------------------------------------------------
async def main():
    """
    Main pipeline execution:
    - Load config
    - Optionally start Dask cluster
    - Load shapefiles and compute metrics
    - Cluster districts
    - Save outputs
    """
    config = load_config()
    if config.use_dask and DASK_AVAILABLE:
        logger.info("Initializing Dask cluster for distributed execution")
        cluster = LocalCluster(n_workers=os.cpu_count())
        client = Client(cluster)
        logger.info("Dask cluster started with %d workers", len(client.scheduler_info()["workers"]))

    metrics_df = await load_and_compute_metrics(config.shapefile_dir, config.features, use_dask=config.use_dask)
    clustered_df = await cluster_districts(metrics_df, config.features, config.dbscan_eps, config.dbscan_min_samples)
    outliers_df = get_outliers(clustered_df)

    save_output(clustered_df, config.output_dir, prefix="clustered_metrics")
    if not outliers_df.empty:
        save_output(outliers_df, config.output_dir, prefix="outliers")

    if config.use_dask and DASK_AVAILABLE:
        client.close()
        cluster.close()

# -------------------- Entry Point ------------------------------------------------------------
if __name__ == "__main__":
    start_http_server(8000)  # Expose Prometheus metrics
    asyncio.run(main())
