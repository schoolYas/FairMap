"""
Enterprise-grade unsupervised district metrics pipeline:
- Async shapefile loading with optional Dask distributed execution
- Metrics computation, validation, clustering, and outlier detection
- Config via YAML or defaults with Pydantic validation
- Structured JSON logging and Prometheus metrics
"""

import asyncio
import logging
import json
import os
from pathlib import Path
from typing import List, Callable, Type

import pandas as pd
import yaml
from pydantic import BaseModel, Field, ValidationError
from prometheus_client import Counter, Histogram, start_http_server

from load_shapefiles import load_all_cd_shapefiles
from calculate_metrics import compute_vectorized as calculate_district_metrics, cluster_metrics

# Optional Dask
try:
    import dask
    from dask.distributed import Client, LocalCluster
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

# ---------------- Logging ----------------
class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "lineNo": record.lineno
        })

handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger = logging.getLogger("enterprise_district_pipeline")
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# ---------------- Prometheus Metrics ----------------
PIPELINE_COUNTER = Counter("district_pipeline_operations_total", "Number of pipeline operations", ["stage"])
PIPELINE_LATENCY = Histogram("district_pipeline_latency_seconds", "Pipeline stage latency", ["stage"])

# ---------------- Exceptions ----------------
class PipelineError(Exception): code = 1000
class CRSValidationError(PipelineError): code = 1001
class FeatureValidationError(PipelineError): code = 1002
class MetricValidationError(PipelineError): code = 1003
class ConfigValidationError(PipelineError): code = 1005

# ---------------- Config ----------------
class PipelineConfig(BaseModel):
    shapefile_dir: str
    dbscan_eps: float = Field(..., gt=0)
    dbscan_min_samples: int = Field(..., gt=0)
    features: List[str]
    retry_attempts: int = Field(3, ge=1)
    retry_backoff_base: int = Field(2, ge=1)
    output_dir: str = "pipeline_outputs"
    use_dask: bool = False

DEFAULT_CONFIG = PipelineConfig(
    shapefile_dir="/Users/yasamean/Documents/FairMap/TIGER_2024_CD",
    dbscan_eps=1.5,
    dbscan_min_samples=5,
    features=["polsby_popper","convex_ratio","schwartzberg","reock","eig_ratio"]
)

# ---------------- Config Loader ----------------
def load_config(path: str = "config.yml") -> PipelineConfig:
    if not Path(path).exists():
        logger.warning("Config file not found. Using default configuration.")
        return DEFAULT_CONFIG
    try:
        with open(path, "r") as f:
            cfg_dict = yaml.safe_load(f)
        return PipelineConfig(**{**DEFAULT_CONFIG.dict(), **cfg_dict})
    except ValidationError as ve:
        logger.error("Configuration validation failed: %s", ve.json())
        raise ConfigValidationError(str(ve))

# ---------------- Retry Decorator ----------------
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
                    logger.warning("Attempt %d/%d failed for %s: %s. Retrying in %ds",
                                   attempt, max_attempts, func.__name__, str(e), wait_time)
                    await asyncio.sleep(wait_time)
            logger.error("All %d attempts failed for %s", max_attempts, func.__name__)
            raise
        return wrapper
    return decorator

# ---------------- Validation ----------------
def validate_crs(gdf, expected_crs="EPSG:4326"):
    if gdf.crs is None:
        raise CRSValidationError("GeoDataFrame has no CRS defined")
    if gdf.crs.to_string() != expected_crs:
        logger.info("Reprojecting GeoDataFrame to %s", expected_crs)
        gdf = gdf.to_crs(expected_crs)
    return gdf

def validate_features(metrics_df: pd.DataFrame, required_features: List[str]):
    missing = [f for f in required_features if f not in metrics_df.columns]
    if missing:
        raise FeatureValidationError(f"Missing required metric columns: {missing}")

def validate_metrics(metrics_df: pd.DataFrame, features: List[str]):
    for f in features:
        if not pd.api.types.is_numeric_dtype(metrics_df[f]):
            raise MetricValidationError(f"Non-numeric values detected in feature '{f}'")

# ---------------- Output ----------------
def save_output(metrics_df: pd.DataFrame, output_dir: str, prefix: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(output_dir)/f"{prefix}.csv"
    json_path = Path(output_dir)/f"{prefix}.json"
    metrics_df.to_csv(csv_path, index=False)
    metrics_df.to_json(json_path, orient="records", indent=2)
    logger.info("Saved outputs: %s and %s", csv_path, json_path)

# ---------------- Shapefile Loader ----------------
async def load_shapefiles_distributed(shapefile_dir: str):
    if DASK_AVAILABLE:
        logger.info("Using Dask for distributed shapefile loading")
        files = list(Path(shapefile_dir).glob("*.shp"))
        futures = [dask.delayed(load_all_cd_shapefiles)(str(f)) for f in files]
        gdfs = dask.compute(*futures)
        return pd.concat(gdfs, ignore_index=True)
    else:
        logger.info("Dask not available, using standard loading")
        return await asyncio.to_thread(load_all_cd_shapefiles, shapefile_dir)

# ---------------- Pipeline ----------------
@retry(max_attempts=DEFAULT_CONFIG.retry_attempts, backoff_base=DEFAULT_CONFIG.retry_backoff_base)
async def load_and_compute_metrics(shapefile_dir: str, features: List[str], use_dask: bool=False) -> pd.DataFrame:
    try:
        with PIPELINE_LATENCY.labels(stage="load_and_compute").time():
            # Load shapefiles
            gdf = await load_shapefiles_distributed(shapefile_dir) if use_dask else await asyncio.to_thread(load_all_cd_shapefiles, shapefile_dir)
            gdf = validate_crs(gdf)
            metrics_df = await asyncio.to_thread(calculate_district_metrics, gdf, features)
            validate_features(metrics_df, features)
            validate_metrics(metrics_df, features)
            PIPELINE_COUNTER.labels(stage="load_and_compute").inc()
            return metrics_df
    except PipelineError as e:
        logger.error("Pipeline stage failed: %s", str(e))
        raise

async def cluster_districts(metrics_df: pd.DataFrame, features: List[str], eps: float, min_samples: int) -> pd.DataFrame:
    validate_features(metrics_df, features)
    validate_metrics(metrics_df, features)
    with PIPELINE_LATENCY.labels(stage="clustering").time():
        clustered_df = cluster_metrics(metrics_df, features, eps, min_samples)
        PIPELINE_COUNTER.labels(stage="clustering").inc()
        return clustered_df

def get_outliers(metrics_df: pd.DataFrame) -> pd.DataFrame:
    outliers = metrics_df[metrics_df["cluster"] == -1]
    logger.info("Detected %d outlier districts", len(outliers))
    return outliers

# ---------------- Main ----------------
async def main():
    config = load_config()
    if config.use_dask and DASK_AVAILABLE:
        cluster = LocalCluster(n_workers=os.cpu_count())
        client = Client(cluster)
        logger.info("Dask cluster started with %d workers", len(client.scheduler_info()["workers"]))

    metrics_df = await load_and_compute_metrics(config.shapefile_dir, config.features, config.use_dask)
    clustered_df = await cluster_districts(metrics_df, config.features, config.dbscan_eps, config.dbscan_min_samples)
    outliers_df = get_outliers(clustered_df)

    save_output(clustered_df, config.output_dir, "clustered_metrics")
    if not outliers_df.empty:
        save_output(outliers_df, config.output_dir, "outliers")

    if config.use_dask and DASK_AVAILABLE:
        client.close()
        cluster.close()

if __name__ == "__main__":
    start_http_server(8000)
    asyncio.run(main())
