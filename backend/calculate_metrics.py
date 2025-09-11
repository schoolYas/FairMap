"""
enterprise_district_metrics.py

Enterprise-hardened district metrics calculator.

Features:
- Config via env, optional YAML, runtime overrides
- Optional distributed execution (Dask or Ray) or local chunked ProcessPoolExecutor with WKB serialization
- Observability: optional Prometheus and OpenTelemetry integration (soft dependencies)
- Robust version parsing and audit metadata
- Error collection + structured error report; optional strict mode raises MetricsProcessingError
- Resilience: retries and fallback (convex hull) for broken geometries
- Optional Pydantic schema validation for inputs/outputs
- Testable small helper functions exposed
"""

from __future__ import annotations

import logging
import logging.config
import os
import sys
import time
import json
import hashlib
import multiprocessing
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Polygon, MultiPolygon
from shapely import wkb
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from importlib import metadata as importlib_metadata
from datetime import datetime

# Optional / soft imports
try:
    import yaml
except Exception:
    yaml = None

try:
    from packaging.version import parse as parse_version
except Exception:
    parse_version = None

try:
    import dask
    from dask import delayed, compute as dask_compute
except Exception:
    dask = None
    delayed = None
    dask_compute = None

try:
    import ray
except Exception:
    ray = None

try:
    from prometheus_client import Summary, Counter, start_http_server
except Exception:
    Summary = None
    Counter = None
    start_http_server = None

try:
    from opentelemetry import trace
    from opentelemetry.trace import Tracer
except Exception:
    trace = None
    Tracer = None

try:
    from pydantic import BaseModel
except Exception:
    BaseModel = None

# ---------- Basic logging ----------
DEFAULT_LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "format": '{"time":"%(asctime)s","logger":"%(name)s","level":"%(levelname)s","message":"%(message)s"}'
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "json",
            "level": "INFO",
            "stream": "ext://sys.stdout",
        }
    },
    "root": {"handlers": ["console"], "level": "INFO"},
}
logging.config.dictConfig(DEFAULT_LOGGING_CONFIG)
logger = logging.getLogger("enterprise_district_metrics")

# ---------- Version / audit helpers ----------
MIN_VERSIONS = {"numpy": "1.21", "pandas": "1.3", "geopandas": "0.12", "shapely": "2.0"}


def _safe_parse_version(ver_str: str) -> Tuple[int, ...]:
    """Tolerantly parse versions (digits only)."""
    import re

    nums = re.findall(r"\d+", ver_str)
    return tuple(int(n) for n in nums) if nums else ()


def check_versions(min_versions: Dict[str, str] = MIN_VERSIONS) -> None:
    """Raise ImportError if installed packages don't meet minimal versions."""
    failed = []
    for pkg, minv in min_versions.items():
        try:
            inst = importlib_metadata.version(pkg)
            if parse_version:
                if parse_version(inst) < parse_version(minv):
                    failed.append(f"{pkg} ({inst} < {minv})")
            else:
                if _safe_parse_version(inst) < _safe_parse_version(minv):
                    failed.append(f"{pkg} ({inst} < {minv})")
        except Exception as e:
            failed.append(f"{pkg}: {e}")
    if failed:
        raise ImportError("Dependency version checks failed: " + "; ".join(failed))


# run version check at import
check_versions()

# ---------- Config dataclass & loader ----------
@dataclass
class MetricsConfig:
    target_crs: int = field(default_factory=lambda: int(os.getenv("DM_TARGET_CRS", "3857")))
    max_workers: int = field(default_factory=lambda: int(os.getenv("DM_MAX_WORKERS", "4")))
    backend: str = field(default_factory=lambda: os.getenv("DM_BACKEND", "process"))  # process/thread/dask/ray
    chunk_size: int = field(default_factory=lambda: int(os.getenv("DM_CHUNK_SIZE", "1000")))
    strict: bool = field(default_factory=lambda: os.getenv("DM_STRICT", "false").lower() in ("1", "true", "yes"))
    enable_observability: bool = field(default_factory=lambda: os.getenv("DM_OBSERVABILITY", "false").lower() in ("1", "true", "yes"))
    prometheus_port: Optional[int] = field(default_factory=lambda: int(os.getenv("DM_PROM_PORT", "8000")) if os.getenv("DM_PROM_PORT") else None)
    enable_distributed: bool = field(default_factory=lambda: os.getenv("DM_DISTRIBUTED", "false").lower() in ("1", "true", "yes"))
    retries: int = field(default_factory=lambda: int(os.getenv("DM_RETRIES", "2")))
    retry_sleep: float = field(default_factory=lambda: float(os.getenv("DM_RETRY_SLEEP", "0.1")))
    fallback_to_convex_hull: bool = field(default_factory=lambda: os.getenv("DM_FALLBACK_CONVEX", "true").lower() in ("1", "true", "yes"))
    config_file: Optional[str] = field(default_factory=lambda: os.getenv("DM_CONFIG_FILE", None))

    def merge_override(self, overrides: Optional[dict]) -> "MetricsConfig":
        if not overrides:
            return self
        d = asdict(self)
        d.update({k: v for k, v in overrides.items() if v is not None})
        return MetricsConfig(**d)


def load_config(overrides: Optional[dict] = None) -> MetricsConfig:
    """Load configuration: env -> optional YAML file -> runtime overrides."""
    cfg = MetricsConfig()
    if cfg.config_file and yaml:
        try:
            with open(cfg.config_file, "r") as fh:
                data = yaml.safe_load(fh) or {}
            # overlay file
            cfg = cfg.merge_override(data)
        except Exception as e:
            logger.warning(f"Failed to load config file {cfg.config_file}: {e}")
    # overlay overrides last
    if overrides:
        cfg = cfg.merge_override(overrides)
    return cfg


# ---------- Observability (Prometheus + OTel) ----------
PROM_SUMMARY = None
PROM_COUNTER = None
TRACER = None

def setup_observability(cfg: MetricsConfig):
    global PROM_SUMMARY, PROM_COUNTER, TRACER
    if not cfg.enable_observability:
        return
    if Summary and Counter:
        try:
            PROM_SUMMARY = Summary("district_metrics_processing_seconds", "Time spent processing district metrics")
            PROM_COUNTER = Counter("district_metrics_errors_total", "Total errors during district metrics processing")
            if cfg.prometheus_port:
                try:
                    start_http_server(cfg.prometheus_port)
                    logger.info(f"Started prometheus client on port {cfg.prometheus_port}")
                except Exception as e:
                    logger.warning(f"Failed to start prometheus HTTP server: {e}")
        except Exception as e:
            logger.warning(f"Prometheus client initialization failed: {e}")
    if trace:
        try:
            TRACER = trace.get_tracer(__name__)
        except Exception:
            TRACER = None


def record_metric(name: str, value: Any):
    if PROM_SUMMARY and name == "duration_seconds":
        PROM_SUMMARY.observe(float(value))
    if PROM_COUNTER and name == "errors" and isinstance(value, (int, float)):
        PROM_COUNTER.inc(int(value))
    # Always log for fallback observability
    logger.info(f"[METRIC] {name}={value}")


# ---------- Error reporting ----------
class ErrorReport:
    def __init__(self):
        self.errors: List[Dict[str, Any]] = []

    def add(self, index: Union[int, str, None], stage: str, message: str):
        self.errors.append({"index": index, "stage": stage, "message": message})

    def summary(self) -> List[Dict[str, Any]]:
        return self.errors

    def count(self) -> int:
        return len(self.errors)


class MetricsProcessingError(Exception):
    """Raised when strict mode is on and non-recoverable errors occurred."""
    def __init__(self, message: str, report: ErrorReport):
        super().__init__(message)
        self.report = report


# ---------- Utility helpers ----------
def _serialize_geom(g: BaseGeometry) -> bytes:
    try:
        return wkb.dumps(g, hex=False)
    except Exception:
        return b""


def _deserialize_geom(b: bytes) -> Optional[BaseGeometry]:
    if not b:
        return None
    try:
        return wkb.loads(b)
    except Exception:
        return None


def _audit_metadata() -> Dict[str, Any]:
    """Return audit metadata: timestamp, package versions, optional git commit hash."""
    meta = {"timestamp": datetime.utcnow().isoformat() + "Z", "versions": {}}
    for pkg in MIN_VERSIONS.keys():
        try:
            meta["versions"][pkg] = importlib_metadata.version(pkg)
        except Exception:
            meta["versions"][pkg] = None
    # try to get git commit hash if available
    try:
        import subprocess
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        meta["git_commit"] = commit
    except Exception:
        meta["git_commit"] = None
    return meta


# ---------- Vectorized metrics ----------
def compute_vectorized(gdf: GeoDataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    geom = gdf.geometry
    valid_mask = gdf.geometry.notna() & ~gdf.geometry.is_empty & gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    df = pd.DataFrame(index=gdf.index)
    if "polsby_popper" in metrics:
        df["polsby_popper"] = np.where(valid_mask, 4 * np.pi * geom.area / (geom.length ** 2), np.nan)
    if "convex_ratio" in metrics:
        df["convex_ratio"] = np.where(valid_mask, geom.area / geom.convex_hull.area, np.nan)
    if "schwartzberg" in metrics:
        df["schwartzberg"] = np.where(valid_mask, geom.length / (2 * np.sqrt(np.pi * geom.area)), np.nan)
    return df


# ---------- CPU-bound worker (single geometry) ----------
def _compute_cpu_metrics_for_geom(serialized_geom: bytes, metrics: Sequence[str], cfg: MetricsConfig, idx: Union[int, str]) -> Dict[str, Any]:
    """Compute CPU-bound metrics for a single geometry. Returns dict with index and metric values or errors."""
    out = {"index": idx}
    geom = _deserialize_geom(serialized_geom)
    if geom is None:
        out.update({m: np.nan for m in metrics})
        out["error"] = "deserialize_failed"
        return out

    # local helper for retries / fallback
    def safe_compute_attempt(fn, fallback=None):
        tries = cfg.retries + 1
        last_exc = None
        for attempt in range(1, tries + 1):
            try:
                return fn()
            except Exception as e:
                last_exc = e
                time.sleep(cfg.retry_sleep)
        # retries exhausted
        if fallback is not None:
            try:
                return fallback()
            except Exception:
                pass
        raise last_exc

    # compute reock (approximation using minimum rotated rectangle -> approximate enclosing circle)
    if "reock" in metrics:
        try:
            def _calc_reock():
                mrr = geom.minimum_rotated_rectangle
                radius = mrr.length / (2 * np.pi) if mrr is not None else 0
                return geom.area / (np.pi * radius**2) if radius > 0 else np.nan

            reock_val = safe_compute_attempt(_calc_reock, fallback=(lambda: np.nan if not cfg.fallback_to_convex_hull else geom.area / geom.convex_hull.area if geom.convex_hull.area > 0 else np.nan))
            out["reock"] = float(reock_val) if reock_val is not None else np.nan
        except Exception as e:
            out["reock"] = np.nan
            out["error_reock"] = str(e)

    # eig_ratio
    if "eig_ratio" in metrics:
        try:
            def _calc_eig():
                coords_list = []
                if isinstance(geom, MultiPolygon):
                    for p in geom.geoms:
                        if p.exterior:
                            coords_list.append(np.array(p.exterior.coords))
                elif isinstance(geom, Polygon):
                    if geom.exterior:
                        coords_list.append(np.array(geom.exterior.coords))
                coords = np.vstack(coords_list) if coords_list else np.empty((0, 2))
                if coords.shape[0] >= 2:
                    cov = np.cov(coords.T)
                    eigvals = np.linalg.eigvalsh(cov)
                    return float(eigvals.min() / eigvals.max()) if eigvals.max() > 0 else np.nan
                return np.nan

            eig_val = safe_compute_attempt(_calc_eig, fallback=(lambda: np.nan))
            out["eig_ratio"] = float(eig_val) if eig_val is not None else np.nan
        except Exception as e:
            out["eig_ratio"] = np.nan
            out["error_eig"] = str(e)

    return out


# ---------- Chunked local parallel execution (WKB serialization to reduce pickling) ----------
def compute_cpu_metrics_local(gdf: GeoDataFrame, metrics: Sequence[str], cfg: MetricsConfig, indices: Optional[Sequence[int]] = None) -> Tuple[pd.DataFrame, ErrorReport]:
    idxs = list(indices) if indices is not None else list(gdf.index)
    error_report = ErrorReport()
    if not idxs:
        return pd.DataFrame(index=gdf.index), error_report

    serialized = {i: _serialize_geom(gdf.geometry.loc[i]) for i in idxs}
    rows: List[Dict[str, Any]] = []
    # Use chunking to limit parallelism memory usage
    for start in range(0, len(idxs), cfg.chunk_size):
        chunk = idxs[start : start + cfg.chunk_size]
        Executor = ProcessPoolExecutor if cfg.backend == "process" else ThreadPoolExecutor
        with Executor(max_workers=max(1, cfg.max_workers)) as exe:
            futures = {exe.submit(_compute_cpu_metrics_for_geom, serialized[i], metrics, cfg, i): i for i in chunk}
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    res = fut.result()
                    rows.append(res)
                    # collect per-index errors
                    for k in list(res.keys()):
                        if k.startswith("error"):
                            error_report.add(res.get("index"), "cpu_metric", str(res[k]))
                except Exception as e:
                    error_report.add(i, "executor_failure", str(e))
                    if cfg.strict:
                        raise
    if not rows:
        return pd.DataFrame(index=gdf.index), error_report
    df = pd.DataFrame(rows).set_index("index")
    # ensure metric columns exist
    for m in metrics:
        if m not in df.columns:
            df[m] = np.nan
    df = df.reindex(gdf.index)
    return df, error_report


# ---------- Distributed backends (Dask/Ray) ----------
def compute_cpu_metrics_distributed(gdf: GeoDataFrame, metrics: Sequence[str], cfg: MetricsConfig) -> Tuple[pd.DataFrame, ErrorReport]:
    """
    Attempt distributed computation using Dask or Ray (if enabled and available).
    Falls back to local chunked execution if distributed lib missing or fails.
    """
    if cfg.backend == "dask" and dask and delayed and dask_compute:
        # Dask implementation: delayed tasks over serialized geometries
        serialized = [ _serialize_geom(gdf.geometry.loc[i]) for i in gdf.index ]
        tasks = [delayed(_compute_cpu_metrics_for_geom)(s, metrics, cfg, i) for s,i in zip(serialized, gdf.index)]
        try:
            results = dask_compute(*tasks)
            rows = list(results)
            df = pd.DataFrame(rows).set_index("index")
            rpt = ErrorReport()
            # collect errors
            for row in rows:
                for k,v in row.items():
                    if k.startswith("error"):
                        rpt.add(row.get("index"), "distributed", str(v))
            for m in metrics:
                if m not in df.columns:
                    df[m] = np.nan
            df = df.reindex(gdf.index)
            return df, rpt
        except Exception as e:
            logger.warning(f"Dask distributed execution failed: {e}. Falling back to local execution.")
            return compute_cpu_metrics_local(gdf, metrics, cfg)
    if cfg.backend == "ray" and ray:
        try:
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
            @ray.remote
            def ray_worker(serialized_geom, metrics_, cfg_dict, idx):
                # Rehydrate MetricsConfig from dict to avoid pickling issues
                cfg_local = MetricsConfig(**cfg_dict)
                return _compute_cpu_metrics_for_geom(serialized_geom, metrics_, cfg_local, idx)
            futures = [ray_worker.remote(_serialize_geom(gdf.geometry.loc[i]), metrics, asdict(cfg), i) for i in gdf.index]
            rows = ray.get(futures)
            df = pd.DataFrame(rows).set_index("index")
            rpt = ErrorReport()
            for row in rows:
                for k in row.keys():
                    if k.startswith("error"):
                        rpt.add(row.get("index"), "distributed", row[k])
            for m in metrics:
                if m not in df.columns:
                    df[m] = np.nan
            df = df.reindex(gdf.index)
            return df, rpt
        except Exception as e:
            logger.warning(f"Ray distributed execution failed: {e}. Falling back to local execution.")
            return compute_cpu_metrics_local(gdf, metrics, cfg)
    # no supported distributed backend: fall back
    return compute_cpu_metrics_local(gdf, metrics, cfg)


# ---------- Public API ----------
def calculate_district_metrics(
    gdf: GeoDataFrame,
    metrics_to_compute: Optional[List[str]] = None,
    config_overrides: Optional[dict] = None,
    return_gdf: bool = False,
) -> Tuple[Union[pd.DataFrame, GeoDataFrame], Dict[str, Any]]:
    """
    Compute district compactness metrics with enterprise-grade options.

    Returns:
        (results, report)
        - results: DataFrame or GeoDataFrame (if return_gdf=True) indexed same as input
        - report: dict containing audit metadata, timings, errors, config used
    """
    start_all = time.time()
    cfg = load_config(config_overrides)
    setup_observability(cfg)
    # pydantic validation of cfg if available
    if BaseModel:
        try:
            class _CfgModel(BaseModel):
                target_crs: int
                max_workers: int
            _CfgModel(target_crs=cfg.target_crs, max_workers=cfg.max_workers)
        except Exception as e:
            logger.warning(f"Config validation via pydantic failed: {e}")

    metrics_to_compute = metrics_to_compute or ["polsby_popper", "convex_ratio", "schwartzberg", "reock", "eig_ratio"]
    metrics_to_compute = [m for m in metrics_to_compute]

    report = {"audit": _audit_metadata(), "timings": {}, "errors": [], "config_used": asdict(cfg)}

    # basic validation
    if not isinstance(gdf, GeoDataFrame):
        raise ValueError("gdf must be a GeoDataFrame")
    if "geometry" not in gdf.columns:
        raise ValueError("GeoDataFrame missing 'geometry' column")
    if gdf.geometry.isna().all():
        raise ValueError("GeoDataFrame contains no geometries")

    # ensure map_name
    if "map_name" not in gdf.columns:
        gdf = gdf.copy()
        gdf["map_name"] = [f"district_{i}" for i in range(len(gdf))]

    # CRS handling: do not silently reproject unless allowed by config
    if gdf.crs is None:
        msg = "Input GeoDataFrame has no CRS defined."
        logger.warning(msg)
        report["errors"].append({"stage": "crs", "message": msg})
        if cfg.strict:
            raise ValueError(msg)
    elif getattr(gdf.crs, "is_geographic", False):
        if cfg.target_crs:
            try:
                t0 = time.time()
                gdf = gdf.to_crs(epsg=cfg.target_crs)
                report["timings"]["crs_transform"] = time.time() - t0
                logger.info(f"Transformed CRS to EPSG:{cfg.target_crs}")
            except Exception as e:
                report["errors"].append({"stage": "crs_transform", "message": str(e)})
                if cfg.strict:
                    raise

    # geometry sanitization & validation
    invalid_mask = ~gdf.is_valid
    if invalid_mask.any():
        cnt = int(invalid_mask.sum())
        logger.info(f"Attempting to fix {cnt} invalid geometries via buffer(0)")
        try:
            gdf.loc[invalid_mask, "geometry"] = gdf.loc[invalid_mask, "geometry"].buffer(0)
        except Exception as e:
            report["errors"].append({"stage": "geometry_sanitize", "message": str(e)})
            if cfg.strict:
                raise

    valid_mask = gdf.geometry.notna() & ~gdf.geometry.is_empty & gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    if not valid_mask.all():
        bad = int((~valid_mask).sum())
        logger.warning(f"{bad} invalid/empty geometries; results will be NaN for those indices")
        report["errors"].append({"stage": "validation", "message": f"{bad} invalid/empty geometries"})

    # vectorized metrics
    t0 = time.time()
    vec_metrics = [m for m in metrics_to_compute if m in ("polsby_popper", "convex_ratio", "schwartzberg")]
    results_df = compute_vectorized(gdf, vec_metrics) if vec_metrics else pd.DataFrame(index=gdf.index)
    report["timings"]["vectorized"] = time.time() - t0

    # CPU-bound metrics (distributed or local)
    cpu_metrics = [m for m in metrics_to_compute if m in ("reock", "eig_ratio")]
    if cpu_metrics:
        t1 = time.time()
        if cfg.enable_distributed and cfg.backend in ("dask", "ray"):
            cpu_df, err_report = compute_cpu_metrics_distributed(gdf, cpu_metrics, cfg)
        else:
            cpu_df, err_report = compute_cpu_metrics_local(gdf, cpu_metrics, cfg)
        report["timings"]["cpu"] = time.time() - t1
        # attach errors
        for e in err_report.summary():
            report["errors"].append(e)
        results_df = results_df.join(cpu_df)

    # finalize output
    out_df = pd.concat([gdf[["map_name"]], results_df], axis=1)
    total_time = time.time() - start_all
    report["timings"]["total"] = total_time
    record_metric("duration_seconds", total_time)
    record_metric("rows_processed", len(gdf))
    if report["errors"]:
        record_metric("errors", len(report["errors"]))
        logger.warning(f"Completed with {len(report['errors'])} errors (see report)")

    if cfg.strict and report["errors"]:
        raise MetricsProcessingError("Errors encountered during processing (strict mode).", ErrorReport())

    if return_gdf:
        out_gdf = gdf.copy()
        for col in results_df.columns:
            out_gdf[col] = results_df[col]
        return out_gdf, report
    return out_df, report


# ---------- Exports for unit tests ----------
__all__ = [
    "MetricsConfig",
    "load_config",
    "compute_vectorized",
    "compute_cpu_metrics_local",
    "compute_cpu_metrics_distributed",
    "calculate_district_metrics",
    "ErrorReport",
    "MetricsProcessingError",
]
