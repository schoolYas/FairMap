"""
Compute compactness and shape metrics for geographic districts with:
- Config via env, YAML, runtime overrides
- Local or distributed computation (Dask/Ray)
- Observability with Prometheus/OTel
- Robust error handling and retries
- Audit metadata for reproducibility
"""

# ---------------- Standard Library Imports ---------------------------------------------------
import os                                               # Environment variable access for config
import logging                                          # Core logging
import logging.config                                   # Structured logging configuration
from datetime import datetime                           # Audit timestamps
from dataclasses import dataclass, field, asdict        # Config and structured data management
from typing import Any, Dict, Optional, Sequence, Tuple,                              # Type hinting
from importlib import metadata as importlib_metadata                                  # Package version introspection
from __future__ import annotations                                                    # Forward references in type hints

# ---------------- Third-Party Imports ------------------------------------------------------
import numpy as np                                                                    # Numerical operations (vectorized)
import pandas as pd                                                                   # Tabular data handling
from geopandas import GeoDataFrame                                                    # Explicit class import
from shapely.geometry.base import BaseGeometry                                        # Base geometry type
from shapely import wkb                                                               # Geometry serialization/deserialization

# ---------------- Optional/Soft Imports --------------------------------------------------
# These are soft dependencies that enhance functionality but have safe fallbacks.
try:
    import yaml                                                                       # YAML config support
except ImportError:
    yaml = None

try:
    from packaging.version import parse as parse_version                              # Version parsing for dependencies
except ImportError:
    parse_version = None

try:
    from prometheus_client import Summary, Counter, start_http_server                 # Observability metrics
except ImportError:
    Summary = None
    Counter = None
    start_http_server = None

try:
    from opentelemetry import trace
    from opentelemetry.trace import Tracer                                            # Tracing
except ImportError:
    trace = None
    Tracer = None

# ---------------- Logging Setup ---------------------------------------------------------
# Structured JSON logging for enterprise observability
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

# ---------------- Dependency Version Checking ---------------------------------------------
MIN_VERSIONS = {"numpy": "1.21", "pandas": "1.3", "geopandas": "0.12", "shapely": "2.0"}

def _safe_parse_version(ver_str: str) -> Tuple[int, ...]:
    """Fallback tuple-based version parsing for comparison."""
    import re
    nums = re.findall(r"\d+", ver_str)
    return tuple(int(n) for n in nums) if nums else ()

def check_versions(min_versions: Dict[str, str] = MIN_VERSIONS) -> None:
    """Ensure required package versions are installed."""
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

check_versions()

# ---------------- Configuration --------------------------------------------------------
@dataclass
class MetricsConfig:
    """
    Centralized configuration for metrics computation.
    Supports env variables, optional YAML, and runtime overrides.
    """
    target_crs: int = field(default_factory=lambda: int(os.getenv("DM_TARGET_CRS", "3857")))
    max_workers: int = field(default_factory=lambda: int(os.getenv("DM_MAX_WORKERS", "4")))
    backend: str = field(default_factory=lambda: os.getenv("DM_BACKEND", "process"))
    chunk_size: int = field(default_factory=lambda: int(os.getenv("DM_CHUNK_SIZE", "1000")))
    strict: bool = field(default_factory=lambda: os.getenv("DM_STRICT", "false").lower() in ("1","true","yes"))
    enable_observability: bool = field(default_factory=lambda: os.getenv("DM_OBSERVABILITY", "false").lower() in ("1","true","yes"))
    prometheus_port: Optional[int] = field(default_factory=lambda: int(os.getenv("DM_PROM_PORT", "8000")) if os.getenv("DM_PROM_PORT") else None)
    enable_distributed: bool = field(default_factory=lambda: os.getenv("DM_DISTRIBUTED", "false").lower() in ("1","true","yes"))
    retries: int = field(default_factory=lambda: int(os.getenv("DM_RETRIES","2")))
    retry_sleep: float = field(default_factory=lambda: float(os.getenv("DM_RETRY_SLEEP","0.1")))
    fallback_to_convex_hull: bool = field(default_factory=lambda: os.getenv("DM_FALLBACK_CONVEX","true").lower() in ("1","true","yes"))
    config_file: Optional[str] = field(default_factory=lambda: os.getenv("DM_CONFIG_FILE", None))

    def merge_override(self, overrides: Optional[dict]) -> "MetricsConfig":
        """Merge runtime overrides into config."""
        if not overrides:
            return self
        d = asdict(self)
        d.update({k: v for k, v in overrides.items() if v is not None})
        return MetricsConfig(**d)

def load_config(overrides: Optional[dict] = None) -> MetricsConfig:
    """Load config from env, optional YAML, and runtime overrides."""
    cfg = MetricsConfig()
    if cfg.config_file and yaml:
        try:
            with open(cfg.config_file, "r") as fh:
                data = yaml.safe_load(fh) or {}
            cfg = cfg.merge_override(data)
        except Exception as e:
            logger.warning(f"Failed to load config file {cfg.config_file}: {e}")
    if overrides:
        cfg = cfg.merge_override(overrides)
    return cfg

# ---------------- Observability Setup -------------------------------------------------
PROM_SUMMARY = None
PROM_COUNTER = None
TRACER = None

def setup_observability(cfg: MetricsConfig):
    """Initialize Prometheus metrics and OpenTelemetry tracing if enabled."""
    global PROM_SUMMARY, PROM_COUNTER, TRACER
    if not cfg.enable_observability:
        return
    if Summary and Counter:
        try:
            PROM_SUMMARY = Summary("district_metrics_processing_seconds", "Processing time")
            PROM_COUNTER = Counter("district_metrics_errors_total", "Error count")
            if cfg.prometheus_port:
                start_http_server(cfg.prometheus_port)
                logger.info(f"Prometheus server started on port {cfg.prometheus_port}")
        except Exception as e:
            logger.warning(f"Prometheus initialization failed: {e}")
    if trace:
        try:
            TRACER = trace.get_tracer(__name__)
        except Exception:
            TRACER = None

def record_metric(name: str, value: Any):
    """Record metrics to Prometheus if available, always log."""
    if PROM_SUMMARY and name == "duration_seconds":
        PROM_SUMMARY.observe(float(value))
    if PROM_COUNTER and name == "errors" and isinstance(value, (int,float)):
        PROM_COUNTER.inc(int(value))
    logger.info(f"[METRIC] {name}={value}")

# ---------------- Geometry Utilities --------------------------------------------------
def _serialize_geom(g: BaseGeometry) -> bytes:
    """Serialize a geometry to WKB bytes, fallback empty bytes on failure."""
    try:
        return wkb.dumps(g, hex=False)
    except Exception:
        return b""

def _deserialize_geom(b: bytes) -> Optional[BaseGeometry]:
    """Deserialize WKB bytes to geometry, return None if invalid."""
    if not b:
        return None
    try:
        return wkb.loads(b)
    except Exception:
        return None

# ---------------- Audit Metadata ------------------------------------------------------
def _audit_metadata() -> Dict[str, Any]:
    """Return timestamp, package versions, and git commit info."""
    meta = {"timestamp": datetime.utcnow().isoformat() + "Z", "versions": {}}
    for pkg in MIN_VERSIONS.keys():
        try:
            meta["versions"][pkg] = importlib_metadata.version(pkg)
        except Exception:
            meta["versions"][pkg] = None
    try:
        import subprocess
        commit = subprocess.check_output(["git","rev-parse","HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        meta["git_commit"] = commit
    except Exception:
        meta["git_commit"] = None
    return meta

# ---------------- Vectorized Metrics ----------------------------------------------------
def compute_vectorized(gdf: GeoDataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    """
    Compute fast vectorized metrics: Polsby-Popper, Convex Ratio, Schwartzberg.
    Returns DataFrame with metric columns.
    """
    geom = gdf.geometry
    valid_mask = gdf.geometry.notna() & ~gdf.geometry.is_empty & gdf.geometry.geom_type.isin(["Polygon","MultiPolygon"])
    df = pd.DataFrame(index=gdf.index)

    if "polsby_popper" in metrics:
        df["polsby_popper"] = np.where(valid_mask, 4 * np.pi * geom.area / (geom.length ** 2), np.nan)
    if "convex_ratio" in metrics:
        df["convex_ratio"] = np.where(valid_mask, geom.area / geom.convex_hull.area, np.nan)
    if "schwartzberg" in metrics:
        df["schwartzberg"] = np.where(valid_mask, geom.length / (2 * np.sqrt(np.pi * geom.area)), np.nan)

    return df


