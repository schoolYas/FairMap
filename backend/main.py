import logging
import mimetypes
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
import geopandas as gpd
import pandas as pd
from pydantic import BaseModel
from io import BytesIO
import os
import zipfile
import tempfile
from concurrent.futures import ThreadPoolExecutor
from time import time
from collections import defaultdict

# Allow non-closed rings if needed
os.environ["OGR_GEOMETRY_ACCEPT_UNCLOSED_RING"] = "YES"

app = FastAPI(title="FairMap Backend", version="1.0")

# --- CORS middleware (added so React frontend can call backend) ---
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input model for predictions
class PredictionInput(BaseModel):
    historical_votes: list  # e.g., [Dem_votes, Rep_votes]
    demographics: dict      # e.g., {"population": 10000, "minority_pct": 30}

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB max upload size

# --- Constants ----------------------------------------------------------------------------
# Supported file types for upload
SUPPORTED_FILE_TYPES = [".geojson", ".shp"]
SHAPEFILE_EXTENSIONS = [".shp", ".shx", ".dbf", ".prj", ".cpg"]
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB max upload size
CHUNK_SIZE = 4 * 1024 * 1024
ALLOWED_MIME_TYPES = {
    ".geojson": "application/geo+json",
    ".zip": "application/zip"
}
ip_upload_history = defaultdict(list)
UPLOAD_RATE_LIMIT = 5  # max uploads per IP per minute
CRS_STANDARD = "EPSG:4326"
executor = ThreadPoolExecutor(max_workers=4)
# --- Constants End -------------------------------------------------------------------------

# --- Utility Functions ---------------------------------------------------------------------
def validate_file_type(filename: str):
    if not filename.endswith((".geojson", ".zip")):
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
async def read_geojson_async(file: BytesIO) -> gpd.GeoDataFrame:
    return await asyncio.to_thread(lambda: gpd.read_file(file))

async def read_shapefile_async(tmp_zip_path: str, tmpdir: str) -> gpd.GeoDataFrame:
    return await asyncio.to_thread(lambda: gpd.read_file(tmp_zip_path))


def read_geojson(file: UploadFile) -> gpd.GeoDataFrame:
    """Read a GeoJSON file into GeoDataFrame with error logging."""
    try:
        content = BytesIO(file.file.read())
        gdf = gpd.read_file(content)
        return gdf
    except Exception as e:
        logger.error(f"Failed to read GeoJSON '{file.filename}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to read GeoJSON: {str(e)}")

def read_shapefile(zip_file: UploadFile) -> gpd.GeoDataFrame:
    """Read a zipped shapefile safely into GeoDataFrame with error logging."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, zip_file.filename)
            with open(zip_path, "wb") as f:
                f.write(zip_file.file.read())

            # Extract zip safely
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmpdir)

            shp_files = [f for f in os.listdir(tmpdir) if f.endswith(".shp")]
            if not shp_files:
                raise HTTPException(status_code=400, detail="No .shp file found in zip")

            shp_path = os.path.join(tmpdir, shp_files[0])
            gdf = gpd.read_file(shp_path)
            return gdf
    except Exception as e:
        logger.error(f"Failed to read shapefile '{zip_file.filename}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to read shapefile: {str(e)}")

def calculate_basic_metrics(gdf: gpd.GeoDataFrame) -> dict:
    """Placeholder fairness metrics (replace with real calculations)."""
    num_districts = len(gdf)

    metrics = {
        "num_districts": num_districts,
        "compactness": 0.75,      # Replace with real calculation
        "efficiency_gap": 0.12    # Replace with real calculation
    }
    return metrics
# --- Utility Functions End ------------------------------------------------------------------

# --- Endpoints ------------------------------------------------------------------------------
@app.post("/upload-map")
async def upload_map(file: UploadFile = File(...), request: Request = None):
    client_ip = request.client.host if request else "unknown"
    
    # --- Rate limiting ---
    now = time()
    window_start = now - 60
    ip_upload_history[client_ip] = [t for t in ip_upload_history[client_ip] if t > window_start]
    if len(ip_upload_history[client_ip]) >= UPLOAD_RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too many uploads; please wait before uploading again")
    ip_upload_history[client_ip].append(now)

    try:
        # --- File type & MIME type validation ---
        validate_file_type(file.filename)
        ext = os.path.splitext(file.filename)[1].lower()
        ALLOWED_MIME_TYPES = {".geojson": "application/geo+json", ".zip": "application/zip"}
        if file.content_type != ALLOWED_MIME_TYPES.get(ext):
            raise HTTPException(status_code=400, detail=f"Invalid MIME type: {file.content_type}")

        # --- Streaming / file size check ---
        CHUNK_SIZE = 4 * 1024 * 1024
        total_bytes = 0
        temp_file = BytesIO()
        while chunk := await file.read(CHUNK_SIZE):
            total_bytes += len(chunk)
            if total_bytes > MAX_FILE_SIZE:
                raise HTTPException(status_code=413, detail="File too large")
            temp_file.write(chunk)
        temp_file.seek(0)

        logger.info(f"Upload attempt: {file.filename}, size={total_bytes} bytes, from IP={client_ip}")

        # --- Read GeoJSON or Shapefile ---
        if ext == ".geojson":
            gdf = await read_geojson_async(temp_file)
        elif ext == ".zip":
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, os.path.basename(file.filename))
                with open(zip_path, "wb") as f:
                    f.write(temp_file.getbuffer())

                # --- Zip traversal safety ---
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    for member in zip_ref.namelist():
                        if os.path.isabs(member) or ".." in member:
                            raise HTTPException(status_code=400, detail="Invalid zip contents")
                    zip_ref.extractall(tmpdir)

                shp_files = [f for f in os.listdir(tmpdir) if f.endswith(".shp")]
                if not shp_files:
                    raise HTTPException(status_code=400, detail="No .shp file found in zip")
                shp_path = os.path.join(tmpdir, shp_files[0])
                gdf = await read_shapefile_async(shp_path, tmpdir)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        # --- Fix invalid geometries ---
        if not gdf.is_valid.all():
            logger.warning(f"Invalid geometries detected in '{file.filename}', attempting to fix with buffer(0)")
            gdf["geometry"] = gdf.buffer(0)
            if not gdf.is_valid.all():
                logger.warning(f"Some geometries in '{file.filename}' could not be fixed")

        # --- CRS normalization ---
        if gdf.crs is None or gdf.crs.to_string() != CRS_STANDARD:
            gdf = gdf.to_crs(CRS_STANDARD)
            logger.info(f"Reprojected '{file.filename}' to {CRS_STANDARD}")

        # --- Prepare response metadata ---
        geojson_obj = gdf.__geo_interface__ if gdf is not None else {"type": "FeatureCollection", "features": []}
        bbox = gdf.total_bounds.tolist() if gdf is not None else []
        num_features = len(gdf) if gdf is not None else 0

        logger.info(f"Upload success: {file.filename}, districts={num_features}")
        return JSONResponse(content={
            "filename": file.filename,
            "num_districts": num_features,
            "geojson": geojson_obj,
            "bbox": bbox,
            "status": "Map uploaded successfully"
        })

    except HTTPException as e:
        logger.warning(f"Upload failed ({file.filename}) from {client_ip}: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error during upload ({file.filename}) from {client_ip}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/calculate-metrics")
async def calculate_metrics(file: UploadFile):
    """Calculate fairness metrics for an uploaded electoral district map."""
    validate_file_type(file.filename)

    if file.filename.endswith(".geojson"):
        gdf = read_geojson(file)
    elif file.filename.endswith(".zip"):
        gdf = read_shapefile(file)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")

    metrics = calculate_basic_metrics(gdf)

    return JSONResponse(content={
        "filename": file.filename,
        "metrics": metrics,
        "status": "Metrics calculated successfully"
    })

@app.post("/run-simulation")
async def run_simulation(simulation_params: dict):
    """Run an ensemble simulation (dummy placeholder)."""
    try:
        num_simulations = simulation_params.get("num_simulations", 100)
        metric_weights = simulation_params.get("metric_weights", {"compactness": 1, "efficiency_gap": 1})

        simulated_results = [{"simulation": i, "score": i * 0.1} for i in range(num_simulations)]

        return JSONResponse(content={
            "num_simulations": num_simulations,
            "metric_weights": metric_weights,
            "results": simulated_results,
            "status": "Simulation completed successfully"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-outcome")
async def predict_outcome(input_data: PredictionInput):
    """Predict partisan outcomes (placeholder logic)."""
    try:
        dem_votes = input_data.historical_votes[0]
        rep_votes = input_data.historical_votes[1]

        predicted_dem_share = (dem_votes / (dem_votes + rep_votes)) * 100
        predicted_rep_share = 100 - predicted_dem_share

        return {
            "predicted_dem_share": round(predicted_dem_share, 2),
            "predicted_rep_share": round(predicted_rep_share, 2),
            "status": "Prediction completed successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/export-report")
async def export_report(request: Request):
    """Export uploaded metrics as a CSV file."""
    try:
        data = await request.json()
        metrics = data.get("metrics")

        if not metrics or not isinstance(metrics, list):
            return JSONResponse(
                status_code=400,
                content={"detail": "Invalid payload: 'metrics' key missing or not a list"}
            )

        df = pd.DataFrame(metrics)
        file_path = "report.csv"
        df.to_csv(file_path, index=False)

        return FileResponse(
            path=file_path,
            filename="report.csv",
            media_type="text/csv"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "FairMap backend is running"}