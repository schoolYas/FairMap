# --- Standard library imports ----------------------------------------------------------------
import asyncio                                      # For async operations, concurrency, and semaphores
import logging                                      # For logging upload attempts, errors, warnings
import os                                           # Environment variables, path manipulations
import tempfile                                     # Temporary directories/files for zip extraction
import time                                         # Timing, rate-limiting, upload latency
import uuid                                         # Generate unique IDs for chunked uploads
import zipfile                                      # Handling uploaded zip files (shapefiles)
from io import BytesIO                              # In-memory file streams for uploads
from concurrent.futures import ThreadPoolExecutor   # Run blocking code (like geopandas) in threads
from collections import defaultdict                 # Store per-IP upload history easily

# --- Third-party imports ----------------------------------------------------------------------
import pyclamd                                      # Scan uploaded zip files for malware with ClamAV
import magic                                        # Detects MIME Types 
import geopandas as gpd                             # Read and manipulate GeoJSON/shapefiles
import pandas as pd                                 # Create DataFrames for exporting metrics/reports
from fastapi import FastAPI, UploadFile, File, HTTPException, Request   # Core FastAPI functionality
from fastapi.responses import JSONResponse, FileResponse                # Send JSON or CSV responses
from pydantic import BaseModel                                          # Define input models for predictions
from contextlib import asynccontextmanager                              # Used in Cleanup and Upload Tracking 

# --- Local imports ----------------------------------------------------------------------------
from logging_setup import logger, UPLOAD_SUCCESS, UPLOAD_FAILURE, UPLOAD_SIZE, UPLOAD_LATENCY
# logger: logging setup for backend
# UPLOAD_SUCCESS/UPLOAD_FAILURE: Prometheus counters for upload results
# UPLOAD_SIZE: Histogram of upload sizes
# UPLOAD_LATENCY: Histogram of upload processing time
#-----------------------------------------------------------------------------------------------

# Allow non-closed polygon geometry rings if needed
os.environ["OGR_GEOMETRY_ACCEPT_UNCLOSED_RING"] = "YES"

#---- Abandoned Upload Cleaner -----------------------------------------------------------------

""" CLEANUP_ABANDONED_UPLOADS
    Periodically removes abandoned chunked uploads from memory to prevent memory bloat.

    - Iterates over all in-progress uploads tracked in UPLOAD_CHUNKS_LAST_MODIFIED.
    - If an upload has not been modified within UPLOAD_CHUNK_TIMEOUT seconds, it is considered abandoned.
    - Deletes the abandoned upload data from both UPLOAD_CHUNKS and UPLOAD_CHUNKS_LAST_MODIFIED.
    - Logs the cleanup of each abandoned upload.
    - Sleeps for 60 seconds before repeating, running continuously in the background.  
"""
async def cleanup_abandoned_uploads():
    while True:
        now = time.time()
        abandoned = [uid for uid, ts in UPLOAD_CHUNKS_LAST_MODIFIED.items() if now - ts > UPLOAD_CHUNK_TIMEOUT]
        for uid in abandoned:
            del UPLOAD_CHUNKS[uid]
            del UPLOAD_CHUNKS_LAST_MODIFIED[uid]
            logger.info(f"Cleaned up abandoned upload: {uid}")
        await asyncio.sleep(60)  # Run cleanup every minute

""" LIFESPAN
    Lifespan context manager for the FastAPI app.

    - Starts background tasks required during app lifetime.
    - Here, it launches cleanup_abandoned_uploads() as a background task to periodically free memory.
    - The 'yield' allows the app to continue running while background tasks operate.
    - Can be extended to include shutdown tasks if needed.
"""
@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(cleanup_abandoned_uploads())
    yield

# Creates App with FastAPI
app = FastAPI(title="FairMap Backend", version="1.0", lifespan = lifespan)
#----------------------------------------------------------------------------------------------

# --- CORS middleware (added so React frontend can call backend) ------------------------------
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    # List of allowed origins that can make requests to this backend
    allow_origins=[
        "http://localhost:3000",                    # Local React dev server
        "http://127.0.0.1:3000",                    # Alternative localhost format
    ],
    allow_credentials=True,                         # Allow authorization headers in requests
    allow_methods=["*"],                            # Allow all HTTP methods
    allow_headers=["*"],                            # Allow all HTTP headers in requests
)
#---------------------------------------------------------------------------------------------

# Define input model for predictions
class PredictionInput(BaseModel):
    historical_votes: list  # e.g., [Dem_votes, Rep_votes]
    demographics: dict      # e.g., {"population": 10000, "minority_pct": 30}

#----- Initialize logger----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)             # Configure root logger to show INFO level and above
logger = logging.getLogger("fairMap")               # Create a named logger for the FairMap backend

# --- File Type and Content Constraints ------------------------------------------------------
SUPPORTED_FILE_TYPES = [".geojson", ".shp"]                     # Allowed individual file types
SHAPEFILE_EXTENSIONS = [".shp", ".shx", ".dbf", ".prj", ".cpg"] # Required shapefile components
ALLOWED_ZIP_CONTENTS = [".shp", ".shx", ".dbf", ".prj", ".cpg"] # Allowed files inside uploaded ZIPs
ALLOWED_MIME_TYPES = {                                          # MIME types for validation
    ".geojson": "application/geo+json",
    ".zip": "application/zip"
}                                                                

# --- Upload Size and Chunking -----------------------------------------------------------------
MAX_FILE_SIZE = 50 * 1024 * 1024                                # 50 MB max upload size
CHUNK_SIZE = 4 * 1024 * 1024                                    # Read in 4 MB chunks

# --- Rate Limiting / Upload Tracking ----------------------------------------------------------
UPLOAD_RATE_LIMIT = 5                                           # Max uploads per IP per minute
ip_upload_history = defaultdict(list)                           # Track per-IP upload timestamps
UPLOAD_CHUNKS = {}                                              # Track in-progress chunked uploads
UPLOAD_CHUNK_TIMEOUT = 15 * 60                                  # 15 minutes
UPLOAD_CHUNKS_LAST_MODIFIED =  {}                               # Track last write timestamp per upload_id


# --- Concurrency / Timeouts -------------------------------------------------------------------
MAX_CONCURRENT_UPLOADS = 4                                      # Max simultaneous uploads
executor = ThreadPoolExecutor(max_workers=4)                    # Thread pool for blocking tasks (e.g., geopandas)
upload_semaphore = asyncio.Semaphore(MAX_CONCURRENT_UPLOADS)    # Limit concurrent async uploads
UPLOAD_TIMEOUT = 30                                             # Timeout per upload in seconds

# --- CRS / Geo-Settings ------------------------------------------------------------------------
CRS_STANDARD = "EPSG:4326"                                      # Standard coordinate reference system
# --- Miscellaneous Constraints End -------------------------------------------------------------

# --- Utility Functions -------------------------------------------------------------------------

# ----------Used In /Upload-Map ------------------------------------------------------------------

""" VALIDATE_FILE_TYPE
    Validate the file extension of an uploaded file.

    Parameters:
        filename (str): Name of the uploaded file.

    Raises:
        HTTPException: If the file extension is not supported (.geojson or .zip).

    Notes:
        This is a quick, preliminary check based on the filename.
        It does not inspect file content — use validate_file_mime() for that.
"""
def validate_file_type(filename: str):
    if not filename.endswith((".geojson", ".zip")):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type",
            headers={"X-Error-Code": "unsupported_file_type"}
        )
    
""" VALIDATE_FILE_MIME
    Validate the MIME type of an uploaded file using python-magic.

    Parameters:
        file (UploadFile): The uploaded file object from FastAPI.

    Raises:
        HTTPException: If the MIME type does not match the expected type
                       for the file extension.

    Steps:
        1. Reset file pointer to the start.
        2. Read the first 1024 bytes to determine the MIME type.
        3. Reset file pointer again for further processing.
        4. Compare detected MIME type to expected type based on extension.
        5. Raise HTTPException if MIME type is invalid.

    Notes:
        This prevents users from renaming malicious files to bypass extension checks.
"""
def validate_file_mime(file: UploadFile):
    file.seek(0)
    mime_type = magic.from_buffer(file.file.read(1024), mime=True)
    file.seek(0)
    ext = os.path.splitext(file.filename)[1].lower()
    expected = ALLOWED_MIME_TYPES.get(ext)
    if expected != mime_type:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid MIME type: {mime_type} for extension {ext}",
            headers={"X-Error-Code": "invalid_mime"}
        )


""" SANITIZE_FILENAME
    Sanitize a filename to prevent path traversal attacks.

    Parameters:
        name (str): Original filename (possibly including directories).

    Returns:
        str: A safe filename containing only the base name (no directories).

    Notes:
        - Removes any path components (../ or absolute paths).
        - Useful when saving user uploads to disk or logging filenames.
"""
def sanitize_filename(name: str):
    return os.path.basename(name)

""" ERROR_RESPONSE
    Generate a structured JSON error response for API clients.

    Parameters:
        detail (str): Human-readable error message.
        code (str): Internal error code for the client.
        status_code (int, optional): HTTP status code. Defaults to 400.

    Returns:
        JSONResponse: FastAPI JSON response containing the error details.

    Notes:
        - Standardizes error responses across endpoints.
        - Can be used instead of raising HTTPException for more controlled output.
"""
def error_response(detail: str, code: str, status_code: int = 400):
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": detail, "code": code, "status": status_code}}
    )
        
""" SCAN_ZIP_FOR_DANGER
    Scan the contents of an uploaded ZIP file for suspicious or disallowed files.

    Parameters:
        zip_file (BytesIO): In-memory ZIP file uploaded by the user.

    Raises:
        HTTPException: 
            - If any filename in the ZIP is absolute or contains '../' (path traversal).
            - If any file inside the ZIP is not in the allowed list (ALLOWED_ZIP_CONTENTS).

    Notes:
        - Protects against ZIP attacks such as path traversal or unwanted files.
        - Only performs a whitelist check; does not scan file contents.
"""
async def scan_zip_for_danger(zip_file: BytesIO):
    with zipfile.ZipFile(zip_file) as zf:
        for f in zf.namelist():
            if os.path.isabs(f) or ".." in f:
                raise HTTPException(status_code=400, detail="Invalid zip contents", 
                                    headers={"X-Error-Code": "invalid_zip_path"})
            ext = os.path.splitext(f)[1].lower()
            if ext not in ALLOWED_ZIP_CONTENTS:
                raise HTTPException(status_code=400, detail=f"Unexpected file in zip: {f}", 
                                    headers={"X-Error-Code": "invalid_zip_file"})

""" CLAMAV_SCAN_ZIP
    Scan an uploaded ZIP file in-memory for malware using ClamAV.

    Parameters:
        temp_file (BytesIO): In-memory ZIP file uploaded by the user.

    Raises:
        HTTPException: 
            - 503 if ClamAV service is unavailable.
            - 400 if malware is detected in the uploaded ZIP.

    Notes:
        - Attempts to connect first via network socket, then Unix socket.
        - Resets file pointer before and after scanning to preserve file content.
        - Protects server from malicious uploads containing viruses or trojans.
"""
async def clamav_scan_zip(temp_file: BytesIO):
    cd = pyclamd.ClamdNetworkSocket()
    if not cd.ping():
        cd = pyclamd.ClamdUnixSocket()
    if not cd.ping():
        raise HTTPException(status_code=503, detail="ClamAV service unavailable", headers={"X-Error-Code": "clamav_down"})
    
    temp_file.seek(0)
    scan_result = cd.scan_stream(temp_file.read())
    if scan_result:
        raise HTTPException(
            status_code=400,
            detail=f"Malware detected in uploaded zip: {scan_result}",
            headers={"X-Error-Code": "malware_detected"}
        )
    temp_file.seek(0)

"""
    Safely extract a ZIP file to a temporary directory while enforcing a whitelist.

    Parameters:
        zip_path (str): Path to the ZIP file on disk.
        extract_to (str): Directory to extract files into.

    Raises:
        HTTPException: 
            - If any file inside the ZIP is absolute or contains "../".
            - If any file extension is not in the allowed whitelist.
            - If required shapefile components are missing (.shp, .shx, .dbf, .prj).

    Notes:
        - Prevents path traversal attacks.
        - Ensures shapefile completeness for GIS processing.
        - Raises early if the ZIP contains unexpected files.
"""
def safe_extract_zip(zip_path: str, extract_to: str):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        files_in_zip = zip_ref.namelist()
        for f in zip_ref.namelist():
            if os.path.isabs(f) or ".." in f:
                raise HTTPException(status_code=400, detail="Invalid zip contents", headers={"X-Error-Code": "invalid_zip_path"})
            ext = os.path.splitext(f)[1].lower()
            if ext not in ALLOWED_ZIP_CONTENTS:
                raise HTTPException(status_code=400, detail=f"Unexpected file in zip: {f}", headers={"X-Error-Code": "invalid_zip_file"})
        zip_ref.extractall(extract_to)
        extracted_exts = {os.path.splitext(f)[1].lower() for f in files_in_zip}
        for req_ext in SHAPEFILE_EXTENSIONS:
            if req_ext not in extracted_exts:
                raise HTTPException(status_code=400, detail=f"Missing required shapefile component: {req_ext}", headers={"X-Error-Code": "missing_shapefile_component"})

""" READ_GEOJSON
    Read an in-memory GeoJSON file into a GeoDataFrame asynchronously.

    Parameters:
        file (BytesIO): Uploaded GeoJSON file.

    Returns:
        gpd.GeoDataFrame: Parsed geospatial data.

    Notes:
        - Uses asyncio.to_thread to offload blocking geopandas I/O.
        - Resets file pointer before reading.
"""
async def read_geojson(file: BytesIO) -> gpd.GeoDataFrame:
    file.seek(0)
    return await asyncio.to_thread(lambda: gpd.read_file(file))


""" READ_SHAPEFILE
    Read an in-memory ZIP containing a shapefile into a GeoDataFrame.

    Parameters:
        file (BytesIO): Uploaded shapefile ZIP.

    Returns:
        gpd.GeoDataFrame: Parsed geospatial data from the .shp file.

    Raises:
        HTTPException:
            - If ClamAV is unavailable or malware detected.
            - If no .shp file is found after extraction.

    Notes:
        - Scans ZIP for malware before processing.
        - Uses a temporary directory for extraction.
        - Reads only the first .shp file found in the ZIP.
        - Offloads blocking geopandas I/O to a separate thread.
"""
async def read_shapefile_zip(file: BytesIO) -> gpd.GeoDataFrame:
    file.seek(0)
    # Scan for malware
    cd = pyclamd.ClamdNetworkSocket()
    if not cd.ping():
        cd = pyclamd.ClamdUnixSocket()
    if not cd.ping():
        raise HTTPException(status_code=503, detail="ClamAV unavailable", headers={"X-Error-Code": "clamav_down"})
    scan_result = cd.scan_stream(file.read())
    if scan_result:
        raise HTTPException(status_code=400, detail=f"Malware detected: {scan_result}", headers={"X-Error-Code": "malware_detected"})
    file.seek(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, str(uuid.uuid4()) + ".zip")
        with open(zip_path, "wb") as f:
            f.write(file.read())
        safe_extract_zip(zip_path, tmpdir)
        shp_files = [f for f in os.listdir(tmpdir) if f.endswith(".shp")]
        if not shp_files:
            raise HTTPException(status_code=400, detail="No .shp file found", headers={"X-Error-Code": "no_shp_in_zip"})
        return await asyncio.to_thread(lambda: gpd.read_file(os.path.join(tmpdir, shp_files[0])))
    
""" FIX_INVALID_GEOMETRIES
    Fix invalid geometries in a GeoDataFrame by buffering.

    Parameters:
        gdf (gpd.GeoDataFrame): GeoDataFrame to validate.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with corrected geometries.

    Notes:
        - Uses buffer(0) trick to repair invalid polygons.
        - Leaves valid geometries unchanged.
"""
def fix_invalid_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if not gdf.is_valid.all():
        gdf["geometry"] = gdf.buffer(0)
    return gdf
#----End of Used In /Upload-Map ------------------------------------------------------------------


def calculate_basic_metrics(gdf: gpd.GeoDataFrame) -> dict:
    return {"num_districts": len(gdf), "compactness": 0.75, "efficiency_gap": 0.12}


#-- Upload Handling ------------------------------------------------------------------------------

async def _handle_upload(file: UploadFile, client_ip: str, upload_id: str, is_final_chunk: bool):
    # Enforce max file size per chunk
    chunk = await file.read(CHUNK_SIZE)
    if upload_id not in UPLOAD_CHUNKS:
        UPLOAD_CHUNKS[upload_id] = BytesIO()
    UPLOAD_CHUNKS[upload_id].write(chunk)
    total_size = UPLOAD_CHUNKS[upload_id].getbuffer().nbytes
    if total_size > MAX_FILE_SIZE:
        del UPLOAD_CHUNKS[upload_id]
        raise HTTPException(status_code=413, detail="File exceeds max size", headers={"X-Error-Code": "max_file_size"})
    UPLOAD_CHUNKS_LAST_MODIFIED[upload_id] = time.time()

    if not is_final_chunk:
        return JSONResponse(content={"upload_id": upload_id, "status": "chunk_received"})

    temp_file = UPLOAD_CHUNKS.pop(upload_id)
    temp_file.seek(0)
    ext = os.path.splitext(file.filename)[1].lower()
    logger.info(f"Upload attempt {upload_id}: {file.filename}, size={total_size} bytes, IP={client_ip}")

    # MIME validation
    validate_file_mime(file)

    # File reading
    if ext == ".geojson":
        gdf = await read_geojson(temp_file)
    elif ext == ".zip":
        gdf = await read_shapefile_zip(temp_file)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format", headers={"X-Error-Code": "invalid_file_format"})

    gdf = fix_invalid_geometries(gdf)
    if gdf.crs is None or gdf.crs.to_string() != CRS_STANDARD:
        gdf = gdf.to_crs(CRS_STANDARD)

    geojson_obj = gdf.__geo_interface__
    bbox = gdf.total_bounds.tolist()
    num_features = len(gdf)

    logger.info(f"Upload success {upload_id}: {file.filename}, features={num_features}")
    UPLOAD_SUCCESS.inc()

    return JSONResponse(content={
        "upload_id": upload_id,
        "filename": sanitize_filename(file.filename),
        "num_districts": num_features,
        "geojson": geojson_obj,
        "bbox": bbox,
        "status": "Map uploaded successfully"
    })
# -- End of Upload Handling -----------------------------------------------------------------
# --- Utility Functions End ------------------------------------------------------------------

# --- Endpoints ------------------------------------------------------------------------------

@app.post("/upload-map")
async def upload_map(file: UploadFile = File(...), request: Request = None):
    client_ip = request.client.host if request else "unknown"

    now = time.time()
    window_start = now - 60
    ip_upload_history[client_ip] = [t for t in ip_upload_history[client_ip] if t > window_start]
    if len(ip_upload_history[client_ip]) >= UPLOAD_RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too many uploads", headers={"X-Error-Code": "rate_limit"})
    ip_upload_history[client_ip].append(now)

    upload_id = request.headers.get("Upload-Id") or str(uuid.uuid4())
    is_final_chunk = request.headers.get("Upload-Final") == "true"

    async with upload_semaphore:
        try:
            validate_file_type(file.filename)
            return await asyncio.wait_for(
                _handle_upload(file, client_ip, upload_id, is_final_chunk),
                timeout=UPLOAD_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.warning(f"Upload timeout for {file.filename} from {client_ip}")
            raise HTTPException(status_code=504, detail="Upload timed out", headers={"X-Error-Code": "upload_timeout"})
        except HTTPException as e:
            logger.warning(f"Upload failed ({file.filename}) from {client_ip}: {e.detail}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error ({file.filename}) from {client_ip}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Unexpected server error", headers={"X-Error-Code": "unknown_error"})     
        

@app.post("/calculate-metrics")
async def calculate_metrics(file: UploadFile):
    """Calculate fairness metrics for an uploaded electoral district map."""
    validate_file_type(file.filename)

    if file.filename.endswith(".geojson"):
        gdf = await read_geojson(file)
    elif file.filename.endswith(".zip"):
        gdf = await read_shapefile_zip(file)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format",headers = {"X-Error-Code": "invalid_file_format"})

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
        raise HTTPException(status_code=500, detail=str(e),headers = {"X-Error-Code": "invalid_exception"})

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
        raise HTTPException(status_code=500, detail=str(e), headers = {"X-Error-Code": "invalid_exception"})

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
        raise HTTPException(status_code=500, detail=str(e), headers = {"X-Error-Code": "invalid_exception"})

@app.get("/")
async def root():
    return {"message": "FairMap backend is running"}