from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
import geopandas as gpd
import pandas as pd
from pydantic import BaseModel
from io import BytesIO
import os
import zipfile
import tempfile

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

# --- Constants ----------------------------------------------------------------------------
# Supported file types for upload
SUPPORTED_FILE_TYPES = [".geojson", ".shp"]
SHAPEFILE_EXTENSIONS = [".shp", ".shx", ".dbf", ".prj", ".cpg"]
# --- Constants End -------------------------------------------------------------------------

# --- Utility Functions ---------------------------------------------------------------------
def validate_file_type(filename: str):
    """Check if the uploaded file has a supported extension."""
    if not any(filename.endswith(ext) for ext in SUPPORTED_FILE_TYPES):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Supported types: {', '.join(SUPPORTED_FILE_TYPES)}"
        )

def read_geojson(file: UploadFile) -> gpd.GeoDataFrame:
    """Read a GeoJSON file into a GeoDataFrame."""
    try:
        content = BytesIO(file.file.read())
        gdf = gpd.read_file(content)
        return gdf
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read GeoJSON: {str(e)}")

def read_shapefile(zip_file: UploadFile) -> gpd.GeoDataFrame:
    """Read a zipped shapefile into a GeoDataFrame safely."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save uploaded zip to temp directory
            zip_path = os.path.join(tmpdir, zip_file.filename)
            with open(zip_path, "wb") as f:
                f.write(zip_file.file.read())

            # Extract zip
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)

            # Find the .shp file
            shp_files = [f for f in os.listdir(tmpdir) if f.endswith(".shp")]
            if not shp_files:
                raise HTTPException(status_code=400, detail="No .shp file found in zip")

            shp_path = os.path.join(tmpdir, shp_files[0])
            gdf = gpd.read_file(shp_path)
            return gdf
    except Exception as e:
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
async def upload_map(file: UploadFile = File(...)):
    """Upload an electoral district map (GeoJSON or zipped shapefile) and return GeoJSON for visualization."""
    validate_file_type(file.filename)

    if file.filename.endswith(".geojson"):
        gdf = read_geojson(file)
    elif file.filename.endswith(".zip"):
        gdf = read_shapefile(file)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")

    # Return GeoJSON object for frontend
    return JSONResponse(content={
        "filename": file.filename,
        "num_districts": len(gdf),
        "geojson": gdf.__geo_interface__,  # object, not string
        "status": "Map uploaded successfully"
    })


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