from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import geopandas as gpd
from io import BytesIO
import os

# Allow non-closed rings if needed
os.environ["OGR_GEOMETRY_ACCEPT_UNCLOSED_RING"] = "YES"

app = FastAPI()

@app.post("/upload-map")
async def upload_map(file: UploadFile = File(...)):
    # Validate file type
    if not (file.filename.endswith(".geojson") or file.filename.endswith(".shp")):
        raise HTTPException(status_code=400, detail="Invalid file type")

    try:
        if file.filename.endswith(".geojson"):
            # Read file into memory and parse with GeoPandas
            content = await file.read()
            gdf = gpd.read_file(BytesIO(content))
            
            return JSONResponse(content={
                "filename": file.filename,
                "num_districts": len(gdf),
                "status": "Map uploaded successfully"
            })
        else:
            # For shapefiles, additional handling required (multiple files)
            return JSONResponse(content={"status": "Shapefile upload not implemented yet"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/")
async def root():
    return {"message": "FairMap backend is running"}


def root():
    return {"message": "FairMap backend running"}