import pytest
from fastapi.testclient import TestClient
from main import app
from io import BytesIO
import zipfile
import json

client = TestClient(app)

# --- Edge case: empty file ---
def test_empty_upload():
    response = client.post("/upload-map", files={"file": ("empty.geojson", BytesIO(b""))})
    assert response.status_code == 500 or response.status_code == 400


# --- Malformed GeoJSON ---
def test_malformed_geojson():
    bad_geojson = b'{ "type": "FeatureCollection", "features": [ { "type": "Feature" } ] }'  # missing geometry
    response = client.post("/upload-map", files={"file": ("bad.geojson", BytesIO(bad_geojson))})
    assert response.status_code == 500

# --- Invalid geometries (can fix with .buffer(0)) ---
def test_invalid_geometry():
    geojson_with_self_intersection = b'''
    {
      "type": "FeatureCollection",
      "features": [
        {
          "type": "Feature",
          "geometry": {"type": "Polygon", "coordinates": [[[0,0],[1,1],[1,0],[0,1],[0,0]]]},
          "properties": {}
        }
      ]
    }
    '''
    response = client.post("/upload-map", files={"file": ("invalid.geojson", BytesIO(geojson_with_self_intersection))})
    assert response.status_code == 200
    data = response.json()
    assert "geojson" in data

# --- Huge file test ---
def test_huge_file():
    huge_geojson = b'{"type":"FeatureCollection","features":[]}' * 10**6  # simulate large file
    response = client.post("/upload-map", files={"file": ("huge.geojson", BytesIO(huge_geojson))})
    assert response.status_code in [413, 500]
