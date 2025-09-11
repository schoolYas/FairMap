import React, { useState } from "react";
import { MapContainer, TileLayer, GeoJSON } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import "leaflet-defaulticon-compatibility/dist/leaflet-defaulticon-compatibility.css";

const API = process.env.REACT_APP_API_BASE || "http://127.0.0.1:8000";

function MapView() {
  const [geoData, setGeoData] = useState(null);
  const [msg, setMsg] = useState("");

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API}/upload-map`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Upload failed (${res.status})`);
      }

      const data = await res.json();
      if (!data.geojson) throw new Error("No geojson returned from server");
      setGeoData(data.geojson);
      setMsg(`Map loaded: ${data.filename}`);
    } catch (err) {
      console.error("Upload failed:", err);
      setMsg(`❌ ${err.message}`);
    }
  };

  return (
    <div style={{ padding: 24 }}>
      <h2>Upload and Preview Map</h2>
      <input type="file" accept=".geojson,.zip" onChange={handleFileUpload} />
      <p>{msg}</p>
      <div style={{ marginTop: 24 }}>
        <MapContainer center={[37.8, -96]} zoom={4} style={{ height: "500px", width: "100%" }}>
          <TileLayer
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution='&copy; <a href="http://osm.org/copyright">OpenStreetMap</a>'
          />
          {geoData && <GeoJSON data={geoData} />}
        </MapContainer>
      </div>
    </div>
  );
}

export default MapView;
