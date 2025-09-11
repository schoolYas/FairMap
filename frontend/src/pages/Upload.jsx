import "leaflet/dist/leaflet.css";
import "leaflet-defaulticon-compatibility/dist/leaflet-defaulticon-compatibility.css";
import React, { useState } from "react"; // ✅ Import React here
import { MapContainer, TileLayer, GeoJSON } from "react-leaflet";


const API = process.env.REACT_APP_API_BASE || "http://127.0.0.1:8000";

function Upload() {
  const [file, setFile] = useState(null);
  const [msg, setMsg] = useState("");
  const [geoData, setGeoData] = useState(null);
  const [filename, setFilename] = useState("");

  async function handleSubmit(e) {
    e.preventDefault();
    if (!file) {
      setMsg("Please choose a .geojson or zipped shapefile (.zip).");
      return;
    }

    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${API}/upload-map`, {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Upload failed (${res.status})`);
      }

      const data = await res.json();
      setFilename(data.filename || file.name);
      setGeoData(data.geojson || null); // display map
      setMsg("Upload successful. Now go to the Metrics page.");
    } catch (err) {
      setMsg(`❌ ${err.message}`);
    }
  }

  return (
    <main style={{ padding: 24 }}>
      <h2>Upload District Map</h2>
      <p>Tip: use the example at <code>backend/example.geojson</code>.</p>
      <form onSubmit={handleSubmit}>
        <input
          type="file"
          accept=".geojson,.zip"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
        />
        <button type="submit" style={{ marginLeft: 12 }}>Upload</button>
      </form>
      <p style={{ marginTop: 12 }}>{msg}</p>

      {geoData && (
        <div style={{ marginTop: 24 }}>
          <h3>Map Preview: {filename}</h3>
          <MapContainer
            center={[37.8, -96]}
            zoom={4}
            style={{ height: "500px", width: "100%", marginTop: 12 }}
          >
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              attribution='&copy; <a href="http://osm.org/copyright">OpenStreetMap</a>'
            />
            <GeoJSON data={geoData} />
          </MapContainer>
        </div>
      )}
    </main>
  );
}

export default Upload;
