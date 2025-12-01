import React, { useState, useRef } from "react";
import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import Upload from "./pages/Upload";
import Glossary from "./pages/Glossary";
import About from "./pages/About";

import { MapContainer, TileLayer, GeoJSON } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import "leaflet-defaulticon-compatibility/dist/leaflet-defaulticon-compatibility.css";
import "./App.css";
import RadarSummary from "./RadarSummary";   // ⭐ NEW IMPORT

/* -----------------------------
   HAMBURGER MENU (SLIDE-IN)
------------------------------*/
function HamburgerMenu({ open, onClose }) {
  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        width: "260px",
        height: "100vh",
        background: "#ffffff",
        boxShadow: "2px 0 8px rgba(0,0,0,0.2)",
        transform: open ? "translateX(0)" : "translateX(-100%)",
        transition: "transform 0.25s ease",
        zIndex: 2000,
        padding: "20px",
        overflowY: "auto",
      }}
    >
      <button
        onClick={onClose}
        style={{
          fontSize: "24px",
          background: "none",
          border: "none",
          cursor: "pointer",
          marginBottom: "20px",
        }}
      >
        ✕
      </button>

      <nav style={{ display: "flex", flexDirection: "column", gap: "18px" }}>
        <Link to="/" style={menuLink}>Home</Link>
        <Link to="/upload" style={menuLink}>Upload</Link>
        <Link to="/glossary" style={menuLink}>Glossary</Link>
        <Link to="/about" style={menuLink}>About</Link>
      </nav>
    </div>
  );
}

const menuLink = {
  fontSize: "18px",
  color: "#333",
  textDecoration: "none",
};

/* -----------------------------
           HOME PAGE
------------------------------*/
function Home() {
  const [geoData, setGeoData] = useState(null);
  const [filename, setFilename] = useState("");
  const [scores, setScores] = useState(null);  // ⭐ for summary chart
  const fileInputRef = useRef(null);

  async function handleFileSelected(event) {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://127.0.0.1:8000/upload-map", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (res.ok) {
        setGeoData(data.geojson);
        setFilename(data.filename);
        setScores(data.scores);  // ⭐ send scores to chart
      } else {
        alert("Upload failed: " + data.detail);
      }
    } catch (err) {
      alert("Upload failed. Check console.");
      console.error(err);
    }
  }

  return (
    <div style={styles.homeContainer}>
      <h1 style={styles.homeTitle}>FairMap</h1>
      <h3 style={styles.homeSubtitle}>Gerrymandering Detection Tool</h3>

      <div style={styles.homeButtons}>
        <button
          style={styles.primaryBtn}
          onClick={() => fileInputRef.current.click()}
        >
          Upload Map
        </button>

        <input
          type="file"
          accept=".zip,.geojson"
          ref={fileInputRef}
          style={{ display: "none" }}
          onChange={handleFileSelected}
        />

        <button style={styles.secondaryBtn}>Learn More</button>
      </div>

      {/* ⭐ SUMMARY PANEL + MAP WRAPPER */}
      <div style={{ display: "flex", gap: "20px", marginTop: "30px" }}>

        {/* ⭐ RADAR SUMMARY TOP-RIGHT */}
        {scores && (
          <div style={{
            width: "380px",
            background: "white",
            padding: "20px",
            borderRadius: "12px",
            boxShadow: "0 0 10px rgba(0,0,0,0.12)",
            height: "fit-content"
          }}>
            <RadarSummary scores={scores} />
          </div>
        )}

        {/* MAP */}
        <div style={{ flex: 1 }}>
          <h2>Map Preview</h2>

          {filename && (
            <p style={{ color: "#555" }}>
              Loaded File: <strong>{filename}</strong>
            </p>
          )}

          <MapContainer
            center={[37.8, -96]}
            zoom={4}
            minZoom={3}
            maxZoom={12}
            scrollWheelZoom={true}
            maxBounds={[
              [49.38, -124.76],
              [24.52, -66.95],
            ]}
            maxBoundsViscosity={1.0}
            style={{
              height: "500px",
              width: "100%",
              marginTop: "20px",
              position: "relative",
            }}
          >
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              className="grayscale"
            />

            {geoData && (
              <GeoJSON
                data={geoData}
                style={{
                  color: "#ff4da6",
                  weight: 3,
                  fillColor: "#ff4da6",
                  fillOpacity: 0.45,
                }}
              />
            )}
          </MapContainer>
        </div>
      </div>

      {/* ⭐ DISTRICT SCORES BELOW EVERYTHING */}
      {scores && (
        <div style={{ marginTop: "30px", textAlign: "left" }}>
          <h3>District Scores</h3>
          {scores.map((s, i) => (
            <div key={i} style={{ marginBottom: "12px" }}>
              <strong>District {i + 1}</strong><br />
              Geometry: {s.geometry_score.toFixed(3)}<br />
              Partisan: {s.partisan_score.toFixed(3)}<br />
              Competitiveness: {s.competitiveness_score.toFixed(3)}<br />
              Demographics: {s.demographics_score.toFixed(3)}<br />
              <strong>Composite: {s.composite_score.toFixed(3)}</strong>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* -----------------------------
            MAIN APP
------------------------------*/
function App() {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <BrowserRouter>
      <header
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          width: "100%",
          height: "60px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "0 16px",
          borderBottom: "1px solid #ddd",
          background: "#ffffff",
          zIndex: 1500,
        }}
      >
        <button
          onClick={() => setMenuOpen(true)}
          style={{
            fontSize: "26px",
            background: "none",
            border: "none",
            cursor: "pointer",
            marginLeft: "4px",
            padding: "4px 8px",
          }}
        >
          ☰
        </button>

        <div style={{ fontWeight: "bold", fontSize: "20px" }}>FairMap</div>

        <a
          href="https://github.com/schoolYas/FairMap"
          target="_blank"
          rel="noopener noreferrer"
          style={styles.githubBtn}
        >
          GitHub
        </a>
      </header>

      <HamburgerMenu open={menuOpen} onClose={() => setMenuOpen(false)} />

      <main style={{ paddingTop: "70px" }}>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/upload" element={<Upload />} />
          <Route path="/glossary" element={<Glossary />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </main>
    </BrowserRouter>
  );
}

export default App;

/* -----------------------------
       STYLES
------------------------------*/
const styles = {
  homeContainer: {
    textAlign: "center",
    padding: "30px 20px",
    background: "#f3f0ff",
    borderBottom: "1px solid #ddd",
  },
  homeTitle: {
    fontSize: "32px",
    fontWeight: "bold",
    marginBottom: "6px",
  },
  homeSubtitle: {
    color: "#444",
    marginTop: "6px",
    marginBottom: "10px",
    fontSize: "16px",
  },
  homeButtons: {
    display: "flex",
    justifyContent: "center",
    gap: "15px",
    marginTop: "10px",
  },
  primaryBtn: {
    backgroundColor: "#5A45DD",
    color: "white",
    padding: "10px 20px",
    borderRadius: "6px",
    border: "none",
    cursor: "pointer",
    fontSize: "14px",
  },
  secondaryBtn: {
    background: "white",
    border: "2px solid #5A45DD",
    color: "#5A45DD",
    padding: "10px 20px",
    borderRadius: "6px",
    cursor: "pointer",
    fontSize: "14px",
  },
  githubBtn: {
    padding: "6px 12px",
    marginRight: "10px",
    borderRadius: "6px",
    background: "#24292f",
    color: "#ffffff",
    textDecoration: "none",
    fontSize: "14px",
  },
};
