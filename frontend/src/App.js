import React, { useState, useEffect } from "react"; // <-- add useState and useEffect
import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import "./App.css";
import "leaflet/dist/leaflet.css";
import "leaflet-defaulticon-compatibility/dist/leaflet-defaulticon-compatibility.css";
import Upload from "./pages/Upload";
import Metrics from "./pages/Metrics";
import About from "./pages/About";

// If you want to use API in App.js (optional)
// const API = process.env.REACT_APP_API_BASE || "http://127.0.0.1:8000";

function Home() {
  const [status, setStatus] = useState("Loading backend status...");

  useEffect(() => {
    fetch("http://127.0.0.1:8000/") // backend root endpoint
      .then((res) => res.json())
      .then((data) => setStatus(data.message))
      .catch(() => setStatus("Backend not reachable"));
  }, []);

  return (
    <main style={{ padding: 24 }}>
      <h1>FairMap</h1>
      <p>{status}</p>
      <p>Upload a district map, calculate metrics, and export a CSV report.</p>
    </main>
  );
}

function App() {
  return (
    <BrowserRouter>
      <div className="App">
        <nav style={{ padding: 12 }}>
          <Link to="/" style={{ marginRight: 12 }}>Home</Link>
          <Link to="/upload" style={{ marginRight: 12 }}>Upload</Link>
          <Link to="/metrics" style={{ marginRight: 12 }}>Metrics</Link>
          <Link to="/about">About</Link>
        </nav>

        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/upload" element={<Upload />} />
          <Route path="/metrics" element={<Metrics />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;
