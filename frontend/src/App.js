import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import "./App.css";

import Upload from "./pages/Upload";
import Metrics from "./pages/Metrics";
import About from "./pages/About";

function App()
{
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

function Home()
{
  return (
    <main style={{ padding: 24 }}>
      <h1>FairMap</h1>
      <p>Upload a district map, calculate metrics, and export a CSV report.</p>
    </main>
  );
}

export default App;