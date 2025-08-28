import { useState } from "react";

const API = process.env.REACT_APP_API_BASE || "http://127.0.0.1:8000";

function Upload()
{
  const [file, setFile] = useState(null);
  const [msg, setMsg] = useState("");

  async function handleSubmit(e)
  {
    e.preventDefault();
    if (!file)
    {
      setMsg("Please choose a .geojson or zipped shapefile (.zip).");
      return;
    }

    try
    {
      const form = new FormData();
      form.append("file", file); // FastAPI expects field named "file"
      const res = await fetch(`${API}/upload-map`, {
        method: "POST",
        body: form,
      });

      if (!res.ok)
      {
        const text = await res.text();
        throw new Error(text || `Upload failed (${res.status})`);
      }

      setMsg("Upload successful. Now go to the Metrics page.");
    }
    catch (err)
    {
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
    </main>
  );
}

export default Upload;