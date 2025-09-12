// src/pages/Metrics.jsx
import { useState } from "react";

const API = process.env.REACT_APP_API_BASE || "http://127.0.0.1:8000";

function Metrics()
{
  const [file, setFile] = useState(null);
  const [rows, setRows] = useState([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  async function calculate(e)
  {
    e.preventDefault();
    if (!file)
    {
      setErr("Please choose the same .geojson (or .zip shapefile) you uploaded.");
      return;
    }

    try
    {
      setBusy(true);
      setErr("");
      const form = new FormData();
      form.append("file", file);

      // Backend expects POST multipart/form-data
      const res = await fetch(`${API}/calculate-metrics`, {
        method: "POST",
        body: form,
      });

      if (!res.ok)
      {
        const text = await res.text();
        throw new Error(text || `HTTP ${res.status}`);
      }

      const data = await res.json();
      // Accept either { metrics: {...} } or an array normalize to table rows
      const m = data.metrics;
      if (Array.isArray(m))
      {
        setRows(m);
      }
      else if (m && typeof m === "object")
      {
        // If it's an object (for example {num_districts, compactness, efficiency_gap})
        // show it as one-row table.
        const objAsRow = { ...m };
        setRows([objAsRow]);
      }
      else
      {
        setRows([]);
      }
    }
    catch (e)
    {
      setErr(e.message);
      setRows([]);
    }
    finally
    {
      setBusy(false);
    }
  }

  async function exportCSV()
  {
    try
    {
      if (rows.length === 0)
      {
        setErr("No metrics to export. Click Calculate first.");
        return;
      }

      const res = await fetch(`${API}/export-report`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ metrics: rows })
      });

      if (!res.ok)
      {
        const text = await res.text();
        throw new Error(text || `Export failed (${res.status})`);
      }

      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "report.csv";
      a.click();
      window.URL.revokeObjectURL(url);
    }
    catch (e)
    {
      setErr(e.message);
    }
  }

  return (
    <main style={{ padding: 24 }}>
      <h2>Metrics</h2>

      <form onSubmit={calculate} style={{ marginBottom: 12 }}>
        <input
          type="file"
          accept=".geojson,.zip"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
        />
        <button type="submit" disabled={busy} style={{ marginLeft: 8 }}>
          {busy ? "Calculating…" : "Calculate"}
        </button>
      </form>

      {err && <p style={{ color: "red" }}>Error: {err}</p>}
      {rows.length === 0 && !err && !busy && (
        <p>Pick your map file, then click Calculate.</p>
      )}

      {rows.length > 0 && (
        <>
          <table border="1" cellPadding="8">
            <thead>
              <tr>
                {Object.keys(rows[0]).map((k) => <th key={k}>{k}</th>)}
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i}>
                  {Object.keys(rows[0]).map((k) => <td key={k}>{String(r[k])}</td>)}
                </tr>
              ))}
            </tbody>
          </table>

          <button onClick={exportCSV} style={{ marginTop: 12 }}>
            Export CSV
          </button>
        </>
      )}
    </main>
  );
}

export default Metrics;