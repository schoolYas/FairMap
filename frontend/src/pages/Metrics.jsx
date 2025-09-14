// src/pages/Metrics.jsx
import { useState } from "react";

const API = process.env.REACT_APP_API_BASE || "http://127.0.0.1:8000";

function Metrics() {
  const [file, setFile] = useState(null);
  const [rows, setRows] = useState([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  async function calculate(e) {
    e.preventDefault();
    if (!file) {
      setErr("Please choose the same .geojson (or .zip shapefile) you uploaded.");
      return;
    }

    try {
      setBusy(true);
      setErr("");
      setRows([]);
      const form = new FormData();
      form.append("file", file);

      const res = await fetch(`${API}/calculate-metrics`, {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `HTTP ${res.status}`);
      }

      const data = await res.json();
      const m = data.metrics;

      if (m && m.summary && Array.isArray(m.per_district)) {
        // flatten into one list of rows
        const rowsOut = [];
        rowsOut.push({ type: "summary", ...m.summary });
        m.per_district.forEach((d) => rowsOut.push({ type: "district", ...d }));
        setRows(rowsOut);
      } else if (Array.isArray(m)) {
        setRows(m);
      } else if (m && typeof m === "object") {
        setRows([{ type: "summary", ...m }]);
      } else {
        setRows([]);
      }
    } catch (e) {
      setErr(e?.message || String(e));
      setRows([]);
    } finally {
      setBusy(false);
    }
  }

  async function exportCSV() {
    try {
      if (rows.length === 0) {
        setErr("No metrics to export. Click Calculate first.");
        return;
      }

      const res = await fetch(`${API}/export-report`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ metrics: rows })
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Export failed (${res.status})`);
      }

      const blob = await res.blob();
      let filename = "report.csv";
      const cd = res.headers.get("content-disposition");
      const match = cd && /filename="?([^"]+)"?/i.exec(cd);
      if (match && match[1]) filename = match[1];

      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (e) {
      setErr(e?.message || String(e));
    }
  }

  // collect all possible keys across rows for headers
  const allKeys = Array.from(new Set(rows.flatMap(r => Object.keys(r))));

  return (
    <main style={{ padding: 24 }}>
      <h2>Metrics</h2>

      <form onSubmit={calculate} style={{ marginBottom: 12 }}>
        <input
          type="file"
          accept=".geojson,.zip"
          disabled={busy}
          onChange={(e) => {
            setFile(e.target.files?.[0] || null);
            setErr("");
            setRows([]);
          }}
        />
        <button
          type="submit"
          disabled={busy || !file}
          style={{ marginLeft: 8 }}
        >
          {busy ? "Calculating…" : "Calculate"}
        </button>

        {file && !busy && (
          <span style={{ marginLeft: 8, opacity: 0.7 }}>
            Selected: {file.name}
          </span>
        )}
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
                {allKeys.map((k) => <th key={k}>{k}</th>)}
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i}>
                  {allKeys.map((k) => (
                    <td key={k}>{r[k] !== undefined ? String(r[k]) : "-"}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>

          <button
            onClick={exportCSV}
            style={{ marginTop: 12 }}
            disabled={busy || rows.length === 0}
          >
            Export CSV
          </button>
        </>
      )}
    </main>
  );
}

export default Metrics;