import React, { useEffect, useState } from "react";

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

export default Home;
