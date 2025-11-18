// frontend/src/components/RadarSummary.js
import React from "react";

import {
  Radar
} from "react-chartjs-2";

import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend
} from "chart.js";

ChartJS.register(
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend
);

export default function RadarSummary({ scores }) {
  if (!scores || scores.length === 0) return null;

  // Compute averages
  const avg = {
    geometry: mean(scores.map(s => s.geometry_score)),
    partisan: mean(scores.map(s => s.partisan_score)),
    competitive: mean(scores.map(s => s.competitiveness_score)),
    demographic: mean(scores.map(s => s.demographics_score)),
    composite: mean(scores.map(s => s.composite_score))
  };

  const chartData = {
    labels: [
      "Geometry",
      "Partisan",
      "Competitiveness",
      "Demographics",
      "Composite"
    ],
    datasets: [
      {
        label: "Fairness Scores",
        data: [
          avg.geometry * 100,
          avg.partisan * 100,
          avg.competitive * 100,
          avg.demographic * 100,
          avg.composite * 100
        ],
        backgroundColor: "rgba(255, 105, 180, 0.35)", // pink fill
        borderColor: "#ff4da6",
        borderWidth: 2,
        pointBackgroundColor: "#ff4da6",
      },
    ],
  };

  const options = {
    scales: {
      r: {
        suggestedMin: 0,
        suggestedMax: 100,
        ticks: { display: false },
        grid: { color: "rgba(0,0,0,0.1)" },
        angleLines: { color: "rgba(0,0,0,0.2)" }
      }
    },
    plugins: {
      legend: { display: false },
    }
  };

  return (
    <div style={{ textAlign: "center" }}>
      <h3 style={{ marginBottom: "10px" }}>Fairness Summary</h3>

      {/* Radar Chart */}
      <Radar data={chartData} options={options} />

      {/* Composite Score */}
      <div style={{
        marginTop: "15px",
        fontSize: "18px",
        fontWeight: "bold"
      }}>
        Total Fairness Score: {(avg.composite * 100).toFixed(1)}
      </div>
    </div>
  );
}

function mean(arr) {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}
