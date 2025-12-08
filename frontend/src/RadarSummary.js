// frontend/src/RadarSummary.js
import React, { useState } from "react";
import { Radar } from "react-chartjs-2";

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

// scores = all district scores
// selectedIndex = index of clicked district (or undefined/null to use averages)
// planScore = plan-level composite from backend
// isGerrymandered = boolean from ensembleStats
export default function RadarSummary({
  scores,
  selectedIndex,
  planScore,
  isGerrymandered
}) {
  const [hoveredLabel, setHoveredLabel] = useState(null);

  if (!scores || scores.length === 0) return null;

  // Do we have a specific district selected?
  const hasSelection =
    typeof selectedIndex === "number" &&
    selectedIndex >= 0 &&
    selectedIndex < scores.length;

  // Averages across all districts
  const avg = {
    geometry: mean(scores.map((s) => s.geometry_score)),
    partisan: mean(scores.map((s) => s.partisan_score)),
    competitive: mean(scores.map((s) => s.competitiveness_score)),
    demographic: mean(scores.map((s) => s.demographics_score)),
    composite: mean(scores.map((s) => s.composite_score))
  };

  // Metrics that will actually feed the radar chart:
  // - if a district is selected → that district’s metrics
  // - otherwise → averages
  const selected = hasSelection ? scores[selectedIndex] : null;

  const geometryVal = hasSelection ? selected.geometry_score : avg.geometry;
  const partisanVal = hasSelection ? selected.partisan_score : avg.partisan;
  const competitiveVal = hasSelection
    ? selected.competitiveness_score
    : avg.competitive;
  const demographicVal = hasSelection
    ? selected.demographics_score
    : avg.demographic;
  const compositeVal = hasSelection ? selected.composite_score : avg.composite;

  // Number shown under the chart
  const displayComposite =
    hasSelection && selected
      ? compositeVal
      : typeof planScore === "number"
        ? planScore
        : avg.composite;

  // Detailed tips that show *under* the chart
  const tipsByLabel = {
    Geometry:
      "Try drawing districts that are compact and follow natural or community boundaries.",
    Partisan:
      "Avoid packing or cracking voters; balance partisan vote share across districts.",
    Competitiveness:
      "Increase the number of districts where either party has a realistic chance to win.",
    Demographics:
      "Avoid unnecessarily concentrating minority communities into a small number of districts.",
    Composite:
      "Adjust boundaries to improve several fairness dimensions at once, not just one."
  };

  // NEW: very short tips just for the tooltip (so it doesn’t get cut off)
  const tooltipShortTips = {
    Geometry: "Draw more compact districts.",
    Partisan: "Balance party support.",
    Competitiveness: "Avoid ultra-safe seats.",
    Demographics: "Don’t pack minority voters."
  };

  // OLD: `const tips = [...]` is removed, we don’t need it anymore

  const chartData = {
    labels: ["Geometry", "Partisan", "Competitiveness", "Demographics"],
    datasets: [
      {
        label: "Fairness Scores",
        data: [
          geometryVal * 100,
          partisanVal * 100,
          competitiveVal * 100,
          demographicVal * 100
        ],
        backgroundColor: "rgba(255, 105, 180, 0.35)", // pink fill
        borderColor: "#ff4da6",
        borderWidth: 2,
        pointBackgroundColor: "#ff4da6"
      }
    ]
  };

  // Hover handler to control the text UNDER the chart
  function handleHover(event, activeElements) {
    // If we’re not treating the plan as gerrymandered, don't show tips
    if (!isGerrymandered) {
      if (hoveredLabel !== null) setHoveredLabel(null);
      return;
    }

    if (activeElements && activeElements.length > 0) {
      const idx = activeElements[0].index;
      const label = chartData.labels[idx];
      setHoveredLabel(label);
    } else if (hoveredLabel !== null) {
      setHoveredLabel(null);
    }
  }
  
  const options = {
    interaction: {
      mode: "nearest",
      intersect: true
    },
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
      tooltip: {
        enabled: true,
        callbacks: {
          label: function (context) {
            const value = `${context.raw.toFixed(1)}%`;

            // If we’re not gerrymandered, just show a simple tooltip
            if (!isGerrymandered) {
              return `${context.label}: ${value}`;
            }
            // If we are gerrymandered, show two short lines
            const label = context.label;
            const shortTip = tooltipShortTips[label];

            if (shortTip) {
              return [`${label}: ${value}`, shortTip];
            }
            return `${label}: ${value}`;
          }
        }
      }
    }
  };

  return (
    <div style={{ textAlign: "center" }}>
      <h3 style={{ marginBottom: "10px" }}>Fairness Summary</h3>

      {isGerrymandered && (
        <div
          style={{
            marginBottom: "6px",
            fontSize: "12px",
            color: "#666"
          }}
        >
          This plan is flagged as gerrymandered. Hover over each axis for
          suggestions on how to make it more equitable.
        </div>
      )}

      {/* Radar Chart */}
      <Radar
        data={chartData}
        options={options}
        onHover={handleHover}
      />

      {/* Hover tips under the chart when plan is gerrymandered */}
      {isGerrymandered && hoveredLabel && (
        <div
          style={{
            marginTop: "10px",
            fontSize: "13px",
            color: "#444",
            textAlign: "left"
          }}
        >
          <strong>{hoveredLabel} tip:</strong>{" "}
          {tipsByLabel[hoveredLabel] ||
            "Adjust this factor to improve fairness."}
        </div>
      )}

      {/* Composite / Total Fairness text */}
      <div
        style={{
          marginTop: "15px",
          fontSize: "18px",
          fontWeight: "bold"
        }}
      >
        Total Fairness Score: {(displayComposite * 100).toFixed(1)}
      </div>

      {!hasSelection && typeof planScore === "number" && (
        <div style={{ marginTop: "4px", fontSize: "13px", color: "#666" }}>
          (Plan-level composite from backend; radar shows district averages)
        </div>
      )}

      {hasSelection && (
        <div style={{ marginTop: "4px", fontSize: "13px", color: "#666" }}>
          (Showing selected district’s scores)
        </div>
      )}
    </div>
  );
}

// Helper function to compute mean of an array
function mean(arr) {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}
