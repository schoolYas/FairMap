import { useState } from "react";
const pink = "#ff4da6";

const TERMS = [
  {
    term: "Compactness",
    def: "Measures how geographically tight and visually tidy a district is. Common metrics include Polsby–Popper, Reock, and Schwartzberg.",
  },
  {
    term: "Efficiency Gap",
    def: "A measure of partisan gerrymandering based on wasted votes. It calculates how many votes did not contribute to a seat victory across parties.",
  },
  {
    term: "Partisan Bias",
    def: "How much advantage one party receives compared to an equally supported opposing party. Indicates asymmetry in seat outcomes.",
  },
  {
    term: "Competitiveness",
    def: "How close elections within a district are expected to be. Highly competitive districts have small vote margins between parties.",
  },
  {
    term: "Contiguity",
    def: "A district is contiguous if all pieces of the district are connected at some point. Islands or separated pieces break contiguity.",
  },
  {
    term: "Community of Interest (COI)",
    def: "A population sharing cultural, historical, economic, or geographic ties. Keeping COIs intact is often a redistricting goal.",
  },
  {
    term: "Minority Opportunity District",
    def: "A district where minority groups have the ability to elect a candidate of choice, often measured using CVAP or racial bloc voting data.",
  },
  {
    term: "Cracking",
    def: "Splitting a cohesive group of voters across many districts to dilute their voting power.",
  },
  {
    term: "Packing",
    def: "Concentrating a group of voters into very few districts, wasting their votes by giving them overwhelming supermajorities.",
  },
  {
    term: "Pairing",
    def: "Drawing two incumbents into the same district to force them to run against each other.",
  },
  {
    term: "Mean–Median Difference",
    def: "A statistical measure comparing the mean district vote share to the median district vote share to detect partisan skew.",
  },
  {
    term: "Seats–Votes Curve",
    def: "A function predicting how many legislative seats a party would win for any given vote share. Used to measure partisan responsiveness.",
  },
  {
    term: "Reock Score",
    def: "Compactness score equal to the ratio of district area to the area of the minimum bounding circle.",
  },
  {
    term: "Polsby–Popper Score",
    def: "Compactness score defined as 4πA / P², where A is area and P is perimeter. Penalizes irregular or spiky shapes.",
  },
  {
    term: "Geographic Polarization",
    def: "Political clustering of voters by geography, which can amplify natural partisan imbalance even without intentional gerrymandering.",
  },
  {
    term: "Voter Dilution",
    def: "Reducing the influence of a group's vote through packing, cracking, or other manipulations.",
  },
  {
    term: "Symmetry",
    def: "A fairness concept where each party should receive the same number of seats for the same share of votes.",
  },
  {
    term: "Ensemble Sampling",
    def: "A computational method generating thousands of alternative district maps to detect whether a plan is an extreme outlier.",
  },
  {
    term: "Convex Hull Ratio",
    def: "Compactness metric comparing a district's area to the area of its convex hull.",
  },
];


export default function Glossary() {

    
  const [query, setQuery] = useState("");

  
  const filtered = TERMS.filter((t) =>
    t.term.toLowerCase().includes(query.toLowerCase())
  );

  return (
    <div style={{ padding: "24px" }}>
      {/* Search Bar */}
      <input
        type="text"
        placeholder="Search glossary…"
        value={query}
        
        onChange={(e) => setQuery(e.target.value)}
        
        style={{
          width: "100%",
          padding: "12px",
          fontSize: "16px",
          borderRadius: "8px",
          border: "2px solid #ddd",
          marginBottom: "20px",
          outlineColor: pink,
        }}
      />

      {/* Glossary Table */}
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr style={{ background: "#f3f0ff" }}>
            <th
              style={{
                textAlign: "left",
                padding: "12px",
                color: pink,
                fontSize: "18px",
              }}
            >
              Term
            </th>
            <th
              style={{
                textAlign: "left",
                padding: "12px",
                color: pink,
                fontSize: "18px",
              }}
            >
              Definition
            </th>
          </tr>
        </thead>

        <tbody>
          {filtered.map((item, idx) => (
            <tr key={idx} style={{ borderBottom: "1px solid #eee" }}>
              <td style={{ padding: "12px", fontWeight: "bold", color: "#5A45DD" }}>
                {item.term}
              </td>
              <td style={{ padding: "12px", color: "#333" }}>{item.def}</td>
            </tr>
          ))}
        </tbody>
      </table>

       {/* If query does not match any of the terms, returns text */}
      {filtered.length === 0 && (
        <p style={{ marginTop: "20px", color: pink }}>No terms match your search.</p>
      )}
    </div>
  );
}


