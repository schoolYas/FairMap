const PURPLE = "#5A45DD";

{/* ----- Python Stack ----------------*/}
const libraries = [
  {
    name: "GeoPandas",
    author: "GeoPandas contributors",
    link: "https://geopandas.org/",
  },
  {
    name: "pandas",
    author: "pandas-dev team",
    link: "https://pandas.pydata.org/",
  },
  {
    name: "NumPy",
    author: "NumPy developers",
    link: "https://numpy.org/",
  },
  {
    name: "scikit-learn (MinMaxScaler)",
    author: "scikit-learn developers",
    link: "https://scikit-learn.org/",
  },
  {
    name: "statsmodels",
    author: "statsmodels developers",
    link: "https://www.statsmodels.org/",
  },
  {
    name: "Shapely",
    author: "Shapely contributors",
    link: "https://shapely.readthedocs.io/",
  },


  {
    name: "React",
    author: "Meta Open Source & community",
    link: "https://react.dev/",
  },
  {
    name: "react-router-dom",
    author: "Remix Software",
    link: "https://reactrouter.com/",
  },
  {
    name: "Leaflet",
    author: "Volodymyr Agafonkin & Leaflet contributors",
    link: "https://leafletjs.com/",
  },
  {
    name: "react-leaflet",
    author: "Paul LeCam & contributors",
    link: "https://react-leaflet.js.org/",
  },
  {
    name: "leaflet-defaulticon-compatibility",
    author: "ghybs",
    link: "https://github.com/ghybs/leaflet-defaulticon-compatibility",
  },
];

export default function LibrariesTable() {
  return (
    <table
      style={{
        width: "100%",
        borderCollapse: "collapse",
        marginTop: "20px"
      }}
    >
        {/* Sets a table with three components: tool, author, and link*/}
      <thead>
        <tr style={{ color: PURPLE, fontWeight: "bold", fontSize: "18px" }}>
          <th style={{ padding: "10px" }}>Component/Tool</th>
          <th style={{ padding: "10px" }}>Author</th>
          <th style={{ padding: "10px" }}>Link</th>
        </tr>
      </thead>

      <tbody>
        {libraries.map((lib) => (
          <tr key={lib.name} style={{ color: PURPLE }}>
            <td style={{ padding: "10px" }}>{lib.name}</td>
            <td style={{ padding: "10px" }}>{lib.author}</td>
            <td style={{ padding: "10px" }}>
              <a
                href={lib.link}
                target="_blank"
                rel="noopener noreferrer"
                style={{ color: PURPLE, textDecoration: "underline" }}
              >
                {lib.link}
              </a>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}