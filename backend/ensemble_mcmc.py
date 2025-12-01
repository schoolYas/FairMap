# ensemble_mcmc.py
import random
from typing import List, Dict, Any, Tuple
import geopandas as gpd
import networkx as nx
import numpy as np

from calculate_metrics import compute_metrics_for_gdf, compute_state_composite


def build_adjacency_graph(gdf: gpd.GeoDataFrame, id_col: str = "_prec_id") -> Tuple[nx.Graph, gpd.GeoDataFrame]:
    """
    Build a precinct adjacency graph from the GeoDataFrame.
    Nodes = precinct indices, edges = polygons that touch.
    """
    # Give each row a stable integer id
    gdf = gdf.copy()
    gdf = gdf.reset_index(drop=True)
    gdf[id_col] = gdf.index

    G = nx.Graph()
    for idx, row in gdf.iterrows():
        G.add_node(idx, population=float(row.get("TOTPOP", 0.0)))

    # naive O(n^2) adjacency (OK for small test states, you can optimize with sindex later)
    geoms = list(gdf.geometry)
    n = len(geoms)
    for i in range(n):
        for j in range(i + 1, n):
            if geoms[i] is not None and geoms[j] is not None and geoms[i].touches(geoms[j]):
                G.add_edge(i, j)

    return G, gdf


def get_boundary_nodes(G: nx.Graph, labels: np.ndarray) -> List[int]:
    """
    Return list of nodes that sit on a boundary between districts.
    """
    boundary = []
    for u in G.nodes:
        for v in G.neighbors(u):
            if labels[u] != labels[v]:
                boundary.append(u)
                break
    return boundary


def district_populations(G: nx.Graph, labels: np.ndarray) -> Dict[int, float]:
    pop_per_d = {}
    for node, data in G.nodes(data=True):
        d = labels[node]
        pop_per_d.setdefault(d, 0.0)
        pop_per_d[d] += float(data.get("population", 0.0))
    return pop_per_d


def can_move_node(
    G: nx.Graph,
    labels: np.ndarray,
    node: int,
    old_d: int,
    new_d: int,
    k: int,
    pop_tol: float,
    ideal_pop: float,
) -> bool:
    """
    Check if moving `node` from old_d -> new_d keeps:
      - old_d contiguous
      - population within tolerance
      - new_d contiguous (node must touch new_d)
    """

    # Ensure node has at least one neighbor in the target district (contiguity)
    if not any(labels[nb] == new_d for nb in G.neighbors(node)):
        return False

    # Check contiguity of old_d AFTER removing node
    remaining = [n for n in G.nodes if labels[n] == old_d and n != node]
    if len(remaining) == 0:
        # would delete a district entirely
        return False
    if len(remaining) > 1:
        subG = G.subgraph(remaining)
        if not nx.is_connected(subG):
            return False

    # Population check
    pops = district_populations(G, labels)
    pop_node = float(G.nodes[node].get("population", 0.0))

    pop_old_new = pops.get(old_d, 0.0) - pop_node
    pop_new_new = pops.get(new_d, 0.0) + pop_node

    # Each district should be within ±pop_tol of ideal_pop
    for pop in (pop_old_new, pop_new_new):
        if ideal_pop > 0:
            if abs(pop - ideal_pop) / ideal_pop > pop_tol:
                return False

    return True


def mcmc_step(
    G: nx.Graph,
    labels: np.ndarray,
    k: int,
    pop_tol: float,
    ideal_pop: float,
) -> np.ndarray:
    """
    One MCMC step:
      - choose a boundary node
      - propose moving it to a neighboring district
      - accept if contiguity + population constraints are satisfied
    """
    labels = labels.copy()

    boundary_nodes = get_boundary_nodes(G, labels)
    if not boundary_nodes:
        return labels  # no moves possible

    node = random.choice(boundary_nodes)
    current_d = labels[node]

    neighbor_ds = {labels[nb] for nb in G.neighbors(node) if labels[nb] != current_d}
    if not neighbor_ds:
        return labels

    # Random order of candidate districts
    neighbors_list = list(neighbor_ds)
    random.shuffle(neighbors_list)

    for new_d in neighbors_list:
        if can_move_node(G, labels, node, current_d, new_d, k, pop_tol, ideal_pop):
            labels[node] = new_d
            break

    return labels


def run_mcmc_ensemble(
    gdf_precincts: gpd.GeoDataFrame,
    k: int,
    n_plans: int = 100,
    thin: int = 10,
    pop_col: str = "TOTPOP",
    pop_tol: float = 0.05,
) -> List[Dict[str, Any]]:
    """
    Run a simple boundary-flip MCMC ensemble over district plans.

    Parameters:
        gdf_precincts: precinct-level GeoDataFrame with a 'CD' column
        k: number of districts
        n_plans: how many sampled plans to record
        thin: keep 1 plan every `thin` steps
        pop_col: name of population column
        pop_tol: allowed relative deviation from ideal population per district (e.g. 0.05 = ±5%)

    Returns:
        List of dicts, each with:
          - "plan_index"
          - "state_composite"
          - optionally anything else you want to inspect later
    """
    if "CD" not in gdf_precincts.columns:
        raise ValueError("gdf_precincts must have a 'CD' column for initial district labels")

    G, gdf = build_adjacency_graph(gdf_precincts, id_col="_prec_id")

    # initial labels from current plan
    labels = gdf["CD"].to_numpy()

    # Pre-compute ideal population
    total_pop = float(gdf[pop_col].sum()) if pop_col in gdf.columns else 0.0
    ideal_pop = total_pop / float(k) if k > 0 else 0.0

    results: List[Dict[str, Any]] = []

    total_steps = n_plans * thin
    for step in range(total_steps):
        labels = mcmc_step(G, labels, k=k, pop_tol=pop_tol, ideal_pop=ideal_pop)

        # Record every `thin` steps
        if (step + 1) % thin == 0:
            gdf_plan = gdf.copy()
            gdf_plan["CD"] = labels

            # re-use your existing metric pipeline
            districts = compute_metrics_for_gdf(gdf_plan)
            state_comp = compute_state_composite(districts)

            results.append({
                "plan_index": len(results),
                "state_composite": float(state_comp),
                # You can add aggregates here if useful, e.g. average partisan_score:
                # "partisan_mean": float(districts["partisan_score"].mean()),
            })

    return results
