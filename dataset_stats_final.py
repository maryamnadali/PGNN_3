# dataset_stats_final.py
import numpy as np
import networkx as nx
from dataset import load_graphs


# -----------------------------
#  Graph / Feature Descriptions
# -----------------------------
GRAPH_TYPE = {
    "grid": "synthetic grid graph",
    "communities": "synthetic connected caveman graph",
    "ppi": "biological protein-protein interaction graph",
    "email": "social email communication graph",
    "protein": "biological protein structure graph",
    "Cora": "citation network of scientific publications",
    "CiteSeer": "citation network of scientific publications",
}

FEATURE_TYPE = {
    "grid": "one-hot features",
    "communities": "permuted one-hot features",
    "ppi": "continuous features",
    "email": "constant scalar feature",
    "protein": "continuous biochemical features",
    "Cora": "binary bag-of-words features",
    "CiteSeer": "binary bag-of-words features",
}


# -----------------------------
#  Degree + Density Statistics
# -----------------------------
def compute_degree_stats(graphs):
    all_degrees = []
    densities = []

    for G in graphs:
        deg_list = [d for _, d in G.degree()]
        all_degrees.extend(deg_list)
        densities.append(nx.density(G))

    all_degrees = np.array(all_degrees, dtype=float)

    return {
        "degree_mean": float(all_degrees.mean()),
        "degree_var": float(all_degrees.var()),
        "degree_min": float(all_degrees.min()),
        "degree_max": float(all_degrees.max()),
        "density_mean": float(np.mean(densities)),
    }


# -----------------------------
#  Clustering Coefficient
# -----------------------------
def compute_clustering(graphs):
    vals = []
    for G in graphs:
        c = nx.clustering(G)
        if len(c) > 0:
            vals.append(np.mean(list(c.values())))
    return float(np.mean(vals)) if len(vals) else np.nan


# -----------------------------
#  Shortest Path / Diameter
# -----------------------------
def compute_distance_stats(graphs, max_nodes=2000):
    avg_sp = []
    diam = []

    for G in graphs:
        if G.number_of_nodes() <= max_nodes and nx.is_connected(G):
            try:
                avg_sp.append(nx.average_shortest_path_length(G))
                diam.append(nx.diameter(G))
            except Exception:
                pass

    return {
        "avg_shortest_path": float(np.mean(avg_sp)) if len(avg_sp) else "N/A",
        "diameter_mean": float(np.mean(diam)) if len(diam) else "N/A",
    }


# -----------------------------
#  Main Analyzer
# -----------------------------
def analyze_dataset(dataset_name):
    graphs, features, edge_labels, node_labels, idx_train, idx_val, idx_test = load_graphs(dataset_name)

    num_graphs = len(graphs)
    V_total = sum(G.number_of_nodes() for G in graphs)
    E_total = sum(G.number_of_edges() for G in graphs)

    deg_stats = compute_degree_stats(graphs)
    clustering_mean = compute_clustering(graphs)
    dist_stats = compute_distance_stats(graphs)

    feature_dim = features[0].shape[1] if len(features) > 0 else 0

    return {
        "dataset": dataset_name,
        "num_graphs": num_graphs,
        "V_total": V_total,
        "E_total": E_total,

        **deg_stats,
        "clustering_mean": clustering_mean,
        **dist_stats,

        "feature_dim": feature_dim,
        "graph_type": GRAPH_TYPE.get(dataset_name, "N/A"),
        "feature_type": FEATURE_TYPE.get(dataset_name, "N/A"),
    }


# -----------------------------
#  Run All
# -----------------------------
if __name__ == "__main__":
    datasets = ["communities", "grid", "ppi", "email", "protein", "Cora", "CiteSeer"]

    header = (
        "dataset\tnum_graphs\tV_total\tE_total\t"
        "degree_mean\tdegree_var\tdegree_min\tdegree_max\t"
        "density_mean\tclustering_mean\tavg_shortest_path\tdiameter_mean\t"
        "feature_dim\tgraph_type\tfeature_type"
    )
    print(header)

    for name in datasets:
        s = analyze_dataset(name)
        print(
            f"{s['dataset']}\t"
            f"{s['num_graphs']}\t"
            f"{s['V_total']}\t"
            f"{s['E_total']}\t"
            f"{s['degree_mean']:.3f}\t"
            f"{s['degree_var']:.3f}\t"
            f"{s['degree_min']:.3f}\t"
            f"{s['degree_max']:.3f}\t"
            f"{s['density_mean']:.6f}\t"
            f"{s['clustering_mean']:.6f}\t"
            f"{s['avg_shortest_path']}\t"
            f"{s['diameter_mean']}\t"
            f"{s['feature_dim']}\t"
            f"{s['graph_type']}\t"
            f"{s['feature_type']}"
        )
