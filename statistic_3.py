import numpy as np
import pandas as pd
import networkx as nx
import json
import pickle as pkl
import scipy.sparse as sp

import torch_geometric as tg
from torch_geometric.utils import to_networkx

from dataset import load_graphs   # ← از پروژهٔ خودت


###############################################################################
# Utility
###############################################################################

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def compute_graph_stats(G, feature, node_labels=None, pair_labels=None):
    V = G.number_of_nodes()
    E = G.number_of_edges()

    if V == 0:
        return dict(
            V=0, E=0,
            avg_degree=0.0, degree_variance=0.0,
            density=0.0, homophily=None,
            feat_dim=int(feature.shape[1]) if feature is not None else 0,
        )

    deg = np.array([d for _, d in G.degree()])
    avg_deg = float(deg.mean())
    var_deg = float(deg.var())
    density = float(nx.density(G))
    feat_dim = int(feature.shape[1])

    homophily = None
    if node_labels is not None:
        y = np.asarray(node_labels)
        same = 0
        for u, v in G.edges():
            if y[u] == y[v]:
                same += 1
        homophily = same / E if E > 0 else None

    elif pair_labels is not None:
        nodes = list(G.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        same = 0
        for u, v in G.edges():
            iu = node_to_idx[u]
            iv = node_to_idx[v]
            if iu > iv:
                val = pair_labels[iu, iv]
            else:
                val = pair_labels[iv, iu]
            if val == 1:
                same += 1
        homophily = same / E if E > 0 else None

    return dict(
        V=V,
        E=E,
        avg_degree=avg_deg,
        degree_variance=var_deg,
        density=density,
        homophily=homophily,
        feat_dim=feat_dim,
    )



###############################################################################
# RAW Protein Loader (1113 graphs, no filtering)
###############################################################################

def load_protein_raw():
    print("Loading PROTEINS_full (RAW, 1113 graphs)…")

    path = "data/PROTEINS_full/"

    edges = np.loadtxt(path + "PROTEINS_full_A.txt", delimiter=",", dtype=int)
    node_attr = np.loadtxt(path + "PROTEINS_full_node_attributes.txt", delimiter=",")
    node_labels = np.loadtxt(path + "PROTEINS_full_node_labels.txt", dtype=int)
    graph_indicator = np.loadtxt(path + "PROTEINS_full_graph_indicator.txt", dtype=int)
    graph_labels = np.loadtxt(path + "PROTEINS_full_graph_labels.txt", dtype=int)

    edges -= 1  # convert to zero-based
    num_graphs = int(graph_indicator.max())

    graphs = []
    features = []
    pair_labels = []

    for gid in range(1, num_graphs + 1):
        nodes = np.where(graph_indicator == gid)[0]
        node_set = set(nodes.tolist())

        G = nx.Graph()
        G.add_nodes_from(nodes.tolist())

        for u, v in edges:
            if (u in node_set) and (v in node_set):
                G.add_edge(int(u), int(v))

        graphs.append(G)
        features.append(node_attr[nodes])

        # pair labels
        n = len(nodes)
        lbl = np.zeros((n, n), dtype=int)
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if node_labels[u] == node_labels[v] and i > j:
                    lbl[i, j] = 1

        pair_labels.append(lbl)

    print("Loaded PROTEINS_full:", len(graphs))
    return graphs, features, pair_labels



###############################################################################
# Planetoid RAW loader (Cora / CiteSeer)
###############################################################################

def load_planetoid_raw(name):
    print(f"Loading {name} raw files…")

    base = f"data/ind.{name}."
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []

    for nm in names:
        with open(base + nm, 'rb') as f:
            objects.append(pkl.load(f, encoding='latin1'))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(f"data/ind.{name}.test.index")
    test_idx_range = np.sort(test_idx_reorder)

    if name == "citeseer":
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_ext = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_ext[test_idx_range - min(test_idx_range), :] = tx
        ty_ext = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_ext[test_idx_range - min(test_idx_range), :] = ty
        tx = tx_ext
        ty = ty_ext

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    feat_dim = features.shape[1]

    G = nx.from_dict_of_lists(graph)
    return [G], [features.toarray()], [None]



###############################################################################
# Main
###############################################################################

GRAPH_TYPE = {
    "grid": "synthetic grid graph",
    "communities": "synthetic connected caveman graph",
    "ppi": "biological protein-protein interaction graph",
    "email": "email communication graph",
    "protein": "biological protein structure graph",
    "Cora": "citation network",
    "CiteSeer": "citation network",
}

FEATURE_TYPE = {
    "grid": "one-hot features",
    "communities": "permuted one-hot features",
    "ppi": "continuous features",
    "email": "constant scalar feature",
    "protein": "continuous biochemical features",
    "Cora": "binary bag-of-words",
    "CiteSeer": "binary bag-of-words",
}


if __name__ == "__main__":

    datasets = ["communities", "grid", "ppi", "email", "protein", "Cora", "CiteSeer"]
    rows = []

    for name in datasets:
        print("\n=== Processing", name, "===")

        if name == "protein":
            graphs, features, pair_labels = load_protein_raw()
            node_labels_list = [None] * len(graphs)

        elif name == "Cora":
            graphs, features, node_labels_list = load_planetoid_raw("cora")
            pair_labels = [None]

        elif name == "CiteSeer":
            graphs, features, node_labels_list = load_planetoid_raw("citeseer")
            pair_labels = [None]

        else:
            graphs, features, edge_labels_raw, node_labels_raw, _, _, _ = load_graphs(name)
            pair_labels = edge_labels_raw
            node_labels_list = node_labels_raw

        stats_list = []
        for i, G in enumerate(graphs):
            feat = features[i]
            nl = node_labels_list[i] if i < len(node_labels_list) else None
            pl = pair_labels[i] if i < len(pair_labels) else None
            s = compute_graph_stats(G, feat, node_labels=nl, pair_labels=pl)
            stats_list.append(s)

        df = pd.DataFrame(stats_list)

        hom = df["homophily"]
        homophily_mean = float(hom.dropna().mean()) if hom.notna().any() else "N/A"

        summary = {
            "dataset": name,
            "num_graphs": len(graphs),
            "V_total": int(df["V"].sum()),
            "E_total": int(df["E"].sum()),
            "V_mean": float(df["V"].mean()),
            "V_min": int(df["V"].min()),
            "V_max": int(df["V"].max()),
            "E_mean": float(df["E"].mean()),
            "E_min": int(df["E"].min()),
            "E_max": int(df["E"].max()),
            "avg_deg_mean": float(df["avg_degree"].mean()),
            "deg_var_mean": float(df["degree_variance"].mean()),
            "density_mean": float(df["density"].mean()),
            "homophily_mean": homophily_mean,
            "feature_dim": stats_list[0]["feat_dim"],
            "graph_type": GRAPH_TYPE[name],
            "feature_type": FEATURE_TYPE[name],
        }

        rows.append(summary)

    df_out = pd.DataFrame(rows)
    df_out.to_csv("dataset_statistics_LLM_ALL.csv", index=False)
    print("\nSaved → dataset_statistics_LLM_ALL.csv")
    print(df_out)
