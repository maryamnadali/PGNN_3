import numpy as np
import pandas as pd
import networkx as nx

import torch_geometric as tg
from torch_geometric.utils import to_networkx

from dataset import load_graphs  # از خود پروژه


# -------------------------------
#  توضیح نوع گراف و نوع فیچر (بر اساس dataset.py و داک رسمی)
# -------------------------------
GRAPH_TYPE = {
    # از grid_2d_graph(20, 20) → گراف شبکه‌ای 2بعدی :contentReference[oaicite:4]{index=4}
    "grid": "synthetic grid graph",

    # از connected_caveman_graph(community_num, community_size) :contentReference[oaicite:5]{index=5}
    "communities": "synthetic connected caveman graph",

    # از node_link_graph روی ppi-G.json (PPI) :contentReference[oaicite:6]{index=6}
    "ppi": "biological protein-protein interaction graph",

    # از read_edgelist روی email.txt (شبکه ایمیل) :contentReference[oaicite:7]{index=7}
    "email": "social email communication graph ",

    # از PROTEINS_full در Graph_load_batch (گراف‌های پروتئین) :contentReference[oaicite:8]{index=8} :contentReference[oaicite:9]{index=9}
    "protein": "biological protein structure graph",

    # طبق داک Planetoid: citation network (nodes=document, edges=citation) :contentReference[oaicite:10]{index=10}
    "Cora": "citation network of scientific publications",

    "CiteSeer": "citation network of scientific publications",
}

FEATURE_TYPE = {
    # feature = I_N (identity) :contentReference[oaicite:11]{index=11}
    "grid": "one-hot features",

    # feature = I_N با permute ستون‌ها :contentReference[oaicite:12]{index=12}
    "communities": "permuted one-hot features",

    # feats از ppi-feats.npy + log-transform روی دو ستون اول :contentReference[oaicite:13]{index=13}
    "ppi": "continuous features",

    # feature = np.ones((n,1)) :contentReference[oaicite:14]{index=14}
    "email": "constant scalar feature",

    # attributes از PROTEINS_full_node_attributes.txt و نرمال‌سازی (mean/std) :contentReference[oaicite:15]{index=15}
    "protein": "continuous biochemical features",

    # طبق تعریف رسمی Cora/CiteSeer: 0/1 bag-of-words over word dictionary :contentReference[oaicite:16]{index=16}
    "Cora": "binary bag-of-words features",
    "CiteSeer": "binary bag-of-words features",
}


def compute_graph_stats(G, feature, node_labels=None, pair_labels=None):
    """
    G:   networkx.Graph
    feature: np.ndarray [num_nodes, feat_dim]
    node_labels: np.ndarray of shape [num_nodes] (class label per node)  یا None
    pair_labels: np.ndarray of shape [n, n] (1 اگر دو نود هم‌کلاس هستند) یا None
    """
    V = G.number_of_nodes()
    E = G.number_of_edges()

    if V == 0:
        return dict(
            V=0, E=0, avg_degree=0.0, degree_variance=0.0,
            density=0.0, homophily=None,
            feat_dim=int(feature.shape[1]) if feature is not None else 0,
        )

    deg = np.array([d for _, d in G.degree()])
    avg_deg = float(deg.mean())
    var_deg = float(deg.var())

    density = float(nx.density(G))
    feat_dim = int(feature.shape[1]) if feature is not None else 0

    homophily = None
    if node_labels is not None:
        # homophily بر اساس label نودی
        y = np.asarray(node_labels)
        same = 0
        if E > 0:
            for u, v in G.edges():
                if y[u] == y[v]:
                    same += 1
            homophily = same / E
    elif pair_labels is not None:
        # homophily بر اساس ماتریس edge_labels (1 = same-class)
        nodes = list(G.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        same = 0
        if E > 0:
            for u, v in G.edges():
                iu = node_to_idx[u]
                iv = node_to_idx[v]
                if iu > iv:
                    val = pair_labels[iu, iv]
                else:
                    val = pair_labels[iv, iu]
                if val == 1:
                    same += 1
            homophily = same / E

    return dict(
        V=V,
        E=E,
        avg_degree=avg_deg,
        degree_variance=var_deg,
        density=density,
        homophily=homophily,
        feat_dim=feat_dim,
    )


if __name__ == "__main__":
    datasets = ["communities", "grid", "ppi", "email", "protein", "Cora", "CiteSeer"]
    summary_rows = []

    for name in datasets:
        print(f"\n=== Loading {name} ===")

        graphs = []
        features = []
        node_labels_list = []
        pair_labels_list = []

        if name in ["communities", "grid", "ppi", "email", "protein"]:
            # استفاده از load_graphs خودت :contentReference[oaicite:17]{index=17}
            graphs_raw, feats_raw, edge_labels_raw, node_labels_raw, _, _, _ = load_graphs(name)

            graphs = graphs_raw
            features = feats_raw
            # edge_labels_raw و node_labels_raw لیست هستند (ممکنه پر یا خالی باشند)
            if edge_labels_raw is None:
                edge_labels_raw = []
            if node_labels_raw is None:
                node_labels_raw = []

            # هم‌اندازه کردن لیست‌ها
            max_len = len(graphs)
            while len(edge_labels_raw) < max_len:
                edge_labels_raw.append(None)
            while len(node_labels_raw) < max_len:
                node_labels_raw.append(None)

            pair_labels_list = edge_labels_raw
            node_labels_list = node_labels_raw

        elif name in ["Cora", "CiteSeer"]:
            # استفاده مستقیم از Planetoid (PyG) :contentReference[oaicite:18]{index=18}
            dataset = tg.datasets.Planetoid(root="datasets/" + name, name=name)
            for data in dataset:
                # to_undirected=True → گراف بدون جهت ساده، هر یال یک‌بار :contentReference[oaicite:19]{index=19}
                G = to_networkx(data, to_undirected=True)
                graphs.append(G)
                features.append(data.x.numpy())
                node_labels_list.append(data.y.numpy())
                pair_labels_list.append(None)

        else:
            raise ValueError(f"Unknown dataset name: {name}")

        # محاسبه آمار برای همه گراف‌ها
        all_stats = []
        for i, G in enumerate(graphs):
            feat = features[i]
            node_lab = node_labels_list[i] if i < len(node_labels_list) else None
            pair_lab = pair_labels_list[i] if i < len(pair_labels_list) else None

            stats = compute_graph_stats(G, feat, node_labels=node_lab, pair_labels=pair_lab)
            all_stats.append(stats)

        df = pd.DataFrame(all_stats)

        hom_col = df["homophily"]
        if hom_col.notna().any():
            homophily_info = float(hom_col.dropna().mean())
        else:
            homophily_info = "No label / N/A"

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

            "homophily_mean": homophily_info,
            "feature_dim": int(df["feat_dim"].iloc[0]) if len(df) > 0 else 0,

            "graph_type": GRAPH_TYPE[name],
            "feature_type": FEATURE_TYPE[name],
        }

        summary_rows.append(summary)

    df_out = pd.DataFrame(summary_rows)
    df_out.to_csv("dataset_statistics_LLM_ALL.csv", index=False)
    print("\nSaved → dataset_statistics_LLM_ALL.csv")
    print(df_out)
