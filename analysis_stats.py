import networkx as nx
import numpy as np
import pandas as pd
from dataset import get_tg_dataset
from args import make_args

def compute_graph_stats(data):
    G = nx.Graph()
    edge_index = data.edge_index.cpu().numpy()
    edges = list(zip(edge_index[0], edge_index[1]))
    G.add_edges_from(edges)

    V = G.number_of_nodes()
    E = G.number_of_edges()

    deg = np.array([d for _, d in G.degree()])
    avg_deg = deg.mean()
    var_deg = deg.var()

    density = nx.density(G)
    clustering = nx.average_clustering(G)

    if nx.is_connected(G):
        diameter = nx.diameter(G)
    else:
        diameter = max(nx.diameter(G.subgraph(c)) for c in nx.connected_components(G))

    from networkx.algorithms.community import greedy_modularity_communities
    comm = greedy_modularity_communities(G)
    modularity = nx.algorithms.community.modularity(G, comm)

    homophily = None
    if hasattr(data, "y") and data.y is not None:
        y = data.y.cpu().numpy()
        same = sum(1 for u, v in G.edges() if y[u] == y[v])
        homophily = same / E

    feat_dim = data.x.size(1)

    return dict(
        V=V,
        E=E,
        avg_degree=avg_deg,
        degree_variance=var_deg,
        density=density,
        clustering=clustering,
        diameter=diameter,
        modularity=modularity,
        homophily=homophily,
        feat_dim=feat_dim,
    )


if __name__ == "__main__":
    args = make_args()
    args.task = "link"

    datasets_link = ["grid", "communities", "ppi"]
    datasets_pair = ["communities", "email", "protein"]

    datasets = datasets_link if args.task == "link" else datasets_pair

    summary_rows = []

    for name in datasets:
        print(f"\n=== Loading {name} ===")
        data_list = get_tg_dataset(args, name, use_cache=False)

        all_stats = [compute_graph_stats(d) for d in data_list]
        df = pd.DataFrame(all_stats)

        summary = {
            "dataset": name,
            "graphs": len(data_list),

            # --- your requested additions ---
            "V_total": df["V"].sum(),
            "E_total": df["E"].sum(),

            "V_mean": df["V"].mean(),
            "V_min": df["V"].min(),
            "V_max": df["V"].max(),

            "E_mean": df["E"].mean(),
            "E_min": df["E"].min(),
            "E_max": df["E"].max(),

            "avg_deg_mean": df["avg_degree"].mean(),
            "deg_var_mean": df["degree_variance"].mean(),
            "density_mean": df["density"].mean(),
            "clustering_mean": df["clustering"].mean(),
            "diameter_mean": df["diameter"].mean(),
            "modularity_mean": df["modularity"].mean(),
            "homophily_mean": df["homophily"].mean() if df["homophily"].notna().any() else None,
            "feature_dim": df["feat_dim"].iloc[0],
        }

        summary_rows.append(summary)

    df_out = pd.DataFrame(summary_rows)
    df_out.to_csv("dataset_statistics.csv", index=False)
    print("\nSaved → dataset_statistics.csv")
