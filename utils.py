import torch
import networkx as nx
import numpy as np
import multiprocessing as mp
import random
import torch.nn.functional as F




# # approximate
def get_edge_mask_link_negative_approximate(mask_link_positive, num_nodes, num_negtive_edges):
    links_temp = np.zeros((num_nodes, num_nodes)) + np.identity(num_nodes)
    mask_link_positive = duplicate_edges(mask_link_positive)
    links_temp[mask_link_positive[0],mask_link_positive[1]] = 1
    # add random noise
    links_temp += np.random.rand(num_nodes,num_nodes)
    prob = num_negtive_edges / (num_nodes*num_nodes-mask_link_positive.shape[1])
    mask_link_negative = np.stack(np.nonzero(links_temp<prob))
    return mask_link_negative


# exact version, slower
def get_edge_mask_link_negative(mask_link_positive, num_nodes, num_negtive_edges):
    mask_link_positive_set = []
    for i in range(mask_link_positive.shape[1]):
        mask_link_positive_set.append(tuple(mask_link_positive[:,i]))
    mask_link_positive_set = set(mask_link_positive_set)

    mask_link_negative = np.zeros((2,num_negtive_edges), dtype=mask_link_positive.dtype)
    for i in range(num_negtive_edges):
        while True:
            mask_temp = tuple(np.random.choice(num_nodes,size=(2,),replace=False))
            if mask_temp not in mask_link_positive_set:
                mask_link_negative[:,i] = mask_temp
                break

    return mask_link_negative

def neighbor_sim_loss(emb, edge_index_np, device, mode='cos'):
    # emb: Tensor [N, D]
    # edge_index_np: np.ndarray [2, E] (یال‌های TRAIN)
    src = torch.from_numpy(edge_index_np[0, :]).long().to(device)
    dst = torch.from_numpy(edge_index_np[1, :]).long().to(device)
    h_src = emb[src]
    h_dst = emb[dst]
    # چون فقط cos می‌خوایم:
    cos = F.cosine_similarity(h_src, h_dst, dim=-1)  # ∈ [-1,1]
    return (1.0 - cos).mean()  # میانگین (1 - cos)


def resample_edge_mask_link_negative(data):
    data.mask_link_negative_train = get_edge_mask_link_negative(data.mask_link_positive_train, num_nodes=data.num_nodes,
                                                      num_negtive_edges=data.mask_link_positive_train.shape[1])
    data.mask_link_negative_val = get_edge_mask_link_negative(data.mask_link_positive, num_nodes=data.num_nodes,
                                                      num_negtive_edges=data.mask_link_positive_val.shape[1])
    data.mask_link_negative_test = get_edge_mask_link_negative(data.mask_link_positive, num_nodes=data.num_nodes,
                                                     num_negtive_edges=data.mask_link_positive_test.shape[1])


def deduplicate_edges(edges):
    edges_new = np.zeros((2,edges.shape[1]//2), dtype=int)
    # add none self edge
    j = 0
    skip_node = {} # node already put into result
    for i in range(edges.shape[1]):
        if edges[0,i]<edges[1,i]:
            edges_new[:,j] = edges[:,i]
            j += 1
        elif edges[0,i]==edges[1,i] and edges[0,i] not in skip_node:
            edges_new[:,j] = edges[:,i]
            skip_node.add(edges[0,i])
            j += 1

    return edges_new

def duplicate_edges(edges):
    return np.concatenate((edges, edges[::-1,:]), axis=-1)


# each node at least remain in the new graph
def split_edges(edges, remove_ratio, connected=False):
    e = edges.shape[1]
    edges = edges[:, np.random.permutation(e)]
    if connected:
        unique, counts = np.unique(edges, return_counts=True)
        node_count = dict(zip(unique, counts))

        index_train = []
        index_val = []
        for i in range(e):
            node1 = edges[0,i]
            node2 = edges[1,i]
            if node_count[node1]>1 and node_count[node2]>1: # if degree>1
                index_val.append(i)
                node_count[node1] -= 1
                node_count[node2] -= 1
                if len(index_val) == int(e * remove_ratio):
                    break
            else:
                index_train.append(i)
        index_train = index_train + list(range(i + 1, e))
        index_test = index_val[:len(index_val)//2]
        index_val = index_val[len(index_val)//2:]

        edges_train = edges[:, index_train]
        edges_val = edges[:, index_val]
        edges_test = edges[:, index_test]
    else:
        split1 = int((1-remove_ratio)*e)
        split2 = int((1-remove_ratio/2)*e)
        edges_train = edges[:,:split1]
        edges_val = edges[:,split1:split2]
        edges_test = edges[:,split2:]

    return edges_train, edges_val, edges_test




def edge_to_set(edges):
    edge_set = []
    for i in range(edges.shape[1]):
        edge_set.append(tuple(edges[:, i]))
    edge_set = set(edge_set)
    return edge_set


def get_link_mask(data, remove_ratio=0.2, resplit=True, infer_link_positive=True):
    if resplit:
        if infer_link_positive:
            data.mask_link_positive = deduplicate_edges(data.edge_index.numpy())
        data.mask_link_positive_train, data.mask_link_positive_val, data.mask_link_positive_test = \
            split_edges(data.mask_link_positive, remove_ratio)
    resample_edge_mask_link_negative(data)


def add_nx_graph(data):
    G = nx.Graph()
    edge_numpy = data.edge_index.numpy()
    edge_list = []
    for i in range(data.num_edges):
        edge_list.append(tuple(edge_numpy[:, i]))
    G.add_edges_from(edge_list)
    data.G = G

def single_source_shortest_path_length_range(graph, node_range, cutoff):
    dists_dict = {}
    for node in node_range:
        dists_dict[node] = nx.single_source_shortest_path_length(graph, node, cutoff)
    return dists_dict

def merge_dicts(dicts):
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result

def all_pairs_shortest_path_length_parallel(graph,cutoff=None,num_workers=4):
    nodes = list(graph.nodes)
    random.shuffle(nodes)
    if len(nodes)<50:
        num_workers = int(num_workers/4)
    elif len(nodes)<400:
        num_workers = int(num_workers/2)

    pool = mp.Pool(processes=num_workers)
    results = [pool.apply_async(single_source_shortest_path_length_range,
            args=(graph, nodes[int(len(nodes)/num_workers*i):int(len(nodes)/num_workers*(i+1))], cutoff)) for i in range(num_workers)]
    output = [p.get() for p in results]
    dists_dict = merge_dicts(output)
    pool.close()
    pool.join()
    return dists_dict


def precompute_dist_data(edge_index, num_nodes, approximate=0):
        '''
        Here dist is 1/real_dist, higher actually means closer, 0 means disconnected
        :return:
        '''
        graph = nx.Graph()
        edge_list = edge_index.transpose(1,0).tolist()
        graph.add_edges_from(edge_list)

        n = num_nodes
        dists_array = np.zeros((n, n))
        # dists_dict = nx.all_pairs_shortest_path_length(graph,cutoff=approximate if approximate>0 else None)
        # dists_dict = {c[0]: c[1] for c in dists_dict}
        dists_dict = all_pairs_shortest_path_length_parallel(graph,cutoff=approximate if approximate>0 else None)
        for i, node_i in enumerate(graph.nodes()):
            shortest_dist = dists_dict[node_i]
            for j, node_j in enumerate(graph.nodes()):
                dist = shortest_dist.get(node_j, -1)
                if dist!=-1:
                    # dists_array[i, j] = 1 / (dist + 1)
                    dists_array[node_i, node_j] = 1 / (dist + 1)
        return dists_array



def get_random_anchorset(n,c=0.5):
    m = int(np.log2(n))
    copy = int(c*m)
    anchorset_id = []
    for i in range(m):
        anchor_size = int(n/np.exp2(i + 1))
        for j in range(copy):
            anchorset_id.append(np.random.choice(n,size=anchor_size,replace=False))
    return anchorset_id

def get_dist_max(anchorset_id, dist, device):
    dist_max = torch.zeros((dist.shape[0],len(anchorset_id))).to(device)
    # print("dist_max=",dist_max.size())
    dist_argmax = torch.zeros((dist.shape[0],len(anchorset_id))).long().to(device)
    # print("dist_argmax=",dist_argmax.size())
    for i in range(len(anchorset_id)):
        temp_id = torch.as_tensor(anchorset_id[i], dtype=torch.long)
        dist_temp = dist[:, temp_id]
        dist_max_temp, dist_argmax_temp = torch.max(dist_temp, dim=-1)
        dist_argmax_temp=dist_argmax_temp.to(torch.device('cpu'))
        dist_max[:,i] = dist_max_temp
        dist_argmax[:,i] = temp_id[dist_argmax_temp]
    return dist_max, dist_argmax


def preselect_anchor(data, layer_num=1, anchor_num=32, anchor_size_num=4, device='cpu', args=None):
    import torch

    data.anchor_size_num = anchor_size_num
    data.anchor_set = []
    anchor_num_per_size = anchor_num//anchor_size_num

    #anchor selection
    if args is None:
        method = 'random'
    else:
        method = args.anchor_method
    
    if method == 'random':
        # روش فعلی، یعنی استفاده از get_random_anchorset
        anchorset_id = get_random_anchorset(data.num_nodes, c=1)
        data.dists_max, data.dists_argmax = get_dist_max(anchorset_id, data.dists, device)

    
    elif method == 'betweenness':
        import math
        import networkx as nx

        m = int(math.log2(data.num_nodes))
        anchor_num = m * m  # مثل random

        # ساخت گراف NetworkX از edge_index
        G = nx.Graph()
        edges = data.edge_index.cpu().numpy()
        G.add_edges_from(edges.T)

        # محاسبه betweenness centrality (نسخه دقیق)
        centrality = nx.betweenness_centrality(G, normalized=True)

        # تبدیل دیکشنری به tensor
        centrality_tensor = torch.zeros(data.num_nodes, device=data.edge_index.device)
        for node, value in centrality.items():
            centrality_tensor[node] = value

        # انتخاب top-k نودها
        topk_nodes = torch.topk(centrality_tensor, anchor_num).indices

        # ساخت anchorset_id
        anchorset_id = [[n.item()] for n in topk_nodes]

        # محاسبه dists_max و dists_argmax
        data.dists_max, data.dists_argmax = get_dist_max(anchorset_id, data.dists, device)
        return


    elif method == 'eigenvector':
        import math
        import networkx as nx

        m = int(math.log2(data.num_nodes))
        anchor_num = m * m  # مثل random

        # ساخت گراف NetworkX از edge_index
        G = nx.Graph()
        edges = data.edge_index.cpu().numpy()
        G.add_edges_from(edges.T)

        # محاسبه eigenvector centrality
        centrality = nx.eigenvector_centrality_numpy(G)

        # تبدیل دیکشنری به tensor
        centrality_tensor = torch.zeros(data.num_nodes, device=data.edge_index.device)
        for node, value in centrality.items():
            centrality_tensor[node] = value

        # انتخاب top-k نودها
        topk_nodes = torch.topk(centrality_tensor, anchor_num).indices

        # ساخت anchorset_id
        anchorset_id = [[n.item()] for n in topk_nodes]

        # محاسبه dists_max و dists_argmax
        data.dists_max, data.dists_argmax = get_dist_max(anchorset_id, data.dists, device)
        return


    elif method == 'degree':
        import math
        m = int(math.log2(data.num_nodes))
        anchor_num = m * m  # مثل random
    
        # Compute degree directly from edge_index
        degrees = torch.zeros(data.num_nodes, device=data.edge_index.device)
        degrees.scatter_add_(0, data.edge_index[0], torch.ones(data.edge_index.size(1), device=data.edge_index.device))
    
        # Select top-k nodes with highest degrees
        topk_nodes = torch.topk(degrees, anchor_num).indices
    
        # Build anchorset_id
        anchorset_id = [[n.item()] for n in topk_nodes]
    
        data.dists_max, data.dists_argmax = get_dist_max(anchorset_id, data.dists, device)
        return

    elif method == 'degree_coverage':
        from torch_geometric.utils import to_undirected, degree
        import heapq
        from collections import defaultdict
        import math

        hop = 1            # ← این عدد را هر بار به شعاع دلخواهت تغییر بده (۱، ۲، ۳، ...)

        m = int(math.log2(data.num_nodes))
        anchor_num = m * m

        edge_index = to_undirected(data.edge_index)
        deg = degree(edge_index[0], data.num_nodes).cpu()

        heap = [(-deg[i].item(), i) for i in range(data.num_nodes)]
        heapq.heapify(heap)

        adj = defaultdict(set)
        for u, v in edge_index.t().tolist():
            adj[u].add(v)
            adj[v].add(u)

        selected, marked = [], set()
        while heap and len(selected) < anchor_num:
            _, v = heapq.heappop(heap)
            if v in marked:
                continue

            selected.append(v)

        # ــ علامت‌گذاری خودش و همهٔ نودهای تا «hop» پله فاصله ــ
            frontier = {v}
            marked.update(frontier)          # لایهٔ صفر (خود انکر)
            for _ in range(hop):
                next_frontier = set()
                for u in frontier:
                    next_frontier.update(adj[u])
                next_frontier -= marked      # فقط نودهایی که قبلاً مارک نشده‌اند
                if not next_frontier:        # اگر دیگر چیزی برای گسترش نیست، تمام
                    break
                marked.update(next_frontier)
                frontier = next_frontier
        # ــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــــ

        anchorset_id = [[n] for n in selected]
        data.dists_max, data.dists_argmax = get_dist_max(anchorset_id, data.dists, device)
        return

    elif method == 'enhanced_degree_coverage':
    # ------------------------------------------------------------
    #  تنظیمات
    # ------------------------------------------------------------
        from torch_geometric.utils import to_undirected, degree
        import heapq
        from collections import defaultdict
        import math, torch

        hop = 1                     # ← شعاع پوشش (۱-هاپ، ۲-هاپ، …)
        m = int(math.log2(data.num_nodes))
        anchor_num = m * m

    # ------------------------------------------------------------
    #  درجه‌گیری و ساخت داده‌های کمکی
    # ------------------------------------------------------------
        edge_index = to_undirected(data.edge_index)
        deg = degree(edge_index[0], data.num_nodes).cpu()            # درجهٔ هر رأس روی CPU

        heap = [(-deg[i].item(), i) for i in range(data.num_nodes)]  # max-heap  (منفی چون heapq)
        heapq.heapify(heap)

    # adjacency list برای پیمایش hop
        adj = defaultdict(set)
        for u, v in edge_index.t().tolist():
            adj[u].add(v)
            adj[v].add(u)

            selected, marked = [], set()

    # ------------------------------------------------------------
    #  حلقهٔ انتخاب انکرها
    # ------------------------------------------------------------
        while heap and len(selected) < anchor_num:

        # ----------- برداشتن همهٔ نودهای با بالاترین درجهٔ فعلی -----------
            top_nodes = []
            while heap:
                d_neg, v = heapq.heappop(heap)
                if v in marked:
                    continue
                deg_max = -d_neg
                top_nodes.append(v)

            # برداشتن بقیهٔ نودهای با همین درجه
                while heap and -heap[0][0] == deg_max:
                    d_neg2, v2 = heapq.heappop(heap)
                    if v2 not in marked:
                        top_nodes.append(v2)
                break     # بعد از جمع کردن هم‌درجه‌ها

            if not top_nodes:
                break  # هیچ نامزد معتبری نمانده است

        # ----------- تعیین بهترین نود بر اساس «دورترین از انکرهای قبلی» -----------
            if len(selected) == 0 or len(top_nodes) == 1:
            # اولین انکر یا فقط یک نامزد وجود دارد
                next_anchor = top_nodes[0]
                for u in top_nodes[1:]:
                    heapq.heappush(heap, (-deg[u].item(), u))     # بقیه را پس می‌دهیم
            else:
            # فاصلهٔ هر نامزد تا نزدیک‌ترین انکر قبلی
                top_tensor = torch.tensor(top_nodes, device=data.dists.device)
                sel_tensor = torch.tensor(selected, device=data.dists.device)
                dist_mat   = data.dists[top_tensor][:, sel_tensor]        # شکل (len(top), len(sel))
                min_dists  = dist_mat.min(dim=1).values                   # نزدیک‌ترین انکر
                idx        = torch.argmax(min_dists).item()               # دورترین = بزرگ‌ترین حداقل فاصله
                next_anchor = top_nodes[idx]

            # بقیهٔ نامزدها را به heap برگردان
                for i, u in enumerate(top_nodes):
                    if i != idx:
                        heapq.heappush(heap, (-deg[u].item(), u))

        # ----------- اضافه‌کردن انکر و مارک‌کردن پوشش hop -----------
            selected.append(next_anchor)

            frontier = {next_anchor}
            marked.update(frontier)           # شعاع صفر (خودش)
            for _ in range(hop):              # گسترش تا hop پله
                nxt = set()
                for u in frontier:
                    nxt.update(adj[u])
                nxt -= marked                 # فقط نودهای جدید
                if not nxt:
                    break
                marked.update(nxt)
                frontier = nxt

    elif method == 'hyper':
        import math, torch
        # -------------------------------
        # 1️⃣ تعداد انکرها
        # -------------------------------
        m = int(math.log2(data.num_nodes))
        anchor_num = m  # مثل مقاله: Q < log2(N)

        # -------------------------------
        # 2️⃣ انتخاب Anchor Nodes (A_V)
        # -------------------------------
        anchor_nodes = np.random.choice(data.num_nodes, size=anchor_num, replace=False)
        anchor_nodes = [[int(a)] for a in anchor_nodes]  # قالب [[a1], [a2], ...]

        # -------------------------------
        # 3️⃣ ساخت هایپرج‌ها (از featureها)
        # هر ستون ویژگی در ماتریس X یک "هایپرج" فرض می‌شود
        # یعنی مجموعه نودهایی که مقدار feature بالاتر از ۰ دارند.
        # -------------------------------
        X = data.x.cpu().numpy()
        num_features = X.shape[1]
        hyperedges = []
        for j in range(num_features):
            nodes_with_feat = np.where(X[:, j] > 0)[0]
            if len(nodes_with_feat) > 0:
                hyperedges.append(nodes_with_feat)

        # -------------------------------
        # 4️⃣ انتخاب Anchor Hyperedges (A_E)
        # -------------------------------
        num_hyper = len(hyperedges)
        anchor_num_hyper = min(anchor_num, num_hyper)
        anchor_hyper = np.random.choice(num_hyper, size=anchor_num_hyper, replace=False)
        # نودهای داخل هر هایپرج به‌عنوان مرجع فاصله در نظر گرفته می‌شوند
        anchor_hyper_nodes = [hyperedges[i].tolist() for i in anchor_hyper]

        # -------------------------------
        # 5️⃣ ادغام Anchor Nodes و Anchor Hyperedges
        # -------------------------------
        anchorset_id = anchor_nodes + anchor_hyper_nodes

        # -------------------------------
        # 6️⃣ محاسبهٔ dist_max و argmax
        # -------------------------------
        data.dists_max, data.dists_argmax = get_dist_max(anchorset_id, data.dists, device)
        return


    # ------------------------------------------------------------
    #  تبدیل به قالب PGNN و محاسبهٔ dists_max / argmax
    # ------------------------------------------------------------
        anchorset_id = [[n] for n in selected]                # [[a1], [a2], ...]
        data.dists_max, data.dists_argmax = get_dist_max(anchorset_id, data.dists, device)
        return


    elif method == 'degree_farthest':
    # ------------------------------------------------------------
    #   وارد کردن ماژول‌های لازم
    # ------------------------------------------------------------
        from torch_geometric.utils import to_undirected, degree
        import heapq, math, torch

    # ------------------------------------------------------------
    #   تعداد انکرها طبق m = log2(N)  →  k = m²
    # ------------------------------------------------------------
        m = int(math.log2(data.num_nodes))
        anchor_num = m * m

    # ------------------------------------------------------------
    #   درجهٔ هر نود روی CPU
    # ------------------------------------------------------------
        edge_index = to_undirected(data.edge_index)
        deg = degree(edge_index[0], data.num_nodes).cpu()           # 1-D tensor (N,)

    # max-heap از (-degree, node)
        heap = [(-deg[i].item(), i) for i in range(data.num_nodes)]
        heapq.heapify(heap)

        selected, selected_set = [], set()                          # انکرهای نهایی

    # ------------------------------------------------------------
    #   حلقهٔ حریصانه
    # ------------------------------------------------------------
        while heap and len(selected) < anchor_num:

        # ---------- برداشتن همهٔ نودهای با بیشترین درجهٔ فعلی ----------
            d_neg, v = heapq.heappop(heap)
            if v in selected_set:           # اگر قبلاً انتخاب شده بود، بپر
                continue
            deg_max = -d_neg                # مقدار درجهٔ بیشینه

            same_degree = [v]               # نودهای با همین درجه
            while heap and -heap[0][0] == deg_max:
                d_neg2, v2 = heapq.heappop(heap)
                if v2 not in selected_set:  # از قبل انتخاب نشده باشد
                    same_degree.append(v2)

        # ---------- اگر بیش از یک نامزدِ هم‌درجه داریم ----------
            if len(selected) > 0 and len(same_degree) > 1:
                cand_t  = torch.tensor(same_degree, device=data.dists.device)
                prev_t  = torch.tensor(selected,    device=data.dists.device)
            # ماتریس فاصلهٔ نامزدها تا انکرهای قبلی
                dist_mat = data.dists[cand_t][:, prev_t]            # (cand, selected)
                min_d    = dist_mat.min(dim=1).values               # نزدیک‌ترین انکر
                idx      = torch.argmax(min_d).item()               # دورترین نامزد
                next_anchor = same_degree[idx]
            # بقیهٔ نامزدها را به heap برگردان
                for i, node in enumerate(same_degree):
                    if i != idx:
                        heapq.heappush(heap, (-deg[node].item(), node))
            else:
            # فقط یک نامزد یا هیچ انکری از قبل انتخاب نشده
                next_anchor = same_degree[0]
                for node in same_degree[1:]:
                    heapq.heappush(heap, (-deg[node].item(), node))

        # ---------- ثبت انکر ----------
            selected.append(next_anchor)
            selected_set.add(next_anchor)

    # ------------------------------------------------------------
    #   تبدیل به قالب PGNN  و به‌روزرسانی data
    # ------------------------------------------------------------
        anchorset_id = [[n] for n in selected]
        data.dists_max, data.dists_argmax = get_dist_max(anchorset_id, data.dists, device)
        return

    
        
    for i in range(anchor_size_num):
        # print("i=",i)
        # print("anchor_size_num=",anchor_size_num)
        anchor_size = 2**(i+1)-1
        # print("anchor_size=",anchor_size)
        anchors = np.random.choice(data.num_nodes, size=(layer_num,anchor_num_per_size,anchor_size), replace=True)
        # print("anchors=",anchors)
        data.anchor_set.append(anchors)
    # print("data.anchor_set=",data.anchor_set)
    data.anchor_set_indicator = np.zeros((layer_num, anchor_num, data.num_nodes), dtype=int)

    anchorset_id = get_random_anchorset(data.num_nodes,c=1)
    # print("len(anchorset_id)=",len(anchorset_id))

    # print("data.dists=",data.dists)
    data.dists_max, data.dists_argmax = get_dist_max(anchorset_id, data.dists, device)
