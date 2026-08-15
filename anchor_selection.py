import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch_geometric.nn import GCNConv


def compute_anchor_budget(
    num_nodes,
    mode="main",
    fixed_k=None,
    reduction=1,
    exact_fixed=False,
):
    """
    Compute the singleton-anchor budget K for one graph.

    Modes
    -----
    main:
        K = min(N, floor(log2(N))^2)

    rule:
        K = min(N, floor(log2(N)))

    progressive:
        Start from K_main and reduce it by `reduction`
        (normally 1, 2, 4, or 8).

    fixed:
        Use a numerical target K.
        If exact_fixed=True, require N >= fixed_k.
        Otherwise use capped K = min(N, fixed_k).

    Notes
    -----
    A one-node graph is treated as a degenerate case with K=1.
    """
    num_nodes = int(num_nodes)

    if num_nodes < 1:
        raise ValueError("num_nodes must be >= 1.")

    # Degenerate single-node graph safeguard.
    if num_nodes == 1:
        if mode == "fixed" and exact_fixed and fixed_k not in (None, 1):
            raise ValueError(
                f"Exact fixed-K={fixed_k} is infeasible for N=1."
            )
        return 1

    m = int(math.log2(num_nodes))
    k_main = min(num_nodes, m * m)

    if mode == "main":
        k = k_main

    elif mode == "rule":
        k = min(num_nodes, m)

    elif mode == "progressive":
        reduction = int(reduction)
        if reduction < 1:
            raise ValueError("reduction must be >= 1.")
        k = max(1, k_main // reduction)

    elif mode == "fixed":
        if fixed_k is None:
            raise ValueError("fixed_k must be provided when mode='fixed'.")

        fixed_k = int(fixed_k)
        if fixed_k < 1:
            raise ValueError("fixed_k must be >= 1.")

        if exact_fixed and num_nodes < fixed_k:
            raise ValueError(
                f"Exact fixed-K={fixed_k} is infeasible for N={num_nodes}."
            )

        k = fixed_k if exact_fixed else min(num_nodes, fixed_k)

    else:
        raise ValueError(
            "mode must be one of: 'main', 'rule', 'progressive', 'fixed'."
        )

    return int(min(num_nodes, max(1, k)))


def hungarian_max_assignment(scores):
    """
    Hard joint one-to-one assignment.

    Parameters
    ----------
    scores : torch.Tensor
        Score matrix with shape [K, N], where rows are slots and
        columns are candidate nodes.

    Returns
    -------
    A_hard : torch.Tensor
        Hard assignment matrix [K, N].
        Every row sums to 1 and every column sums to at most 1.

    anchor_idx : torch.LongTensor
        Length-K vector. anchor_idx[k] is the node assigned to slot k.
    """
    if scores.dim() != 2:
        raise ValueError("scores must have shape [K, N].")

    k, n = scores.shape

    if k < 1 or n < 1:
        raise ValueError("K and N must both be >= 1.")

    if k > n:
        raise ValueError(
            f"One-to-one singleton assignment requires K <= N, got K={k}, N={n}."
        )

    # Hungarian is a discrete forward solver, so it must not be part
    # of the autograd graph.
    cost = (-scores.detach()).cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)

    if len(row_ind) != k:
        raise RuntimeError(
            f"Hungarian assigned {len(row_ind)} rows, expected {k}."
        )

    row_ind_t = torch.as_tensor(
        row_ind, dtype=torch.long, device=scores.device
    )
    col_ind_t = torch.as_tensor(
        col_ind, dtype=torch.long, device=scores.device
    )

    A_hard = torch.zeros_like(scores)
    A_hard[row_ind_t, col_ind_t] = 1.0

    # Explicitly store node indices in slot order.
    anchor_idx = torch.empty(k, dtype=torch.long, device=scores.device)
    anchor_idx[row_ind_t] = col_ind_t

    return A_hard, anchor_idx


def slack_sinkhorn(scores, temperature=1.0, num_iters=30, eps=1e-12):
    """
    Slack-Sinkhorn soft relaxation in log-space.

    The score matrix has shape [K, N].

    For K < N, a non-learnable slack row is added with normalized
    marginals:

        r = [1/N, ..., 1/N, (N-K)/N]
        c = [1/N, ..., 1/N]

    If T is the resulting transport matrix:

        A_soft = N * T[:K, :]

    Therefore each anchor row approximately sums to 1, while the
    total anchor mass of each node column is at most 1.

    For K == N, the slack row would have zero mass, so it is omitted.
    """
    if scores.dim() != 2:
        raise ValueError("scores must have shape [K, N].")

    k, n = scores.shape

    if k < 1 or n < 1:
        raise ValueError("K and N must both be >= 1.")

    if k > n:
        raise ValueError(
            f"Slack-Sinkhorn requires K <= N, got K={k}, N={n}."
        )

    if temperature <= 0:
        raise ValueError("temperature must be > 0.")

    num_iters = int(num_iters)
    if num_iters < 1:
        raise ValueError("num_iters must be >= 1.")

    dtype = scores.dtype
    device = scores.device

    anchor_logits = scores / float(temperature)

    if k < n:
        # Fixed, non-learnable slack score for v1.
        slack_logits = torch.zeros(
            1, n, dtype=dtype, device=device
        )
        log_transport = torch.cat(
            [anchor_logits, slack_logits], dim=0
        )

        anchor_row_mass = torch.full(
            (k,), 1.0 / n, dtype=dtype, device=device
        )
        slack_mass = torch.tensor(
            [(n - k) / n], dtype=dtype, device=device
        )
        row_marginals = torch.cat(
            [anchor_row_mass, slack_mass], dim=0
        )
    else:
        # K == N: zero-mass slack row is unnecessary.
        log_transport = anchor_logits
        row_marginals = torch.full(
            (k,), 1.0 / n, dtype=dtype, device=device
        )

    col_marginals = torch.full(
        (n,), 1.0 / n, dtype=dtype, device=device
    )

    log_r = torch.log(row_marginals.clamp_min(eps))
    log_c = torch.log(col_marginals.clamp_min(eps))

    # Alternating log-domain row/column normalization.
    for _ in range(num_iters):
        log_transport = (
            log_transport
            - torch.logsumexp(log_transport, dim=1, keepdim=True)
            + log_r.unsqueeze(1)
        )

        log_transport = (
            log_transport
            - torch.logsumexp(log_transport, dim=0, keepdim=True)
            + log_c.unsqueeze(0)
        )

    transport = torch.exp(log_transport)
    A_soft = float(n) * transport[:k, :]

    return A_soft


class SlotAnchorSelector(nn.Module):
    """
    Anchor Selection v1:
      2-layer shared GCN selector
      + learnable role/slot queries
      + Hungarian hard assignment
      + Slack-Sinkhorn soft relaxation
      + straight-through estimator.

    Q_max is shared across graphs. For a graph with K_g anchors,
    only Q_max[:K_g] is used.
    """

    def __init__(
        self,
        input_dim,
        selector_dim,
        k_max,
        temperature=1.0,
        sinkhorn_iters=30,
    ):
        super().__init__()

        input_dim = int(input_dim)
        selector_dim = int(selector_dim)
        k_max = int(k_max)

        if input_dim < 1:
            raise ValueError("input_dim must be >= 1.")
        if selector_dim < 1:
            raise ValueError("selector_dim must be >= 1.")
        if k_max < 1:
            raise ValueError("k_max must be >= 1.")
        if temperature <= 0:
            raise ValueError("temperature must be > 0.")
        if int(sinkhorn_iters) < 1:
            raise ValueError("sinkhorn_iters must be >= 1.")

        self.input_dim = input_dim
        self.selector_dim = selector_dim
        self.k_max = k_max
        self.temperature = float(temperature)
        self.sinkhorn_iters = int(sinkhorn_iters)

        self.conv1 = GCNConv(input_dim, selector_dim)
        self.conv2 = GCNConv(selector_dim, selector_dim)

        # Shared slot bank for single- or multi-graph datasets.
        self.slot_queries = nn.Parameter(
            torch.empty(k_max, selector_dim)
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        nn.init.xavier_uniform_(self.slot_queries)

    def encode_nodes(self, x, edge_index):
        """
        Selector-only node representation H_sel [N, d_s].
        This representation is used only to choose anchors.
        """
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        return h

    def forward(self, x, edge_index, k):
        """
        Parameters
        ----------
        x : torch.Tensor
            Node features [N, input_dim].

        edge_index : torch.LongTensor
            Graph edges [2, E]. For link prediction this must be
            the training-observable graph.

        k : int
            Number of singleton anchors for this graph.

        Returns
        -------
        dict with:
            H_sel      [N, d_s]
            Q          [K, d_s]
            scores     [K, N]
            A_hard     [K, N]
            A_soft     [K, N]
            A_st       [K, N]
            anchor_idx [K]
        """
        if x.dim() != 2:
            raise ValueError("x must have shape [N, input_dim].")

        n = int(x.size(0))
        k = int(k)

        if k < 1:
            raise ValueError("k must be >= 1.")
        if k > n:
            raise ValueError(
                f"Singleton assignment requires K <= N, got K={k}, N={n}."
            )
        if k > self.k_max:
            raise ValueError(
                f"K={k} exceeds shared slot bank k_max={self.k_max}."
            )

        H_sel = self.encode_nodes(x, edge_index)
        Q = self.slot_queries[:k]

        # L2-normalized role-to-node scores.
        H_norm = F.normalize(H_sel, p=2, dim=-1)
        Q_norm = F.normalize(Q, p=2, dim=-1)

        scores = Q_norm @ H_norm.transpose(0, 1)  # [K, N]

        A_hard, anchor_idx = hungarian_max_assignment(scores)

        A_soft = slack_sinkhorn(
            scores,
            temperature=self.temperature,
            num_iters=self.sinkhorn_iters,
        )

        # Forward is hard; backward follows A_soft.
        A_st = A_hard - A_soft.detach() + A_soft

        return {
            "H_sel": H_sel,
            "Q": Q,
            "scores": scores,
            "A_hard": A_hard,
            "A_soft": A_soft,
            "A_st": A_st,
            "anchor_idx": anchor_idx,
        }

class GlobalTopKSelector(nn.Module):
    """
    Global scalar scorer + Top-K ablation.

    Same 2-layer GCN selector backbone as SlotAnchorSelector,
    but every node receives only ONE global scalar score.

    There are no role-specific slot queries and no joint
    one-to-one Sinkhorn assignment.

    Forward:
        exactly K unique top-scoring singleton nodes.

    Backward:
        straight-through relaxation based on a shared
        soft global ranking distribution.
    """

    def __init__(
        self,
        input_dim,
        selector_dim,
        temperature=1.0,
    ):
        super().__init__()

        input_dim = int(input_dim)
        selector_dim = int(selector_dim)

        if input_dim < 1:
            raise ValueError("input_dim must be >= 1.")

        if selector_dim < 1:
            raise ValueError("selector_dim must be >= 1.")

        if temperature <= 0:
            raise ValueError("temperature must be > 0.")

        self.input_dim = input_dim
        self.selector_dim = selector_dim
        self.temperature = float(temperature)

        # Same selector backbone as our proposed method
        self.conv1 = GCNConv(input_dim, selector_dim)
        self.conv2 = GCNConv(selector_dim, selector_dim)

        # ONE global scalar scorer
        self.score_head = nn.Linear(selector_dim, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.score_head.reset_parameters()

    def encode_nodes(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        return h

    def forward(self, x, edge_index, k):

        if x.dim() != 2:
            raise ValueError(
                "x must have shape [N, input_dim]."
            )

        n = int(x.size(0))
        k = int(k)

        if k < 1:
            raise ValueError("k must be >= 1.")

        if k > n:
            raise ValueError(
                f"Top-K singleton selection requires "
                f"K <= N, got K={k}, N={n}."
            )

        # -------------------------------------------------
        # 1) Shared GCN representation
        # -------------------------------------------------
        H_sel = self.encode_nodes(
            x,
            edge_index
        )

        # -------------------------------------------------
        # 2) ONE scalar score per node
        # -------------------------------------------------
        scores = self.score_head(
            H_sel
        ).squeeze(-1)                     # [N]

        """
        # Stabilize score scale.
        scores = scores / scores.norm(
            p=2
        ).clamp_min(1e-12)
        """

        # -------------------------------------------------
        # 3) Hard global Top-K
        # -------------------------------------------------
        anchor_idx = torch.topk(
            scores,
            k=k,
            largest=True,
            sorted=True
        ).indices                         # [K]

        A_hard = torch.zeros(
            k,
            n,
            dtype=x.dtype,
            device=x.device
        )

        A_hard[
            torch.arange(k, device=x.device),
            anchor_idx
        ] = 1.0

        # -------------------------------------------------
        # 4) Soft global ranking relaxation
        #
        # Unlike Slot-Joint:
        #   - no role-specific distributions
        #   - no Sinkhorn
        #   - no column-capacity constraint
        #
        # All K rows share one global ranking distribution.
        # -------------------------------------------------
        probs = torch.softmax(
            scores / self.temperature,
            dim=0
        )                                 # [N]

        A_soft = probs.unsqueeze(0).expand(
            k,
            -1
        )                                 # [K,N]

        # -------------------------------------------------
        # 5) Straight-through
        # Forward = exact Top-K
        # Backward = global soft scores
        # -------------------------------------------------
        A_st = (
            A_hard
            - A_soft.detach()
            + A_soft
        )

        return {
            "H_sel": H_sel,
            "scores": scores,
            "A_hard": A_hard,
            "A_soft": A_soft,
            "A_st": A_st,
            "anchor_idx": anchor_idx,
        }
