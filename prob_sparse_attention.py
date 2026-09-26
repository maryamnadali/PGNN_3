import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistanceAwareProbSparseCrossAttention(nn.Module):
    """
    Distance-aware multi-head ProbSparse node-anchor cross-attention.

    Main ideas
    ----------
    1) Nodes are queries.
    2) Anchors are keys/values.
    3) K and V are NOT pre-gated by distance.
    4) Positional information enters the attention score as a learnable bias:
           score = QK/sqrt(d_h) + beta * positional_score
    5) Sigmoid is used instead of Softmax so anchors do not need to compete
       for a probability mass that sums to one.
    6) ProbSparse query selection is performed independently per head.
    7) Only Top-q node queries receive full node-anchor attention.

    Stage-1 optimization
    --------------------
    The mathematical formulation is unchanged. The previous per-head Python
    loop, active-node union/mapping, and repeated index_copy operations are
    replaced by a batched/vectorized implementation across attention heads.

    Parameters
    ----------
    input_dim : int
        Input node/anchor feature dimension.

    output_dim : int
        Total output dimension across all heads.

    num_heads : int
        Number of attention heads.

    factor : int
        ProbSparse factor c.
        sample_k ~ c * ln(K)
        top_q    ~ c * ln(N)

    min_top : int
        Minimum number of sampled anchors / selected queries.

    beta_init : float
        Initial positive weight of the positional bias.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        num_heads=4,
        factor=5,
        min_top=1,
        beta_init=1.0,
    ):
        super().__init__()

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_heads = int(num_heads)
        self.factor = int(factor)
        self.min_top = int(min_top)

        if self.output_dim % self.num_heads != 0:
            raise ValueError(
                "output_dim must be divisible by num_heads. "
                f"Got output_dim={output_dim}, "
                f"num_heads={num_heads}."
            )

        if self.factor < 1:
            raise ValueError("factor must be >= 1.")

        if self.min_top < 1:
            raise ValueError("min_top must be >= 1.")

        if beta_init <= 0:
            raise ValueError("beta_init must be > 0.")

        self.head_dim = self.output_dim // self.num_heads

        # --------------------------------------------------
        # Q / K / V projections
        # --------------------------------------------------
        self.Wq = nn.Linear(
            self.input_dim,
            self.output_dim,
            bias=False
        )

        self.Wk = nn.Linear(
            self.input_dim,
            self.output_dim,
            bias=False
        )

        self.Wv = nn.Linear(
            self.input_dim,
            self.output_dim,
            bias=False
        )

        # --------------------------------------------------
        # Learnable positive beta
        #
        # beta = softplus(raw_beta)
        # Initialization chosen so beta starts at beta_init.
        # --------------------------------------------------
        raw_beta_init = math.log(
            math.expm1(float(beta_init))
        )

        self.raw_beta = nn.Parameter(
            torch.tensor(
                raw_beta_init,
                dtype=torch.float32
            )
        )

    @property
    def beta(self):
        """
        Positive learnable positional-bias strength.
        """
        return F.softplus(self.raw_beta)

    def _compute_sparse_sizes(self, num_nodes, num_anchors):
        """
        Informer-style logarithmic sampling/query budgets.

        sample_k:
            sampled anchors used to estimate query sparsity.

        top_q:
            node queries selected for full attention.
        """

        # log(1)=0, so max(..., 2) prevents zero-sized budgets.
        sample_k = self.factor * math.ceil(
            math.log(max(num_anchors, 2))
        )

        top_q = self.factor * math.ceil(
            math.log(max(num_nodes, 2))
        )

        sample_k = min(
            num_anchors,
            max(self.min_top, sample_k)
        )

        top_q = min(
            num_nodes,
            max(self.min_top, top_q)
        )

        return int(sample_k), int(top_q)

    def _build_mean_context(
        self,
        anchor_features,
        positional_scores,
    ):
        """
        Position-aware mean fallback context for inactive queries.

        First compute the mean Value representation across anchors:

            V_mean(v) = mean_a V(v,a)

        Then preserve anchor-specific positional information:

            C(v,a) = s(v,a) * V_mean(v)

        Output shape:
            [N, K, output_dim]
        """

        # --------------------------------------------------
        # Mean anchor representation
        #
        # Because Wv has no bias:
        #
        # mean_a Wv(h_a)
        # =
        # Wv(mean_a h_a)
        # --------------------------------------------------
        mean_anchor_features = anchor_features.mean(
            dim=1
        )  # [N, input_dim]

        mean_value = self.Wv(
            mean_anchor_features
        )  # [N, output_dim]

        # --------------------------------------------------
        # Restore anchor-specific positional information
        #
        # C(v,a) = s(v,a) * V_mean(v)
        # --------------------------------------------------
        context = (
            mean_value.unsqueeze(1)
            * positional_scores.unsqueeze(-1)
        )  # [N, K, output_dim]

        return context

    def forward(
        self,
        node_features,
        anchor_features,
        positional_scores,
        context_init=None,
    ):
        """
        Parameters
        ----------
        node_features : Tensor [N, input_dim]
            Target-node representations.

        anchor_features : Tensor [N, K, input_dim]
            Anchor representation associated with each node.

            For singleton learned anchors, the K anchors are shared
            across nodes.

            For legacy P-GNN anchor sets, the selected representative
            anchor may differ per target node.

        positional_scores : Tensor [N, K]
            Distance/proximity scores after the PGNN distance transform.

        context_init : Tensor [N, K, output_dim] or None
            If None:
                use the distance-aware mean fallback context.

            If provided:
                use this external context (e.g. P-GNN-style concat).

        Returns
        -------
        messages : Tensor [N, K, output_dim]

        info : dict
            Diagnostic information.
        """

        if node_features.dim() != 2:
            raise ValueError(
                "node_features must have shape [N, input_dim]."
            )

        if anchor_features.dim() != 3:
            raise ValueError(
                "anchor_features must have shape "
                "[N, K, input_dim]."
            )

        if positional_scores.dim() != 2:
            raise ValueError(
                "positional_scores must have shape [N, K]."
            )

        N = node_features.size(0)
        K = anchor_features.size(1)

        if anchor_features.size(0) != N:
            raise ValueError(
                "node_features and anchor_features "
                "must have the same N."
            )

        if positional_scores.shape != (N, K):
            raise ValueError(
                "positional_scores must match [N, K]."
            )

        H = self.num_heads
        Dh = self.head_dim

        # ==================================================
        # 1) Queries for all nodes
        # ==================================================
        Q_all = self.Wq(
            node_features
        ).reshape(
            N,
            H,
            Dh
        )  # [N, H, Dh]

        # ==================================================
        # 2) Initial context for inactive queries
        # ==================================================
        if context_init is None:

            # Position-aware mean fallback:
            #
            # V_mean(v) = mean_a V(v,a)
            # C(v,a)    = s(v,a) * V_mean(v)
            messages = self._build_mean_context(
                anchor_features,
                positional_scores,
            )  # [N, K, output_dim]

        else:

            expected_shape = (
                N,
                K,
                self.output_dim
            )

            if tuple(context_init.shape) != expected_shape:
                raise ValueError(
                    "context_init must have shape "
                    f"{expected_shape}, got "
                    f"{tuple(context_init.shape)}."
                )

            messages = context_init.clone()

        # Split fallback representation into heads.
        context_heads = messages.reshape(
            N,
            K,
            H,
            Dh
        )  # [N, K, H, Dh]

        # ==================================================
        # 3) Informer-style sparse budgets
        # ==================================================
        sample_k, top_q = self._compute_sparse_sizes(
            num_nodes=N,
            num_anchors=K
        )

        device = node_features.device

        # ==================================================
        # 4) Sample anchors for each node query
        # ==================================================
        idx_sample = torch.randint(
            low=0,
            high=K,
            size=(N, sample_k),
            device=device,
        )

        batch_idx = (
            torch.arange(
                N,
                device=device
            )
            .unsqueeze(1)
            .expand(-1, sample_k)
        )

        sampled_anchor_features = anchor_features[
            batch_idx,
            idx_sample,
            :
        ]  # [N, sample_k, input_dim]

        # Only sampled K are projected here.
        K_sample = self.Wk(
            sampled_anchor_features
        ).reshape(
            N,
            sample_k,
            H,
            Dh
        )

        sampled_positional_scores = positional_scores[
            batch_idx,
            idx_sample
        ].unsqueeze(-1)  # [N, sample_k, 1]

        # ==================================================
        # 5) Approximate sparsity measurement
        #
        # score = QK/sqrt(Dh) + beta * positional_score
        # ==================================================
        sampled_scores = (
            (
                Q_all.unsqueeze(1)
                * K_sample
            ).sum(dim=-1)
            / math.sqrt(Dh)
        )  # [N, sample_k, H]

        sampled_scores = (
            sampled_scores
            + self.beta * sampled_positional_scores
        )

        # --------------------------------------------------
        # Informer-style max-minus-mean approximation.
        #
        # IMPORTANT:
        # denominator = total number of anchors K,
        # not sample_k.
        # --------------------------------------------------
        sparsity_measure = (
            sampled_scores.max(dim=1).values
            -
            sampled_scores.sum(dim=1) / float(K)
        )  # [N, H]

        # ==================================================
        # 6) Top queries PER HEAD -- unchanged mathematically
        # ==================================================
        top_idx_per_head = torch.topk(
            sparsity_measure,
            k=top_q,
            dim=0,
            largest=True,
            sorted=False,
        ).indices  # [top_q, H]

        # ==================================================
        # 7) Stage-1 vectorized full attention for Top-q
        #
        # Previous implementation:
        #   - torch.unique over active nodes
        #   - global-node -> active-row mapping
        #   - Python loop over heads
        #   - one index_copy per head
        #
        # New implementation:
        #   - preserve independent Top-q selection per head
        #   - gather selected queries for all heads at once
        #   - project only the weight slice belonging to each head
        #   - compute all head-specific attention in one batched path
        #   - scatter all updates back to the fallback context at once
        #
        # This changes implementation only, not the equations.
        # ==================================================

        # [H, top_q]
        top_idx_h = (
            top_idx_per_head
            .transpose(0, 1)
            .contiguous()
        )

        # --------------------------------------------------
        # Selected Q for every head
        #
        # Q_all:       [N, H, Dh]
        # Q_by_head:   [H, N, Dh]
        # Q_top:       [H, top_q, Dh]
        # --------------------------------------------------
        Q_by_head = Q_all.permute(
            1,
            0,
            2
        )

        Q_top = torch.gather(
            Q_by_head,
            dim=1,
            index=top_idx_h.unsqueeze(-1).expand(
                -1,
                -1,
                Dh
            ),
        )

        # --------------------------------------------------
        # Selected node-anchor features for every head
        #
        # [H, top_q, K, input_dim]
        #
        # A node may be selected by more than one head.
        # Keeping the head dimension explicit removes the need
        # for active_union and node_to_active.
        # --------------------------------------------------
        anchor_top = anchor_features[
            top_idx_h
        ]

        # --------------------------------------------------
        # Head-specific K / V projection
        #
        # Wk/Wv are ordinary output_dim x input_dim linear maps.
        # Reshaping their weights into [H, Dh, input_dim] and
        # applying the matching slice for each head is exactly
        # equivalent to:
        #
        #   self.Wk(anchor_top)[..., head, :]
        #   self.Wv(anchor_top)[..., head, :]
        #
        # but avoids computing unused output heads.
        # --------------------------------------------------
        Wk_by_head = self.Wk.weight.reshape(
            H,
            Dh,
            self.input_dim
        )

        Wv_by_head = self.Wv.weight.reshape(
            H,
            Dh,
            self.input_dim
        )

        # [H, top_q, K, Dh]
        K_top = torch.matmul(
            anchor_top,
            Wk_by_head.transpose(
                -1,
                -2
            ).unsqueeze(1)
        )

        # [H, top_q, K, Dh]
        V_top = torch.matmul(
            anchor_top,
            Wv_by_head.transpose(
                -1,
                -2
            ).unsqueeze(1)
        )

        # --------------------------------------------------
        # Distance / positional scores for selected nodes
        # [H, top_q, K]
        # --------------------------------------------------
        positional_top = positional_scores[
            top_idx_h
        ]

        # --------------------------------------------------
        # Full content score for selected queries
        #
        # score = QK/sqrt(Dh) + beta * positional_score
        # --------------------------------------------------
        scores_top = (
            (
                Q_top.unsqueeze(2)
                * K_top
            ).sum(dim=-1)
            / math.sqrt(Dh)
        )  # [H, top_q, K]

        scores_top = (
            scores_top
            + self.beta * positional_top
        )

        # --------------------------------------------------
        # Independent anchor relevance -- unchanged
        # --------------------------------------------------
        attention_top = torch.sigmoid(
            scores_top
        )  # [H, top_q, K]

        messages_top = (
            V_top
            * attention_top.unsqueeze(-1)
        )  # [H, top_q, K, Dh]

        # Keep current PGNN behavior -- unchanged.
        messages_top = F.relu(
            messages_top
        )

        # --------------------------------------------------
        # Vectorized replacement of active-query contexts
        #
        # context_by_head: [H, N, K, Dh]
        # scatter_index:   [H, top_q, K, Dh]
        #
        # torch.scatter is out-of-place here, so gradients flow
        # both through the fallback context and messages_top.
        # --------------------------------------------------
        context_by_head = context_heads.permute(
            2,
            0,
            1,
            3
        )  # [H, N, K, Dh]

        scatter_index = (
            top_idx_h
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(
                -1,
                -1,
                K,
                Dh
            )
        )

        updated_by_head = context_by_head.scatter(
            dim=1,
            index=scatter_index,
            src=messages_top,
        )  # [H, N, K, Dh]

        # Back to the original layout:
        # [H, N, K, Dh] -> [N, K, H, Dh] -> [N, K, output_dim]
        messages = (
            updated_by_head
            .permute(1, 2, 0, 3)
            .contiguous()
            .reshape(
                N,
                K,
                self.output_dim
            )
        )

        # --------------------------------------------------
        # Diagnostics
        #
        # active_union is intentionally not constructed in the
        # optimized forward path because torch.unique was one of
        # the avoidable GPU overheads in the previous version.
        # The key is retained as None for compatibility.
        # --------------------------------------------------
        info = {
            "sample_k": sample_k,
            "top_q": top_q,
            "beta": self.beta.detach(),
            "active_union": None,
            "top_idx_per_head": (
                top_idx_per_head.detach()
            ),
        }

        return messages, info
