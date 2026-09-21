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
                f"Got output_dim={self.output_dim}, "
                f"num_heads={self.num_heads}."
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
        )

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
        # 6) Top queries PER HEAD
        # ==================================================
        top_idx_per_head = torch.topk(
            sparsity_measure,
            k=top_q,
            dim=0,
            largest=True,
            sorted=False,
        ).indices  # [top_q, H]

        # --------------------------------------------------
        # Union of active nodes across heads.
        #
        # K and V for all anchors are computed only for
        # nodes that are active in at least one head.
        # --------------------------------------------------
        active_union = torch.unique(
            top_idx_per_head.reshape(-1)
        )

        active_anchor_features = anchor_features[
            active_union
        ]  # [U, K, input_dim]

        K_active = self.Wk(
            active_anchor_features
        ).reshape(
            active_union.numel(),
            K,
            H,
            Dh
        )

        V_active = self.Wv(
            active_anchor_features
        ).reshape(
            active_union.numel(),
            K,
            H,
            Dh
        )

        # Map global node id -> active_union row id.
        node_to_active = torch.full(
            (N,),
            -1,
            dtype=torch.long,
            device=device,
        )

        node_to_active[active_union] = torch.arange(
            active_union.numel(),
            device=device,
        )

        # ==================================================
        # 7) Full node-anchor cross-attention only for
        #    Top-q queries of each head
        # ==================================================
        updated_heads = []

        for head in range(H):

            active_nodes_h = top_idx_per_head[
                :,
                head
            ]  # [top_q]

            active_rows_h = node_to_active[
                active_nodes_h
            ]

            Q_h = Q_all[
                active_nodes_h,
                head,
                :
            ]  # [top_q, Dh]

            K_h = K_active[
                active_rows_h,
                :,
                head,
                :
            ]  # [top_q, K, Dh]

            V_h = V_active[
                active_rows_h,
                :,
                head,
                :
            ]  # [top_q, K, Dh]

            # ----------------------------------------------
            # Full content score
            # ----------------------------------------------
            scores_h = (
                (
                    Q_h.unsqueeze(1)
                    * K_h
                ).sum(dim=-1)
                / math.sqrt(Dh)
            )  # [top_q, K]

            # ----------------------------------------------
            # Distance-aware positional bias
            # ----------------------------------------------
            scores_h = (
                scores_h
                + self.beta
                * positional_scores[
                    active_nodes_h
                ]
            )

            # ----------------------------------------------
            # Independent anchor relevance.
            #
            # No Softmax competition across anchors.
            # ----------------------------------------------
            attention_h = torch.sigmoid(
                scores_h
            )  # [top_q, K]

            messages_h = (
                V_h
                * attention_h.unsqueeze(-1)
            )  # [top_q, K, Dh]

            # Keep current PGNN behavior:
            # nonlinear message activation.
            messages_h = F.relu(
                messages_h
            )

            # Current fallback values for this head.
            base_head = context_heads[
                :,
                :,
                head,
                :
            ]  # [N, K, Dh]

            # Out-of-place replacement keeps autograd safe.
            updated_head = base_head.index_copy(
                0,
                active_nodes_h,
                messages_h,
            )

            updated_heads.append(
                updated_head
            )

        # [N, K, H, Dh]
        updated_heads = torch.stack(
            updated_heads,
            dim=2
        )

        messages = updated_heads.reshape(
            N,
            K,
            self.output_dim
        )

        info = {
            "sample_k": sample_k,
            "top_q": top_q,
            "beta": self.beta.detach(),
            "active_union": active_union.detach(),
            "top_idx_per_head": (
                top_idx_per_head.detach()
            ),
        }

        return messages, info
