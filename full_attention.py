import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistanceAwareFullCrossAttention(nn.Module):
    """
    Dense counterpart of the proposed distance-aware
    multi-head ProbSparse node-anchor cross-attention.

    Difference from ProbSparse:
        - No anchor sampling for query selection.
        - No Top-q query selection.
        - Every node query attends to all K anchors.

    The attention formulation is otherwise the same:
        score = QK / sqrt(d_h) + beta * positional_score
        attention = sigmoid(score)
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        num_heads=4,
        beta_init=1.0,
    ):
        super().__init__()

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_heads = int(num_heads)

        if self.num_heads < 1:
            raise ValueError(
                "num_heads must be >= 1."
            )

        if self.output_dim % self.num_heads != 0:
            raise ValueError(
                "output_dim must be divisible by num_heads. "
                f"Got output_dim={self.output_dim}, "
                f"num_heads={self.num_heads}."
            )

        if beta_init <= 0:
            raise ValueError(
                "beta_init must be > 0."
            )

        self.head_dim = (
            self.output_dim // self.num_heads
        )

        # --------------------------------------------------
        # Q / K / V projections
        # Exactly the same parameterization as ProbSparse.
        # --------------------------------------------------
        self.Wq = nn.Linear(
            self.input_dim,
            self.output_dim,
            bias=False,
        )

        self.Wk = nn.Linear(
            self.input_dim,
            self.output_dim,
            bias=False,
        )

        self.Wv = nn.Linear(
            self.input_dim,
            self.output_dim,
            bias=False,
        )

        # --------------------------------------------------
        # Positive learnable positional coefficient
        #
        # beta = softplus(raw_beta)
        # --------------------------------------------------
        raw_beta_init = math.log(
            math.expm1(float(beta_init))
        )

        self.raw_beta = nn.Parameter(
            torch.tensor(
                raw_beta_init,
                dtype=torch.float32,
            )
        )

    @property
    def beta(self):
        return F.softplus(
            self.raw_beta
        )

    def forward(
        self,
        node_features,
        anchor_features,
        positional_scores,
    ):
        """
        Parameters
        ----------
        node_features : Tensor [N, input_dim]
            Node/query representations.

        anchor_features : Tensor [N, K, input_dim]
            Anchor representations associated with each node.

        positional_scores : Tensor [N, K]
            Learned/transformed positional scores.

        Returns
        -------
        messages : Tensor [N, K, output_dim]
        """

        if node_features.dim() != 2:
            raise ValueError(
                "node_features must have shape "
                "[N, input_dim]."
            )

        if anchor_features.dim() != 3:
            raise ValueError(
                "anchor_features must have shape "
                "[N, K, input_dim]."
            )

        if positional_scores.dim() != 2:
            raise ValueError(
                "positional_scores must have shape "
                "[N, K]."
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
        # 1) Queries for ALL nodes
        # ==================================================
        Q = self.Wq(
            node_features
        ).reshape(
            N,
            H,
            Dh,
        )  # [N, H, Dh]

        # ==================================================
        # 2) Keys and values for ALL node-anchor pairs
        # ==================================================
        K_all = self.Wk(
            anchor_features
        ).reshape(
            N,
            K,
            H,
            Dh,
        )  # [N, K, H, Dh]

        V_all = self.Wv(
            anchor_features
        ).reshape(
            N,
            K,
            H,
            Dh,
        )  # [N, K, H, Dh]

        # ==================================================
        # 3) Full content score
        #
        # QK / sqrt(Dh)
        # ==================================================
        scores = (
            (
                Q.unsqueeze(1)
                * K_all
            ).sum(dim=-1)
            / math.sqrt(Dh)
        )  # [N, K, H]

        # ==================================================
        # 4) Position-aware additive bias
        #
        # score = QK/sqrt(Dh) + beta*s(v,a)
        # ==================================================
        scores = (
            scores
            + self.beta
            * positional_scores.unsqueeze(-1)
        )

        # ==================================================
        # 5) Independent anchor relevance
        # ==================================================
        attention = torch.sigmoid(
            scores
        )  # [N, K, H]

        # ==================================================
        # 6) Messages
        # ==================================================
        messages_h = (
            V_all
            * attention.unsqueeze(-1)
        )  # [N, K, H, Dh]

        # Same nonlinear message activation as ProbSparse.
        messages_h = F.relu(
            messages_h
        )

        messages = messages_h.reshape(
            N,
            K,
            self.output_dim,
        )

        return messages
