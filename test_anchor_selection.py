import math

import torch

from anchor_selection import (
    SlotAnchorSelector,
    compute_anchor_budget,
)


def make_toy_graph():
    # 8-node undirected ring with both directions in edge_index.
    n = 8
    src = []
    dst = []

    for i in range(n):
        j = (i + 1) % n
        src.extend([i, j])
        dst.extend([j, i])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    x = torch.randn(n, 5)

    return x, edge_index


def test_budget():
    # N=400 -> floor(log2(400)) = 8 -> K_main = 64
    assert compute_anchor_budget(400, mode="main") == 64
    assert compute_anchor_budget(400, mode="rule") == 8
    assert compute_anchor_budget(
        400, mode="progressive", reduction=2
    ) == 32
    assert compute_anchor_budget(
        20, mode="fixed", fixed_k=32, exact_fixed=False
    ) == 20

    print("PASS: anchor budget")


def test_assignment_and_gradients():
    torch.manual_seed(123)

    x, edge_index = make_toy_graph()
    n = x.size(0)
    k = 4

    selector = SlotAnchorSelector(
        input_dim=x.size(1),
        selector_dim=6,
        k_max=k,
        temperature=1.0,
        sinkhorn_iters=50,
    )

    out = selector(x, edge_index, k=k)

    S = out["scores"]
    A_hard = out["A_hard"]
    A_soft = out["A_soft"]
    A_st = out["A_st"]
    anchor_idx = out["anchor_idx"]

    # ---- Shapes ----
    assert S.shape == (k, n)
    assert A_hard.shape == (k, n)
    assert A_soft.shape == (k, n)
    assert A_st.shape == (k, n)

    print("PASS: shapes")

    # ---- Hard one-to-one assignment ----
    assert torch.all(A_hard.sum(dim=1) == 1)
    assert torch.all(A_hard.sum(dim=0) <= 1)
    assert anchor_idx.numel() == k
    assert torch.unique(anchor_idx).numel() == k

    print("PASS: hard assignment constraints")

    # ---- Soft marginals ----
    row_sums = A_soft.sum(dim=1)
    col_sums = A_soft.sum(dim=0)

    assert torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=5e-3,
        rtol=5e-3,
    )
    assert torch.all(col_sums <= 1.0 + 5e-3)

    print("PASS: soft marginals")

    # ---- Straight-through forward equality ----
    assert torch.allclose(
        A_st.detach(),
        A_hard,
        atol=1e-7,
        rtol=0.0,
    )

    print("PASS: straight-through forward equality")

    # ---- Proximity equivalence ----
    R = torch.randn(n, n)

    z_from_assignment = R @ A_hard.transpose(0, 1)
    z_from_gather = R[:, anchor_idx]

    assert torch.allclose(
        z_from_assignment,
        z_from_gather,
        atol=1e-6,
        rtol=1e-6,
    )

    print("PASS: proximity equivalence")

    # ---- Feature equivalence ----
    F = torch.randn(n, 7)

    f_from_assignment = A_hard @ F
    f_from_gather = F[anchor_idx]

    assert torch.allclose(
        f_from_assignment,
        f_from_gather,
        atol=1e-6,
        rtol=1e-6,
    )

    print("PASS: feature equivalence")

    # ---- Gradient flow through soft path ----
    # Use a non-symmetric probe so the loss is not constant under
    # row/column marginal constraints.
    probe = torch.randn_like(A_st)
    loss = (A_st * probe).sum()
    loss.backward()

    q_grad = selector.slot_queries.grad
    assert q_grad is not None
    assert torch.isfinite(q_grad).all()
    assert q_grad.abs().sum().item() > 0

    gcn_grad_total = 0.0
    for name, param in selector.named_parameters():
        if name.startswith("conv") and param.grad is not None:
            assert torch.isfinite(param.grad).all()
            gcn_grad_total += param.grad.abs().sum().item()

    assert gcn_grad_total > 0

    print("PASS: non-zero finite gradient to Q")
    print("PASS: non-zero finite gradient to selector GCN")


def test_k_equals_n():
    torch.manual_seed(123)

    x, edge_index = make_toy_graph()
    n = x.size(0)

    selector = SlotAnchorSelector(
        input_dim=x.size(1),
        selector_dim=6,
        k_max=n,
        temperature=1.0,
        sinkhorn_iters=50,
    )

    out = selector(x, edge_index, k=n)

    assert out["A_soft"].shape == (n, n)
    assert torch.allclose(
        out["A_soft"].sum(dim=1),
        torch.ones(n),
        atol=5e-3,
        rtol=5e-3,
    )
    assert torch.allclose(
        out["A_soft"].sum(dim=0),
        torch.ones(n),
        atol=5e-3,
        rtol=5e-3,
    )

    print("PASS: K == N Sinkhorn edge case")


def test_multi_graph_slot_slicing():
    torch.manual_seed(123)

    # Simulate two graphs with different K values but one shared Q_max.
    k1 = compute_anchor_budget(20, mode="main")
    k2 = compute_anchor_budget(40, mode="main")
    k_max = max(k1, k2)

    selector = SlotAnchorSelector(
        input_dim=5,
        selector_dim=6,
        k_max=k_max,
        temperature=1.0,
        sinkhorn_iters=30,
    )

    q1 = selector.slot_queries[:k1]
    q2 = selector.slot_queries[:k2]

    assert q1.shape[0] == k1
    assert q2.shape[0] == k2
    assert q1.shape[1] == q2.shape[1] == 6

    print("PASS: multi-graph Q_max slicing")


if __name__ == "__main__":
    test_budget()
    test_assignment_and_gradients()
    test_k_equals_n()
    test_multi_graph_slot_slicing()

    print("\nALL ANCHOR-SELECTION UNIT TESTS PASSED")
