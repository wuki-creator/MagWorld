# -*- coding: utf-8 -*-
"""Local numerical tests for the v38 explicit pole--field--cell term."""
import numpy as np
import torch

import model_world_h1_v37 as m37
import model_world_h1_v38 as m38

G = 48
B = 8
KW = dict(n_genes=G, d_model=64, d_z=32, d_hidden=48, d_dir=16,
          graph_k=8, n_layers=1, response_rank=16)
rng = np.random.RandomState(0)
ei = rng.randint(0, G, size=(2, 120))
es = rng.choice([-1.0, 1.0], size=120).astype(np.float32)
ew = rng.rand(120).astype(np.float32)
np.savez("tiny_prior.npz", edge_index=ei, edge_sign=es, edge_w=ew)

torch.manual_seed(7)
x_ctrl = torch.rand(B, G) * 3.0
pert = torch.tensor([3, 7, 11, 15, 19, 23, 31, 40])


def build(cls):
    torch.manual_seed(42)
    return cls(prior_graph_paths=["tiny_prior.npz"], **KW)


v37 = build(m37.WorldModelH1V37)
v38 = build(m38.WorldModelH1V38)
with torch.no_grad():
    out37 = v37.predict_delta(x_ctrl, pert)
    closed = v38.predict_delta(x_ctrl, pert)
assert out37.shape == closed.shape == (B, G)
assert torch.isfinite(closed).all()
assert (out37 - closed).abs().max().item() < 1e-5
print("OK: zero-initialized v38 is exactly v37")

# The new gate receives a nonzero gradient even while it is exactly closed.
closed = v38.predict_delta(x_ctrl, pert)
(closed.square().mean()).backward()
grad = v38.pole_mix.grad
assert grad is not None and torch.isfinite(grad) and grad.abs().item() > 0
v38.zero_grad()
print("OK: pole mixing gradient flows")

with torch.no_grad():
    v38.pole_mix.fill_(1.0)
    opened = v38.predict_delta(x_ctrl, pert)
    base_deviation = opened - closed
    assert base_deviation.abs().max().item() > 1e-6

    # On receivers whose own poles are unchanged, flipping the source pole
    # must reverse the force. Perturbed genes are excluded here because the
    # same gene parameter also represents their receiver pole in other rows.
    v38.pole_sign[pert] = -v38.pole_sign[pert]
    source_flipped = v38.predict_delta(x_ctrl, pert)
    source_deviation = source_flipped - closed
    unchanged_receivers = torch.ones(G, dtype=torch.bool)
    unchanged_receivers[pert] = False
    residual = (base_deviation[:, unchanged_receivers]
                + source_deviation[:, unchanged_receivers]).abs().max().item()
    scale = base_deviation[:, unchanged_receivers].abs().max().item() + 1e-12
    assert residual / scale < 1e-4

    # Flipping every pole leaves q_source * q_receiver unchanged.
    v38.pole_sign[pert] = -v38.pole_sign[pert]  # restore sources
    globally_flipped_input = v38.predict_delta(x_ctrl, pert)
    assert (globally_flipped_input - opened).abs().max().item() < 1e-5
    v38.pole_sign.neg_()
    globally_flipped = v38.predict_delta(x_ctrl, pert)
    assert (globally_flipped - opened).abs().max().item() < 1e-4

print("OK: source-pole reversal negates force; global reversal preserves it")
print("ALL V38 LOCAL TESTS PASSED")
