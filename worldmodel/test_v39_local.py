# -*- coding: utf-8 -*-
"""Local numerical and ablation tests for v39 recursive field mixtures."""
import numpy as np
import torch
import torch.nn.functional as F

import model_world_h1_v37 as m37
import model_world_h1_v39 as m39

G = 48
B = 8
KW = dict(n_genes=G, d_model=64, d_z=32, d_hidden=48, d_dir=16,
          graph_k=8, n_layers=3, response_rank=16)
rng = np.random.RandomState(0)
ei = rng.randint(0, G, size=(2, 120))
es = rng.choice([-1.0, 1.0], size=120).astype(np.float32)
ew = rng.rand(120).astype(np.float32)
np.savez("tiny_prior.npz", edge_index=ei, edge_sign=es, edge_w=ew)

torch.manual_seed(7)
x_ctrl = torch.rand(B, G) * 3.0
pert = torch.tensor([3, 7, 11, 15, 19, 23, 31, 40])


def build(cls, **overrides):
    torch.manual_seed(42)
    config = dict(KW)
    config.update(overrides)
    return cls(prior_graph_paths=["tiny_prior.npz"], **config)


v37 = build(m37.WorldModelH1V37)
v39 = build(m39.WorldModelH1V39)
with torch.no_grad():
    out37 = v37.predict_delta(x_ctrl, pert)
    closed = v39.predict_delta(x_ctrl, pert)
assert out37.shape == closed.shape == (B, G)
assert torch.isfinite(closed).all()
assert (out37 - closed).abs().max().item() < 1e-5
print("OK: zero-initialized v39 is exactly v37")

deep = build(m39.WorldModelH1V39, n_layers=8)
closed_base = build(m39.WorldModelH1V39Base)
assert deep.n_layers == 8
assert not closed_base.field_mix.requires_grad
with torch.no_grad():
    base_output = closed_base.predict_delta(x_ctrl, pert)
assert (base_output - out37).abs().max().item() < 1e-5
print("OK: eight-step recursion is preserved and the closed base matches v37")

warm = build(m39.WorldModelH1V39Warm)
warm_static = build(m39.WorldModelH1V39WarmNonRecursive)
assert abs(warm.field_mix.item() - 0.05) < 1e-8
assert not warm_static.recursive_gates
with torch.no_grad():
    warm_output = warm.predict_delta(x_ctrl, pert)
assert torch.isfinite(warm_output).all()
assert (warm_output - out37).abs().max().item() > 1e-6
print("OK: predefined warm-gate control activates a finite non-v37 field path")

closed = v39.predict_delta(x_ctrl, pert)
closed.square().mean().backward()
grad = v39.field_mix.grad
assert grad is not None and torch.isfinite(grad) and grad.abs().item() > 0
v39.zero_grad()
print("OK: mixed-field output gradient flows while closed")

with torch.no_grad():
    v39.field_mix.fill_(1.0)
    opened = v39.predict_delta(x_ctrl, pert)
assert torch.isfinite(opened).all()
assert (opened - closed).abs().max().item() > 1e-6
opened_for_grad = v39.predict_delta(x_ctrl, pert)
opened_for_grad.square().mean().backward()
for name in ("recursive_state", "recursive_direction"):
    param = getattr(v39, name)
    assert param.weight.grad is not None
    assert torch.isfinite(param.weight.grad).all()
    assert param.weight.grad.abs().sum().item() > 0
assert v39.field_gate[-1].weight.grad is not None
assert v39.field_gate[-1].weight.grad.abs().sum().item() > 0
v39.zero_grad()
print("OK: recursive state, direction and neural gate receive gradients")

# A single recursion has h=0 at the field call, so recursive and static modes
# must be identical. Multiple recursions must be allowed to diverge.
rec1 = build(m39.WorldModelH1V39, n_layers=1)
static1 = build(m39.WorldModelH1V39NonRecursive, n_layers=1)
rec3 = build(m39.WorldModelH1V39, n_layers=3)
static3 = build(m39.WorldModelH1V39NonRecursive, n_layers=3)
with torch.no_grad():
    for model in (rec1, static1, rec3, static3):
        model.field_mix.fill_(1.0)
    one_step_difference = (rec1.predict_delta(x_ctrl, pert)
                           - static1.predict_delta(x_ctrl, pert)).abs().max().item()
    multi_step_difference = (rec3.predict_delta(x_ctrl, pert)
                             - static3.predict_delta(x_ctrl, pert)).abs().max().item()
assert one_step_difference < 1e-6
assert multi_step_difference > 1e-6
print("OK: recursion is identical at step one and recombined at later steps")

# Validate normalized neural gates and exact one-hot single-expert ablations.
state = torch.randn(B, G, KW["d_dir"])
d_src = torch.rand(B, G) + 0.01
u_src = F.normalize(torch.randn(B, G, KW["d_z"]), dim=-1)
charge = F.softplus(torch.randn(G))
for expert_index, cls in enumerate((m39.WorldModelH1V39Pole, m39.WorldModelH1V39Dipole,
                                    m39.WorldModelH1V39Local, m39.WorldModelH1V39Free)):
    model = build(cls)
    experts, compatibility, signed_charge, source_signed = model._field_experts(
        charge, pert, state, d_src, u_src
    )
    assert experts.shape == (B, G, 4)
    assert torch.isfinite(experts).all()
    weights = model._field_weights(
        state, experts, compatibility, signed_charge, source_signed, d_src, pert
    )
    expected = torch.zeros(B, G, 4)
    expected[..., expert_index] = 1.0
    assert torch.equal(weights, expected)
    with torch.no_grad():
        model.field_mix.fill_(1.0)
        output = model.predict_delta(x_ctrl, pert)
    assert torch.isfinite(output).all()
print("OK: all four single-expert ablations are finite and exactly one-hot")

weights = v39._field_weights(
    state, *v39._field_experts(charge, pert, state, d_src, u_src), d_src, pert
)
assert weights.shape == (B, G, 4)
assert torch.isfinite(weights).all()
assert (weights.sum(-1) - 1.0).abs().max().item() < 1e-6
assert (weights - 0.25).abs().max().item() < 1e-6
print("OK: zero-initialized neural gate gives a normalized uniform mixture")
print("ALL V39 LOCAL TESTS PASSED")
