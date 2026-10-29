# -*- coding: utf-8 -*-
"""MagWorld v39: neural mixtures of latent fields with recursive recombination.

Four radial field experts act along the regularized source--receiver direction:
signed pole, anisotropic dipole, local exponential and low-rank free fields.
A cell/gene-conditioned neural gate mixes their scalar radial coefficients.
When recursive_gates=True, the accumulated response h updates the gate state,
susceptibility, response direction and amplitude before every mixture step.

All fields are latent computational analogies, not physical magnetic claims.
The output gate field_mix is zero-initialized, so a new v39 model has exactly
the same response function as v37. Non-recursive and single-expert subclasses
provide matched ablations without modifying the shared training pipeline.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_world_h1_v38 import WorldModelH1V38


class WorldModelH1V39(WorldModelH1V38):
    EXPERT_NAMES = ("pole", "dipole", "local", "free")

    def __init__(self, *args, field_gate_hidden: int = 32,
                 recursive_gates: bool = True, active_expert: int | None = None,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # The legacy backbone caps n_layers at three. v39 explicitly supports
        # deeper recursion while preserving the requested one/two-layer runs.
        self.n_layers = max(1, int(kwargs.get("n_layers", self.n_layers)))
        self.recursive_gates = bool(recursive_gates)
        self.active_expert = active_expert
        if active_expert is not None and not 0 <= active_expert < len(self.EXPERT_NAMES):
            raise ValueError(f"active_expert must be in [0, {len(self.EXPERT_NAMES)})")

        self.dipole_axis = nn.Parameter(torch.randn(self.n_genes, self.d_z) * 0.02)
        self.local_raw_width = nn.Parameter(torch.tensor(0.0))
        self.field_source = nn.Linear(self.d_model, self.d_dir, bias=False)
        self.field_target = nn.Linear(self.d_model, self.d_dir, bias=False)
        self.recursive_state = nn.Linear(1, self.d_dir, bias=False)
        self.recursive_direction = nn.Linear(1, self.d_z, bias=False)

        n_experts = len(self.EXPERT_NAMES)
        self.field_gate = nn.Sequential(
            nn.Linear(self.d_dir + n_experts + 4, field_gate_hidden),
            nn.SiLU(),
            nn.Linear(field_gate_hidden, n_experts),
        )
        # Uniform initial expert weights; the closed output gate still makes
        # the initial function exactly v37 and permits a nonzero mixed-field
        # derivative with respect to field_mix.
        nn.init.zeros_(self.field_gate[-1].weight)
        nn.init.zeros_(self.field_gate[-1].bias)
        self.field_mix = nn.Parameter(torch.zeros(()))

    def _field_experts(self, charge, indices, state, d_src, u_src):
        """Return B x G x 4 radial coefficients and gate-side features."""
        e = self.gene_emb.weight
        signs = torch.tanh(self.pole_sign)
        signed_charge = charge * signs
        source_signed = signed_charge[indices]
        source_positive = charge[indices]
        chi = torch.sigmoid(self.pole_susceptibility(state).squeeze(-1))

        off_source = (d_src > 1e-3).to(d_src.dtype)
        inv_square = off_source / (d_src.square() + self.pole_epsilon)
        inv_square = inv_square / inv_square.mean(dim=1, keepdim=True).clamp_min(1e-6)

        pole = (source_signed[:, None] * signed_charge[None, :]
                * chi * inv_square)

        dipole_axis = F.normalize(self.dipole_axis, dim=-1)
        cos_theta = (u_src * dipole_axis[None, :, :]).sum(-1)
        anisotropy = 0.5 * (3.0 * cos_theta.square() - 1.0)
        dipole = (source_positive[:, None] * charge[None, :]
                  * chi * anisotropy * inv_square)

        local_width = F.softplus(self.local_raw_width) + 0.05
        local_kernel = off_source * torch.exp(-d_src.square() / local_width)
        local_kernel = local_kernel / local_kernel.mean(dim=1, keepdim=True).clamp_min(1e-6)
        local = source_positive[:, None] * charge[None, :] * chi * local_kernel

        source_embedding = self.field_source(e[indices])
        target_embedding = self.field_target(e)
        compatibility = ((source_embedding[:, None, :] * target_embedding[None, :, :]).sum(-1)
                         / math.sqrt(self.d_dir))
        free = torch.tanh(compatibility) * chi * inv_square

        experts = torch.stack((pole, dipole, local, free), dim=-1)
        return experts, compatibility, signed_charge, source_signed

    def _field_weights(self, state, experts, compatibility, signed_charge,
                       source_signed, d_src, indices):
        batch_size, n_genes, n_experts = experts.shape
        if self.active_expert is not None:
            return F.one_hot(
                torch.tensor(self.active_expert, device=experts.device),
                num_classes=n_experts,
            ).to(experts.dtype).view(1, 1, n_experts).expand(batch_size, n_genes, -1)

        gate_features = torch.cat((
            state,
            experts,
            d_src.unsqueeze(-1),
            source_signed[:, None, None].expand(-1, n_genes, -1),
            signed_charge[None, :, None].expand(batch_size, -1, -1),
            compatibility.unsqueeze(-1),
        ), dim=-1)
        return torch.softmax(self.field_gate(gate_features), dim=-1)

    def pole_interaction(self, pos, charge, indices, moment, src_diff, d_src,
                         u_src, response_direction, h=None):
        """Mix expert fields and project them onto the current response axis."""
        if h is None or not self.recursive_gates:
            state = moment
            current_direction = response_direction
        else:
            state = moment + self.recursive_state(h.unsqueeze(-1))
            current_direction = F.normalize(
                response_direction + self.recursive_direction(h.unsqueeze(-1)), dim=-1
            )

        experts, compatibility, signed_charge, source_signed = self._field_experts(
            charge, indices, state, d_src, u_src
        )
        weights = self._field_weights(
            state, experts, compatibility, signed_charge, source_signed, d_src, indices
        )
        mixed_radial = (weights * experts).sum(-1)
        projected_force = mixed_radial * (u_src * current_direction).sum(-1)
        amplitude = torch.tanh(self.pole_response(state).squeeze(-1))
        return self.field_mix * projected_force * amplitude

    @torch.no_grad()
    def magnetic_diagnostics(self, pert_idx) -> dict:
        diag = super().magnetic_diagnostics(pert_idx)
        diag.update({
            "field_mix": float(self.field_mix),
            "field_mode": "recursive" if self.recursive_gates else "nonrecursive",
            "active_expert": (self.EXPERT_NAMES[self.active_expert]
                              if self.active_expert is not None else "mixture"),
            "field_gate_weight_l1": float(self.field_gate[-1].weight.abs().sum()),
            "recursive_state_weight_l1": float(self.recursive_state.weight.abs().sum()),
            "recursive_direction_weight_l1": float(self.recursive_direction.weight.abs().sum()),
        })
        return diag


def _activate_field_for_warm_ablation(model):
    """Predefined activation control; unlike the default, this is not v37-equivalent."""
    with torch.no_grad():
        model.field_mix.fill_(0.05)
        nn.init.normal_(model.field_gate[-1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(model.field_gate[-1].bias)
    return model


class WorldModelH1V39Warm(WorldModelH1V39):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        _activate_field_for_warm_ablation(self)


class WorldModelH1V39WarmNonRecursive(WorldModelH1V39):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, recursive_gates=False, **kwargs)
        _activate_field_for_warm_ablation(self)


class WorldModelH1V39Base(WorldModelH1V39):
    """v37-equivalent backbone with a permanently closed v39 field path."""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, recursive_gates=False, **kwargs)
        self.field_mix.requires_grad_(False)


class WorldModelH1V39NonRecursive(WorldModelH1V39):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, recursive_gates=False, **kwargs)


class WorldModelH1V39Pole(WorldModelH1V39):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, active_expert=0, **kwargs)


class WorldModelH1V39Dipole(WorldModelH1V39):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, active_expert=1, **kwargs)


class WorldModelH1V39Local(WorldModelH1V39):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, active_expert=2, **kwargs)


class WorldModelH1V39Free(WorldModelH1V39):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, active_expert=3, **kwargs)


WorldModel = WorldModelH1V39
