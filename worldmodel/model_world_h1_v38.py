# -*- coding: utf-8 -*-
"""MagWorld v38: explicit pole--field--cell interaction in latent space.

The magnetic terms are computational analogies, not claims that cells contain
physical magnetic monopoles. Gene i has latent position p_i, effective signed
pole q_i and cell-state-dependent susceptibility chi_i(x); the positive source
scale is absorbed into q_t. For perturbation t:

    N_t = mean_{i != t} (||p_i - p_t||^2 + epsilon)^-1
    B_ti = q_t (p_i - p_t) /
           [||p_i - p_t|| * (||p_i - p_t||^2 + epsilon) * N_t]
    F_ti(x) = q_i chi_i(x) B_ti
    dx_pole = gamma <F_ti(x), d_i(x)> rho_i(x)

Here d_i(x) is the learned response direction and rho_i(x) is a bounded
response amplitude. Thus q_t q_i > 0 is repulsive and q_t q_i < 0 is
attractive before projection onto the response direction. The singular
self-interaction is excluded because direct target effects are modeled by a
separate self-effect term. A zero-initialized mixing coefficient makes the
initial v38 function identical to v37 while allowing end-to-end training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_world_h1_v18 import _slog
from model_world_h1_v37 import WorldModelH1V37


class WorldModelH1V38(WorldModelH1V37):
    def __init__(self, *args, pole_epsilon: float = 0.05, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pole_epsilon = float(pole_epsilon)
        # Nonzero signs permit immediate gradients after the zero-initialized
        # mixing gate opens. tanh bounds the signed latent pole to (-1, 1).
        self.pole_sign = nn.Parameter(torch.randn(self.n_genes) * 0.1)
        self.pole_susceptibility = nn.Linear(self.d_dir, 1)
        self.pole_response = nn.Linear(self.d_dir, 1)
        self.pole_mix = nn.Parameter(torch.zeros(()))

    def pole_interaction(self, pos, charge, indices, moment, src_diff, d_src,
                         u_src, response_direction, h=None):
        """Return the scalar pole force projected onto each response axis."""
        signs = torch.tanh(self.pole_sign)
        signed_charge = charge * signs
        source_charge = signed_charge[indices]
        chi = torch.sigmoid(self.pole_susceptibility(moment).squeeze(-1))

        # Regularized inverse-square law; remove the source-self singularity.
        inv_square = 1.0 / (d_src.square() + self.pole_epsilon)
        off_source = (d_src > 1e-3).to(inv_square.dtype)
        inv_square = inv_square * off_source
        inv_square = inv_square / inv_square.mean(dim=1, keepdim=True).clamp_min(1e-6)

        radial_force = (source_charge[:, None] * signed_charge[None, :]
                        * chi * inv_square)
        projected_force = radial_force * (u_src * response_direction).sum(-1)
        amplitude = torch.tanh(self.pole_response(moment).squeeze(-1))
        return self.pole_mix * projected_force * amplitude

    def predict_delta(self, x_ctrl: torch.Tensor, pert_idx) -> torch.Tensor:
        if self.n_relations == 0:
            return super().predict_delta(x_ctrl, pert_idx)
        device = x_ctrl.device
        indices = self._indices(pert_idx).to(device)
        batch_size = x_ctrl.shape[0]
        e = self.gene_emb.weight
        n_genes = e.shape[0]

        u = self._prior_encode(e)
        charge_prior = self.prior_charge(u).squeeze(-1)
        amp_prior = self.prior_amp(u).squeeze(-1)
        src_prior = 1.0 + torch.tanh(self.prior_src(u[indices]).squeeze(-1))

        pos = F.normalize(self.pole(e), dim=-1)
        charge = F.softplus(self.charge(e).squeeze(-1) + self.charge_bias + charge_prior)

        nb = self._knn_edges(pos.detach())
        p_nb = pos[nb]
        diff = pos[:, None, :] - p_nb
        r_nb = diff.norm(dim=-1).clamp_min(1e-4)
        k_nb = self._kernel(r_nb)
        k_nb = k_nb / (k_nb.sum(-1, keepdim=True) + 1e-6)
        uvec = diff / r_nb.unsqueeze(-1)

        ctx_in = F.layer_norm(x_ctrl, (n_genes,)) if self.normalize_context else x_ctrl
        z = self.context_encoder(ctx_in)
        s = self.state_gene(e)[None, :, :] + self.state_ctx(z)[:, None, :]

        expr = torch.log1p(x_ctrl.clamp_min(0.0))
        moment = charge[None, :] * (
            1.0 + torch.tanh(self.moment_expr_w * expr + self.moment_expr_b)
        )

        kappa = F.softplus(self.raw_source_kappa)
        p_src = pos[indices]
        src_diff = pos[None, :, :] - p_src[:, None, :]
        d_src = src_diff.norm(dim=-1).clamp_min(1e-4)
        k_src = self._kernel(d_src)
        k_src = k_src / k_src.mean(dim=1, keepdim=True).clamp_min(1e-6)
        u_src = src_diff / d_src.unsqueeze(-1)

        f = getattr(self, "dgidb_feat", None)
        if f is not None:
            f_src = f[indices]
            src_gain = (1.0 + torch.tanh(self.src_proj(f_src)).squeeze(-1)) * src_prior
            amp_bias = self.amp_dg(f).squeeze(-1) + amp_prior
        else:
            src_gain = src_prior
            amp_bias = amp_prior

        phi = kappa * charge[indices][:, None] * k_src * src_gain[:, None]
        vec = phi[:, :, None] * u_src

        src_vec = torch.tanh(self.src_tf(torch.cat([e[indices], z], dim=-1)))
        q = self.kq_q(s)
        k = self.kq_k(src_vec)
        force_embedding = k[:, None, :] * q
        value = self.kq_v(s)
        induced = force_embedding * value
        axis = self.gene_axis / (1.0 + self.gene_axis.norm(dim=-1, keepdim=True))
        dx_tf = (induced * axis[None]).sum(-1)

        h = torch.zeros(batch_size, n_genes, device=device)
        relax = F.softplus(self.raw_relax)
        h_max = F.softplus(self.raw_h_max)[None, :]
        for _ in range(self.n_layers):
            m_nb = moment[:, nb]
            phi = phi + (k_nb[None] * m_nb).sum(-1)
            vec = vec + ((k_nb[None] * m_nb).unsqueeze(-1) * uvec[None]).sum(2)
            v_norm = vec.norm(dim=-1).clamp_min(1e-6)
            if self.use_langevin:
                scale = h_max * self._langevin(v_norm / h_max) / v_norm
            else:
                scale = 1.0 / v_norm
            vec_s = vec * scale.unsqueeze(-1)

            h_ext = v_norm.mean(1, keepdim=True)
            pre_a = (
                self.amp_gene(s).squeeze(-1)
                + self.amp_phi(_slog(phi).unsqueeze(-1)).squeeze(-1)
                + self.amp_v(_slog(v_norm).unsqueeze(-1)).squeeze(-1)
                + self.amp_ext(h_ext)
                + self.amp_bias
                + amp_bias[None, :]
            )
            a = torch.sigmoid(pre_a)

            d0 = F.normalize(self.dir_proj(s), dim=-1)
            vproj = vec_s - (vec_s * d0).sum(-1, keepdim=True) * d0
            eta = torch.sigmoid(self.raw_eta) * 2.0
            d = F.normalize(d0 + eta * vproj, dim=-1)
            gate = torch.sigmoid(self.gate_proj(d).squeeze(-1) + self.gate_bias)

            phi_n = _slog(phi)
            phi_n = phi_n / phi_n.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-6)
            r = self.response_right(phi_n) @ self.response_left.weight

            dx_pole = self.pole_interaction(
                pos, charge, indices, s, src_diff, d_src, u_src, d0, h=h
            )
            dx = a * gate * (torch.tanh(r) + dx_tf + dx_pole)
            h = h + dx
            moment = moment + relax * dx

        ctx_rec = F.normalize(self.context_receiver(e), dim=-1)
        ctx_gate = self.context_strength * torch.tanh(z @ ctx_rec.T / (self.d_z ** 0.5))
        shared = self.effective_shared_bias_scale()
        base = shared * self.gene_bias[None, :] * (1.0 + ctx_gate)
        raw = base + h
        self_effect = -F.softplus(self.raw_self_effect)
        raw = raw.scatter_add(1, indices[:, None], self_effect.expand(len(indices), 1))
        return self.max_delta * torch.tanh(raw / self.max_delta)

    @torch.no_grad()
    def magnetic_diagnostics(self, pert_idx) -> dict:
        diag = super().magnetic_diagnostics(pert_idx)
        signs = torch.tanh(self.pole_sign)
        diag.update({
            "pole_epsilon": self.pole_epsilon,
            "pole_mix": float(self.pole_mix),
            "pole_positive_frac": round(float((signs > 0).float().mean()), 4),
            "pole_susceptibility_w": float(self.pole_susceptibility.weight.abs().sum()),
            "pole_response_w": float(self.pole_response.weight.abs().sum()),
        })
        return diag


WorldModel = WorldModelH1V38
