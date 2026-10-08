
"""EdgeCompat variant with signed transverse-impact-parameter regression."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from omtf_gmt.features import N_FEATURES
from omtf_gmt.features_tps import (
    F_TPS_DEPTH_REGION,
    F_TPS_ETA1,
    F_TPS_PHI_REL,
    F_TPS_TF_LAYER,
)

K_MAX = 3
_MAX_LAYER_GAP = 2
_MAX_STATION_GAP = 2
_MAX_DELTA_PHI = 0.12
_MAX_DELTA_ETA = 0.25


def _structured_edge_mask(
    stubs: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Build TPS edges between nearby stubs in layer/station and eta-phi."""
    phi = stubs[..., F_TPS_PHI_REL]
    eta = stubs[..., F_TPS_ETA1]
    layer = (stubs[..., F_TPS_TF_LAYER] * 4).round()
    station = (stubs[..., F_TPS_DEPTH_REGION] * 3 + 1).round()

    delta_phi = phi.unsqueeze(2) - phi.unsqueeze(1)
    delta_phi = torch.remainder(delta_phi + torch.pi, 2 * torch.pi) - torch.pi
    delta_eta = eta.unsqueeze(2) - eta.unsqueeze(1)
    layer_gap = (layer.unsqueeze(2) - layer.unsqueeze(1)).abs()
    station_gap = (station.unsqueeze(2) - station.unsqueeze(1)).abs()

    layer_compatible = (layer_gap >= 1) & (layer_gap <= _MAX_LAYER_GAP)
    station_compatible = (station_gap >= 1) & (station_gap <= _MAX_STATION_GAP)
    topology_compatible = layer_compatible | station_compatible
    geometry_compatible = (
        (delta_phi.abs() <= _MAX_DELTA_PHI)
        & (delta_eta.abs() <= _MAX_DELTA_ETA)
    )

    pair_valid = valid_mask.bool().unsqueeze(2) & valid_mask.bool().unsqueeze(1)
    n_stubs = stubs.shape[1]
    no_self = ~torch.eye(n_stubs, dtype=torch.bool, device=stubs.device).unsqueeze(0)
    return pair_valid & no_self & topology_compatible & geometry_compatible


def _mlp(dims: list[int], dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class GMTEdgeCompatDxy(nn.Module):
    """EdgeCompat encoder with candidate classification, pT, and dxy heads.

    Candidate slots follow the cache's target ordering. ``dxy_pred`` is signed
    transverse impact parameter in cm and is trained only for occupied slots.
    """

    def __init__(self, hidden: int = 64, K: int = K_MAX, dropout: float = 0.0):
        super().__init__()
        self.K = K
        self.node_encoder = _mlp([N_FEATURES, hidden, hidden], dropout)
        self.edge_encoder = _mlp([2 * hidden, hidden, 1], dropout)
        self.node_updater = _mlp([2 * hidden, hidden, hidden], dropout)
        self.node_head = nn.Linear(hidden, 1)
        self.cand_head = _mlp([hidden, hidden, K], dropout)
        self.pt_head = _mlp([hidden, hidden, K], dropout)
        self.dxy_head = _mlp([hidden, hidden, K], dropout)

    def forward(
        self,
        stubs: torch.Tensor,
        valid_mask: torch.Tensor,
        **_,
    ) -> dict[str, torch.Tensor]:
        batch_size, n_stubs, _ = stubs.shape
        node_emb = self.node_encoder(stubs)

        emb_i = node_emb.unsqueeze(2).expand(-1, -1, n_stubs, -1)
        emb_j = node_emb.unsqueeze(1).expand(-1, n_stubs, -1, -1)
        edge_score = torch.sigmoid(
            self.edge_encoder(torch.cat([emb_i, emb_j], dim=-1)).squeeze(-1)
        )

        edge_mask = _structured_edge_mask(stubs, valid_mask)
        weights = edge_score * edge_mask.float()
        weights = weights / weights.sum(dim=2, keepdim=True).clamp(min=1e-6)
        context = (weights.unsqueeze(-1) * emb_j).sum(dim=2)
        node_upd = self.node_updater(torch.cat([node_emb, context], dim=-1))

        valid = valid_mask.unsqueeze(-1).float()
        global_ctx = (node_upd * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

        assign_logits = self.cand_head(node_upd).transpose(1, 2)
        valid_slots = valid_mask.bool().unsqueeze(1)
        has_valid = valid_mask.bool().any(dim=1).view(batch_size, 1, 1)
        assign_logits = assign_logits.masked_fill(~valid_slots, float("-inf"))
        assign_logits = torch.where(has_valid, assign_logits, torch.zeros_like(assign_logits))
        assign_weights = F.softmax(assign_logits, dim=-1) * valid_slots.float()
        assign_weights = assign_weights / assign_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        slot_ctx = torch.bmm(assign_weights, node_upd)
        dxy_slot_outputs = self.dxy_head(slot_ctx)
        dxy_pred = torch.diagonal(dxy_slot_outputs, dim1=1, dim2=2)

        return {
            "node_logit": self.node_head(node_upd).squeeze(-1),
            "candidate_logits": self.cand_head(global_ctx),
            "pt_pred": F.softplus(self.pt_head(global_ctx)),
            "dxy_pred": dxy_pred,
            "assign_weights": assign_weights,
        }


def build_edge_compat_dxy(
    hidden: int = 64, K: int = K_MAX, dropout: float = 0.0
) -> GMTEdgeCompatDxy:
    return GMTEdgeCompatDxy(hidden=hidden, K=K, dropout=dropout)