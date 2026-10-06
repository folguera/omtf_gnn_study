
"""EdgeCompat variant with signed transverse-impact-parameter regression."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from omtf_gmt.features import N_FEATURES

K_MAX = 3


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

        pair_valid = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)
        no_self = ~torch.eye(n_stubs, dtype=torch.bool, device=stubs.device).unsqueeze(0)
        edge_mask = pair_valid & no_self
        weights = edge_score * edge_mask.float()
        weights = weights / weights.sum(dim=2, keepdim=True).clamp(min=1e-6)
        context = (weights.unsqueeze(-1) * emb_j).sum(dim=2)
        node_upd = self.node_updater(torch.cat([node_emb, context], dim=-1))

        valid = valid_mask.unsqueeze(-1).float()
        global_ctx = (node_upd * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

        return {
            "node_logit": self.node_head(node_upd).squeeze(-1),
            "candidate_logits": self.cand_head(global_ctx),
            "pt_pred": F.softplus(self.pt_head(global_ctx)),
            "dxy_pred": self.dxy_head(global_ctx),
        }


def build_edge_compat_dxy(
    hidden: int = 64, K: int = K_MAX, dropout: float = 0.0
) -> GMTEdgeCompatDxy:
    return GMTEdgeCompatDxy(hidden=hidden, K=K, dropout=dropout)