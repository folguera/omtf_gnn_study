"""
EdgeCompatAssign — EdgeCompat encoder with an explicit stub-to-candidate
assignment decoder (Phase B6).

Architecture
------------
Encoder: identical to GMTEdgeCompat (Phase B5).

New decoder: a small assignment head maps each updated stub embedding to K
logit scores, one per candidate slot.  Softmax over stubs (masked to valid
ones) gives assignment weights, which are used to compute per-slot context
vectors for the candidate / pT / charge heads.

  assign_head   : MLP(H → H → K)   per-stub logits  (B, Nmax, K)
  assign_weights: softmax over stubs after masking padding  (B, K, Nmax)
  slot_ctx[k]   : Σ_i assign_weights[k, i] · node_upd[i]   (B, K, H)
  cand_head     : MLP(H → H → 1) on slot_ctx  (B, K)
  pt_head       : same
  chg_head      : same

Slot convention: slot k ↔ track_id = k + 1 (fixed, no Hungarian matching).

Outputs
-------
  node_logit       : (B, Nmax)
  candidate_logits : (B, K)
  pt_pred          : (B, K)  softplus
  charge_pred      : (B, K)  tanh
  assign_logits    : (B, K, Nmax)  pre-softmax
  assign_weights   : (B, K, Nmax)  post-softmax (normalised over valid stubs)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from omtf_gmt.features import N_FEATURES

K_MAX: int = 3


def _mlp(dims: list[int], dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class EdgeCompatAssign(nn.Module):
    def __init__(self, hidden: int = 64, K: int = K_MAX, dropout: float = 0.0):
        super().__init__()
        self.K = K
        H = hidden

        # ---- shared encoder (identical to GMTEdgeCompat) ----
        self.node_encoder = _mlp([N_FEATURES, H, H], dropout)
        self.edge_encoder = _mlp([2 * H, H, 1],      dropout)
        self.node_updater = _mlp([2 * H, H, H],       dropout)
        self.node_head    = nn.Linear(H, 1)

        # ---- assignment decoder ----
        self.assign_head  = _mlp([H, H, K], dropout)   # (B, Nmax, K)

        # ---- per-slot candidate heads (H → 1, applied to slot context) ----
        self.cand_head = _mlp([H, H, 1], dropout)
        self.pt_head   = _mlp([H, H, 1], dropout)
        self.chg_head  = _mlp([H, H, 1], dropout)

    def forward(
        self,
        stubs:      torch.Tensor,   # (B, Nmax, F)
        valid_mask: torch.Tensor,   # (B, Nmax) bool
        **_,
    ) -> dict[str, torch.Tensor]:
        B, Nmax, _ = stubs.shape
        device = stubs.device
        H = self.node_encoder[-1].out_features

        # ---- encoder (shared with GMTEdgeCompat) ----
        node_emb = self.node_encoder(stubs)                        # (B, Nmax, H)

        ei = node_emb.unsqueeze(2).expand(-1, -1, Nmax, -1)
        ej = node_emb.unsqueeze(1).expand(-1, Nmax, -1, -1)
        edge_score = torch.sigmoid(
            self.edge_encoder(torch.cat([ei, ej], dim=-1)).squeeze(-1)
        )                                                          # (B, Nmax, Nmax)

        vm = valid_mask
        pair_valid = vm.unsqueeze(2) & vm.unsqueeze(1)
        no_self    = ~torch.eye(Nmax, dtype=torch.bool, device=device).unsqueeze(0)
        edge_mask  = pair_valid & no_self

        w     = edge_score * edge_mask.float()
        denom = w.sum(dim=2, keepdim=True).clamp(min=1e-6)
        w_n   = (w / denom).unsqueeze(-1)
        ctx   = (w_n * ej).sum(dim=2)                             # (B, Nmax, H)

        node_upd = self.node_updater(torch.cat([node_emb, ctx], dim=-1))  # (B, Nmax, H)

        node_logit = self.node_head(node_upd).squeeze(-1)         # (B, Nmax)

        # ---- assignment decoder ----
        assign_logits = self.assign_head(node_upd)                # (B, Nmax, K)
        assign_logits = assign_logits.transpose(1, 2)             # (B, K, Nmax)
        # mask padding stubs to −∞ so they get ~0 weight after softmax
        assign_logits = assign_logits.masked_fill(
            ~valid_mask.unsqueeze(1).expand(-1, self.K, -1), float('-inf')
        )
        assign_weights = F.softmax(assign_logits, dim=-1)         # (B, K, Nmax)

        # slot context: weighted sum of updated stub embeddings
        slot_ctx = torch.bmm(assign_weights, node_upd)            # (B, K, H)

        # ---- per-slot heads ----
        candidate_logits = self.cand_head(slot_ctx).squeeze(-1)   # (B, K)
        pt_pred          = F.softplus(self.pt_head(slot_ctx)).squeeze(-1)   # (B, K)
        charge_pred      = torch.tanh(self.chg_head(slot_ctx)).squeeze(-1)  # (B, K)

        return {
            "node_logit":       node_logit,
            "candidate_logits": candidate_logits,
            "pt_pred":          pt_pred,
            "charge_pred":      charge_pred,
            "assign_logits":    assign_logits,
            "assign_weights":   assign_weights,
        }


# ---- assignment supervision loss ------------------------------------------

def assignment_supervision_loss(
    assign_weights: torch.Tensor,   # (B, K, Nmax)
    track_id:       torch.Tensor,   # (B, Nmax)  int8, values 0..K
    valid_mask:     torch.Tensor,   # (B, Nmax)  bool
    gen_pt:         torch.Tensor,   # (B, K)     float32
    target_track_id: torch.Tensor | None = None,  # (B, K) packed-slot → original track ID
) -> torch.Tensor:
    """
    Weighted cross-entropy: each active slot attends to its own true stubs.
    When supplied, target_track_id maps packed slots to original track IDs.
    Otherwise slots are assumed to map to IDs 1..K.

    Only applies to slots that have at least one true stub.
    Empty slots are not supervised (no NULL-token loss).
    """
    B, K, Nmax = assign_weights.shape
    total = assign_weights.new_tensor(0.0)
    n_terms = 0

    log_aw = torch.log(assign_weights.clamp(min=1e-8))            # (B, K, Nmax)

    for k in range(K):
        active   = gen_pt[:, k] > 0                               # (B,)
        if not active.any():
            continue

        target_id = (
            target_track_id[:, k] if target_track_id is not None
            else torch.full_like(gen_pt[:, k], k + 1, dtype=track_id.dtype)
        )
        true_stubs = ((track_id == target_id.unsqueeze(1)) & valid_mask).float()
        n_true     = true_stubs.sum(dim=1, keepdim=True).clamp(min=1)
        target     = true_stubs / n_true                          # uniform over true stubs

        # CE:  −Σ_i target[i] * log(p[i])
        per_sample = -(target * log_aw[:, k, :]).sum(dim=1)       # (B,)

        has_true = true_stubs.sum(dim=1) > 0                      # (B,)
        mask     = active & has_true
        if mask.any():
            total   = total + per_sample[mask].sum()
            n_terms += int(mask.sum().item())

    return total / max(n_terms, 1)


# ---- factory ---------------------------------------------------------------

def build_edge_compat_assign(
    hidden: int = 64, K: int = K_MAX, dropout: float = 0.0
) -> EdgeCompatAssign:
    return EdgeCompatAssign(hidden=hidden, K=K, dropout=dropout)
