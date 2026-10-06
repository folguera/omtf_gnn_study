"""
GMT branch training script — Phase B1 baseline.

Trains any registered GMT model on the pre-built GMT cache.
Saves checkpoints and per-epoch metrics.

Usage
-----
  python src/omtf_gmt/train.py \\
      --cache-dir build/omtf_gmt/cache \\
      --datasets S1 S2 S3 S4 S5 B1 B2 B3 B4 \\
      --repeat B4:8 \\
      --model deepsets \\
      --hidden 64 \\
      --epochs 50 \\
      --output-dir build/omtf_gmt/checkpoints/deepsets_B1a

Per-dataset repeat factors (--repeat DS:N) oversample the training split of dataset DS
by a factor of N without touching the validation split.  B4 should be oversampled so
the 0-candidate regime is adequately represented.  See docs/omtf_gmt/TRAINING_MIX_DECISION.md.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from omtf_gmt.dataset import GMTCachedDataset, collate_gmt, expand_datasets, expand_repeats
from omtf_gmt.models  import build_deepsets, build_edge_compat, build_edge_compat_dxy, build_edge_compat_assign, build_slot_model, build_seq_slot, build_count_model, build_detr_model
from omtf_gmt.models.edge_compat_assign import assignment_supervision_loss
from omtf_gmt.models.slot_model import (
    candidate_count_loss,
    attention_diversity_loss,
    null_attention_empty_slot_loss,
    attention_supervision_loss,
)

K_MAX        = 3
ALL_DATASETS   = ["S1", "S2", "S3", "S4", "S5", "B1", "B2", "B3", "B4"]
ALL_G_DATASETS = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "B4"]


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #

def compute_loss(
    out:    dict[str, torch.Tensor],
    batch:  dict[str, torch.Tensor],
    w_node:           float = 1.0,
    w_cand:           float = 1.0,
    w_pt:             float = 0.5,
    w_count:          float = 0.0,
    w_div:            float = 0.0,
    w_null:           float = 0.0,
    w_attn:           float = 0.0,
    w_hard_neg:       float = 0.0,
    unmatched_weight: float = 1.0,
    w_assign:         float = 0.0,
    w_dxy:            float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    vm  = batch["valid_mask"]              # (N, 24)
    nl  = batch["node_label"]              # (N, 24)
    gpt = batch["gen_pt"]                  # (N, K)

    # node BCE (valid stubs only)
    # optionally down-weight unmatched stubs (truth_source == 0)
    node_logit = out["node_logit"]
    if unmatched_weight != 1.0 and "truth_source" in batch:
        ts = batch["truth_source"]                          # (N, 24) int
        w  = torch.ones_like(nl, dtype=torch.float32)
        w[ts == 0] = unmatched_weight
        w = w * vm.float()                                  # zero out padding
        raw = F.binary_cross_entropy_with_logits(
            node_logit, nl.float(), reduction="none"
        )
        node_loss = (raw * w).sum() / w.sum().clamp(min=1.0)
    else:
        node_loss = F.binary_cross_entropy_with_logits(
            node_logit[vm], nl[vm].float(), reduction="mean"
        )

    # candidate BCE: slot k is positive if gen_pt[k] > 0
    cand_target = (gpt > 0).float()
    cand_loss   = F.binary_cross_entropy_with_logits(
        out["candidate_logits"], cand_target, reduction="mean"
    )

    # pT regression (log-space MSE, signal slots only)
    sig_mask = cand_target > 0.5
    pt_loss  = torch.tensor(0.0, device=node_logit.device)
    if sig_mask.any():
        pt_pred_pos = out["pt_pred"]   # models apply F.softplus internally
        pt_loss = F.mse_loss(
            torch.log1p(pt_pred_pos[sig_mask]),
            torch.log1p(gpt[sig_mask]),
        )

    total = w_node * node_loss + w_cand * cand_loss + w_pt * pt_loss
    breakdown = {
        "node_loss": node_loss.item(),
        "cand_loss": cand_loss.item(),
        "pt_loss":   pt_loss.item(),
    }

    if w_dxy > 0.0 and "dxy_pred" in out:
        dxy_mask = cand_target > 0.5
        dxy_loss = torch.tensor(0.0, device=node_logit.device)
        if dxy_mask.any():
            dxy_loss = F.smooth_l1_loss(
                out["dxy_pred"][dxy_mask], batch["gen_dxy"][dxy_mask]
            )
        total = total + w_dxy * dxy_loss
        breakdown["dxy_loss"] = dxy_loss.item()

    # explicit hard-negative candidate loss: push all slots negative for G7/G8/G9/G10 windows
    if w_hard_neg > 0.0:
        hard_neg_mask = None

        if "meta_is_hard_neg" in batch:
            hard_neg_mask = batch["meta_is_hard_neg"].to(
                device=node_logit.device, dtype=torch.bool,
            )
        elif "meta" in batch:
            hard_neg_mask = torch.tensor(
                [int(m.get("is_hard_neg", 0)) for m in batch["meta"]],
                dtype=torch.bool, device=node_logit.device,
            )

        if hard_neg_mask is not None and hard_neg_mask.any():
            hn_logits = out["candidate_logits"][hard_neg_mask]
            hn_loss = F.binary_cross_entropy_with_logits(
                hn_logits,
                torch.zeros_like(hn_logits),
                reduction="mean",
            )
            total = total + w_hard_neg * hn_loss
            breakdown["hard_neg_loss"] = hn_loss.item()

    # slot-model auxiliary losses — active only when model returns these keys
    if w_count > 0 and "null_attn" in out:
        cnt_loss = candidate_count_loss(out["candidate_logits"], gpt)
        total    = total + w_count * cnt_loss
        breakdown["count_loss"] = cnt_loss.item()

    if w_div > 0 and "attn_weights" in out and "null_attn" in out:
        div_loss = attention_diversity_loss(out["attn_weights"], vm)
        total    = total + w_div * div_loss
        breakdown["div_loss"] = div_loss.item()

    if w_null > 0 and "null_attn" in out:
        nul_loss = null_attention_empty_slot_loss(out["null_attn"], gpt)
        total    = total + w_null * nul_loss
        breakdown["null_loss"] = nul_loss.item()

    if w_attn > 0 and "attn_weights" in out and "null_attn" in out:
        sup_loss = attention_supervision_loss(
            out["attn_weights"], out["null_attn"],
            batch["track_id"], vm, gpt,
        )
        total    = total + w_attn * sup_loss
        breakdown["attn_sup_loss"] = sup_loss.item()

    # count CE loss — active when model returns count_logits (count_model)
    if "count_logits" in out:
        true_count = (gpt > 0).sum(dim=1).long().clamp(max=out["count_logits"].shape[-1] - 1)
        cnt_ce = F.cross_entropy(out["count_logits"], true_count)
        total  = total + w_count * cnt_ce
        breakdown["count_ce_loss"] = cnt_ce.item()

    if w_assign > 0.0 and "assign_weights" in out:
        a_loss = assignment_supervision_loss(
            out["assign_weights"], batch["track_id"],
            batch["valid_mask"],   batch["gen_pt"],
        )
        total = total + w_assign * a_loss
        breakdown["assign_loss"] = a_loss.item()

    breakdown["loss"] = total.item()
    return total, breakdown


# --------------------------------------------------------------------------- #
# DETR Hungarian matching loss
# --------------------------------------------------------------------------- #


# All K! permutations of K slot indices, precomputed as a module-level constant.
# K=3 → 6 permutations. Used by detr_loss for vectorised matching.
_SLOT_PERMS = torch.tensor(
    list(itertools.permutations(range(K_MAX))), dtype=torch.long
)  # (6, 3)


def detr_loss(
    out:        dict,
    batch:      dict,
    w_node:     float = 1.0,
    w_pt:       float = 0.5,
    w_no_obj:   float = 0.1,
    w_pt_match: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Permutation-invariant loss via exhaustive permutation search (fully on GPU).

    With K=3 there are only 6 permutations.  For each window we evaluate the
    total cost of all 6 assignments in parallel and pick the cheapest one.
    No Python loops; no scipy; no CPU round-trips.

    Cost of assigning prediction k to target j:
      real target  (gen_pt[j] > 0): -log σ(logit_k) + w_pt_match · |Δ log pT|
      null target  (gen_pt[j] = 0): w_no_obj · -log σ(−logit_k)
    """
    vm          = batch["valid_mask"]
    nl          = batch["node_label"]
    gpt         = batch["gen_pt"]           # (N, K)  — 0 for null slots
    cand_logits = out["candidate_logits"]   # (N, K)
    pt_pred     = out["pt_pred"]    # models apply F.softplus internally
    node_logit  = out["node_logit"]

    N, K   = cand_logits.shape
    device = cand_logits.device

    # ---- node BCE ----
    node_loss = F.binary_cross_entropy_with_logits(
        node_logit[vm], nl[vm].float(), reduction="mean"
    )

    # ---- build cost matrix C[b, k, j] ----
    is_real = (gpt > 0)                                       # (N, K)  bool

    bce_obj    = -F.logsigmoid(cand_logits)                   # (N, K)  cost: predict object
    bce_no_obj = -F.logsigmoid(-cand_logits)                  # (N, K)  cost: predict no-object

    log_pt_pred = torch.log1p(pt_pred)                        # (N, K)
    log_pt_true = torch.log1p(gpt)                            # (N, K)
    pt_cost = (log_pt_pred.unsqueeze(2) - log_pt_true.unsqueeze(1)).abs()  # (N, K, K)

    real_cost = bce_obj.unsqueeze(2) + w_pt_match * pt_cost   # (N, K, K)
    null_cost = w_no_obj * bce_no_obj.unsqueeze(2).expand(N, K, K)

    # is_real indexed over target axis j: (N, 1, K) → (N, K, K)
    cost = torch.where(is_real.unsqueeze(1).expand(N, K, K), real_cost, null_cost)

    # ---- find best permutation (vectorised over batch) ----
    perms = _SLOT_PERMS.to(device)                            # (P, K)  P=6
    P     = perms.shape[0]

    # cost_perm[b, p, k] = cost[b, k, perms[p, k]]
    perm_col_idx = perms.view(1, P, K).expand(N, P, K)        # (N, P, K)
    # expand cost to (N, P, K, K) and gather along last dim
    cost_exp  = cost.unsqueeze(1).expand(N, P, K, K)
    perm_col  = perm_col_idx.unsqueeze(-1)                    # (N, P, K, 1)
    cost_perm = cost_exp.gather(3, perm_col).squeeze(-1)      # (N, P, K)
    perm_total = cost_perm.sum(-1)                            # (N, P)

    best_p = perm_total.argmin(dim=-1)                        # (N,)

    # ---- recover target assignment for best permutation ----
    # target_idx[b, k] = perms[best_p[b], k]
    best_p_exp  = best_p.view(N, 1, 1).expand(N, 1, K)
    target_idx  = perms.unsqueeze(0).expand(N, P, K).gather(1, best_p_exp).squeeze(1)  # (N, K)

    # ---- compute training loss using best assignment ----
    is_real_matched = is_real.gather(1, target_idx)           # (N, K)
    gpt_matched     = gpt.gather(1, target_idx)               # (N, K)

    # BCE: target=1 for real, target=0 for null; null slots down-weighted
    cand_targets = is_real_matched.float()
    cand_weights = torch.where(
        is_real_matched,
        torch.ones_like(cand_targets),
        torch.full_like(cand_targets, w_no_obj),
    )
    cand_loss = (
        F.binary_cross_entropy_with_logits(cand_logits, cand_targets, reduction="none")
        * cand_weights
    ).mean()

    # pT MSE (real matched slots only)
    pt_loss_mat = F.mse_loss(
        torch.log1p(pt_pred), torch.log1p(gpt_matched), reduction="none"
    )
    n_real  = is_real_matched.float().sum().clamp(min=1.0)
    pt_loss = (pt_loss_mat * is_real_matched.float()).sum() / n_real

    total = w_node * node_loss + cand_loss + w_pt * pt_loss

    breakdown = {
        "node_loss": node_loss.item(),
        "cand_loss": cand_loss.item(),
        "pt_loss":   pt_loss.item(),
        "loss":      total.item(),
    }
    return total, breakdown


# --------------------------------------------------------------------------- #
# Metrics (simple, for smoke-test)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def quick_metrics(out: dict, batch: dict) -> dict[str, float]:
    vm  = batch["valid_mask"]
    nl  = batch["node_label"]
    gpt = batch["gen_pt"]

    # node AUC approximation via positive-vs-negative mean logit gap
    nl_b  = nl[vm].bool()
    logit = out["node_logit"][vm]
    if nl_b.any() and (~nl_b).any():
        node_gap = (logit[nl_b].mean() - logit[~nl_b].mean()).item()
    else:
        node_gap = 0.0

    # stub recall: fraction of signal stubs with logit > 0
    stub_recall = ((logit > 0) & nl_b).float().sum().item() / max(1, nl_b.float().sum().item())

    # candidate recovery: fraction of gen-muon slots correctly fired
    cand_logit = out["candidate_logits"]
    sig_slots  = (gpt > 0)
    cand_rec   = ((cand_logit > 0) & sig_slots).float().sum().item() / max(1, sig_slots.float().sum().item())

    # false-positive rate on zero-candidate windows (B4 regime)
    zero_win = sig_slots.sum(dim=1) == 0           # (N,) windows with no true candidate
    if zero_win.any():
        zero_fp = (cand_logit[zero_win] > 0).any(dim=1).float().mean().item()
    else:
        zero_fp = 0.0

    # extra-slot false positives: fired slots where no true candidate exists
    neg_slots = ~sig_slots
    extra_slot_fp = (
        ((cand_logit > 0) & neg_slots).float().sum().item()
        / max(1, neg_slots.float().sum().item())
    )

    return {
        "node_logit_gap": node_gap,
        "stub_recall":    stub_recall,
        "cand_recovery":  cand_rec,
        "zero_win_fp":    zero_fp,
        "extra_slot_fp":  extra_slot_fp,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GMT branch training")
    p.add_argument("--cache-dir",    type=Path, required=True)
    p.add_argument("--datasets",     nargs="+", default=ALL_DATASETS)
    p.add_argument("--repeat",       nargs="*", default=["B4:8"], metavar="DS:N",
                   help="per-dataset train-split repeat factors, e.g. --repeat B4:8 S4:2")
    p.add_argument("--model",        default="deepsets",
                   choices=["deepsets", "edge_compat", "edge_compat_dxy", "edge_compat_assign",
                            "slot_model", "seq_slot", "count_model", "detr_model"])
    p.add_argument("--hidden",       type=int,   default=64)
    p.add_argument("--dropout",      type=float, default=0.0)
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch-size",   type=int,   default=512)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--w-node",       type=float, default=1.0)
    p.add_argument("--w-cand",       type=float, default=1.0)
    p.add_argument("--w-pt",         type=float, default=0.5)
    p.add_argument("--w-dxy",        type=float, default=1.0,
                   help="signed gen-muon dxy Smooth L1 loss weight (cm; edge_compat_dxy only)")
    p.add_argument("--w-count",      type=float, default=0.0,
                   help="candidate-count MSE loss weight (slot_model only)")
    p.add_argument("--w-div",        type=float, default=0.0,
                   help="attention diversity loss weight (slot_model only)")
    p.add_argument("--w-null",       type=float, default=0.0,
                   help="null-attention empty-slot loss weight (slot_model only)")
    p.add_argument("--w-attn",       type=float, default=0.0,
                   help="track_id attention supervision loss weight (slot_model only)")
    p.add_argument("--w-no-obj",     type=float, default=0.1,
                   help="no-object slot loss weight (detr_model only)")
    p.add_argument("--w-hard-neg",   type=float, default=0.0,
                   help="explicit hard-negative candidate loss weight (pushes G7/G8 slots to zero)")
    p.add_argument("--w-assign",     type=float, default=0.0,
                   help="assignment supervision loss weight (edge_compat_assign only)")
    p.add_argument("--unmatched-stub-weight", type=float, default=1.0,
                   help="node-loss weight for truth_source==0 (unmatched) stubs; <1.0 down-weights ambiguous stubs")
    p.add_argument("--num-workers",  type=int,   default=4,
                   help="DataLoader worker processes (0 = main process only)")
    p.add_argument("--amp",          action="store_true", default=False,
                   help="enable automatic mixed precision (fp16) — recommended on V100+")
    p.add_argument("--scheduler",    default="none", choices=["none", "cosine"],
                   help="LR scheduler: cosine annealing (eta_min = lr * 0.05)")
    p.add_argument("--save-epochs",  nargs="*", type=int, default=[],
                   help="extra epochs to save checkpoints at, e.g. --save-epochs 50 75 100")
    p.add_argument("--output-dir",   type=Path,  default=Path("build/omtf_gmt/checkpoints"))
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = expand_datasets(args.datasets)
    if args.model == "edge_compat_dxy":
        cache_manifest = json.loads((args.cache_dir / "manifest.json").read_text())
        unchecked = [
            ds for ds in dataset_names
            if not cache_manifest.get("datasets", {}).get(ds, {}).get("gen_dxy_required", False)
        ]
        if unchecked:
            raise ValueError(
                "The edge_compat_dxy model requires caches rebuilt with "
                f"--require-dxy; unverified datasets: {', '.join(unchecked)}"
            )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark        = True  # autotune kernels for fixed input shapes
        torch.backends.cuda.matmul.allow_tf32 = True  # A100 TF32 tensor cores for fp32 matmuls
        torch.backends.cudnn.allow_tf32       = True

    # --- parse per-dataset repeat factors (alias-aware) ---
    repeats = expand_repeats(args.repeat)

    # --- data ---
    train_parts: list = []
    val_parts:   list = []
    ds_sizes: list[tuple[str, int, int]] = []   # (name, n_train_1x, n_repeat)

    for ds in dataset_names:
        full  = GMTCachedDataset(args.cache_dir, ds)
        n     = len(full)
        n_tr  = int(0.85 * n)
        n_va  = n - n_tr
        tr, va = torch.utils.data.random_split(
            full, [n_tr, n_va],
            generator=torch.Generator().manual_seed(42),
        )
        train_parts.append(tr)
        val_parts.append(va)
        ds_sizes.append((ds, n_tr, repeats.get(ds, 1)))

    # WeightedRandomSampler: each dataset's samples get weight = repeat factor.
    # Draws are independent with replacement, so high-weight datasets appear
    # more often but in random positions throughout the epoch — no block patterns.
    sample_weights: list[float] = []
    for _, n_tr, n_rep in ds_sizes:
        sample_weights.extend([float(n_rep)] * n_tr)

    n_train_eff = sum(n_tr * n_rep for _, n_tr, n_rep in ds_sizes)
    n_val_total = sum(len(d) for d in val_parts)

    pin = device.type == "cuda"
    pw  = args.num_workers > 0

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float64),
        num_samples=n_train_eff,
        replacement=True,
    )
    train_loader = DataLoader(
        ConcatDataset(train_parts), batch_size=args.batch_size,
        sampler=sampler, collate_fn=collate_gmt,
        num_workers=args.num_workers, pin_memory=pin, persistent_workers=pw,
    )
    val_loader = DataLoader(
        ConcatDataset(val_parts), batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_gmt,
        num_workers=args.num_workers, pin_memory=pin, persistent_workers=pw,
    )

    print("Effective training mix (WeightedRandomSampler):")
    for ds_name, n_tr, n_rep in ds_sizes:
        eff = n_tr * n_rep
        pct = 100.0 * eff / max(1, n_train_eff)
        rep_str = f" ×{n_rep}" if n_rep > 1 else ""
        print(f"  {ds_name}{rep_str}: {eff:,} ({pct:.1f}%)")
    print(f"  Total train: {n_train_eff:,}  |  Val: {n_val_total:,}  |  Device: {device}")
    # Note: n_train_eff > raw_train_size when any repeat > 1 — epoch is proportionally longer.

    mix_info = {
        "datasets": [
            {
                "name": ds, "n_train_1x": n_tr, "repeat": n_rep,
                "effective": n_tr * n_rep,
                "fraction": (n_tr * n_rep) / max(1, n_train_eff),
            }
            for ds, n_tr, n_rep in ds_sizes
        ],
        "n_train_eff": n_train_eff,
        "n_val_total": n_val_total,
        "sampler": "WeightedRandomSampler(replacement=True)",
        "amp": args.amp,
        "num_workers": args.num_workers,
        "batch_size": args.batch_size,
    }
    (args.output_dir / "training_mix.json").write_text(json.dumps(mix_info, indent=2))

    # --- model ---
    if args.model == "deepsets":
        model = build_deepsets(hidden=args.hidden, dropout=args.dropout)
    elif args.model == "edge_compat":
        model = build_edge_compat(hidden=args.hidden, dropout=args.dropout)
    elif args.model == "edge_compat_dxy":
        model = build_edge_compat_dxy(hidden=args.hidden, dropout=args.dropout)
    elif args.model == "edge_compat_assign":
        model = build_edge_compat_assign(hidden=args.hidden, dropout=args.dropout)
    elif args.model == "slot_model":
        model = build_slot_model(hidden=args.hidden, dropout=args.dropout)
    elif args.model == "seq_slot":
        model = build_seq_slot(hidden=args.hidden, dropout=args.dropout)
    elif args.model == "count_model":
        model = build_count_model(hidden=args.hidden, dropout=args.dropout)
    elif args.model == "detr_model":
        model = build_detr_model(hidden=args.hidden, dropout=args.dropout)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}  params={n_params:,}")

    opt    = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs, eta_min=args.lr * 0.05
        )
        print(f"Scheduler: CosineAnnealingLR  T_max={args.epochs}  eta_min={args.lr * 0.05:.2e}")

    save_epochs = set(args.save_epochs)
    history = []
    best_val_loss = float("inf")
    ckpt_path = args.output_dir / f"gmt_{args.model}_best.pt"
    last_path = args.output_dir / f"gmt_{args.model}_last.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # train
        model.train()
        tr_loss    = 0.0
        tr_hn_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=pin) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=args.amp):
                out  = model(batch["stubs"], batch["valid_mask"])
                if args.model == "detr_model":
                    loss, bd = detr_loss(out, batch, args.w_node, args.w_pt, args.w_no_obj)
                else:
                    loss, bd = compute_loss(out, batch, args.w_node, args.w_cand, args.w_pt,
                                       args.w_count, args.w_div, args.w_null, args.w_attn,
                                       args.w_hard_neg, args.unmatched_stub_weight,
                                       args.w_assign, args.w_dxy)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            tr_loss    += loss.item()
            tr_hn_loss += bd.get("hard_neg_loss", 0.0)
        if scheduler is not None:
            scheduler.step()
        tr_loss    /= max(1, len(train_loader))
        tr_hn_loss /= max(1, len(train_loader))

        # validate
        model.eval()
        val_loss = 0.0
        val_metrics: dict[str, float] = {
            "node_logit_gap": 0, "stub_recall": 0, "cand_recovery": 0,
            "zero_win_fp": 0, "extra_slot_fp": 0,
        }
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device, non_blocking=pin) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                with torch.cuda.amp.autocast(enabled=args.amp):
                    out  = model(batch["stubs"], batch["valid_mask"])
                    if args.model == "detr_model":
                        loss, _ = detr_loss(out, batch, args.w_node, args.w_pt, args.w_no_obj)
                    else:
                        loss, _ = compute_loss(out, batch, args.w_node, args.w_cand, args.w_pt,
                                       args.w_count, args.w_div, args.w_null, args.w_attn,
                                       args.w_hard_neg, args.unmatched_stub_weight,
                                       args.w_assign, args.w_dxy)
                val_loss += loss.item()
                m = quick_metrics(out, batch)
                for k in val_metrics:
                    val_metrics[k] += m[k]
        val_loss /= max(1, len(val_loader))
        for k in val_metrics:
            val_metrics[k] /= max(1, len(val_loader))

        elapsed = time.time() - t0
        hn_str = f"  hn={tr_hn_loss:.4f}" if tr_hn_loss > 0.0 else ""
        print(
            f"[{epoch:3d}/{args.epochs}] "
            f"tr={tr_loss:.4f}  val={val_loss:.4f}{hn_str}  "
            f"recall={val_metrics['stub_recall']:.3f}  "
            f"cand_rec={val_metrics['cand_recovery']:.3f}  "
            f"zero_fp={val_metrics['zero_win_fp']:.3f}  "
            f"xslot_fp={val_metrics['extra_slot_fp']:.3f}  "
            f"({elapsed:.0f}s)"
        )

        row = {"epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss, **val_metrics}
        history.append(row)

        ckpt_payload = {
            "model": model.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "args": vars(args),
        }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_payload["best_val_loss"] = best_val_loss
            torch.save(ckpt_payload, ckpt_path)

        # always overwrite last.pt
        torch.save(ckpt_payload, last_path)

        # fixed-epoch snapshots
        if epoch in save_epochs:
            snap_path = args.output_dir / f"gmt_{args.model}_epoch_{epoch:04d}.pt"
            torch.save(ckpt_payload, snap_path)
            print(f"  Snapshot saved: {snap_path}")

    # save history
    hist_path = args.output_dir / f"gmt_{args.model}_history.json"
    hist_path.write_text(json.dumps(history, indent=2))
    print(f"\nBest val_loss={best_val_loss:.4f} at checkpoint: {ckpt_path}")
    print(f"History: {hist_path}")


if __name__ == "__main__":
    main()
