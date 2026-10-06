#!/usr/bin/env python
"""
GMT model evaluation — Phase B1 eval suite.

This script evaluates a trained GMT-visible-stub model on the pre-built GMT cache.

Evaluation passes
-----------------
1. Per-slot / per-window metrics on the held-out validation split:
   - Stub recall and stub fake rate
   - Slot efficiency and negative-slot fake rate
   - Candidate multiplicity confusion matrix
   - Efficiency vs gen-muon pT and |d0|
   - pT regression quality
   - Count-based slot-level threshold scan
   - Zero-candidate false-positive rate

2. Event-level trigger efficiency on the full cached dataset:
   - Window-level trigger efficiency at pT thresholds
   - Event-level trigger efficiency after OR across processor windows
   - Background acceptance for B4
   - Count-based pooled event-level summary

Important scope
---------------
This is a Phase B1 model/cache evaluation. It is not yet a full L1 trigger-rate
study with complete GMT integration and absolute luminosity normalization.

Usage
-----
  python scripts/omtf_gmt/eval_gmt.py \
      --checkpoint build/omtf_gmt/checkpoints/deepsets_B1a/gmt_deepsets_best.pt \
      --cache-dir  build/omtf_gmt/cache \
      --datasets   S1 S2 S3 S4 S5 B1 B2 B3 B4 \
      --threshold  0.0 \
      --output     build/omtf_gmt/eval/deepsets_B1a_eval.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from omtf_gmt.dataset import GMTCachedDataset, collate_gmt, expand_datasets
from omtf_gmt.models import build_deepsets, build_edge_compat, build_slot_model, build_seq_slot, build_count_model, build_detr_model


ALL_DS        = ["S1", "S2", "S3", "S4", "S5", "B1", "B2", "B3", "B4"]
ALL_G_DS      = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10", "B4"]
ALL_DAS_DS    = ["single_muon_flatpt", "displaced_lowpt", "displaced_midpt",
                 "dy_prompt", "llp_addon", "minbias"]
ZERO_CAND_DS  = {"B4", "G7", "G8", "G9_pos", "G9_neg", "G10_pos", "G10_neg",
                 "minbias"}   # datasets where zero overlap candidates are expected
K_MAX = 3

PT_BINS = [0, 2, 5, 10, 15, 20, 30, 50, 100, 9999]
D0_BINS = [0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 9999]
THR_SCAN = np.linspace(-3.0, 3.0, 61)
PT_TRIG_THR = [0, 5, 10, 15, 20]


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #

def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _sigma68(values: np.ndarray) -> float | None:
    """
    Robust 68% width.

    For a centered residual distribution, this is approximately the half-width
    between the 16th and 84th percentiles.
    """
    if values.size == 0:
        return None
    p16, p84 = np.percentile(values, [16, 84])
    return float(0.5 * (p84 - p16))


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def _require_shard_keys(shard: dict[str, Any], shard_path: Path, required: list[str]) -> None:
    missing = [k for k in required if k not in shard]
    if missing:
        raise KeyError(
            f"Shard {shard_path} is missing required keys: {missing}. "
            "Rebuild the GMT cache or update the evaluator to match the cache schema."
        )


def _bin_efficiency(values: np.ndarray, recovered: np.ndarray, bins: list[float]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (values >= lo) & (values < hi)
        n = int(mask.sum())
        nrec = int((recovered & mask).sum()) if n > 0 else 0
        result.append({
            "lo": lo,
            "hi": hi if hi < 9000 else None,
            "n": n,
            "n_rec": nrec,
            "efficiency": nrec / n if n > 0 else None,
        })
    return result


def _format_opt_float(x: float | None, ndigits: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and not np.isfinite(x):
        return "—"
    return f"{x:.{ndigits}f}"


# --------------------------------------------------------------------------- #
# Model loading                                                               #
# --------------------------------------------------------------------------- #

def load_model(ckpt_path: Path, device: torch.device):
    """
    Load model from checkpoint.

    The checkpoint is expected to contain:
      - "model": state_dict
      - "args": training args, including model name, hidden dim, dropout
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})

    mname = args.get("model", "deepsets")
    hdim = int(args.get("hidden", 64))
    dropout = float(args.get("dropout", 0.0))

    if mname == "deepsets":
        model = build_deepsets(hidden=hdim, dropout=dropout)
    elif mname == "edge_compat":
        model = build_edge_compat(hidden=hdim, dropout=dropout)
    elif mname == "edge_compat_dxy":
        from omtf_gmt.models import build_edge_compat_dxy
        model = build_edge_compat_dxy(hidden=hdim, dropout=dropout)
    elif mname == "edge_compat_assign":
        from omtf_gmt.models.edge_compat_assign import build_edge_compat_assign
        model = build_edge_compat_assign(hidden=hdim, dropout=dropout)
    elif mname == "slot_model":
        model = build_slot_model(hidden=hdim, dropout=dropout)
    elif mname == "seq_slot":
        model = build_seq_slot(hidden=hdim, dropout=dropout)
    elif mname == "count_model":
        model = build_count_model(hidden=hdim, dropout=dropout)
    elif mname == "detr_model":
        model = build_detr_model(hidden=hdim, dropout=dropout)
    else:
        raise ValueError(f"Unknown model type in checkpoint: {mname!r}")

    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    return (
        model,
        mname,
        hdim,
        dropout,
        ckpt.get("epoch", "?"),
        ckpt.get("best_val_loss", float("nan")),
    )


def positive_pt(raw_pt: torch.Tensor) -> torch.Tensor:
    """Return pt_pred unchanged — all model heads already apply F.softplus internally."""
    return raw_pt


# --------------------------------------------------------------------------- #
# Per-dataset validation-split evaluation                                     #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def eval_dataset(
    model,
    cache_dir: Path,
    ds: str,
    threshold: float,
    device: torch.device,
    batch_size: int = 2048,
) -> dict[str, Any]:
    full = GMTCachedDataset(cache_dir, ds)
    n = len(full)
    n_tr = int(0.85 * n)
    _, val = torch.utils.data.random_split(
        full,
        [n_tr, n - n_tr],
        generator=torch.Generator().manual_seed(42),
    )
    loader = DataLoader(
        val,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_gmt,
        num_workers=2,
    )

    # Stub-level accumulators.
    n_sig_stubs = 0
    n_noise_stubs = 0
    n_recall_stubs = 0
    n_fake_stubs = 0

    # Per-slot candidate stats.
    slot_n_true = np.zeros(K_MAX, dtype=np.int64)
    slot_n_rec = np.zeros(K_MAX, dtype=np.int64)
    slot_n_false = np.zeros(K_MAX, dtype=np.int64)
    slot_n_neg = np.zeros(K_MAX, dtype=np.int64)

    # Candidate multiplicity confusion matrix: true multiplicity rows, predicted multiplicity columns.
    # Indices 0..3 because K_MAX=3.
    mult_conf = np.zeros((K_MAX + 1, K_MAX + 1), dtype=np.int64)

    # Efficiency vs pT and d0.
    pt_true: list[float] = []
    pt_rec: list[bool] = []
    d0_true: list[float] = []
    d0_rec: list[bool] = []

    # pT regression residuals on positive slots.
    pt_rel_err: list[float] = []
    pt_log_err: list[float] = []
    pt_abs_err: list[float] = []
    dxy_abs_err: list[float] = []
    dxy_signed_err: list[float] = []

    # Zero-window stats.
    n_zero_windows = 0
    n_zero_fp = 0
    n_zero_fake_cands = 0

    # assignment diagnostics (only populated for edge_compat_assign)
    assign_purity_sum   = np.zeros(K_MAX, dtype=np.float64)
    assign_leakage_sum  = np.zeros(K_MAX, dtype=np.float64)
    assign_noise_sum    = np.zeros(K_MAX, dtype=np.float64)
    assign_n_slots      = np.zeros(K_MAX, dtype=np.int64)

    # Threshold-scan count accumulators.
    roc_counts: dict[float, dict[str, int]] = {
        round(float(thr), 2): {"tp": 0, "fp": 0, "n_pos": 0, "n_neg": 0}
        for thr in THR_SCAN
    }
    b4_scan: dict[float, dict[str, int]] = {
        round(float(thr), 2): {"zero_fp_windows": 0, "zero_windows": 0, "fake_cands": 0}
        for thr in THR_SCAN
    }

    for batch in loader:
        batch = _to_device(batch, device)

        stubs = batch["stubs"]
        vm = batch["valid_mask"]
        nl = batch["node_label"]
        gpt = batch["gen_pt"]
        gd0 = batch["gen_dxy"]

        out = model(stubs, vm)
        node_logit = out["node_logit"]
        cand_logits = out["candidate_logits"]
        pt_pred = positive_pt(out["pt_pred"])
        dxy_pred = out.get("dxy_pred")

        # ---- stub-level recall and fake rate ----
        nl_valid = nl[vm].bool()
        node_logits_valid = node_logit[vm]

        n_sig_stubs += int(nl_valid.sum().item())
        n_noise_stubs += int((~nl_valid).sum().item())
        n_recall_stubs += int(((node_logits_valid > threshold) & nl_valid).sum().item())
        n_fake_stubs += int(((node_logits_valid > threshold) & (~nl_valid)).sum().item())

        # ---- candidate stats ----
        sig_slots = gpt > 0
        fired = cand_logits > threshold

        for k in range(K_MAX):
            pos = sig_slots[:, k]
            neg = ~pos
            slot_n_true[k] += int(pos.sum().item())
            slot_n_rec[k] += int((fired[:, k] & pos).sum().item())
            slot_n_neg[k] += int(neg.sum().item())
            slot_n_false[k] += int((fired[:, k] & neg).sum().item())

        # ---- multiplicity confusion ----
        true_mult = sig_slots.sum(dim=1).clamp(0, K_MAX).cpu().numpy().astype(np.int64)
        pred_mult = fired.sum(dim=1).clamp(0, K_MAX).cpu().numpy().astype(np.int64)
        for t, p in zip(true_mult, pred_mult):
            mult_conf[t, p] += 1

        # ---- pT and d0 efficiency ----
        sig_np = sig_slots.cpu().numpy().astype(bool)
        fired_np = fired.cpu().numpy().astype(bool)
        gpt_np = gpt.cpu().numpy()
        gd0_np = gd0.cpu().numpy()
        pt_pred_np = pt_pred.cpu().numpy()
        dxy_pred_np = dxy_pred.cpu().numpy() if dxy_pred is not None else None

        for k in range(K_MAX):
            mask = sig_np[:, k]
            if not mask.any():
                continue

            pt_true.extend(gpt_np[:, k][mask].tolist())
            pt_rec.extend(fired_np[:, k][mask].tolist())

            d0_true.extend(np.abs(gd0_np[:, k][mask]).tolist())
            d0_rec.extend(fired_np[:, k][mask].tolist())

            tgt = gpt_np[:, k][mask]
            pred = pt_pred_np[:, k][mask]
            safe_tgt = np.maximum(tgt, 1e-6)
            safe_pred = np.maximum(pred, 1e-6)

            pt_abs_err.extend(np.abs(pred - tgt).tolist())
            pt_rel_err.extend(((pred - tgt) / safe_tgt).tolist())
            pt_log_err.extend((np.log1p(safe_pred) - np.log1p(safe_tgt)).tolist())
            if dxy_pred_np is not None:
                dxy_residual = dxy_pred_np[:, k][mask] - gd0_np[:, k][mask]
                dxy_abs_err.extend(np.abs(dxy_residual).tolist())
                dxy_signed_err.extend(dxy_residual.tolist())

        # ---- zero-candidate stats at chosen threshold ----
        zero_win = sig_slots.sum(dim=1) == 0
        n_zero_windows += int(zero_win.sum().item())
        if zero_win.any():
            zero_fired = fired[zero_win]
            n_zero_fp += int(zero_fired.any(dim=1).sum().item())
            n_zero_fake_cands += int(zero_fired.sum().item())

        # ---- assignment diagnostics (edge_compat_assign only) ----
        if "assign_weights" in out:
            aw = out["assign_weights"]          # (B, K, Nmax)
            tid = batch["track_id"]             # (B, Nmax) int
            for k in range(K_MAX):
                active = sig_slots[:, k]        # (B,) bool — slot has a real candidate
                if not active.any():
                    continue
                aw_k = aw[active, k, :]         # (n_active, Nmax)
                tid_a = tid[active]             # (n_active, Nmax)
                vm_a  = vm[active]              # (n_active, Nmax)
                # purity: weight on correct track_id
                purity   = aw_k[tid_a == k + 1].sum().item()
                # leakage: weight on wrong positive track_id
                other_pos = ((tid_a > 0) & (tid_a != k + 1))
                leakage  = aw_k[other_pos].sum().item()
                # noise attention: weight on noise stubs (track_id == 0, valid)
                noise_m  = (tid_a == 0) & vm_a
                noise_at = aw_k[noise_m].sum().item()
                n_act    = int(active.sum().item())
                assign_purity_sum[k]  += purity
                assign_leakage_sum[k] += leakage
                assign_noise_sum[k]   += noise_at
                assign_n_slots[k]     += n_act

        # ---- threshold-scan counts ----
        labels_flat = sig_slots.cpu().numpy().reshape(-1).astype(np.int8)
        logits_flat = cand_logits.cpu().numpy().reshape(-1)
        n_pos = int(labels_flat.sum())
        n_neg = int(len(labels_flat) - n_pos)

        for thr in THR_SCAN:
            tkey = round(float(thr), 2)
            fired_flat = logits_flat > thr
            tp = int((fired_flat & (labels_flat == 1)).sum())
            fp = int((fired_flat & (labels_flat == 0)).sum())
            roc_counts[tkey]["tp"] += tp
            roc_counts[tkey]["fp"] += fp
            roc_counts[tkey]["n_pos"] += n_pos
            roc_counts[tkey]["n_neg"] += n_neg

            if zero_win.any():
                fired_thr = cand_logits[zero_win] > thr
                b4_scan[tkey]["zero_fp_windows"] += int(fired_thr.any(dim=1).sum().item())
                b4_scan[tkey]["zero_windows"] += int(zero_win.sum().item())
                b4_scan[tkey]["fake_cands"] += int(fired_thr.sum().item())

    # ---- Derived summaries ----
    pt_arr = np.array(pt_true, dtype=np.float32)
    pt_rec_arr = np.array(pt_rec, dtype=bool)
    d0_arr = np.array(d0_true, dtype=np.float32)
    d0_rec_arr = np.array(d0_rec, dtype=bool)

    pt_eff = _bin_efficiency(pt_arr, pt_rec_arr, PT_BINS)
    d0_eff = _bin_efficiency(d0_arr, d0_rec_arr, D0_BINS)

    pt_rel_np = np.array(pt_rel_err, dtype=np.float32)
    pt_log_np = np.array(pt_log_err, dtype=np.float32)
    pt_abs_np = np.array(pt_abs_err, dtype=np.float32)
    dxy_abs_np = np.array(dxy_abs_err, dtype=np.float32)
    dxy_signed_np = np.array(dxy_signed_err, dtype=np.float32)

    roc = []
    for thr in sorted(roc_counts):
        c = roc_counts[thr]
        roc.append({
            "threshold": thr,
            "tp": c["tp"],
            "fp": c["fp"],
            "n_pos": c["n_pos"],
            "n_neg": c["n_neg"],
            "efficiency": _safe_div(c["tp"], c["n_pos"]),
            "fake_rate": _safe_div(c["fp"], c["n_neg"]),
        })

    zero_scan = []
    for thr in sorted(b4_scan):
        c = b4_scan[thr]
        zero_scan.append({
            "threshold": thr,
            "zero_fp_rate": _safe_div(c["zero_fp_windows"], c["zero_windows"]),
            "mean_fake_cands_per_zero_window": _safe_div(c["fake_cands"], c["zero_windows"]),
            "zero_windows": c["zero_windows"],
        })

    return {
        "ds": ds,
        "n_val": len(val),
        "threshold": threshold,
        "stub_recall": _safe_div(n_recall_stubs, n_sig_stubs),
        "stub_fake_rate": _safe_div(n_fake_stubs, n_noise_stubs),
        "noise_frac": _safe_div(n_noise_stubs, n_sig_stubs + n_noise_stubs),
        "slot_efficiency": (slot_n_rec / np.maximum(slot_n_true, 1)).tolist(),
        "slot_fake_rate": (slot_n_false / np.maximum(slot_n_neg, 1)).tolist(),
        "slot_n_true": slot_n_true.tolist(),
        "slot_n_rec": slot_n_rec.tolist(),
        "slot_n_neg": slot_n_neg.tolist(),
        "slot_n_false": slot_n_false.tolist(),
        "overall_efficiency": float(slot_n_rec.sum() / max(1, slot_n_true.sum())),
        "overall_fake_rate": float(slot_n_false.sum() / max(1, slot_n_neg.sum())),
        "zero_win_fp_rate": _safe_div(n_zero_fp, n_zero_windows) if n_zero_windows > 0 else None,
        "zero_win_mean_fake_cands": _safe_div(n_zero_fake_cands, n_zero_windows) if n_zero_windows > 0 else None,
        "n_zero_windows": n_zero_windows,
        "multiplicity_confusion": mult_conf.tolist(),
        "pt_efficiency": pt_eff,
        "d0_efficiency": d0_eff,
        "pt_metrics": {
            "n": int(pt_abs_np.size),
            "mae_abs_pt": float(np.mean(pt_abs_np)) if pt_abs_np.size else None,
            "median_rel_err": float(np.median(pt_rel_np)) if pt_rel_np.size else None,
            "sigma68_rel_err": _sigma68(pt_rel_np),
            "mae_log_pt": float(np.mean(np.abs(pt_log_np))) if pt_log_np.size else None,
            "sigma68_log_pt": _sigma68(pt_log_np),
        },
        "dxy_metrics": {
            "n": int(dxy_abs_np.size),
            "mae_cm": float(np.mean(dxy_abs_np)) if dxy_abs_np.size else None,
            "bias_cm": float(np.mean(dxy_signed_np)) if dxy_signed_np.size else None,
        },
        "roc": roc,
        "zero_threshold_scan": zero_scan,
        "assign_purity":  (assign_purity_sum  / np.maximum(assign_n_slots, 1)).tolist(),
        "assign_leakage": (assign_leakage_sum / np.maximum(assign_n_slots, 1)).tolist(),
        "assign_noise":   (assign_noise_sum   / np.maximum(assign_n_slots, 1)).tolist(),
    }


# --------------------------------------------------------------------------- #
# Event-level trigger efficiency                                              #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def eval_event_level(
    model,
    cache_dir: Path,
    ds: str,
    threshold: float,
    device: torch.device,
    batch_size: int = 2048,
) -> dict[str, Any]:
    """
    Compute window-level and event-level trigger efficiency.

    A window triggers if any candidate slot fires.
    An event triggers if any of its processor windows triggers.

    File boundaries are detected by event_num decreasing. This is acceptable for
    Phase B1, but the robust future schema should store meta_file_id explicitly.
    """
    ds_dir = cache_dir / ds
    shards = sorted(ds_dir.glob("shard_*.pt"))
    if not shards:
        return {}

    t0 = time.time()
    all_fires: list[torch.Tensor] = []
    all_gpt_max: list[torch.Tensor] = []
    all_en: list[torch.Tensor] = []
    all_ip: list[torch.Tensor] = []

    required = ["stubs", "valid_mask", "gen_pt", "meta_event_num", "meta_i_proc"]

    for sp in shards:
        shard = torch.load(sp, map_location="cpu", weights_only=False)
        _require_shard_keys(shard, sp, required)

        n_samples = int(shard["stubs"].shape[0])
        fires_list: list[torch.Tensor] = []

        for i in range(0, n_samples, batch_size):
            j = min(i + batch_size, n_samples)
            stubs = shard["stubs"][i:j].to(device)
            vm = shard["valid_mask"][i:j].to(device)
            out = model(stubs, vm)
            fires = (out["candidate_logits"] > threshold).any(dim=-1)
            fires_list.append(fires.cpu())

        all_fires.append(torch.cat(fires_list))
        all_gpt_max.append(shard["gen_pt"].max(dim=-1).values)
        all_en.append(shard["meta_event_num"])
        all_ip.append(shard["meta_i_proc"])

    fires_full = torch.cat(all_fires)
    gpt_full = torch.cat(all_gpt_max)
    en_full = torch.cat(all_en)
    ip_full = torch.cat(all_ip)
    n_total = len(fires_full)

    is_noise = ds in ZERO_CAND_DS

    # ---- Per-window metrics and count payload ----
    win: dict[str, Any] = {"n_windows": n_total}
    count_payload: dict[str, dict[str, int]] = {"window": {}, "event": {}}

    if is_noise:
        bg_pass = int(fires_full.sum().item())
        win["window_bg_accept"] = _safe_div(bg_pass, n_total)
        count_payload["window"]["bg_pass"] = bg_pass
        count_payload["window"]["bg_total"] = int(n_total)
    else:
        for pt in PT_TRIG_THR:
            mask = gpt_full > pt
            total = int(mask.sum().item())
            passed = int((fires_full[mask]).sum().item()) if total else 0
            win[f"window_trig_eff@{pt}"] = _safe_div(passed, total) if total else None
            count_payload["window"][f"pass@{pt}"] = passed
            count_payload["window"][f"total@{pt}"] = total

    # ---- File-boundary detection ----
    en_list = en_full.tolist()
    cuts = [0]
    for i in range(1, len(en_list)):
        if en_list[i] < en_list[i - 1]:
            cuts.append(i)
    cuts.append(len(en_list))
    n_files = len(cuts) - 1

    # ---- Event-level grouping within each file segment ----
    ev_fires_list: list[bool] = []
    ev_gpt_list: list[float] = []
    ev_procs_list: list[int] = []

    for seg in range(n_files):
        s, e = cuts[seg], cuts[seg + 1]
        seg_en = en_full[s:e].tolist()
        seg_fires = fires_full[s:e].tolist()
        seg_gpt = gpt_full[s:e].tolist()
        seg_ip = ip_full[s:e].tolist()

        ev_f: dict[int, bool] = defaultdict(lambda: False)
        ev_g: dict[int, float] = defaultdict(float)
        ev_p: dict[int, set[int]] = defaultdict(set)

        for en, f, g, ip in zip(seg_en, seg_fires, seg_gpt, seg_ip):
            ev_f[int(en)] = bool(ev_f[int(en)] or bool(f))
            ev_g[int(en)] = max(float(ev_g[int(en)]), float(g))
            ev_p[int(en)].add(int(ip))

        for key in ev_f:
            ev_fires_list.append(bool(ev_f[key]))
            ev_gpt_list.append(float(ev_g[key]))
            ev_procs_list.append(len(ev_p[key]))

    n_events = len(ev_fires_list)
    ev_fires_t = torch.tensor(ev_fires_list, dtype=torch.bool)
    ev_gpt_t = torch.tensor(ev_gpt_list, dtype=torch.float)
    mean_procs = sum(ev_procs_list) / max(n_events, 1)

    elapsed = time.time() - t0
    print(
        f"    {n_total:,} windows / {n_files} files → {n_events:,} events "
        f"(mean {mean_procs:.2f} proc/event)  [{elapsed:.0f}s]"
    )

    ev: dict[str, Any] = {
        "n_events": n_events,
        "n_files": n_files,
        "mean_procs_per_event": mean_procs,
    }

    if is_noise:
        bg_pass = int(ev_fires_t.sum().item())
        ev["event_bg_accept"] = _safe_div(bg_pass, n_events)
        count_payload["event"]["bg_pass"] = bg_pass
        count_payload["event"]["bg_total"] = int(n_events)
    else:
        for pt in PT_TRIG_THR:
            mask = ev_gpt_t > pt
            total = int(mask.sum().item())
            passed = int(ev_fires_t[mask].sum().item()) if total else 0
            ev[f"event_trig_eff@{pt}"] = _safe_div(passed, total) if total else None
            count_payload["event"][f"pass@{pt}"] = passed
            count_payload["event"][f"total@{pt}"] = total

    return {**win, **ev, "counts": count_payload}


# --------------------------------------------------------------------------- #
# Report rendering                                                            #
# --------------------------------------------------------------------------- #

def _combine_bin_efficiencies(results: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    combined: dict[tuple[float, float | None], list[int]] = {}
    for r in results:
        for b in r[key]:
            k = (b["lo"], b["hi"])
            if k not in combined:
                combined[k] = [0, 0]
            combined[k][0] += int(b["n"])
            combined[k][1] += int(b["n_rec"])

    out: list[dict[str, Any]] = []
    for (lo, hi), (n, nrec) in combined.items():
        out.append({
            "lo": lo,
            "hi": hi,
            "n": n,
            "n_rec": nrec,
            "efficiency": _safe_div(nrec, n) if n else None,
        })
    return out


def _combine_roc_counts(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    combined: dict[float, dict[str, int]] = {}
    for r in results:
        for point in r["roc"]:
            thr = float(point["threshold"])
            if thr not in combined:
                combined[thr] = {"tp": 0, "fp": 0, "n_pos": 0, "n_neg": 0}
            combined[thr]["tp"] += int(point["tp"])
            combined[thr]["fp"] += int(point["fp"])
            combined[thr]["n_pos"] += int(point["n_pos"])
            combined[thr]["n_neg"] += int(point["n_neg"])

    out: list[dict[str, Any]] = []
    for thr in sorted(combined):
        c = combined[thr]
        out.append({
            "threshold": thr,
            "tp": c["tp"],
            "fp": c["fp"],
            "n_pos": c["n_pos"],
            "n_neg": c["n_neg"],
            "efficiency": _safe_div(c["tp"], c["n_pos"]),
            "fake_rate": _safe_div(c["fp"], c["n_neg"]),
        })
    return out


def _combine_event_counts(ev_results: dict[str, dict[str, Any]], pt: int) -> tuple[float | None, float | None]:
    win_pass = win_total = ev_pass = ev_total = 0
    for ds, ev in ev_results.items():
        if ds in ZERO_CAND_DS:
            continue
        counts = ev.get("counts", {})
        wc = counts.get("window", {})
        ec = counts.get("event", {})
        win_pass += int(wc.get(f"pass@{pt}", 0))
        win_total += int(wc.get(f"total@{pt}", 0))
        ev_pass += int(ec.get(f"pass@{pt}", 0))
        ev_total += int(ec.get(f"total@{pt}", 0))

    win_eff = _safe_div(win_pass, win_total) if win_total else None
    ev_eff = _safe_div(ev_pass, ev_total) if ev_total else None
    return win_eff, ev_eff


def render_report(
    results: list[dict[str, Any]],
    ev_results: dict[str, dict[str, Any]],
    model_name: str,
    hdim: int,
    dropout: float,
    ckpt_epoch: int | str,
    best_loss: float,
    threshold: float,
    ckpt_path: str,
) -> str:
    lines: list[str] = [
        "# GMT Model Evaluation\n",
        f"Model: `{model_name}` hidden={hdim} dropout={dropout}  |  "
        f"Checkpoint: `{ckpt_path}`  |  "
        f"Best epoch: {ckpt_epoch}  best_val_loss: {best_loss:.4f}\n",
        f"Threshold: logit > {threshold:.2f}\n",
        "**Scope:** Phase B1 cache-level evaluation. This is not yet a full absolute L1 trigger-rate study.\n",
        "---\n",
    ]

    # ---- Overall table ----
    lines += ["## Overall validation-split metrics\n"]
    lines += [
        "| Dataset | Val samples | Stub recall | Stub fake | Cand eff | Neg-slot fake | Zero-win FP | Mean fake cands / zero win |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        zfp = _format_opt_float(r["zero_win_fp_rate"])
        zfc = _format_opt_float(r["zero_win_mean_fake_cands"])
        lines.append(
            f"| {r['ds']} | {r['n_val']:,} "
            f"| {r['stub_recall']:.3f} "
            f"| {r['stub_fake_rate']:.3f} "
            f"| {r['overall_efficiency']:.3f} "
            f"| {r['overall_fake_rate']:.3f} "
            f"| {zfp} | {zfc} |"
        )
    lines.append("")

    # ---- Per-slot table ----
    lines += ["## Per-slot efficiency and negative-slot fake rate\n"]
    lines += [
        "| Dataset | Eff slot 0 | Eff slot 1 | Eff slot 2 | Fake slot 0 | Fake slot 1 | Fake slot 2 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        eff = r["slot_efficiency"]
        fake = r["slot_fake_rate"]
        eff_str = " | ".join(
            f"{eff[k]:.3f} (n={r['slot_n_true'][k]:,})" for k in range(K_MAX)
        )
        fake_str = " | ".join(
            f"{fake[k]:.3f} (n={r['slot_n_neg'][k]:,})" for k in range(K_MAX)
        )
        lines.append(f"| {r['ds']} | {eff_str} | {fake_str} |")
    lines.append("")

    # ---- Candidate multiplicity confusion ----
    lines += ["## Candidate multiplicity confusion matrices\n"]
    lines += [
        "Rows are true candidate multiplicity. Columns are predicted candidate multiplicity at the chosen threshold.\n"
    ]
    for r in results:
        mat = np.array(r["multiplicity_confusion"], dtype=np.int64)
        lines += [f"### {r['ds']}\n"]
        lines += ["| true \\ pred | 0 | 1 | 2 | 3 |", "| --- | ---: | ---: | ---: | ---: |"]
        for i in range(K_MAX + 1):
            lines.append(f"| {i} | " + " | ".join(f"{int(mat[i, j]):,}" for j in range(K_MAX + 1)) + " |")
        lines.append("")

    # ---- pT regression metrics ----
    lines += ["## pT regression metrics\n"]
    lines += [
        "| Dataset | N signal slots | MAE pT [GeV] | Median rel err | sigma68 rel err | MAE log(1+pT) | sigma68 log |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        pm = r["pt_metrics"]
        lines.append(
            f"| {r['ds']} | {pm['n']:,} "
            f"| {_format_opt_float(pm['mae_abs_pt'])} "
            f"| {_format_opt_float(pm['median_rel_err'])} "
            f"| {_format_opt_float(pm['sigma68_rel_err'])} "
            f"| {_format_opt_float(pm['mae_log_pt'])} "
            f"| {_format_opt_float(pm['sigma68_log_pt'])} |"
        )
    lines.append("")

    # ---- dxy regression metrics ----
    lines += ["## Signed dxy regression metrics\n"]
    lines += [
        "| Dataset | N signal slots | MAE dxy [cm] | Mean residual [cm] |",
        "| --- | ---: | ---: | ---: |",
    ]
    for r in results:
        dm = r["dxy_metrics"]
        lines.append(
            f"| {r['ds']} | {dm['n']:,} "
            f"| {_format_opt_float(dm['mae_cm'])} "
            f"| {_format_opt_float(dm['bias_cm'])} |"
        )
    lines.append("")

    # ---- pT efficiency ----
    lines += ["## Efficiency vs gen-muon pT  (all signal slots, all datasets pooled by counts)\n"]
    lines += ["| pT bin [GeV] | N true | N rec | Efficiency |", "| --- | ---: | ---: | ---: |"]
    for b in _combine_bin_efficiencies(results, "pt_efficiency"):
        hi_str = f"{b['hi']}" if b["hi"] is not None else "∞"
        eff_s = _format_opt_float(b["efficiency"])
        lines.append(f"| [{b['lo']}, {hi_str}) | {b['n']:,} | {b['n_rec']:,} | {eff_s} |")
    lines.append("")

    # ---- d0 efficiency ----
    lines += ["## Efficiency vs gen-muon |d0|  (all signal slots, all datasets pooled by counts)\n"]
    lines += ["| |d0| bin [cm] | N true | N rec | Efficiency |", "| --- | ---: | ---: | ---: |"]
    for b in _combine_bin_efficiencies(results, "d0_efficiency"):
        hi_str = f"{b['hi']}" if b["hi"] is not None else "∞"
        eff_s = _format_opt_float(b["efficiency"])
        lines.append(f"| [{b['lo']}, {hi_str}) | {b['n']:,} | {b['n_rec']:,} | {eff_s} |")
    lines.append("")

    # ---- Count-based threshold scan ----
    lines += ["## Threshold scan  (slot-level ROC, all datasets pooled by counts)\n"]
    lines += ["| Threshold | Efficiency | Negative-slot fake rate | TP | FP | N pos | N neg |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    shown_thrs = {-2.0, -1.0, 0.0, 1.0, 2.0}
    for point in _combine_roc_counts(results):
        t = round(float(point["threshold"]), 1)
        if t not in shown_thrs:
            continue
        lines.append(
            f"| {point['threshold']:.1f} | {point['efficiency']:.3f} | {point['fake_rate']:.3f} "
            f"| {point['tp']:,} | {point['fp']:,} | {point['n_pos']:,} | {point['n_neg']:,} |"
        )
    lines.append("")

    # ---- Zero-candidate threshold scan ----
    lines += ["## Zero-candidate threshold scan  (B4 / zero-window diagnostics)\n"]
    lines += ["| Dataset | Threshold | Zero-window FP | Mean fake cands / zero window | Zero windows |", "| --- | ---: | ---: | ---: | ---: |"]
    for r in results:
        if r["n_zero_windows"] <= 0:
            continue
        for point in r["zero_threshold_scan"]:
            t = round(float(point["threshold"]), 1)
            if t not in shown_thrs:
                continue
            lines.append(
                f"| {r['ds']} | {point['threshold']:.1f} | {point['zero_fp_rate']:.3f} "
                f"| {point['mean_fake_cands_per_zero_window']:.3f} | {point['zero_windows']:,} |"
            )
    lines.append("")

    # ---- Event-level trigger efficiency ----
    if ev_results:
        lines += ["## Event-level trigger efficiency\n"]
        lines += [
            "Window = any slot fires. Event = OR over all processor windows for that CMS event. "
            "File-boundary grouping is based on event-number reset.\n"
        ]
        lines += ["| Dataset | win@10 | evt@10 | gain | procs/evt | n_events | note |", "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
        for ds, ev in ev_results.items():
            mp = ev.get("mean_procs_per_event", 0.0)
            ne = ev.get("n_events", 0)
            if ds in ZERO_CAND_DS:
                wbg = ev.get("window_bg_accept")
                ebg = ev.get("event_bg_accept")
                lines.append(
                    f"| {ds} | — | — | — | {mp:.2f} | {ne:,} | "
                    f"bg: win={_format_opt_float(wbg)} evt={_format_opt_float(ebg)} |"
                )
            else:
                w = ev.get("window_trig_eff@10")
                e = ev.get("event_trig_eff@10")
                gain = (e - w) if (w is not None and e is not None) else None
                lines.append(
                    f"| {ds} | {_format_opt_float(w, 4)} | {_format_opt_float(e, 4)} | {_format_opt_float(gain, 4)} "
                    f"| {mp:.2f} | {ne:,} | |"
                )
        lines.append("")

        lines += ["### Event-level trig_eff vs pT threshold  (all signal datasets pooled by counts)\n"]
        lines += ["| pT cut [GeV] | win_eff | evt_eff | gain |", "| ---: | ---: | ---: | ---: |"]
        for pt in PT_TRIG_THR:
            win_eff, ev_eff = _combine_event_counts(ev_results, pt)
            gain = (ev_eff - win_eff) if (win_eff is not None and ev_eff is not None) else None
            lines.append(
                f"| > {pt} | {_format_opt_float(win_eff, 4)} | {_format_opt_float(ev_eff, 4)} | {_format_opt_float(gain, 4)} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="GMT model evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("build/omtf_gmt/cache"))
    parser.add_argument("--datasets", nargs="+", default=ALL_DS)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("build/omtf_gmt/eval/eval.md"))
    args = parser.parse_args()

    device = torch.device(args.device)
    model, model_name, hidden_dim, dropout, epoch, best_loss = load_model(args.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"Model: {model_name} hidden={hidden_dim} dropout={dropout} "
        f"params={n_params:,} epoch={epoch} best_val_loss={best_loss:.4f}"
    )

    # Expand logical aliases (G1 → G1_pos, G1_neg) to physical cache names.
    datasets = expand_datasets(args.datasets)

    # Pass 1: validation-split metrics.
    results: list[dict[str, Any]] = []
    for ds in datasets:
        if not (args.cache_dir / ds).exists():
            print(f"  [{ds}] not in cache — skipping")
            continue
        print(f"  Evaluating {ds} ...", end="", flush=True)
        r = eval_dataset(model, args.cache_dir, ds, args.threshold, device, args.batch_size)
        results.append(r)
        print(
            f" n_val={r['n_val']:,} eff={r['overall_efficiency']:.3f} "
            f"neg_fake={r['overall_fake_rate']:.3f} stub_fake={r['stub_fake_rate']:.3f}"
            + (f" zero_fp={r['zero_win_fp_rate']:.3f}" if r["zero_win_fp_rate"] is not None else "")
        )

    if not results:
        print("No datasets found.")
        return

    # Pass 2: full-dataset event-level metrics.
    print("\n  Event-level eval (full dataset, shard order):")
    ev_results: dict[str, dict[str, Any]] = {}
    for ds in datasets:
        if not (args.cache_dir / ds).exists():
            continue
        print(f"  {ds} ...", end="", flush=True)
        ev_results[ds] = eval_event_level(
            model,
            args.cache_dir,
            ds,
            args.threshold,
            device,
            args.batch_size,
        )

    report = render_report(
        results=results,
        ev_results=ev_results,
        model_name=model_name,
        hdim=hidden_dim,
        dropout=dropout,
        ckpt_epoch=epoch,
        best_loss=best_loss,
        threshold=args.threshold,
        ckpt_path=str(args.checkpoint),
    )

    print("\n" + report)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)

    out_json = {"per_window": results, "event_level": ev_results}
    json_path = args.output.with_suffix(".json")
    json_path.write_text(json.dumps(out_json, indent=2))

    print(f"Report: {args.output}")
    print(f"Data:   {json_path}")


if __name__ == "__main__":
    main()
