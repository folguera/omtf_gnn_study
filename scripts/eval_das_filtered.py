#!/usr/bin/env python
"""
DAS external validation with OMTF overlap eta filter.

The unfiltered eval_gmt.py counts all windows as signal denominator, including
windows where the gen muon is outside the OMTF overlap band (0.82–1.24), which
the model is not supposed to trigger on.  This script applies an overlap filter:
a window is in the efficiency denominator only if it has at least one signal
stub (node_label > 0.5) with stub_in_overlap = 1 (feature index 11).

Metrics reported:
  - Overlap efficiency  : fires / (true target AND overlap signal)
  - Out-of-overlap FP   : fires / (true target AND no overlap signal)
  - Zero-target FP rate : fires / (no true target)  — background acceptance
  - pT efficiency bins  (overlap windows only)
  - Event-level efficiency after OR over processor windows (eta-filtered)
  - Threshold scan

Usage:
  python scripts/omtf_gmt/eval_das_filtered.py \\
      --checkpoint build/omtf_gmt/checkpoints/frozen_fp32_tps_edgecompat_h64/model.pt \\
      --cache-dir  build/omtf_gmt/cache_das_tps \\
      --datasets   single_muon_flatpt displaced_lowpt displaced_midpt \\
                   dy_prompt llp_addon minbias \\
      --output     build/omtf_gmt/eval/frozen_fp32_das_filtered_eval.md
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
from torch.utils.data import DataLoader, random_split

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from omtf_gmt.dataset import GMTCachedDataset, collate_gmt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEAT_STUB_IN_OVERLAP = 11   # 1.0 if 0.83 ≤ |eta| ≤ 1.24
FEAT_ABS_ETA         = 12   # |offlineEta1|

PT_BINS     = [0, 2, 5, 10, 15, 20, 30, 50, 100, 9999]
PT_TRIG_THR = [0, 5, 10, 15, 20]
THR_SCAN    = np.linspace(-3.0, 3.0, 61)
K_MAX       = 3

BACKGROUND_DS = {"minbias"}   # pure-noise datasets — no efficiency, only FP rate


# ---------------------------------------------------------------------------
# Model loading (mirrors eval_gmt.py)
# ---------------------------------------------------------------------------

def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    mname  = args.get("model", "edge_compat")
    hdim   = int(args.get("hidden", 64))
    drop   = float(args.get("dropout", 0.0))

    if mname == "edge_compat":
        from omtf_gmt.models import build_edge_compat
        model = build_edge_compat(hidden=hdim, dropout=drop)
    elif mname == "edge_compat_dxy":
        from omtf_gmt.models import build_edge_compat_dxy
        model = build_edge_compat_dxy(hidden=hdim, dropout=drop)
    elif mname == "edge_compat_assign":
        from omtf_gmt.models.edge_compat_assign import build_edge_compat_assign
        model = build_edge_compat_assign(hidden=hdim, dropout=drop)
    elif mname == "deepsets":
        from omtf_gmt.models import build_deepsets
        model = build_deepsets(hidden=hdim, dropout=drop)
    else:
        from omtf_gmt.models import build_edge_compat
        model = build_edge_compat(hidden=hdim, dropout=drop)

    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, mname, hdim, ckpt.get("epoch", "?"), ckpt.get("best_val_loss", float("nan"))


# ---------------------------------------------------------------------------
# Overlap filter
# ---------------------------------------------------------------------------

def overlap_signal_mask(stubs: torch.Tensor,
                        node_label: torch.Tensor,
                        valid_mask: torch.Tensor) -> torch.Tensor:
    """
    Return a (B,) bool tensor: True if the window has at least one
    signal stub (node_label > 0.5, valid) with stub_in_overlap == 1.
    Uses feature index 11 (stub_in_overlap).
    """
    sig   = (node_label > 0.5) & valid_mask                # (B, Nmax)
    in_ov = stubs[:, :, FEAT_STUB_IN_OVERLAP] > 0.5        # (B, Nmax)
    return (sig & in_ov).any(dim=1)                        # (B,)


# ---------------------------------------------------------------------------
# Per-dataset validation-split evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_dataset_filtered(
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
    _, val = random_split(full, [n_tr, n - n_tr],
                          generator=torch.Generator().manual_seed(42))
    loader = DataLoader(val, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_gmt, num_workers=2)

    is_bg = ds in BACKGROUND_DS

    # Window-level counters
    n_total          = 0
    n_ov_sig         = 0   # true target AND overlap signal
    n_ov_sig_fires   = 0   # model fires on overlap signal windows
    n_out_sig        = 0   # true target AND no overlap signal (barrel/endcap muon)
    n_out_sig_fires  = 0
    n_no_tgt         = 0   # no true target
    n_no_tgt_fires   = 0   # false positive on no-target windows

    # pT efficiency (overlap-filtered)
    pt_bins = PT_BINS
    pt_true  = [0] * (len(pt_bins) - 1)
    pt_rec   = [0] * (len(pt_bins) - 1)

    # Threshold scan
    thr_counts: dict[float, dict[str, int]] = {
        round(float(t), 2): {"ov_tp": 0, "ov_fp": 0, "ov_n_pos": 0, "ov_n_neg": 0,
                              "bg_fp": 0, "bg_total": 0}
        for t in THR_SCAN
    }

    for batch in loader:
        stubs      = batch["stubs"].to(device)
        vm         = batch["valid_mask"].to(device)
        nl         = batch["node_label"].to(device)
        gpt        = batch["gen_pt"].to(device)

        out        = model(stubs, vm)
        logits     = out["candidate_logits"]          # (B, K)
        fires      = (logits > threshold).any(dim=-1) # (B,)
        max_logit  = logits.max(dim=-1).values         # (B,)

        has_tgt    = (gpt.max(dim=-1).values > 0)     # (B,)
        ov_sig     = overlap_signal_mask(stubs, nl, vm)  # (B,)

        # Categorise windows
        ov_sig_w   = ov_sig & has_tgt
        out_sig_w  = (~ov_sig) & has_tgt
        no_tgt_w   = ~has_tgt

        b = stubs.shape[0]
        n_total         += b
        n_ov_sig        += int(ov_sig_w.sum())
        n_ov_sig_fires  += int((fires & ov_sig_w).sum())
        n_out_sig       += int(out_sig_w.sum())
        n_out_sig_fires += int((fires & out_sig_w).sum())
        n_no_tgt        += int(no_tgt_w.sum())
        n_no_tgt_fires  += int((fires & no_tgt_w).sum())

        # pT efficiency (overlap signal windows only)
        if not is_bg:
            gpt_max = gpt.max(dim=-1).values.cpu().numpy()
            fires_np = fires.cpu().numpy()
            ov_np    = ov_sig_w.cpu().numpy()
            for bi_lo, bi_hi, pt_lo, pt_hi in zip(
                range(len(pt_bins) - 1), range(1, len(pt_bins)),
                pt_bins[:-1], pt_bins[1:]
            ):
                mask = ov_np & (gpt_max >= pt_lo) & (gpt_max < pt_hi)
                pt_true[bi_lo] += int(mask.sum())
                pt_rec[bi_lo]  += int((mask & fires_np).sum())

        # Threshold scan
        max_logit_np = max_logit.cpu().numpy()
        ov_sig_np    = ov_sig_w.cpu().numpy()
        out_sig_np   = out_sig_w.cpu().numpy()
        no_tgt_np    = no_tgt_w.cpu().numpy()

        for thr_k, counts in thr_counts.items():
            fire_t = max_logit_np > thr_k
            counts["ov_tp"]    += int((fire_t & ov_sig_np).sum())
            counts["ov_fp"]    += int((fire_t & (~ov_sig_np)).sum())
            counts["ov_n_pos"] += int(ov_sig_np.sum())
            counts["ov_n_neg"] += int((~ov_sig_np).sum())
            counts["bg_fp"]    += int((fire_t & no_tgt_np).sum())
            counts["bg_total"] += int(no_tgt_np.sum())

    def _eff(n, d): return n / d if d > 0 else None
    def _fmt(v): return round(v, 6) if v is not None else None

    pt_eff = []
    for i, (lo, hi) in enumerate(zip(pt_bins[:-1], pt_bins[1:])):
        n, r = pt_true[i], pt_rec[i]
        pt_eff.append({"lo": lo, "hi": hi if hi < 9000 else None,
                       "n": n, "n_rec": r,
                       "efficiency": _fmt(_eff(r, n))})

    thr_scan_out = []
    for thr_k, c in sorted(thr_counts.items()):
        thr_scan_out.append({
            "threshold": thr_k,
            "ov_eff":    _fmt(_eff(c["ov_tp"], c["ov_n_pos"])),
            "fp_rate":   _fmt(_eff(c["ov_fp"], c["ov_n_neg"])),
            "bg_fp":     _fmt(_eff(c["bg_fp"], c["bg_total"])),
        })

    return {
        "n_total":          n_total,
        "n_ov_sig":         n_ov_sig,
        "n_out_sig":        n_out_sig,
        "n_no_tgt":         n_no_tgt,
        "ov_efficiency":    _fmt(_eff(n_ov_sig_fires, n_ov_sig)),
        "out_ov_fp_rate":   _fmt(_eff(n_out_sig_fires, n_out_sig)),
        "zero_tgt_fp_rate": _fmt(_eff(n_no_tgt_fires, n_no_tgt)),
        "pt_efficiency":    pt_eff,
        "threshold_scan":   thr_scan_out,
    }


# ---------------------------------------------------------------------------
# Event-level (eta-filtered)
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_event_level_filtered(
    model,
    cache_dir: Path,
    ds: str,
    threshold: float,
    device: torch.device,
    batch_size: int = 2048,
) -> dict[str, Any]:
    """
    Event-level trigger efficiency with overlap eta filter.

    An event is in the signal denominator if at least one of its
    processor windows has an overlap signal stub.
    """
    ds_dir = cache_dir / ds
    shards = sorted(ds_dir.glob("shard_*.pt"))
    if not shards:
        return {}

    is_bg = ds in BACKGROUND_DS

    all_fires:   list[torch.Tensor] = []
    all_ov_sig:  list[torch.Tensor] = []
    all_has_tgt: list[torch.Tensor] = []
    all_gpt_max: list[torch.Tensor] = []
    all_en:      list[torch.Tensor] = []

    for sp in shards:
        shard = torch.load(sp, map_location="cpu", weights_only=False)
        n = int(shard["stubs"].shape[0])
        fires_l, ov_l, tgt_l = [], [], []

        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            stubs = shard["stubs"][i:j].to(device)
            vm    = shard["valid_mask"][i:j].to(device)
            nl    = shard["node_label"][i:j].to(device)
            gpt_b = shard["gen_pt"][i:j].to(device)

            out    = model(stubs, vm)
            fires  = (out["candidate_logits"] > threshold).any(dim=-1).cpu()
            ov_sig = overlap_signal_mask(stubs, nl, vm).cpu()
            has_t  = (gpt_b.max(dim=-1).values > 0).cpu()

            fires_l.append(fires); ov_l.append(ov_sig); tgt_l.append(has_t)

        all_fires.append(torch.cat(fires_l))
        all_ov_sig.append(torch.cat(ov_l))
        all_has_tgt.append(torch.cat(tgt_l))
        all_gpt_max.append(shard["gen_pt"].max(dim=-1).values)
        all_en.append(shard["meta_event_num"])

    fires_full   = torch.cat(all_fires).numpy()
    ov_full      = torch.cat(all_ov_sig).numpy()
    has_tgt_full = torch.cat(all_has_tgt).numpy()
    gpt_full     = torch.cat(all_gpt_max).numpy()
    en_full      = torch.cat(all_en).numpy().astype(np.int64)

    # Group by event (file boundaries detected by event_num decreasing)
    event_dict: dict[tuple, dict] = defaultdict(
        lambda: {"fires": False, "ov_sig": False, "has_tgt": False, "gpt_max": 0.0}
    )
    file_id = 0
    prev_en = en_full[0]
    for i in range(len(en_full)):
        en = int(en_full[i])
        if en < prev_en:
            file_id += 1
        prev_en = en
        key = (file_id, en)
        ev  = event_dict[key]
        ev["fires"]   = ev["fires"]   or bool(fires_full[i])
        ev["ov_sig"]  = ev["ov_sig"]  or bool(ov_full[i])
        ev["has_tgt"] = ev["has_tgt"] or bool(has_tgt_full[i])
        ev["gpt_max"] = max(ev["gpt_max"], float(gpt_full[i]))

    n_events      = len(event_dict)
    evs           = list(event_dict.values())

    # Filtered efficiency: fires & ov_sig & has_tgt / ov_sig & has_tgt
    sig_denom = sum(1 for e in evs if e["ov_sig"] and e["has_tgt"])
    sig_num   = sum(1 for e in evs if e["fires"] and e["ov_sig"] and e["has_tgt"])

    # Background accept: fires & !has_tgt / !has_tgt  (or full event for bg datasets)
    if is_bg:
        bg_denom = n_events
        bg_num   = sum(1 for e in evs if e["fires"])
    else:
        bg_denom = sum(1 for e in evs if not e["has_tgt"])
        bg_num   = sum(1 for e in evs if e["fires"] and not e["has_tgt"])

    # pT-threshold scan (signal events with overlap signal)
    trig_eff: dict[int, dict] = {}
    for pt_cut in PT_TRIG_THR:
        denom = sum(1 for e in evs if e["ov_sig"] and e["has_tgt"] and e["gpt_max"] >= pt_cut)
        numer = sum(1 for e in evs if e["fires"] and e["ov_sig"] and e["has_tgt"] and e["gpt_max"] >= pt_cut)
        trig_eff[pt_cut] = {"n_denom": denom, "n_numer": numer,
                             "eff": round(numer / denom, 6) if denom else None}

    def _r(v): return round(v, 6) if v is not None else None
    return {
        "n_events":          n_events,
        "n_sig_denom":       sig_denom,
        "event_ov_eff":      _r(sig_num / sig_denom if sig_denom else None),
        "event_bg_accept":   _r(bg_num  / bg_denom  if bg_denom  else None),
        "trig_eff_vs_pt":    trig_eff,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _f(v, fmt=".3f"):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return format(v, fmt)


def render_report(results: dict, ev_results: dict, model_info: str, threshold: float) -> str:
    lines = [
        "# DAS External Validation — Overlap-Filtered",
        "",
        f"Model: {model_info}",
        f"Threshold: logit > {threshold:.2f}",
        "",
        "> **Eta filter:** a window/event enters the signal denominator only if it",
        "> has at least one signal stub with `stub_in_overlap = 1` (0.83 ≤ |η| ≤ 1.24).",
        "> This removes out-of-acceptance muons from the efficiency calculation.",
        "",
        "---",
        "",
        "## Window-level summary (validation split)",
        "",
        "| Dataset | Total | Overlap-sig | Out-ovlp-sig | No-target |"
        " Overlap eff | Out-ovlp FP | Zero-tgt FP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for ds, r in results.items():
        lines.append(
            f"| {ds} | {r['n_total']:,} | {r['n_ov_sig']:,} | {r['n_out_sig']:,}"
            f" | {r['n_no_tgt']:,}"
            f" | {_f(r['ov_efficiency'])}"
            f" | {_f(r['out_ov_fp_rate'])}"
            f" | {_f(r['zero_tgt_fp_rate'])} |"
        )

    lines += ["", "---", "", "## pT efficiency (overlap-filtered windows)", ""]
    pt_bins = PT_BINS
    hdr = "| pT bin [GeV] |" + "".join(f" {ds} |" for ds in results if ds not in BACKGROUND_DS)
    sep = "| --- |" + " ---: |" * len([d for d in results if d not in BACKGROUND_DS])
    lines += [hdr, sep]
    for i, (lo, hi) in enumerate(zip(pt_bins[:-1], pt_bins[1:])):
        hi_str = "∞" if hi > 9000 else str(hi)
        row = f"| [{lo}, {hi_str}) |"
        for ds, r in results.items():
            if ds in BACKGROUND_DS:
                continue
            entry = r["pt_efficiency"][i] if i < len(r["pt_efficiency"]) else {}
            eff = entry.get("efficiency")
            n   = entry.get("n", 0)
            row += f" {_f(eff)} (n={n:,}) |"
        lines.append(row)

    lines += ["", "---", "", "## Event-level efficiency (overlap-filtered)", ""]
    lines += [
        "| Dataset | Events | Sig denom (η-filtered) | Overlap evt-eff"
        " | bg accept | pT>0 | pT>5 | pT>10 | pT>15 | pT>20 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for ds, ev in ev_results.items():
        te = ev.get("trig_eff_vs_pt", {})
        row = (
            f"| {ds} | {ev['n_events']:,} | {ev['n_sig_denom']:,}"
            f" | {_f(ev.get('event_ov_eff'))}"
            f" | {_f(ev.get('event_bg_accept'))}"
            + "".join(f" | {_f(te.get(pt, {}).get('eff'))}" for pt in PT_TRIG_THR)
            + " |"
        )
        lines.append(row)

    lines += ["", "---", "", "## Threshold scan (overlap-filtered, @0.0 operating point)",
              "",
              "| Dataset | Thr | Overlap eff | FP rate | Zero-tgt FP |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for ds, r in results.items():
        for entry in r.get("threshold_scan", []):
            t = entry["threshold"]
            if abs(t) < 0.01 or abs(t - 1.0) < 0.01 or abs(t + 1.0) < 0.01 or abs(t - 2.0) < 0.01 or abs(t + 2.0) < 0.01:
                lines.append(
                    f"| {ds} | {t:.1f}"
                    f" | {_f(entry.get('ov_eff'))}"
                    f" | {_f(entry.get('fp_rate'))}"
                    f" | {_f(entry.get('bg_fp'))} |"
                )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="DAS eval with overlap eta filter")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cache-dir",  type=Path, default=Path("build/omtf_gmt/cache_das_tps"))
    p.add_argument("--datasets",   nargs="+",
                   default=["single_muon_flatpt", "displaced_lowpt", "displaced_midpt",
                             "dy_prompt", "llp_addon", "minbias"])
    p.add_argument("--threshold",  type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output",     type=Path,
                   default=Path("build/omtf_gmt/eval/frozen_fp32_das_filtered_eval.md"))
    args = p.parse_args()

    device = torch.device(args.device)
    model, mname, hdim, epoch, best_loss = load_model(args.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    model_info = (f"`{mname}` hidden={hdim}  epoch={epoch}"
                  f"  best_val_loss={best_loss:.4f}  params={n_params:,}")
    print(f"Model: {model_info}")

    val_results: dict[str, dict] = {}
    ev_results:  dict[str, dict] = {}

    for ds in args.datasets:
        print(f"  [{ds}] val-split ...", end="", flush=True)
        t0 = time.time()
        val_results[ds] = eval_dataset_filtered(
            model, args.cache_dir, ds, args.threshold, device, args.batch_size
        )
        r = val_results[ds]
        print(f" ov_eff={_f(r['ov_efficiency'])}  zero_fp={_f(r['zero_tgt_fp_rate'])}  ({time.time()-t0:.0f}s)")

        print(f"  [{ds}] event-level ...", end="", flush=True)
        t0 = time.time()
        ev_results[ds] = eval_event_level_filtered(
            model, args.cache_dir, ds, args.threshold, device, args.batch_size
        )
        ev = ev_results[ds]
        print(f" evt_ov_eff={_f(ev.get('event_ov_eff'))}  ({time.time()-t0:.0f}s)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    md = render_report(val_results, ev_results, model_info, args.threshold)
    args.output.write_text(md)

    json_path = args.output.with_suffix(".json")
    json_path.write_text(json.dumps({"per_window": val_results, "event_level": ev_results}, indent=2))

    print(f"\nReport: {args.output}")
    print(f"Data:   {json_path}")


if __name__ == "__main__":
    main()
