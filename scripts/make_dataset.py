#!/usr/bin/env python
"""
GMT-visible-stub dataset builder — TPS source (Phase B4).

Parallel to make_gmt_dataset.py (KMTF, Phase B1/B2), but uses
MuonStubTps stubs from NanoAOD instead of MuonStubKmtf.

For each entry in OMTFAllInputTree (one OMTF processor window per event):
  1. Collect all TPS stubs from NanoAOD whose offlineCoord1 falls inside
     the processor's 120° phi window.
  2. Transfer truth labels (trackId, ambiguous) from OMTF stubs via
     phi+BX matching (same strategy as KMTF, station check relaxed).
  3. Attach gen-muon kinematics from NanoAOD GenMuon collection.
  4. Pad/truncate to Nmax stubs and write sharded .pt output.

Output schema — identical to KMTF cache (reusable with train.py / eval_gmt.py):
  stubs            (N, Nmax, 14) float32  — per-stub node features (TPS schema)
  valid_mask       (N, Nmax)     bool
  track_id         (N, Nmax)     int8
  ambiguous        (N, Nmax)     uint8
  node_label       (N, Nmax)     float32  — 1 if stub belongs to an overlap target
  truth_source     (N, Nmax)     int8     — 2=omtf_transfer, 1=omtf_noise, 0=unmatched
  gen_pt           (N, K)        float32
  gen_charge       (N, K)        float32
  gen_dxy          (N, K)        float32
  target_track_id  (N, K)        int8
  meta_event_num   (N,)          int32
  meta_i_proc      (N,)          int32
  meta_n_stubs     (N,)          int32
  meta_n_gen       (N,)          int32
  meta_is_hard_neg (N,)          int8

Feature schema: see src/omtf_gmt/features_tps.py (14 features, same dim as KMTF).
Do NOT mix KMTF and TPS checkpoints — feature indices 7–10 differ.

Usage
-----
  python scripts/omtf_gmt/make_gmt_dataset_tps.py \\
      --datasets G1_pos G1_neg G7 G8 B4 \\
      --data-dir data/prod \\
      --output-dir build/omtf_gmt/cache_v2_tps \\
      --max-files 10

  # All datasets
  python scripts/omtf_gmt/make_gmt_dataset_tps.py \\
      --data-dir data/prod \\
      --output-dir build/omtf_gmt/cache_v2_tps
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from omtf_gmt.regioning      import stubs_in_window, phi_rel, omtf_phi_to_global_rad, angle_diff
from omtf_gmt.features_tps   import build_node_features_tps, N_FEATURES_TPS, FEATURE_NAMES_TPS
from omtf_gmt.truth_transfer  import PHI_MATCH_THRESH

try:
    import uproot
    import awkward as ak
except ImportError:
    print("uproot and awkward are required.  pip install uproot awkward")
    sys.exit(1)

# ----- constants -----------------------------------------------------------

ALL_DATASETS = [
    "G1_pos", "G1_neg",
    "G2_pos", "G2_neg",
    "G3_pos", "G3_neg",
    "G4_pos", "G4_neg",
    "G5_pos", "G5_neg",
    "G6_pos", "G6_neg",
    "G7", "G8", "G9_pos", "G9_neg", "G10_pos", "G10_neg", "B4",
]
HARD_NEG_DATASETS = {"G7", "G8", "G9", "G10"}

SCHEMA_VERSION = 2
SHARD_SIZE     = 5_000
_NMAX          = 24
_K_MAX         = 3

_TRUTH_SRC_ENC: dict[str, int] = {
    "omtf_transfer": 2,
    "omtf_noise":    1,
    "unmatched":     0,
}

OMTF_HITS_BRANCHES = [
    "reg_eventNum", "reg_iProcessor",
    "reg_stub_phiHw", "reg_stub_layer", "reg_stub_bx",
    "reg_stub_trackId", "reg_stub_ambiguous",
]
TPS_BRANCHES_REQUIRED = [
    "event", "nGenMuon",
    "GenMuon_pt", "GenMuon_charge",
    "nMuonStubTps",
    "MuonStubTps_offlineCoord1", "MuonStubTps_offlineCoord2",
    "MuonStubTps_offlineEta1",   "MuonStubTps_offlineEta2",
    "MuonStubTps_quality",       "MuonStubTps_etaQuality",
    "MuonStubTps_bxNum",
    "MuonStubTps_tfLayer",       "MuonStubTps_depthRegion",
    "MuonStubTps_stubType",
]
TPS_BRANCHES_OPTIONAL = ["GenMuon_dXY"]


# ----- TPS truth transfer --------------------------------------------------
#
# Same phi+BX matching strategy as KMTF truth_transfer.transfer(), but without
# the DT-station compatibility check (TPS includes both barrel and endcap stubs
# whose OMTF layer→station mapping is ambiguous on the endcap side).

def _transfer_tps(
    tps_phi_rad:   np.ndarray,   # (N,) offlineCoord1 global phi [rad]
    tps_bx:        np.ndarray,   # (N,) bxNum
    omtf_phi_hw:   np.ndarray,   # (M,) processor-local phiHw
    omtf_bx:       np.ndarray,   # (M,) reg_stub_bx
    omtf_track_id: np.ndarray,   # (M,) reg_stub_trackId
    omtf_ambiguous:np.ndarray,   # (M,) reg_stub_ambiguous
    omtf_proc:     int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (track_id, ambiguous, truth_source) arrays for N TPS stubs."""
    N = len(tps_phi_rad)
    out_tid = np.zeros(N, dtype=np.int8)
    out_amb = np.zeros(N, dtype=np.uint8)
    out_src = ["unmatched"] * N

    if len(omtf_phi_hw) == 0:
        return out_tid, out_amb, out_src

    omtf_phi_rad = omtf_phi_to_global_rad(omtf_phi_hw.astype(np.float64), omtf_proc)

    for i in range(N):
        k_phi = float(tps_phi_rad[i])
        k_bx  = int(tps_bx[i])

        bx_ok  = (omtf_bx == k_bx)
        dphi   = np.abs(angle_diff(omtf_phi_rad, k_phi))
        phi_ok = dphi < PHI_MATCH_THRESH
        cands  = bx_ok & phi_ok

        if not cands.any():
            # relax BX and retry with phi only
            cands = phi_ok
        if not cands.any():
            continue

        best = int(np.argmin(dphi + np.where(cands, 0.0, 1e9)))
        out_tid[i] = int(omtf_track_id[best])
        out_amb[i] = int(omtf_ambiguous[best])
        out_src[i] = "omtf_transfer" if omtf_track_id[best] != 0 else "omtf_noise"

    return out_tid, out_amb, out_src


# ----- file-level builder --------------------------------------------------

def _load_nano_event_map(nano_path: Path, require_dxy: bool = False) -> dict[int, dict]:
    """Return dict: uint32(event) → {gen_pt, gen_charge, gen_dxy, tps_stubs}."""
    tree = uproot.open(str(nano_path))["Events"]
    available = set(tree.keys())
    branches  = TPS_BRANCHES_REQUIRED + [b for b in TPS_BRANCHES_OPTIONAL if b in available]
    has_dxy   = "GenMuon_dXY" in available
    if require_dxy and not has_dxy:
        raise ValueError(f"{nano_path} does not contain required GenMuon_dXY branch")
    arr = tree.arrays(branches, library="ak")

    event_map: dict[int, dict] = {}
    for i in range(len(arr)):
        ev = int(arr["event"][i]) & 0xFFFFFFFF

        n_gen        = int(arr["nGenMuon"][i])
        gen_pt_full  = ak.to_numpy(arr["GenMuon_pt"][i]).astype(np.float32)
        gen_ch_full  = ak.to_numpy(arr["GenMuon_charge"][i]).astype(np.float32)
        gen_dxy_full = (ak.to_numpy(arr["GenMuon_dXY"][i]).astype(np.float32)
                        if has_dxy else np.zeros(n_gen, dtype=np.float32))

        c1  = ak.to_numpy(arr["MuonStubTps_offlineCoord1"][i]).astype(np.float64)
        c2  = ak.to_numpy(arr["MuonStubTps_offlineCoord2"][i]).astype(np.float32)
        e1  = ak.to_numpy(arr["MuonStubTps_offlineEta1"][i]).astype(np.float32)
        e2  = ak.to_numpy(arr["MuonStubTps_offlineEta2"][i]).astype(np.float32)
        q   = ak.to_numpy(arr["MuonStubTps_quality"][i]).astype(np.int16)
        eq  = ak.to_numpy(arr["MuonStubTps_etaQuality"][i]).astype(np.int16)
        bx  = ak.to_numpy(arr["MuonStubTps_bxNum"][i]).astype(np.int8)
        lay = ak.to_numpy(arr["MuonStubTps_tfLayer"][i]).astype(np.int8)
        dep = ak.to_numpy(arr["MuonStubTps_depthRegion"][i]).astype(np.int8)
        sty = ak.to_numpy(arr["MuonStubTps_stubType"][i]).astype(np.int8)

        event_map[ev] = {
            "gen_pt_full":     gen_pt_full,
            "gen_charge_full": gen_ch_full,
            "gen_dxy_full":    gen_dxy_full,
            "tps": {
                "c1": c1, "c2": c2, "e1": e1, "e2": e2,
                "q": q, "eq": eq, "bx": bx, "lay": lay, "dep": dep, "sty": sty,
            },
        }
    return event_map


def _process_omtf_entry(
    event_num:     int,
    proc:          int,
    phi_hw:        np.ndarray,
    layer:         np.ndarray,
    bx_omtf:       np.ndarray,
    track_id_omtf: np.ndarray,
    ambig_omtf:    np.ndarray,
    nano_ev:       dict,
    is_hard_neg:   bool = False,
) -> dict | None:
    """Build one GMT-TPS sample from an OMTF processor-window entry."""
    tps = nano_ev["tps"]
    c1  = tps["c1"]

    if len(c1) == 0:
        return None

    # 1. select TPS stubs in this processor's phi window
    mask = stubs_in_window(c1, proc)
    if not mask.any():
        return None

    t_c1  = c1[mask].astype(np.float64)
    t_c2  = tps["c2"][mask]
    t_e1  = tps["e1"][mask]
    t_e2  = tps["e2"][mask]
    t_q   = tps["q"][mask]
    t_eq  = tps["eq"][mask]
    t_bx  = tps["bx"][mask]
    t_lay = tps["lay"][mask]
    t_dep = tps["dep"][mask]
    t_sty = tps["sty"][mask]

    n_real = len(t_c1)

    # 2. truth transfer (BX+phi matching, no station constraint)
    tr_tid, tr_amb, tr_src = _transfer_tps(
        tps_phi_rad    = t_c1,
        tps_bx         = t_bx,
        omtf_phi_hw    = phi_hw,
        omtf_bx        = bx_omtf,
        omtf_track_id  = track_id_omtf,
        omtf_ambiguous = ambig_omtf,
        omtf_proc      = proc,
    )

    # 3. build node features (processor-centred phi)
    phi_r = phi_rel(t_c1, proc).astype(np.float32)
    X = build_node_features_tps(phi_r, t_c2, t_e1, t_e2, t_q, t_eq, t_bx, t_lay, t_dep, t_sty)

    # 4. truncate by quality (descending) if needed
    tr_src_arr = np.array(tr_src, dtype=object)
    if n_real > _NMAX:
        order    = np.argsort(-t_q.astype(float))[:_NMAX]
        X        = X[order]
        tr_tid   = tr_tid[order]
        tr_amb   = tr_amb[order]
        tr_src_arr = tr_src_arr[order]
        n_real   = _NMAX

    # 5. window-level overlap candidate targets
    positive_ids = sorted(set(int(x) for x in tr_tid[:n_real] if int(x) > 0))

    gen_pt     = np.zeros(_K_MAX, dtype=np.float32)
    gen_charge = np.zeros(_K_MAX, dtype=np.float32)
    gen_dxy    = np.zeros(_K_MAX, dtype=np.float32)
    tgt_tid    = np.zeros(_K_MAX, dtype=np.int8)

    if not is_hard_neg:
        pt_full  = nano_ev["gen_pt_full"]
        ch_full  = nano_ev["gen_charge_full"]
        dxy_full = nano_ev["gen_dxy_full"]
        for slot, tid_val in enumerate(positive_ids[:_K_MAX]):
            idx = tid_val - 1
            if idx < len(pt_full):
                gen_pt[slot]     = pt_full[idx]
                gen_charge[slot] = ch_full[idx]
                gen_dxy[slot]    = dxy_full[idx]
            tgt_tid[slot] = tid_val

    # 6. pad to Nmax
    stubs_pad = np.zeros((_NMAX, N_FEATURES_TPS), dtype=np.float32)
    stubs_pad[:n_real] = X
    vm  = np.zeros(_NMAX, dtype=bool)
    vm[:n_real] = True
    tid = np.zeros(_NMAX, dtype=np.int8)
    tid[:n_real] = tr_tid
    amb = np.zeros(_NMAX, dtype=np.uint8)
    amb[:n_real] = tr_amb
    nl  = (tid != 0).astype(np.float32) * vm.astype(np.float32)

    ts = np.zeros(_NMAX, dtype=np.int8)
    for j in range(n_real):
        ts[j] = _TRUTH_SRC_ENC.get(str(tr_src_arr[j]), 0)

    return {
        "stubs":           torch.from_numpy(stubs_pad),
        "valid_mask":      torch.from_numpy(vm),
        "track_id":        torch.from_numpy(tid),
        "ambiguous":       torch.from_numpy(amb),
        "node_label":      torch.from_numpy(nl),
        "truth_source":    torch.from_numpy(ts),
        "gen_pt":          torch.from_numpy(gen_pt),
        "gen_charge":      torch.from_numpy(gen_charge),
        "gen_dxy":         torch.from_numpy(gen_dxy),
        "target_track_id": torch.from_numpy(tgt_tid),
        "meta_event_num":    torch.tensor(event_num,            dtype=torch.int32),
        "meta_i_proc":       torch.tensor(proc,                 dtype=torch.int32),
        "meta_n_stubs":      torch.tensor(n_real,               dtype=torch.int32),
        "meta_n_gen":        torch.tensor(0 if is_hard_neg else len(positive_ids), dtype=torch.int32),
        "meta_is_hard_neg":  torch.tensor(int(is_hard_neg),     dtype=torch.int8),
    }


def _stack_shard(samples: list[dict]) -> dict:
    keys_tensor = [
        "stubs", "valid_mask", "track_id", "ambiguous",
        "node_label", "truth_source",
        "gen_pt", "gen_charge", "gen_dxy", "target_track_id",
    ]
    keys_scalar = [
        "meta_event_num", "meta_i_proc", "meta_n_stubs",
        "meta_n_gen", "meta_is_hard_neg",
    ]
    shard = {}
    for k in keys_tensor + keys_scalar:
        shard[k] = torch.stack([s[k] for s in samples])
    return shard


def process_file_pair(
    hits_path:   Path,
    nano_path:   Path,
    is_hard_neg: bool = False,
    verbose:     bool = False,
    require_dxy: bool = False,
) -> list[dict]:
    nano_map = _load_nano_event_map(nano_path, require_dxy=require_dxy)

    hits_tree = uproot.open(str(hits_path))["simOmtfPhase2Digis/OMTFAllInputTree"]
    arr = hits_tree.arrays(OMTF_HITS_BRANCHES, library="ak")

    samples: list[dict] = []
    n_skipped = 0

    for i in range(len(arr)):
        ev   = int(arr["reg_eventNum"][i]) & 0xFFFFFFFF
        proc = int(arr["reg_iProcessor"][i])

        if ev not in nano_map:
            n_skipped += 1
            continue

        phi_hw = ak.to_numpy(arr["reg_stub_phiHw"][i]).astype(np.int32)
        layer  = ak.to_numpy(arr["reg_stub_layer"][i]).astype(np.int8)
        bx_o   = ak.to_numpy(arr["reg_stub_bx"][i]).astype(np.int8)
        tid_o  = ak.to_numpy(arr["reg_stub_trackId"][i]).astype(np.int8)
        amb_o  = ak.to_numpy(arr["reg_stub_ambiguous"][i]).astype(np.uint8)

        sample = _process_omtf_entry(
            ev, proc, phi_hw, layer, bx_o, tid_o, amb_o, nano_map[ev],
            is_hard_neg=is_hard_neg,
        )
        if sample is not None:
            samples.append(sample)

    if verbose and n_skipped:
        print(f"    skipped {n_skipped} OMTF entries (event not in NanoAOD)")
    return samples


# ----- main ----------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build GMT-visible-stub dataset (TPS, Phase B4)")
    p.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    p.add_argument("--data-dir", type=Path, default=Path("data/prod"))
    p.add_argument("--output-dir", type=Path, default=Path("build/omtf_gmt/cache_v2_tps"))
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    p.add_argument("--require-dxy", action="store_true",
                   help="fail if any NanoAOD file lacks GenMuon_dXY")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    verbose = not args.quiet

    manifest: dict = {
        "schema_version":       SCHEMA_VERSION,
        "branch":               "omtf_gmt_tps",
        "stage":                "B4_tps_with_overlap_meta",
        "stub_source":          "MuonStubTps",
        "nmax":                 _NMAX,
        "k_max":                _K_MAX,
        "n_features":           N_FEATURES_TPS,
        "feature_names":        FEATURE_NAMES_TPS,
        "truth_source_encoding":{v: k for k, v in _TRUTH_SRC_ENC.items()},
        "datasets":             {},
    }

    for ds in args.datasets:
        ds_dir     = args.data_dir / ds
        hits_files = sorted(ds_dir.glob("omtf_hits_*.root"))
        if args.max_files:
            hits_files = hits_files[:args.max_files]
        if not hits_files:
            print(f"  [{ds}] no hits files found in {ds_dir}, skipping")
            continue

        ds_out = outdir / ds
        ds_out.mkdir(exist_ok=True)
        for old in ds_out.glob("shard_*.pt"):
            old.unlink()

        t0 = time.time()
        base_ds     = ds.removesuffix("_pos").removesuffix("_neg")
        is_hard_neg = base_ds in HARD_NEG_DATASETS

        pending:   list[dict] = []
        shard_idx = 0
        n_total   = 0
        n_files   = 0

        for hits_path in hits_files:
            nano_path = hits_path.parent / hits_path.name.replace("omtf_hits_", "omtf_nano_")
            if not nano_path.exists():
                if verbose:
                    print(f"  [{ds}] missing nano for {hits_path.name}, skipping")
                continue

            samples = process_file_pair(
                hits_path, nano_path, is_hard_neg=is_hard_neg,
                require_dxy=args.require_dxy,
            )
            pending.extend(samples)
            n_files += 1

            while len(pending) >= args.shard_size:
                shard = _stack_shard(pending[:args.shard_size])
                torch.save(shard, ds_out / f"shard_{shard_idx:04d}.pt")
                shard_idx += 1
                n_total   += args.shard_size
                pending    = pending[args.shard_size:]

            if verbose:
                print(
                    f"  [{ds}] {n_files}/{len(hits_files)} files  "
                    f"samples_so_far={n_total + len(pending):,}",
                    end="\r",
                )

        if pending:
            shard = _stack_shard(pending)
            torch.save(shard, ds_out / f"shard_{shard_idx:04d}.pt")
            n_total += len(pending)

        elapsed = time.time() - t0
        manifest["datasets"][ds] = {
            "n_samples":    n_total,
            "n_shards":     shard_idx + (1 if pending else 0),
            "n_files":      n_files,
            "is_hard_neg":  is_hard_neg,
            "base_dataset": base_ds,
            "gen_dxy_required": args.require_dxy,
        }
        if verbose:
            print(f"\n  [{ds}] done — {n_total:,} samples in {elapsed:.0f}s")

    manifest_path = outdir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        existing["datasets"].update(manifest["datasets"])
        manifest = existing
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest written to {manifest_path}")


if __name__ == "__main__":
    main()
