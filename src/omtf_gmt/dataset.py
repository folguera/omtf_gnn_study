"""
GMTDataset — one sample per (event, OMTF-processor) pair.

Each sample contains:
  stubs        (Nmax, 10) float32  — KMTF barrel stub features
  valid_mask   (Nmax,)    bool     — True for real stubs
  track_id     (Nmax,)    int8     — 0=noise, 1..K=gen-muon slot
  ambiguous    (Nmax,)    uint8    — transferred ambiguity flag
  node_label   (Nmax,)    float32  — 1.0 if track_id != 0
  gen_pt       (K,)       float32  — gen-muon pT, slot-ordered
  gen_charge   (K,)       float32
  gen_dxy      (K,)       float32
  meta: event, processor, n_stubs, n_gen, source

Nmax and K_MAX are configurable; defaults match the OMTF-internal branch.

The dataset is NOT built here — use scripts/omtf_gmt/make_gmt_dataset.py
to pre-build sharded .pt files, then load them with GMTCachedDataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset

SCHEMA_VERSION: int = 2
_NMAX_DEFAULT: int  = 24
_K_MAX: int         = 3

# G1–G6 are produced in pos/neg eta-hemisphere pairs.  These aliases let callers
# pass logical names (G1, G2, …) which expand to the two physical cache directories.
DATASET_ALIASES: dict[str, list[str]] = {
    "G1": ["G1_pos", "G1_neg"],
    "G2": ["G2_pos", "G2_neg"],
    "G3": ["G3_pos", "G3_neg"],
    "G4": ["G4_pos", "G4_neg"],
    "G5": ["G5_pos", "G5_neg"],
    "G6": ["G6_pos", "G6_neg"],
    "G9":  ["G9_pos",  "G9_neg"],
    "G10": ["G10_pos", "G10_neg"],
}


def expand_datasets(names: list[str]) -> list[str]:
    """Expand logical dataset names to physical cache directory names."""
    result: list[str] = []
    for name in names:
        result.extend(DATASET_ALIASES.get(name, [name]))
    return result


def expand_repeats(specs: list[str] | None) -> dict[str, int]:
    """Parse 'DS:N' repeat specs and expand aliases to physical names.

    Example: ['B4:8', 'G7:2', 'G8:4'] → {'B4': 8, 'G7': 2, 'G8': 4}
    Example: ['G1:2'] → {'G1_pos': 2, 'G1_neg': 2}
    """
    repeats: dict[str, int] = {}
    for spec in (specs or []):
        ds_name, n_str = spec.split(":", 1)
        n = int(n_str)
        for physical in DATASET_ALIASES.get(ds_name, [ds_name]):
            repeats[physical] = n
    return repeats


# --------------------------------------------------------------------------- #
# Shard I/O helpers                                                           #
# --------------------------------------------------------------------------- #

def _load_shard(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def iter_shards(cache_dir: Path, dataset: str) -> Iterator[dict]:
    ds_dir = Path(cache_dir) / dataset
    for p in sorted(ds_dir.glob("shard_*.pt")):
        yield _load_shard(p)


# --------------------------------------------------------------------------- #
# Dataset                                                                     #
# --------------------------------------------------------------------------- #

class GMTCachedDataset(Dataset):
    """
    Load GMT-visible-stub samples from pre-built .pt shards.

    Parameters
    ----------
    cache_dir : directory with manifest.json and dataset sub-directories
    dataset   : name, e.g. "S1"
    """

    def __init__(self, cache_dir: str | Path, dataset: str):
        cache_dir = Path(cache_dir)
        self._validate_manifest(cache_dir, dataset)

        stubs_l, vm_l, tid_l, amb_l, nl_l, ts_l, target_tid_l = [], [], [], [], [], [], []
        gpt_l, gc_l, gd_l = [], [], []
        men_l, mip_l, mns_l, mng_l, mhn_l = [], [], [], [], []

        for p in sorted((cache_dir / dataset).glob("shard_*.pt")):
            s = _load_shard(p)
            stubs_l.append(s["stubs"])
            vm_l.append(s["valid_mask"])
            tid_l.append(s["track_id"])
            if "target_track_id" in s:
                target_tid_l.append(s["target_track_id"])
            amb_l.append(s["ambiguous"])
            nl_l.append(s["node_label"])
            ts_l.append(s["truth_source"])
            gpt_l.append(s["gen_pt"])
            gc_l.append(s["gen_charge"])
            gd_l.append(s["gen_dxy"])
            men_l.append(s["meta_event_num"])
            mip_l.append(s["meta_i_proc"])
            mns_l.append(s["meta_n_stubs"])
            mng_l.append(s["meta_n_gen"])
            mhn_l.append(s["meta_is_hard_neg"])

        self.stubs        = torch.cat(stubs_l)
        self.valid_mask   = torch.cat(vm_l)
        self.track_id     = torch.cat(tid_l)
        self.target_track_id = (
            torch.cat(target_tid_l) if len(target_tid_l) == len(tid_l) else None
        )
        self.ambiguous    = torch.cat(amb_l)
        self.node_label   = torch.cat(nl_l)
        self.truth_source = torch.cat(ts_l)
        self.gen_pt       = torch.cat(gpt_l)
        self.gen_charge   = torch.cat(gc_l)
        self.gen_dxy      = torch.cat(gd_l)
        self.meta_event   = torch.cat(men_l)
        self.meta_proc    = torch.cat(mip_l)
        self.meta_n_stubs = torch.cat(mns_l)
        self.meta_n_gen   = torch.cat(mng_l)
        self.meta_is_hard_neg = torch.cat(mhn_l)

    def __len__(self) -> int:
        return self.stubs.shape[0]

    def __getitem__(self, idx: int) -> dict:
        sample = {
            "stubs":        self.stubs[idx],
            "valid_mask":   self.valid_mask[idx],
            "track_id":     self.track_id[idx],
            "ambiguous":    self.ambiguous[idx],
            "node_label":   self.node_label[idx],
            "truth_source": self.truth_source[idx],
            "gen_pt":       self.gen_pt[idx],
            "gen_charge":   self.gen_charge[idx],
            "gen_dxy":      self.gen_dxy[idx],
            "meta": {
                "event_num":    int(self.meta_event[idx]),
                "i_processor":  int(self.meta_proc[idx]),
                "n_stubs":      int(self.meta_n_stubs[idx]),
                "n_gen_tracks": int(self.meta_n_gen[idx]),
                "is_hard_neg":  bool(self.meta_is_hard_neg[idx]),
            },
        }
        if self.target_track_id is not None:
            sample["target_track_id"] = self.target_track_id[idx]
        return sample

    @staticmethod
    def _validate_manifest(cache_dir: Path, dataset: str) -> None:
        mf = json.loads((cache_dir / "manifest.json").read_text())
        if mf.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"GMT cache schema version mismatch: {mf.get('schema_version')}")
        if dataset not in mf.get("datasets", {}):
            raise KeyError(f"Dataset {dataset!r} not in GMT cache manifest")


def collate_gmt(batch: list[dict]) -> dict:
    """Collate a list of samples into batched tensors."""
    collated = {
        "stubs":        torch.stack([b["stubs"] for b in batch]),
        "valid_mask":   torch.stack([b["valid_mask"] for b in batch]),
        "track_id":     torch.stack([b["track_id"] for b in batch]),
        "ambiguous":    torch.stack([b["ambiguous"] for b in batch]),
        "node_label":   torch.stack([b["node_label"] for b in batch]),
        "truth_source": torch.stack([b["truth_source"] for b in batch]),
        "gen_pt":       torch.stack([b["gen_pt"] for b in batch]),
        "gen_charge":   torch.stack([b["gen_charge"] for b in batch]),
        "gen_dxy":      torch.stack([b["gen_dxy"] for b in batch]),
        "meta":         [b["meta"] for b in batch],
    }
    if "target_track_id" in batch[0]:
        collated["target_track_id"] = torch.stack([b["target_track_id"] for b in batch])
    return collated
