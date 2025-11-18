#!/usr/bin/env python3
# filter_candidates.py
import argparse, yaml
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Filter candidates.yaml by sel_candidates.yaml include list.")
    ap.add_argument("--src", default="ur_sensor_sim/mesh_sampling/candidates.yaml",
                    help="Full candidates YAML (with 'candidates': [...]).")
    ap.add_argument("--sel", default="ur_sensor_sim/mesh_sampling/sel_candidates.yaml",
                    help="Selection YAML containing 'include' (or 'included') indices.")
    ap.add_argument("--out", default="ur_sensor_sim/mesh_sampling/selected_candidates.yaml",
                    help="Output filtered candidates YAML.")
    args = ap.parse_args()

    src = yaml.safe_load(Path(args.src).read_text())
    sel = yaml.safe_load(Path(args.sel).read_text())

    # support both keys: include / included
    idx_list = sel.get("include", sel.get("included", None))
    if idx_list is None:
        raise SystemExit(f"{args.sel} must contain 'include' or 'included'.")

    # Safety: coerce to unique ints, but keep the given order
    seen = set()
    indices = []
    for x in idx_list:
        i = int(x)
        if i not in seen:
            indices.append(i)
            seen.add(i)

    all_cands = src.get("candidates", [])
    n_total = len(all_cands)

    # Filter with bounds check
    filtered = []
    bad = []
    for i in indices:
        if 0 <= i < n_total:
            # Shallow copy so we don't mutate the original structure by accident
            item = dict(all_cands[i])
            filtered.append(item)
        else:
            bad.append(i)

    if bad:
        print(f"[WARN] Skipping out-of-range indices: {bad} (0..{n_total-1})")

    out_doc = {
        # keep top-level metadata if present; fall back to sensible defaults
        "spacing_m": src.get("spacing_m", 0.03),
        "offset_m": src.get("offset_m", 0.005),
        "candidate_count": len(filtered),
        "candidates": filtered,
    }

    Path(args.out).write_text(yaml.safe_dump(out_doc, sort_keys=False))
    print(f"[OK] Wrote {args.out} | kept {len(filtered)}/{n_total} candidates")

if __name__ == "__main__":
    main()
