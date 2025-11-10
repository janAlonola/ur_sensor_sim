#!/usr/bin/env python3
# merge_heatmaps.py
import argparse, math
from pathlib import Path
import numpy as np
import yaml

def load_yaml(p: Path): return yaml.safe_load(p.read_text())
def save_yaml(p: Path, d: dict): p.write_text(yaml.safe_dump(d, sort_keys=False))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="ur_sensor_sim/tmp/weighted_heatmaps",
                    help="Folder with per-pose *_heatmap.yaml (from compute_visibility_all_poses.py)")
    ap.add_argument("--out", default="ur_sensor_sim/tmp/combined_heatmap.yaml",
                    help="Output combined YAML")
    ap.add_argument("--aggregate-weights", choices=["none","max","mean","softmax"],
                    default="max", help="Aggregate voxel weights across poses")
    ap.add_argument("--temperature", type=float, default=0.5,
                    help="Softmax temperature if aggregate-weights=softmax")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    files = sorted([p for p in in_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in (".yaml",".yml")])
    if not files: raise SystemExit(f"No YAMLs in {in_dir}")

    # Load all and sanity-check voxel grid & sensor count
    docs = [load_yaml(f) for f in files]
    V = int(docs[0]["voxel_count"])
    S = int(docs[0]["sensor_count"])
    vox0 = np.asarray(docs[0]["voxels"], dtype=np.float32)
    for f,d in zip(files, docs):
        if int(d["voxel_count"]) != V: raise SystemExit(f"{f.name}: voxel_count mismatch")
        if int(d["sensor_count"]) != S: raise SystemExit(f"{f.name}: sensor_count mismatch")
        v = np.asarray(d["voxels"], dtype=np.float32)
        if v.shape != vox0.shape or not np.allclose(v, vox0):
            raise SystemExit(f"{f.name}: voxel grid differs")

    # Build visible_by: [pose][V] -> list[int]
    visible_by = []
    pose_names = []
    for f,d in zip(files, docs):
        pose_names.append(d.get("pose_name", f.stem))
        vbp = d["visible_by"]  # already [V] -> list[int]
        if len(vbp) != V: raise SystemExit(f"{f.name}: visible_by length != V")
        visible_by.append(vbp)

    # Aggregate weights if present
    weights = None
    if args.aggregate-weights != "none":
        has_w = [("weights" in d) for d in docs]
        if all(has_w):
            W = np.stack([np.asarray(d["weights"], dtype=np.float64) for d in docs], axis=1)  # (V,P)
            if args.aggregate-weights == "max":
                w = np.max(W, axis=1)
            elif args.aggregate-weights == "mean":
                w = np.mean(W, axis=1)
            else:  # softmax
                tau = max(1e-6, float(args.temperature))
                Sx = np.exp(W / tau)
                P = Sx / np.clip(np.sum(Sx, axis=1, keepdims=True), 1e-12, None)
                w = np.sum(P * W, axis=1)
            # normalize to [0,1] for sanity (optional; comment out if not wanted)
            wmin, wmax = float(w.min()), float(w.max())
            weights = ((w - wmin) / (wmax - wmin)) if wmax > wmin else np.zeros_like(w)
        else:
            print("[WARN] Not all inputs carry 'weights'; skipping weight aggregation.")
            weights = None

    out = {
        "voxel_size_m": docs[0].get("voxel_size_m"),
        "voxel_count": V,
        "sensor_count": S,
        "pose_count": len(files),
        "fov_deg": docs[0].get("fov_deg"),
        "max_range_m": docs[0].get("max_range_m"),
        "voxels": vox0.tolist(),
        "visible_by": visible_by,   # <<< shape [P][V] -> list[int]
        "pose_names": pose_names,
        "weight_aggregation": {
            "method": args.aggregate-weights,
            "temperature": float(args.temperature) if args.aggregate-weights=="softmax" else None
        }
    }
    if weights is not None:
        out["weights"] = weights.astype(float).tolist()

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    save_yaml(outp, out)
    print(f"[OK] wrote {outp} | P={len(files)} | V={V} | S={S} | weights={'yes' if weights is not None else 'no'}")

if __name__ == "__main__":
    main()
