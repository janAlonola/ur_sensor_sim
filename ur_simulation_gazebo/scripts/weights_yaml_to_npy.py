#!/usr/bin/env python3
# weights_yaml_to_npy.py
import argparse, numpy as np, yaml
from pathlib import Path

def yaml_to_npy(yaml_path: Path, key: str, expected_V: int|None, out_dir: Path|None):
    data = yaml.safe_load(yaml_path.read_text())
    if key not in data:
        raise SystemExit(f"Key '{key}' not found in {yaml_path}")
    w = np.asarray(data[key], dtype=float)
    if expected_V is not None and w.shape != (expected_V,):
        raise SystemExit(f"{yaml_path.name}: weights.shape={w.shape} != expected V={expected_V}")
    out = (out_dir / (yaml_path.stem + ".npy")) if out_dir else yaml_path.with_suffix(".npy")
    np.save(out, w)
    print(f"[OK] {yaml_path.name} -> {out.name} | V={w.size}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-yaml", default="ur_sensor_sim/tmp/weighted_poses_robot",
                    help="YAML path OR directory containing multiple YAMLs with weights")
    ap.add_argument("--key", default="weights", help="YAML key holding weights array")
    ap.add_argument("--out", default="ur_sensor_sim/tmp/weighted_poses_robot",
                    help="Output .npy path (only when converting a single YAML). "
                         "If --weights-yaml is a directory, this is treated as an output directory.")
    ap.add_argument("--expected-V", type=int, default=None, help="Optional sanity check")
    args = ap.parse_args()

    src = Path(args.weights_yaml)
    if src.is_dir():
        out_dir = Path(args.out) if args.out else src
        out_dir.mkdir(parents=True, exist_ok=True)
        # convert all *.yaml / *.yml in folder
        files = sorted([p for p in src.iterdir() if p.suffix.lower() in (".yaml",".yml") and p.is_file()])
        if not files:
            raise SystemExit(f"No YAMLs found in {src}")
        for y in files:
            yaml_to_npy(y, args.key, args.expected_V, out_dir)
    else:
        # single file
        out_path = Path(args.out) if args.out else None
        yaml_to_npy(src, args.key, args.expected_V, out_dir=None if out_path else None)
        if out_path is not None:
            # move/rename produced .npy to requested path
            produced = src.with_suffix(".npy")
            produced.rename(out_path)
            print(f"[OK] Renamed to {out_path}")

if __name__ == "__main__":
    main()
