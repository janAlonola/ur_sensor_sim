#!/usr/bin/env python3
# weights_yaml_to_npy.py
import argparse, numpy as np, yaml
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-yaml", default="ur_sensor_sim/tmp/capsule_weighted.yaml")
    ap.add_argument("--key", default="weights", help="YAML key holding the weights array")
    ap.add_argument("--out", default="ur_sensor_sim/tmp/capsule_weighted.npy")
    ap.add_argument("--expected-V", type=int, default=None, help="Optional sanity check")
    args = ap.parse_args()

    data = yaml.safe_load(Path(args.weights_yaml).read_text())
    if args.key not in data:
        raise SystemExit(f"Key '{args.key}' not found in {args.weights_yaml}")
    w = np.asarray(data[args.key], dtype=float)
    if args.expected_V is not None and w.shape != (args.expected_V,):
        raise SystemExit(f"weights.shape={w.shape} != expected V={args.expected_V}")

    out = Path(args.out) if args.out else Path(args.weights_yaml).with_suffix(".npy")
    np.save(out, w)
    print(f"[OK] Saved {out} | V={w.size}")

if __name__ == "__main__":
    main()
