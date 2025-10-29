import yaml, numpy as np
from pathlib import Path

def load_visible_by(yaml_path):
    """
    Expects YAML with:
      - voxel_count
      - sensor_count
      - pose_count
      - visible_by: list[pose][voxel] -> list of sensor indices
    Returns:
      sensor_sets: list of sets; sensor_sets[j] = set of voxel indices covered by sensor j across all poses
    """
    data = yaml.safe_load(Path(yaml_path).read_text())
    V = data["voxel_count"]
    P = data["pose_count"]
    S = data["sensor_count"]
    vis_by = data["visible_by"]   # shape: [P][V] -> list[int]
    sensor_sets = [set() for _ in range(S)]
    for p in range(P):
        vox_lists = vis_by[p]
        for v, sens_list in enumerate(vox_lists):
            for s in sens_list:
                sensor_sets[s].add(v)
    return sensor_sets, V

def greedy_max_coverage(sensor_sets, universe_size, k=None, target_frac=None, weights=None, verbose=True):
    """
    Greedy selector for maximum coverage.
    - k: max number of sensors (budget). If None, continue until target_frac reached.
    - target_frac: stop when covered >= target_frac * universe_size. If None, run for k steps.
    - weights: optional array (len=universe_size) for voxel importance.
    Returns:
      selected (list of sensor indices),
      covered_set (set of voxel indices),
      curve (list of cumulative covered counts after each pick),
      gains (list of marginal gains)
    """
    U = set(range(universe_size))
    covered = set()
    selected = []
    curve, gains = [], []

    # use weights if provided; default weight=1 per voxel
    if weights is None:
        weights = np.ones(universe_size, dtype=np.float64)

    remaining = set(range(len(sensor_sets)))
    goal = None
    if target_frac is not None:
        goal = int(np.ceil(target_frac * universe_size))

    step = 0
    while True:
        if goal is not None and len(covered) >= goal:
            break
        if k is not None and len(selected) >= k:
            break
        best_s, best_gain, best_vox = None, -1.0, None

        # evaluate marginal gain for each remaining sensor
        for s in list(remaining):
            new_vox = sensor_sets[s] - covered
            if not new_vox:
                continue
            # weighted gain
            g = float(np.sum([weights[v] for v in new_vox]))
            if g > best_gain:
                best_gain = g
                best_s = s
                best_vox = new_vox

        if best_s is None:
            # no more improvement possible
            break

        # take it
        selected.append(best_s)
        covered.update(best_vox)
        remaining.remove(best_s)
        curve.append(len(covered))
        gains.append(best_gain)
        step += 1
        if verbose:
            print(f"[{step:02d}] pick sensor {best_s}  +{len(best_vox)} (cum {len(covered)}/{universe_size})")

    return selected, covered, curve, gains

def sensors_rank_by_total_coverage(sensor_sets):
    """Convenience: rank sensors by total unique voxels they can see (union across poses)."""
    sizes = [(j, len(s)) for j, s in enumerate(sensor_sets)]
    return sorted(sizes, key=lambda x: x[1], reverse=True)

# ---------- Example usage ----------
# 1) Build per-sensor voxel sets from YAML
sensor_sets, V = load_visible_by("heatmap_with_v_by_s.yaml")
print("loaded")

# 2a) Minimal set for ≥95% coverage
selected_95, covered_95, curve_95, gains_95 = greedy_max_coverage(
    sensor_sets, V, k=None, target_frac=0.95, verbose=True
)
print(f"Selected {len(selected_95)} sensors for ≥95% coverage.")

# 2b) Best k sensors (e.g., k=10)
selected_k, covered_k, curve_k, gains_k = greedy_max_coverage(
    sensor_sets, V, k=10, target_frac=None, verbose=True
)
print(f"Top-10 sensors cover {len(covered_k)}/{V} voxels ({100*len(covered_k)/V:.1f}%).")

# 3) Optional: see which sensors are strongest individually
ranking = sensors_rank_by_total_coverage(sensor_sets)
print("Top 10 individual sensors by union coverage:")
print(ranking[:10])