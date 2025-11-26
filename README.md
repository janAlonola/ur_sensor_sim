# Universal_Robots_ROS2_Gazebo_Simulation + Sensor Placement Optimization

This repository is based on  
[UniversalRobots/Universal_Robots_ROS2_Gazebo_Simulation](https://github.com/UniversalRobots/Universal_Robots_ROS2_Gazebo_Simulation)  
and extends the UR Gazebo simulation with an **offline sensor placement optimizer** for
e.g. ToF sensors mounted around a UR robot.

The optimizer works on

- a **voxelized workspace**,
- a set of **candidate sensor poses**, and
- multiple **robot poses**,

and selects a single set of sensors that gives good visibility across **all** poses.

---

## Build status / supported ROS versions

Since Gazebo Classic is no longer supported from ROS 2 Jazzy onwards, this repository is configured
for **ROS 2 Humble**.

<table width="100%">
  <tr>
    <th></th>
    <th>Humble</th>
  </tr>
  <tr>
    <th>Branch</th>
    <td><a href="https://github.com/UniversalRobots/Universal_Robots_ROS2_Gazebo_Simulation/tree/humble">humble</a></td>
  </tr>
  <tr>
    <th>Build status</th>
    <td>
      <a href="https://github.com/UniversalRobots/Universal_Robots_ROS2_Gazebo_Simulation/actions/workflows/humble-binary-main.yml?query=event%3Aschedule++">
         <img src="https://github.com/UniversalRobots/Universal_Robots_ROS2_Gazebo_Simulation/actions/workflows/humble-binary-main.yml/badge.svg?event=schedule"
              alt="Humble Binary Main"/>
      </a> <br />
    </td>
  </tr>
</table>
---

## Using the repository

### 1. Setup ROS workspace

```bash
export COLCON_WS=~/workspaces/ur_gazebo
mkdir -p $COLCON_WS/src
````

Feel free to change `~/workspaces/ur_gazebo` to any absolute path.
It is usually convenient to keep multiple ROS workspaces in a dedicated `workspaces/` folder and
include the ROS version in the workspace name.

### 2. Clone repository and install dependencies

```bash
cd $COLCON_WS/src
git clone -b humble https://github.com/janAlonola/ur_sensor_sim.git
# or your fork/branch
rosdep update && rosdep install --ignore-src --from-paths . -y
```

### 3. Configure and build

```bash
cd $COLCON_WS
colcon build --symlink-install
```

Then source the workspace as usual.

---

## Running the simulation (for Validitation)

Start UR robot + Gazebo Classic:

```bash
ros2 launch ur_simulation_gazebo ur_sim_control.launch.py
```

Move the robot using the test script from `ur_robot_driver` (if installed):

```bash
ros2 launch ur_robot_driver test_joint_trajectory_controller.launch.py
```

Example using MoveIt with the simulated robot:

```bash
ros2 launch ur_simulation_gazebo ur_sim_moveit.launch.py
```

---

## Sensor placement optimization – overview

The high-level workflow is:
ToDo: Sensor Meshing and Xacro Generation for Validation
1. Create a **voxel map** of the workspace.
2. Define a set of **robot poses** (`poses.yaml`).
3. Compute **per-voxel weights** for each pose (distance to robot, etc.).
4. Compute **sensor visibility** with occlusion for all poses.
5. Run a **multi-pose GRASP optimizer** to choose a single set of sensors that
   performs well across all poses.

The optimizer does **not** give extra score for the same voxel being seen by multiple sensors
in the same pose: per pose it only matters whether a voxel is covered at least once.

---

## Important files

* `capsule.yaml`
  Voxel description of the workspace (output of `voxel_grid.py`).

* `poses.yaml`
  List of robot joint configurations. Each entry has a `name` (e.g. `b1_w1`) and
  `joints_deg`/`joints_rad` for all UR joints.

* `selected_candidates.yaml`
  Candidate sensor poses (reduced set from `mesh_sampling` / `sensor_picker`).

* `weighted_poses_*.yaml` & `weighted_poses_*.npy` ToDo: Update
  Per-pose voxel weights. The YAML files contain metadata + weights; the `.npy` files
  contain just the weight vector for fast loading.

* `occlusion_heatmaps/*.yaml` ToDo: Update
  Per-pose visibility: for each voxel, which sensors see it (`visible_by`), including
  occlusion by the robot based on ray casting.

* `optimizer_all_poses.py`
  GRASP-based optimizer that selects one global sensor set over all poses.

---

## Step-by-step workflow

### 1. Create workspace voxel map

Script: `voxel_grid.py`

* Input: workspace definition / URDF
* Output: `capsule.yaml` with e.g.

  * `voxels`: list of `[x, y, z]`
  * `voxel_size_m`, `voxel_count`

This can be visualized in RViz or a custom visualizer if desired.

---

### 2. Define robot poses

File: `poses.yaml`

* Contains:

  * `joint_names`: order of UR joints
  * `poses`: list of

    * `name`: e.g. `b1_w1`
    * `joints_deg` / `joints_rad`: joint angles

These poses are used both for voxel weighting and for the visibility computation.

---

### 3. Weight voxels (distance to robot)

Scripts: e.g. `weighted_voxel.py` / `weighted_voxel_robot.py`

* Inputs:

  * `capsule.yaml`
  * `poses.yaml`
  * URDF (`ur10.urdf`)
  * robot collision meshes via `trimesh.closest_point`

* Idea:

  * For each pose and voxel, compute the distance to the robot surface.
  * Map this distance to a weight using a falloff function (linear, gamma, exp,
    or a sigmoid/saturation curve):

    * within a minimum distance (sensor blind zone) → weight 0
    * around a preferred distance band → high weight
    * very far away → weight decreases again

* Output:

  * One weighted voxel YAML per pose, e.g. `weighted_poses_sigmoid/b1_w1.yaml`
    containing `weights: [...]` and metadata.

Optionally, visualize the weighted voxels.

---

### 4. Convert weights YAML → `.npy`

Script: `weights_yaml_to_npy.py`

* Reads each `weighted_poses_*.yaml`
* Writes `weighted_poses_*.npy` containing a single 1-D NumPy array (length = voxel_count).

The optimizer uses these numpy files because they are much faster to load.

---

### 5. Compute visibility with occlusion

Script: e.g. `compute_visibility_all_poses.py`

* Inputs:

  * `poses.yaml`
  * `capsule.yaml`
  * `selected_candidates.yaml`
  * URDF + collision meshes (via `trimesh` and ray casting)

* For each pose:

  * Transform all candidate sensors into the world frame.
  * For each voxel:

    * check if it is within sensor FOV and range,
    * cast a ray and test for intersections with the robot mesh
      (`trimesh.ray.intersects_location` or similar).
  * Store for each voxel the list of sensor indices that can see it: `visible_by`.

* Output:

  * Heatmap YAMLs in `occlusion_heatmaps/`, one per pose (or per group of poses), each containing

    * `voxel_count`, `sensor_count`
    * `visible_by` as `[pose][voxel] -> list[int]` or `[voxel] -> list[int]`

These can be visualized as coverage/heatmap plots.

---

### 6. Multi-pose sensor optimization

Script: `optimize_across_poses_multipose.py`

* Inputs:

  * `--heatmaps`: directory with `occlusion_heatmaps/*.yaml`
  * `--weights-list`: list or glob for `weighted_poses_*.npy` (one per pose)

* Internally:

  * For each pose, builds a sparse CSR matrix `A` of shape `(S, V)`:

    * rows = sensors, columns = voxels, entries = 1 if sensor sees voxel.
  * Loads pose-specific weights `W` of shape `(V, P)`:

    * `V` = voxel count, `P` = pose count.
  * Runs a GRASP algorithm:

    * greedy construction with a **Restricted Candidate List** (`--rcl-size`),
    * multiple restarts (`--iters`),
    * then 1-swap local search (`--local-rounds`).

* Objective modes (`--objective`):

  * `sum`:
    Maximize
    `sum_p sum_v W[v,p] * covered[p,v]`
    (no extra credit for seeing the same voxel with multiple sensors in one pose).
  * `softmin`:
    “Robust” objective using a soft minimum across poses (`--softmin-temp`).
  * `frac`:
    A voxel only counts if it is covered in at least `alpha` fraction of poses
    (`--frac-alpha`).

* Important note:
  For a given pose, if several sensors see the same voxel, the voxel is counted
  **once**. Extra redundant coverage does not increase the score in that pose.

* Output:

  * Printed list of selected sensor indices.
  * JSON file (via `--export-json`) including:

    * `selected_sensors`
    * objective value and mode
    * numbers of sensors, voxels, poses
    * source heatmap and weight files
    * optimizer parameters (k, iters, etc.)

These selected sensors can then be used to generate Xacro/URDF files and visualized in Gazebo / RViz.

---

## Short workflow recap

1. **Voxel map:** `voxel_grid.py` → `capsule.yaml`
2. **Poses:** edit `poses.yaml`
3. **Weights:** `weighted_voxel*.py` → `weighted_poses_*.yaml`
   → `weights_yaml_to_npy.py` → `weighted_poses_*.npy`
4. **Visibility:** `compute_visibility_all_poses.py` → `occlusion_heatmaps/*.yaml`
5. **Optimization:** `optimize_across_poses_multipose.py`
   → final list of selected sensor indices
6. **Visualization / export:** build Xacro/URDF with the selected sensors
   and inspect in Gazebo / RViz.