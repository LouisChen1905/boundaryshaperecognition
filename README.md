# Boundary shape recognition

## Grass-only boundary exploration

`boundary_exploration.py` is an independent, stateful grid explorer. Create
`BoundaryExplorer(ExplorationConfig(robot_radius=..., sensor_range=...))`, then
call `update(grass_points, boundary_points, (x, y, yaw), timestamp)` for each
time-aligned frame. Two-column point arrays are global XY. Three-column camera
XYZ arrays require a full 4x4 `sensor_to_world` transform; camera Y is not
discarded. The result contains the sparse known-grass, boundary, and safe-cell
sets, a boundary-closure flag, the next observation pose and a path, or a
status explaining why no path is available. Unknown cells are implicit and
never traversable. Evidence is counted once per cell per frame; grass needs
two observations by default, while even one boundary observation blocks a
cell. Consecutive boundary points are connected only across gaps no larger
than `boundary_link_gap`.

```python
from boundary_exploration import BoundaryExplorer, ExplorationConfig

explorer = BoundaryExplorer(ExplorationConfig(robot_radius=0.2, sensor_range=2.0))
result = explorer.update(grass_world_xy, boundary_world_xy,
                         (robot_x, robot_y, robot_yaw), timestamp_s)
if result.status == "exploring":
    send_goal_to_local_planner(result.goal_pose, result.path_xy)
```

The robot's circular footprint, pose margin, and boundary clearance are
applied before routing. Goals are selected from reachable safe cells for
their view of grass-to-unknown frontiers, with extra weight near observed
boundary. `trajectory_is_safe` and `command_is_safe` check a local planner's
proposed motion before execution. If the sensor cannot confirm grass beneath
the robot at startup, the explorer returns `waiting_for_observation`; an
externally verified initial grass disk can be supplied explicitly through
`seed_verified_grass_disk`. A pose alone does not certify grass. The algorithm
uses a circular footprint and a finite-range sensor model; callers must supply
calibrated dimensions, localization uncertainty, and camera extrinsics. A
closed confirmed boundary reports `complete`; an open contour with no reachable
safe viewpoint reports `incomplete_unreachable`. Input timestamps must be
monotonic; skip malformed or zero-timestamp sensor packets before calling
`update`.

Detailed fitting steps and direction conventions are documented in [拟合算法说明.md](拟合算法说明.md).

Run `python replay_boundary.py` to load the log and replay each frame simultaneously in global XY (left) and camera XZ (right) coordinates. The green dots are measured boundary points, the blue curves are independently estimated boundary trends, and the bright orange arrows and labels show the **tracking reference direction** near each purple curve's midpoint. The red marker and line show the robot position and heading in the global view. Both views use the same frame controls: Play/Pause, Previous/Next, the slider, or the left/right arrow keys. `--speed 2` plays at twice the logged rate.

Navigation data packages in JSONL format use a live exploration replay. For a bounded inspection of the supplied large file, run `python replay_boundary.py "D:\Downloads\0047X\navigation-data.jsonl" --max-records 1000`. The reader streams records and pairs each `visual_boundary_points` packet with the latest preceding `location` pose. Full camera XYZ coordinates are transformed into global XY before updating `BoundaryExplorer`; zero sensor timestamps fall back to the packet system time. Only the current map and 50 recent frame snapshots are retained. The left view shows confirmed grass (light green), observed boundary (red), robot-center safe cells (teal), the robot, a selected goal and safe path when available, and the original single-frame fit (blue). The right view shows camera boundary and grass points. Play/Pause, Previous/Next and arrow keys work with the bounded history; moving backward only displays a cached result and does not add evidence again. `--speed 2` doubles playback speed.

The replay uses example values `--exploration-robot-radius 0.2` and `--exploration-sensor-range 2.0`; set these to measured values before interpreting goals or paths. `--grid-resolution` controls map detail. If the starting grass patch has been independently verified, `--verified-start-radius R` seeds that patch. Without verified grass under the robot, the replay correctly reports `waiting_for_observation` even if grass is visible farther ahead. The JSONL replay displays exploration output but does not send commands to a robot.

The dashed purple line is one tracking reference shown in both views. It is computed in global XY by moving each fitted boundary point 0.4 m along its local normal toward the lawn side. Its point order is then chosen so the boundary lies on the **right** of the tracking direction. On a closed outer lawn boundary, this corresponds to counterclockwise travel with lawn on the left. Grass points determine the lawn side when available; otherwise the robot position is a provisional interior-side estimate, marked as such in the plot. Curves with bends too tight for a smooth 0.4 m inner offset fall back to a simpler trend. Consecutive frames with nearly unchanged robot position and yaw are fitted together to keep the reference stable while the robot is stationary. The global reference is then mapped into camera XZ using that frame's pose and camera extrinsics, assuming camera Y=0 as for the logged points. It is not a separate 0.4 m offset of the camera XZ fit.

The figure header separately reports the recorded pose loop direction and the direction of the sampled reference midpoints. These summaries require a nearly closed route; they report "open or incomplete" when the start/end gap is too large. A positive signed area and pose yaw increase indicate CCW in global XY. This geometric summary does not replace verification against grass coverage or the robot's driving limits.

Optional grass points can be included in each log line after `boundary_xz` as `grass_xz: [(x, z); (x, z)]`, before `location info`. Alternatively pass `--grass-csv grass.csv` with either `frame,x,y` (global XY) or `frame,camera_x,camera_z` (camera XZ) columns; `frame` is 1-based. The grass cloud is shown in light green. At least five points with 70% on a consistent side are needed to override the robot-side estimate. When enough grass points are present, the plot reports the fraction lying to the left of the directed reference and the fraction of reference samples with grass within 0.25 m. Set `--grass-support-radius` to match the real point-cloud density. These are supporting observations, not proof that every reference point is inside grass. The provided sample log has no grass points, so it cannot verify that the reference is inside the lawn.

Set `--min-turn-radius R` when the robot's minimum turning radius is known. The boundary fit is then limited to curvature at most `1 / (R + 0.4)` m⁻¹, with the existing 1.5 m⁻¹ cap retained. This is a geometric turning-radius constraint; it does not model steering transients, speed, or obstacles. Without `R`, trackability cannot be guaranteed from the point cloud alone.

Set `--robot-width W` to report the nominal distance from the tracked robot's right body edge to the fitted boundary: `0.4 - W/2` meters. For an asymmetric robot, use `--right-half-width H` instead to calculate `0.4 - H`. The options are mutually exclusive. A nonpositive value is highlighted in red. This assumes the reference tracks the robot center and ignores pose error, body shape, and corner sweep during turns. The provided log does not specify robot width.

For every frame, `boundary_shape.recognize(points_xyz)` estimates a centerline from global boundary points:

1. Use principal component analysis to find the long axis of the cloud.
2. Divide that axis into bins and take the median transverse coordinate in each populated bin.
3. Reject transverse outliers and fit a cubic, quadratic, or linear trend through the bin medians, subject to a curvature limit.
4. Calculate tangent directions, length, net turning angle, and a rough shape label: straight, left bend, right bend, or S bend.

`orient_shape(shape, robot_xy)` orders the blue centerline from the endpoint closest to the robot; its left/right shape labels refer to that order. The orange tracking arrow follows a separate right-side-boundary rule and may point opposite to the blue centerline's order. Direction angles are counterclockwise from global +X in the global view, and from camera +X toward +Z in the camera view. Frames with too few points or too little extent return `None` and display “insufficient shape data”.

The analysis is a geometric approximation for a single, mostly continuous boundary. Branches, loops, widely separated boundaries, and heavy occlusion can make a single trend misleading. The labels and thresholds have not been calibrated against annotated ground truth.
